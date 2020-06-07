from MLDL.models import FrankenCaRL
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gc
from MLDL.utils import *

class TherapyFrankenCaRL(FrankenCaRL):
    def __init__(self, net, SpecialistModel, K=2000, custom_loss=None, loss_params=None, use_exemplars=True, distillation=True, all_data_means=True):
        super().__init__(net, K=K, custom_loss=custom_loss, loss_params=loss_params, use_exemplars=use_exemplars, distillation=distillation, all_data_means=all_data_means)
        self.SpecialistModel = SpecialistModel
        self.specialist_yellow_pages = dict()


    def train_specialist(self, dataset):
        """
        Trains a specialist using the dataset, assigns it to the corresponding mapping.
        Returns:
            A sound mind
        """
        incoming_labels = np.unique(dataset.targets)
        print(f"Training specialist for labels {incoming_labels}...")
        specialist = self.SpecialistModel().to(self.DEVICE)

        # The parameters of iCaRL are used in the model, except for the number of epochs
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        optimizer = optim.SGD(specialist.parameters(), lr=0.2, weight_decay=5e-5, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)

        dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, drop_last=False)

        NUM_EPOCHS_SPECIALIST = 30
        for epoch in range(NUM_EPOCHS_SPECIALIST):
          # print(f'SPECIALIST EPOCH {epoch+1}/{NUM_EPOCHS_SPECIALIST}, LR = {scheduler.get_last_lr()}')

          mean_loss_epoch = 0
          for images, labels in dataloader:
            images = images.to(self.DEVICE)
            labels = labels.to(self.DEVICE)

            specialist.train()
            optimizer.zero_grad()

            outputs = specialist(images)

            # One hot encoding labels for binary cross-entropy loss
            labels_onehot = nn.functional.one_hot(labels, 100).type_as(outputs)
            loss = criterion(outputs, labels_onehot).sum(dim=1).mean()

            mean_loss_epoch += loss.item()
            loss.backward()
            optimizer.step()
            # --- end batch
          scheduler.step()
          # print(f"Mean batch loss: {mean_loss_epoch/len(dataloader):.5}")
          # --- end epoch

        torch.cuda.empty_cache()
        # Append specialist to appropriate values in the dictionary
        specialist.eval()
        self.specialist_yellow_pages[tuple(incoming_labels)] = specialist


    def classify(self, X, num_min_distances=3, specialist_confront=True):
      """
      Does classification, optionally by asking specialists.

      """
      torch.cuda.empty_cache()
      with torch.no_grad():
        self.net.eval()
        # Compute feature mappings of batch
        X = X.to(self.DEVICE)
        phi_X = self.net.feature_extractor(X)

        # Find nearest mean for each phi_x
        labels = []
        ex_means = torch.stack(self.class_means)
        for i, x in enumerate(phi_X): # changed from norm_phi_X
            # broadcasting x to shape of exemaplar_means
            distances_from_class = (ex_means - x).norm(dim=1)
            closest_classes = torch.argsort(distances_from_class)[:num_min_distances]
            # Clostest classes are now the candidates classes. We ask the corresponding specialists for consultation
            # Pass the unmapped x to the specialist
            if specialist_confront:
                y = self.ask_specialists_confront(X[i], closest_classes)
            else:
                y = self.ask_specialists_hard(X[i], closest_classes)
            labels.append(y)
        labels = torch.stack(labels).type(torch.long)
        torch.cuda.empty_cache
        return labels

    def KL(self, P, Q):
        return (P * (P / Q).log()).sum()

    def classify_fc(self, X, num_ask_specialists=3, unique_specialist=True):
        """
        Classify using fc and asking specialists

        Returns:
            Labels predicted for batch X as a torch.Tensor
        """
        self.net.eval()

        X = X.to(self.DEVICE)

        # Forward images and get the most activated label(s)
        pred_labels = []
        for x in X:
            activations = self.net(x.unsqueeze(0)).squeeze()[:self.num_tot_classes]
            pg = nn.functional.softmax(activations, dim=0)

            most_active_labels = torch.argsort(pg, descending=True)[:num_ask_specialists]

            if unique_specialist:
                to_ask_labels = np.unique(np.floor(most_active_labels.cpu().numpy() / 10).astype(int) * 10)
            else:
                to_ask_labels = most_active_labels

            specialist_opinions = [self.get_specialist_opinion(x, label) for label in to_ask_labels] # list of prob distros

            # Minimize KL divergenge starting from uniform distro or something
            q = (torch.ones(pg.size()) * 1/pg.size(0)).to(self.DEVICE)
            q.requires_grad = True
            # Use sgd to minimize KL distance
            sgd = optim.SGD([q], lr=0.01)
            for i in range(100):
                sgd.zero_grad()
                qp = nn.functional.softmax(q, dim=0)
                loss = self.KL(pg.detach(), qp) + sum([self.KL(pm.detach(), qp) for pm in specialist_opinions])
                loss.backward()
                sgd.step()
            # Security check
            if torch.isnan(qp).any():
                print(f"qp went to nan while minimizing KL!")
            final_prob = nn.functional.softmax(qp, dim=0)
            # We can now take the highest probability
            pred_label = torch.argmax(final_prob)
            pred_labels.append(pred_label.squeeze().item())
        # concatenate the predicted labels and return them as tensor
        return torch.Tensor(pred_labels).type(torch.long)


    def get_specialist_opinion(self, x, candidate_label):
        """
        Given image x and a candidate label, return the distribution given by its specialist.
        Thus, returs a probability distribution pm.
        """
        for specialized_labels, specialist in self.specialist_yellow_pages.items():
            if candidate_label in specialized_labels:
                break # dirty trick: break to keep right specialist and specialized labels
        # We now have the right specialist
        activation = specialist(x.unsqueeze(0)).squeeze()[:self.num_tot_classes]
        pm = nn.functional.softmax(activation, dim=0)
        return pm


    def test_fc(self, test_dataset):
        """
        Overrides FrankenCaRL's default test_fc method by using specialist classify.
        Computes accuracy over entire test set and prints it onscreen.
        Interestingly enough, does not return any stupid stuff.
        """
        self.net.eval()

        test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=4)
        running_corrects = 0
        t = self.num_tot_classes
        matrix = new_confusion_matrix(lenx=t, leny=t)

        for images, labels in test_dataloader:
            # print(f"Test labels: {np.unique(labels.numpy())}")
            images = images.to(self.DEVICE)
            # labels = labels.to(self.DEVICE)

            preds = self.classify_fc(images)

            update_confusion_matrix(matrix, preds, labels)

            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()

        # Calculate Accuracy and mean loss
        accuracy = running_corrects / len(test_dataloader.dataset)
        self.accuracies_fc.append(accuracy)
        print(f'\033[94mAccuracy on test set with fc :{accuracy}\x1b[0m')
        show_confusion_matrix(matrix)


    def ask_specialists_hard(self, x, candidate_classes):
        """
        Call the specialists of the candidate classes and ask for therapy to help in the prediction.
        Returns:
            Some peace of mind, hopefully. Also the predicted label.
        """
        probabilities = []
        for candidate_class in candidate_classes:
            # Find the appropriate specialist
            for specialized_labels, specialist in self.specialist_yellow_pages.items():
                if candidate_class in specialized_labels:
                    break # dirty trick: break to keep right specialist and specialized labels

            # Forward the image into the specialist
            out_prob = nn.functional.softmax(specialist(x.unsqueeze(0)).squeeze(), dim=0)[candidate_class]
            out_prob = out_prob.cpu().item()
            probabilities.append(out_prob)

        # Now take the highest probability
        predicted_label = candidate_classes[np.argmax(probabilities)]
        # print(f"Calling specialist resulted in {predicted_label}. ", end="")
        return predicted_label

    def ask_specialists_confront(self, x, candidate_classes, trust_specialists=False):
        """
        Given a list of candidate classes, in order, return the first label that matches the specialist prediction.
        If none match, return ...?

        Params:
            candidate_classes: candidate labels, sorted by relevance
        Returns:
            Predicted label after consulting all specialists (i.e. the final prediction)
        """
        specialist_strongest_opinion = (-1, 0.0)
        for candidate_class in candidate_classes:
            # Find the appropriate specialist
            for specialized_labels, specialist in self.specialist_yellow_pages.items():
                if candidate_class in specialized_labels:
                    break # dirty trick: break to keep right specialist and specialized labels

            # Forward the image into the specialist and take SPECIALIST'S prediction
            out_disto = nn.functional.softmax(specialist(x.unsqueeze(0)).squeeze(), dim=0)
            specialist_prediction = out_disto.argmax()
            specialist_prediction_prob = out_disto[specialist_prediction]
            if specialist_prediction_prob > specialist_strongest_opinion[1]:
                specialist_strongest_opinion = (specialist_prediction, specialist_prediction_prob)
            # If specialist and general model predictions match, confirm the prediction
            if specialist_prediction == candidate_class:
                return candidate_class

        if trust_specialists:
            # If the loop did not find any match between general model and specialists, return the strongest opinion amongst all specialists
            return specialist_strongest_opinion[0]
        else:
            # If the loop did not find any match between general model and specialist, return whatever the general model would have returned (i.e. the most relevant label)
            return candidate_classes[0]


    def incremental_train(self, train_dataset, train_dataset_no_aug, test_dataset):
        labels = train_dataset.targets
        new_classes = np.unique(labels)
        print(f'Arriving new classes {new_classes}')

        # Compute number of total labels
        num_old_labels = len(self.exemplar_sets)
        num_new_labels = len(new_classes)

        t = num_old_labels + num_new_labels

        self.update_representation(train_dataset)
        self.train_specialist(train_dataset)

        m = int(self.K/t)
        D = self.reduce_exemplar_set(m=m)

        gc.collect()

        for label in new_classes:
          bool_idx = (train_dataset_no_aug.targets == label)
          idx = np.argwhere(bool_idx).flatten()
          print(f'Constructing exemplar set for label {label} (memory: {len(gc.get_objects())})')
          images_of_y = []

          for single_index in idx:
            img, label = train_dataset_no_aug[single_index]
            images_of_y.append(img)

          images_of_y = torch.stack(images_of_y)

          if self.use_exemplars:
            self.construct_exemplar_set(X=images_of_y, y=label, m=m)

          if self.all_data_means:
            self.compute_class_means_with_training(images_of_y)

          torch.no_grad()
          gc.collect()
          del bool_idx
          del idx


        if not self.all_data_means:
          self.compute_exemplars_means()

        #if self.use_exemplars:
        #    self.test_ncm(test_dataset)
        self.test_fc(test_dataset)
