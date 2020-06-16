from sklearn.neighbors import KNeighborsClassifier
import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch import optim
from torchvision.transforms import Compose
from torchvision import transforms
from torchvision import models
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import gc
from copy import deepcopy
from MLDL.utils import *
from MLDL.datasets.project_dataset1 import MergeDataset


class KNNiCaRL():
    """
    FrankenCaRL's evil brother, KNNiCaRL. Instead of a refined, mathematically sound NCM, uses a rougher and more barbaric KNN as classifier.
    (it can also run a basic NCM and hybrid1 FC setup though)
    By default, KNN is trained on the exemplars only.
    Also by defualt, parameter K of the KNN is proportional to the size of the exemplars , and thus changes as training batches arrive.
    Stop kNN shaming. They are beautiful too.

    *** Internally has a method for random herding, but no options to activate it are currently available (god knows why) ***
    """
    def __init__(self, net, K=2000, custom_loss=None, loss_params=None, remove_duplicates=True):
        self.exemplar_sets = []
        self.class_means = []
        self.K = K
        # Inizializing with parameters from paper ** insert reference **
        self.MOMENTUM = 0.9
        self.LR = 2
        self.BATCH_SIZE = 128
        self.MILESTONE = [48, 62]
        self.WEIGHT_DECAY = 1e-5
        self.GAMMA = 0.2
        self.NUM_EPOCHS = 70
        self.DEVICE = 'cuda'

        self.exemplar_dataset = []
        self.remove_duplicates = remove_duplicates

        # Keep internal copy of the network
        self.net = deepcopy(net).to(self.DEVICE)

        # Other internal parameters
        self.num_tot_classes = 0
        self.accuracies = {
        'accuracy_nmc': [],
        'accuracy_fc': [],
        'accuracy_knn': [],
        'accuracy_nmc_old': [],
        'accuracy_nmc_new': [],
        'accuracy_fc_old': [],
        'accuracy_fc_new': [],
        'accuracy_knn_old': [],
        'accuracy_knn_new': []
    }
        # Set loss to use
        self.custom_loss = custom_loss

        if loss_params is None:
            self.loss_params = {}
        else:
            self.loss_params = loss_params

    def set_params(self, params):
        self.MOMENTUM = params['MOMENTUM']
        self.LR = params['LR']
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.MILESTONE = params['MILESTONE']
        self.WEIGHT_DECAY = params['WEIGHT_DECAY']
        self.GAMMA = params['GAMMA']
        self.NUM_EPOCHS = params['NUM_EPOCHS']

    def compute_exemplars_means(self):
        """
        Compute means of exemplars and store them in a class variable.
        Returns:
        Gandalf's pointy hat
        """
        # First obtain feature extractor
        self.net.eval()

        with torch.no_grad():
            self.class_means = []
            for label, Py in enumerate(self.exemplar_sets):
                print(f"Computing means for label {label}")
                # show_image_label(Py[random.choice(range(len(Py)))], label)

                phi_Py = self.net.feature_extractor(Py.to(self.DEVICE))
                #print(f"FOR DEBUG -- norm of mapped exemplar set {phi_Py.norm(dim=1)}")
                mu_y = phi_Py.mean(dim = 0)
                mu_y.data = mu_y.data / mu_y.data.norm()
                self.class_means.append(mu_y)

    def classify_NCM(self, X):
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.net.eval()
            # Compute feature mappings of batch
            X = X.to(self.DEVICE)
            phi_X = self.net.feature_extractor(X)
            # Normalize each mapped input
            norm_phi_X = []

            # Find nearest mean for each phi_x
            labels = []
            ex_means = torch.stack(self.class_means)
            for x in phi_X: # changed from norm_phi_X
                # broadcasting x to shape of exemaplar_means
                distances_from_class = (ex_means - x).norm(dim=1)
                y = distances_from_class.argmin()
                labels.append(y)

            labels = torch.stack(labels).type(torch.long)
            torch.cuda.empty_cache
            return labels


    def update_representation(self, train_dataset):
        """
        Update something
        Returns:
        La bici di Bibbona
        """
        old_net = deepcopy(self.net)
        # Concatenate current exemplar sets with respective labels
        exemplars_dataset = []
        for label, exemplar_set in enumerate(self.exemplar_sets):
            for exemplar in exemplar_set:
                exemplars_dataset.append((exemplar, label))


        num_old_classes = len(self.exemplar_sets)
        num_new_classes = len(np.unique(train_dataset.targets))
        num_tot_classes = num_old_classes + num_new_classes
        self.num_tot_classes = num_tot_classes

        # Create big D dataset
        D = MergeDataset(train_dataset, exemplars_dataset, augment2=False)
        old_net = deepcopy(self.net)

        optimizer = optim.SGD(self.net.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY, momentum=self.MOMENTUM)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.MILESTONE, gamma=self.GAMMA)

        criterion = nn.BCEWithLogitsLoss(reduction='none')

        dataloader = DataLoader(D, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)

        for epoch in range(self.NUM_EPOCHS):
            print(f'EPOCH {epoch+1}/{self.NUM_EPOCHS}, LR = {scheduler.get_last_lr()}')

            mean_loss_epoch = 0
            for images, labels in dataloader:
                images = images.to(self.DEVICE)
                labels = labels.to(self.DEVICE)

                self.net.train()
                optimizer.zero_grad()

                outputs = self.net(images)[:, :num_tot_classes]
                labels_onehot = nn.functional.one_hot(labels, self.num_tot_classes).type_as(outputs)

                if num_old_classes == 0:
                    loss = criterion(outputs, labels_onehot).sum(dim=1).mean()
                else:
                    labels_onehot = labels_onehot.type_as(outputs)[:, num_old_classes:]
                    out_old = torch.sigmoid(old_net(images))[:,:num_old_classes]
                    target = torch.cat((out_old, labels_onehot), dim=1)
                    loss = criterion(outputs, target).sum(dim=1).mean()

                soft_margin_loss = self.soft_nearest_mean_class_loss(images, labels, old_net)

                mean_loss_epoch += loss.item()
                total_loss = loss + soft_margin_loss
                total_loss.backward()
                optimizer.step()
                # --- end batch
            scheduler.step()
            print(f"Mean batch loss: {mean_loss_epoch/len(dataloader):.5}")
            # --- end epoch

        torch.cuda.empty_cache()
        return D

    def train_KNN(self, n_neighbors=50):
        """
        The function that performs the training, after the network training on the KNN
        Uses exemplars as training set.
        """
        print(f'Training KNN with {n_neighbors} neighbors (ex. set size: {len(self.exemplar_sets[0])})')
        exemplars_dataset = []
        for label, exemplar_set in enumerate(self.exemplar_sets):
            for exemplar in exemplar_set:
                exemplars_dataset.append((exemplar, label))

        self.knn = KNeighborsClassifier(n_neighbors)
        dataloader = DataLoader(exemplars_dataset, batch_size=100, shuffle=False, drop_last=False, num_workers=4)

        with torch.no_grad():
            labels_list = []
            fts_list = []
            for images, labels in dataloader:
                fts_map = self.net.feature_extractor(images.to(self.DEVICE))
                labels_list.append(labels)
                fts_list.append(fts_map.cpu())

            all_labels = torch.cat(labels_list)
            all_fts = torch.cat(fts_list)
            self.knn.fit(all_fts, all_labels)

    def random_construct_exemplar_set(self, X, y, m):
        """
        X only contains elements of a single label y
        """
        with torch.no_grad():
            indexes = torch.randperm(X.size(0))[:m]
            exemplar_set = X[indexes]
            self.exemplar_sets.append(exemplar_set)

    def incremental_train(self, train_dataset, test_dataset, n_neighbors=3/4):
        """
        Params:
          n_neighbors: if float, fraction of exemplars sets to use for K.
            if int, value to use for K.
        """
        labels = train_dataset.targets
        new_classes = np.unique(labels)
        print(f'Arriving new classes {new_classes}')

        # Compute number of total labels
        num_old_labels = len(self.exemplar_sets)
        num_new_labels = len(new_classes)

        t = num_old_labels + num_new_labels
        D = self.update_representation(train_dataset)

        m = int(self.K/t)
        self.reduce_exemplar_set(m=m)

        for label in new_classes:
            bool_idx = (train_dataset.targets == label)
            idx = np.argwhere(bool_idx).flatten()
            print(f'Constructing exemplar set for label {label} (memory: {len(gc.get_objects())})')
            images_of_y = []

            for single_index in idx:
                img, label = train_dataset[single_index]
                images_of_y.append(img)

            images_of_y = torch.stack(images_of_y)
            self.construct_exemplar_set(X=images_of_y, y=label, m=m)

        if isinstance(n_neighbors, int):
          internal_n_neighbors = n_neighbors
        else:
          internal_n_neighbors = int(m * n_neighbors)

        self.train_KNN(internal_n_neighbors)
        self.compute_exemplars_means()

        self.test_knn(test_dataset, num_old_labels)
        self.test_fc(test_dataset, num_old_labels)
        self.test_nmc(test_dataset, num_old_labels)

    def reduce_exemplar_set(self, m):
      """
      The function reduces the number of images for each exampler set at m
      Params:
        m: number of elements that has to be collected
      Return:
        the list of exemplar_sets updated
      """
      for i, exemplar_set in enumerate(self.exemplar_sets):
        self.exemplar_sets[i] = exemplar_set[:m]
      return self.exemplar_sets


    def construct_exemplar_set(self, X, y, m):
      """
      X only contains elements of a single label y
      """
      with torch.no_grad():
        self.net.eval()

        # Compute class mean of X
        loader = DataLoader(X,batch_size=self.BATCH_SIZE, shuffle=True, drop_last=False, num_workers = 4)
        phi_X = []
        for images in loader:
          images = images.to(self.DEVICE)
          phi_X_batch = self.net.feature_extractor(images)
          phi_X.append(phi_X_batch)
          del images

        phi_X = torch.cat(phi_X).to('cpu')
        mu_y = phi_X.mean(dim=0)

        Py = []
        size_mapped_images = phi_X[0].size()[0]
        # Accumulates sum of exemplars
        sum_taken_exemplars = torch.zeros(1, phi_X.size()[1])

        for k in range(1, int(m+1)):
          # Using broadcast: expanding mu_y and sum_taken_exemplars to phi_X shape
          mean_distances = (mu_y - (1/k)*(phi_X + sum_taken_exemplars)).norm(dim=1)
          min_index = mean_distances.argmin(dim=0).item()
          p = X[min_index]
          Py.append(p)
          p = p.unsqueeze(0)
          phi_p = self.net.feature_extractor(p.to(self.DEVICE))
          sum_taken_exemplars = sum_taken_exemplars + phi_p.to('cpu')
          if self.remove_duplicates:
            X = torch.cat((X[:min_index], X[min_index+1:]), dim = 0)
            phi_X = torch.cat((phi_X[:min_index], phi_X[min_index+1:]), dim = 0)
          del phi_p

        Py = torch.stack(Py)
        self.exemplar_sets.append(Py) # for dictionary version: self.exemplar_sets[y] = Py

    

    def test_nmc(self, test_dataset, num_old_classes):
        self.net.eval()
        test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
        running_corrects = 0
        old_corrects = 0
        n_old = 0
        t = self.num_tot_classes
        matrix = new_confusion_matrix(lenx=t, leny=t)
        tot_loss = 0
        for images, labels in test_dataloader:
            # print(f"Test labels: {np.unique(labels.numpy())}")
            images = images.to(self.DEVICE)
            labels = labels.to(self.DEVICE)
            old_idx = (labels.cpu().numpy() < num_old_classes)
            # Get prediction with  NMC
            preds = self.classify_NCM(images).to(self.DEVICE)
            # Update Corrects
            old_corrects += torch.sum(preds[old_idx] == labels[old_idx].data).data.item()
            n_old += np.sum(old_idx)
            running_corrects += torch.sum(preds == labels.data).data.item()
            update_confusion_matrix(matrix, preds, labels)

        # Calculate Accuracy and mean loss
        accuracy = running_corrects / len(test_dataloader.dataset)
        old_accuracy = old_corrects / n_old
        new_corrects = running_corrects - old_corrects
        new_accuracy = new_corrects / (len(test_dataloader.dataset) - n_old)

        self.accuracies['accuracy_nmc'].append(accuracy)
        self.accuracies['accuracy_nmc_old'].append(old_accuracy)
        self.accuracies['accuracy_nmc_new'].append(new_accuracy)
        print(f'\033[94mAccuracy on test set with NMC :{accuracy}\x1b[0m')
        print(f'\033[94mOld accuracy on test set with NMC :{old_accuracy}\x1b[0m')
        print(f'\033[94mNew accuracy on test set with NMC :{new_accuracy}\x1b[0m')
        show_confusion_matrix(matrix)

    def test_fc(self, test_dataset, num_old_classes):
        self.net.eval()
        test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
        running_corrects = 0
        old_corrects = 0
        n_old = 0
        t = self.num_tot_classes
        matrix = new_confusion_matrix(lenx=t, leny=t)
        tot_loss = 0
        for images, labels in test_dataloader:
            # print(f"Test labels: {np.unique(labels.numpy())}")
            images = images.to(self.DEVICE)
            labels = labels.to(self.DEVICE)
            old_idx = (labels.cpu().numpy() < num_old_classes)

            if self.custom_loss is not None and self.custom_loss.__name__ == 'less_forget_loss':
                outputs = self.net.forward_cosine(images)[:,:self.num_tot_classes]
            else:
                outputs = self.net(images)[:,:self.num_tot_classes]
            _, preds = torch.max(outputs.data, 1)

            update_confusion_matrix(matrix, preds, labels)

            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()
            old_corrects += torch.sum(preds[old_idx] == labels[old_idx].data).data.item()
            n_old += np.sum(old_idx)
        # Calculate Accuracy and mean loss
        accuracy = running_corrects / len(test_dataloader.dataset)
        old_accuracy = old_corrects / n_old
        new_corrects = running_corrects - old_corrects
        new_accuracy = new_corrects / (len(test_dataloader.dataset) - n_old)
        self.accuracies['accuracy_fc'].append(accuracy)
        self.accuracies['accuracy_fc_old'].append(old_accuracy)
        self.accuracies['accuracy_fc_new'].append(new_accuracy)
        print(f'\033[94mAccuracy on test set with FC :{accuracy}\x1b[0m')
        print(f'\033[94mOld accuracy on test set with FC :{old_accuracy}\x1b[0m')
        print(f'\033[94mNew accuracy on test set with FC :{new_accuracy}\x1b[0m')
        show_confusion_matrix(matrix)


    def test_knn(self, test_dataset, num_old_classes):
        self.net.eval()
        with torch.no_grad():
            test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
            running_corrects = 0
            old_corrects = 0
            n_old = 0
            t = self.num_tot_classes
            matrix = new_confusion_matrix(lenx=t, leny=t)
            tot_loss = 0
            for images, labels in test_dataloader:
                # print(f"Test labels: {np.unique(labels.numpy())}")
                images = images.to(self.DEVICE)
                fts_map = self.net.feature_extractor(images)
                preds = self.knn.predict(fts_map.cpu())
                old_idx = (labels.cpu().numpy() < num_old_classes)
                update_confusion_matrix(matrix, torch.Tensor(preds).type_as(labels), labels)

                # Update Corrects
                running_corrects += torch.sum(torch.Tensor(preds) == labels.data).data.item()
                old_corrects += torch.sum(torch.Tensor(preds)[old_idx] == labels[old_idx].data).data.item()
                n_old += np.sum(old_idx)

            # Calculate Accuracy and mean loss
            accuracy = running_corrects / len(test_dataloader.dataset)
            old_accuracy = old_corrects / n_old
            new_corrects = running_corrects - old_corrects
            new_accuracy = new_corrects / (len(test_dataloader.dataset) - n_old)
            self.accuracies['accuracy_fc'].append(accuracy)
            self.accuracies['accuracy_fc_old'].append(old_accuracy)
            self.accuracies['accuracy_fc_new'].append(new_accuracy)
            print(f'\033[94mAccuracy on test set with FC :{accuracy}\x1b[0m')
            print(f'\033[94mOld accuracy on test set with FC :{old_accuracy}\x1b[0m')
            print(f'\033[94mNew accuracy on test set with FC :{new_accuracy}\x1b[0m')
            show_confusion_matrix(matrix)

    def soft_nearest_mean_class_loss(self, images, labels, old_net, T=2):
        """
        Compute soft nearest mean class loss, which has been proven to have the longest name in all loss functions history.
        This is probably the only goal we'll achieve with that.
        Returns:
            loss as a scalar for the whole batch, ready to call backward on
        """
        self.net.eval()
        X = self.net.feature_extractor(images)

        all_logs = []
        for i, x in enumerate(X):
            #for the DENOMINATOR
            bool_idx = torch.sum(X!=x, dim=1).type(torch.bool)
            # extracting all the images that not corrispond to x
            all_X_except_x = X[bool_idx]
            # computing the square distance
            all_distances_squared = (x - all_X_except_x).pow(2).sum(dim=1)
            denominator = torch.exp(-all_distances_squared/T).sum()
            # for the NUMERATOR
            # extracting all the labels of images different from x
            all_y_except_x = labels[bool_idx]
            # finding all images with the same label of x
            X_same_label_as_x = all_X_except_x[all_y_except_x != labels[i]]
            # computing the square distance
            all_distances_squared = (x - X_same_label_as_x).pow(2).sum(dim=1)
            numerator = torch.exp(-all_distances_squared/T).sum()

            x_contribution_to_loss = torch.log(numerator/denominator)
            all_logs.append(x_contribution_to_loss)
        # Sum all contributions (outer sum)
        b = images.size(0)
        loss = - torch.Tensor(all_logs).sum() / b
        return loss
