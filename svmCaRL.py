from sklearn.svm import SVC

class SVMiCaRL():
    """
    Implements iCaRL + SVM, i don't even know if it's the right one
    """
    def __init__(self, net, K=2000, custom_loss=None, loss_params=None, use_exemplars=True, distillation=True, all_data_means=True):
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

        # Internal flags to set FrankenCaRL's behavior
        self.use_exemplars = use_exemplars
        self.distillation = distillation
        self.all_data_means = all_data_means
        self.svm = SVC(kernel='linear', tol=0.0001)

        # Keep internal copy of the network
        self.net = deepcopy(net).to(self.DEVICE)

        # Other internal parameters
        self.num_tot_classes = 0
        self.accuracies_NCM = []
        self.accuracies_FC = []
        self.accuracies_SVM = []
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
        if self.use_exemplars:
            D = MergeDataset(train_dataset, exemplars_dataset, augment2=False)
        else:
            D = train_dataset

        # If this is not the first training, we save the old network

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

                # One hot encoding labels for binary cross-entropy loss
                labels_onehot = nn.functional.one_hot(labels, self.num_tot_classes).type_as(outputs)

                if num_old_classes == 0 or not self.distillation:
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

    def train_SVM(self, dataset):
        print('Training SVM...')
        svc = SVC(kernel='linear', tol=0.0001)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False, drop_last=False, num_workers=4)

        with torch.no_grad():
            self.net.eval()
            labels_list = []
            fts_list = []
            for images, labels in dataloader:
                fts_map = self.net.feature_extractor(images.to(self.DEVICE))
                labels_list.append(labels)
                fts_list.append(fts_map.cpu())

            all_labels = torch.cat(labels_list)
            all_fts = torch.cat(fts_list)
            svc.fit(all_fts, all_labels)
            self.svc = svc

    def construct_exemplar_set_SVM(self, D, m):
        support_indexes = self.svc.support_

        X = torch.stack([img for img, _ in D])
        y = np.array([label for _, label in D])

        # We obliterate the previous exemplars set and forge a new one by the power of support vectors
        self.exemplar_sets = []

        print(f'y : {y}')
        # Note that unique sorts labels so the index-label correspondence is mantenuta
        for label in np.unique(y):
            print(f'Creating exemplar set for label {label} with support vectors')
            idx = (y[support_indexes] == int(label))
            support_set = X[support_indexes][idx]
            support_set = support_set[:m]
            self.exemplar_sets.append(support_set)

    def train_SVM_for_classification(self):
        """
        Non prende niente a parametro.
        Trains SVM only on exemlars for classification purposes.
        Won't work.
        """
        # Concatenate current exemplar sets with respective labels
        exemplars_dataset = []
        for label, exemplar_set in enumerate(self.exemplar_sets):
            for exemplar in exemplar_set:
                exemplars_dataset.append((exemplar, label))

        svc = SVC(kernel='linear', tol=0.0001)
        dataloader = DataLoader(exemplars_dataset, batch_size=100, shuffle=False, drop_last=False, num_workers=4)
        with torch.no_grad():
            self.net.eval()
            labels_list = []
            fts_list = []
            for images, labels in dataloader:
                fts_map = self.net.feature_extractor(images.to(self.DEVICE))
                labels_list.append(labels)
                fts_list.append(fts_map.cpu())

            all_labels = torch.cat(labels_list)
            all_fts = torch.cat(fts_list)
            svc.fit(all_fts, all_labels)
            self.svc = svc

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
            bool_idx = torch.sum(X!=x, dim=1).type(torch.bool)
            all_X_except_x = X[bool_idx]
            all_distances_squared = (x - all_X_except_x).pow(2).sum(dim=1)
            denominator = torch.exp(-all_distances_squared/T).sum()

            all_y_except_x = labels[bool_idx]
            X_same_label_as_x = all_X_except_x[all_y_except_x != labels[i]] # This is very readable
            all_distances_squared = (x - X_same_label_as_x).pow(2).sum(dim=1)
            numerator = torch.exp(-all_distances_squared/T).sum()

            x_contribution_to_loss = torch.log(numerator/denominator)
            all_logs.append(x_contribution_to_loss)
        # Sum all contributions (outer sum)
        b = images.size(0)
        loss = - torch.Tensor(all_logs).sum() / b
        return loss



    def incremental_train(self, train_dataset, train_dataset_no_aug, test_dataset):
        labels = train_dataset.targets
        new_classes = np.unique(labels)
        print(f'Arriving new classes {new_classes}')

        # Compute number of total labels
        num_old_labels = len(self.exemplar_sets)
        num_new_labels = len(new_classes)

        t = num_old_labels + num_new_labels
        D = self.update_representation(train_dataset)
        # SVM is first trained to select exemplars
        self.train_SVM(D)
        m = int(self.K/t)
        self.construct_exemplar_set_SVM(D, m)
        self.compute_exemplars_means()

        # We now retrain it to classify based on exemplars only
        self.train_SVM_for_classification()


        self.test_SVM(test_dataset)
        self.test_FC(test_dataset)
        self.test_NCM(test_dataset)

    def test_NCM(self, test_dataset):
        self.net.eval()
        with torch.no_grad():
            test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
            running_corrects = 0
            t = self.num_tot_classes
            matrix = new_confusion_matrix(lenx=t, leny=t)
            tot_loss = 0
            for images, labels in test_dataloader:
                # print(f"Test labels: {np.unique(labels.numpy())}")
                images = images.to(self.DEVICE)
                labels = labels.to(self.DEVICE)

                # Get prediction with  NMC
                preds = self.classify_NCM(images).to(self.DEVICE)

                # Update Corrects
                running_corrects += torch.sum(preds == labels.data).data.item()
                update_confusion_matrix(matrix, preds, labels)

            # Calculate Accuracy and mean loss
            accuracy = running_corrects / len(test_dataloader.dataset)
            self.accuracies_NCM.append(accuracy)
            print(f'\033[94mAccuracy on test set with NMC :{accuracy}\x1b[0m')
            show_confusion_matrix(matrix)

    def test_FC(self, test_dataset):
        self.net.eval()
        with torch.no_grad():
            test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
            running_corrects = 0
            t = self.num_tot_classes
            matrix = new_confusion_matrix(lenx=t, leny=t)
            tot_loss = 0
            for images, labels in test_dataloader:
                # print(f"Test labels: {np.unique(labels.numpy())}")
                images = images.to(self.DEVICE)
                labels = labels.to(self.DEVICE)

                outputs = self.net(images)[:, :self.num_tot_classes]
                _, preds = torch.max(nn.Softmax(outputs).dim, 1)

                update_confusion_matrix(matrix, preds, labels)

                # Update Corrects
                running_corrects += torch.sum(preds == labels.data).data.item()

            # Calculate Accuracy and mean loss
            accuracy = running_corrects / len(test_dataloader.dataset)
            self.accuracies_FC.append(accuracy)
            print(f'\033[94mAccuracy on test set with fc :{accuracy}\x1b[0m')
            show_confusion_matrix(matrix)

    def test_SVM(self, test_dataset):
        self.net.eval()
        with torch.no_grad():
            test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
            running_corrects = 0
            t = self.num_tot_classes
            matrix = new_confusion_matrix(lenx=t, leny=t)
            tot_loss = 0
            for images, labels in test_dataloader:
                # print(f"Test labels: {np.unique(labels.numpy())}")
                images = images.to(self.DEVICE)
                fts_map = self.net.feature_extractor(images)
                preds = self.svc.predict(fts_map.cpu())

                update_confusion_matrix(matrix, torch.Tensor(preds).type_as(labels), labels)

                # Update Corrects
                running_corrects += torch.sum(torch.Tensor(preds) == labels.data).data.item()

            # Calculate Accuracy and mean loss
            accuracy = running_corrects / len(test_dataloader.dataset)
            self.accuracies_SVM.append(accuracy)
            print(f'\033[94mAccuracy on test set with SVM:{accuracy}\x1b[0m')
            show_confusion_matrix(matrix)
