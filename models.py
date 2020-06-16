import torch
from torch.utils.data import Subset, DataLoader, TensorDataset, Dataset
from torch.backends import cudnn
import torch.optim as optim
from torchvision.transforms import Compose
from torchvision import transforms
import torchvision.models as models
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
from MLDL.loss_variations import soft_nearest_mean_class_loss
from MLDL.exemplars_generator import generate_exemplars_smote, generate_images_with_network, generate_new_image, generate_exemplar_max_activation

class FrankenCaRL():
  """
  Implements iCaRL as decribed in *insert paper* (the actual name of the paper is *insert paper*)

  The behavior of "distillation" flag is overridden if a custom loss is used.
  """
  def __init__(self, net, K=2000, custom_loss=None, loss_params=None, use_exemplars=True, distillation=True, all_data_means=True, remove_duplicates=True, soft_nm=False, exemplars_generator='none'):
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
    self.remove_duplicates = remove_duplicates
    self.soft_nm = soft_nm

    # Keep internal copy of the network
    self.net = deepcopy(net).to(self.DEVICE)

    # set the generator of exemplars
    if exemplars_generator is 'max_act':
        self.exemplars_generator = generate_exemplar_max_activation
    elif exemplars_generator is 'mean':
        self.exemplars_generator = generate_new_image
    elif exemplars_generator is 'smote':
        self.exemplars_generator = 'smote'
    elif exemplars_generator is 'network':
        self.exemplars_generator = generate_images_with_network
    else:
        self.exemplars_generator = 'none'

    # Other internal parameters
    self.num_tot_classes = 0
    self.accuracies = {
        'accuracy_nmc': [],
        'accuracy_fc': [],
        'accuracy_nmc_old': [],
        'accuracy_nmc_new': [],
        'accuracy_fc_old': [],
        'accuracy_fc_new': []
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


  def compute_class_means_with_training(self, X):
    """
      Compute class means with data passed as argument
      Params:
        - X: images that belong to a single class label
      Returns:
        A delicious ham sandwich
    """
    self.net.eval()

    with torch.no_grad():
      phi_X = self.net.feature_extractor(X.to(self.DEVICE))
      mean = phi_X.mean(dim = 0)
      mean.data = mean.data / mean.data.norm()
      self.class_means.append(mean)

  def classify(self, X):

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
      The merged dataset D
    """

    # Concatenate current exemplar sets with respective labels
    exemplars_dataset = []
    for label, exemplar_set in enumerate(self.exemplar_sets):
      for exemplar in exemplar_set:
        exemplars_dataset.append((exemplar, label))

      if self.exemplars_generator is not 'none' and self.exemplars_generator is not 'smote':
        X = exemplar_set[:100]
        new_images = self.exemplars_generator(self, label, 500 - exemplar_set.size(0) , X)

        for new_image in new_images:
          exemplars_dataset.append((new_image, label))

    num_new_classes = len(np.unique(train_dataset.targets))
    #if use_exemplars:
    #  num_old_classes = len(self.exemplar_sets)
    #else:
    num_old_classes = self.num_tot_classes
    num_tot_classes = num_old_classes + num_new_classes
    self.num_tot_classes = num_tot_classes

    # Create big D dataset
    if self.use_exemplars:
      D = MergeDataset(train_dataset, exemplars_dataset, augment2=False)
    else:
      D = train_dataset

    if self.exemplars_generator is 'smote' and len(self.exemplar_sets) >0:
      # converting into tensor
      X = torch.stack([img for (img, lb) in D])
      labels = torch.Tensor([lb for (img, lb) in D])
      # generating new exemplar with SMOTE
      new_X, new_labels = generate_exemplars_smote(self, labels, X)
      # appending to the new dataset
      D = [(new_x, new_lb) for (new_x, new_lb) in zip(new_X, new_labels)]

    # Save the old network for distillation
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
        # option for cosine resnet
        if self.custom_loss is not None and self.custom_loss.__name__ == 'less_forget_loss':
          outputs = self.net.forward_cosine(images)[:, :num_tot_classes]
        else:
          outputs = self.net(images)[:, :num_tot_classes]

        # One hot encoding labels for binary cross-entropy loss
        labels_onehot = nn.functional.one_hot(labels, self.num_tot_classes).type_as(outputs)

        if self.custom_loss is None:
            # If custom loss is not specified, original iCaRL loss is used
            if num_old_classes == 0 or not self.distillation:
                loss = criterion(outputs, labels_onehot).sum(dim=1).mean()
            else:
                labels_onehot = labels_onehot.type_as(outputs)[:, num_old_classes:]
                out_old = torch.sigmoid(old_net(images))[:,:num_old_classes]
                target = torch.cat((out_old, labels_onehot), dim=1)
                loss = criterion(outputs, target).sum(dim=1).mean()
        else:
            loss = self.custom_loss(self, images, labels, old_net, **self.loss_params)

        if self.soft_nm:
            loss = loss + soft_nearest_mean_class_loss(self, images, labels, old_net, T=2)

        mean_loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
        # --- end batch
      scheduler.step()
      print(f"Mean batch loss: {mean_loss_epoch/len(dataloader):.5}")
      # --- end epoch

    torch.cuda.empty_cache()
    return D

  def incremental_train(self, train_dataset, test_dataset):
    labels = train_dataset.targets
    new_classes = np.unique(labels)
    print(f'Arriving new classes {new_classes}')

    # Compute number of total labels
    num_old_labels = len(self.exemplar_sets)
    num_new_labels = len(new_classes)

    t = num_old_labels + num_new_labels

    self.update_representation(train_dataset)

    m = int(self.K/t)
    self.reduce_exemplar_set(m=m)

    gc.collect()

    for label in new_classes:
      bool_idx = (train_dataset.targets == label)
      idx = np.argwhere(bool_idx).flatten()
      print(f'Constructing exemplar set for label {label} (memory: {len(gc.get_objects())})')
      images_of_y = []

      for single_index in idx:
        img, label = train_dataset[single_index]
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

    if self.use_exemplars:
        self.test_ncm(test_dataset, num_old_labels)
    self.test_fc(test_dataset, num_old_labels)


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


  def test_ncm(self, test_dataset, num_old_classes):
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
      preds = self.classify(images).to(self.DEVICE)

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
