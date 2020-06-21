#from MLDL.nets.custom_resnet import ResNet18
from copy import deepcopy
import gc
import torch
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
from torch.backends import cudnn
from MLDL.nets.custom_resnet import ResNet18
import torch.optim as optim
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from MLDL.utils import *
from matplotlib import cm
import random
import pandas as pd
import os

class FamiliCaRL():
  """
  Support gay marriage. More equity, less discrimination.
  """
  def __init__(self, K=2000, clf_loss=None, dist_loss=None, clf_params=None, dist_params=None, all_data_means=False, remove_duplicates=True, custom_model=None):
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

    # Internal flags to set behavior
    self.all_data_means = all_data_means

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
    # Optional losses and model
    if custom_model is not None:
      self.ancestor_model = custom_model
    else:
      self.ancestor_model = ResNet18

    if clf_loss is not None:
      self.clf_loss = clf_loss
    else:
      self.clf_loss = nn.BCEWithLogitsLoss(reduction='none')

    if dist_loss is not None:
      self.dist_loss = dist_loss
    else:
      self.dist_loss = nn.BCEWithLogitsLoss(reduction='none')

    if clf_params is None:
      self.clf_params = {}
    else:
      self.clf_params = clf_params

    if dist_params is None:
      self.dist_params = {}
    else:
      self.dist_params = dist_params

  def compute_exemplars_means(self):
    # First obtain feature extractor
    self.old_parent.eval()

    with torch.no_grad():
      self.class_means = []
      for label, Py in enumerate(self.exemplar_sets):
        print(f"Computing means for label {label}")

        phi_Py = self.old_parent.feature_extractor(Py.to(self.DEVICE))
        mu_y = phi_Py.mean(dim = 0)
        mu_y.data = mu_y.data / mu_y.data.norm()
        self.class_means.append(mu_y)

  def compute_class_means_with_training(self, X):
    """
      Compute class means with data passed as argument
      Params:
        - X: images that belong to a single class label
    """
    self.old_parent.eval()

    with torch.no_grad():
      phi_X = self.old_parent.feature_extractor(X.to(self.DEVICE))
      mean = phi_X.mean(dim = 0)
      mean.data = mean.data / mean.data.norm()
      self.class_means.append(mean)

  def classify_NMC(self, X):
    torch.cuda.empty_cache()
    with torch.no_grad():

      self.old_parent.eval()

      # Compute feature mappings of batch
      X = X.to(self.DEVICE)
      phi_X = self.old_parent.feature_extractor(X)
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

  def random_construct_exemplar_set(self, X, y, m):
      """
      X only contains elements of a single label y
      """
      with torch.no_grad():
          indexes = torch.randperm(X.size(0))[:m]
          exemplar_set = X[indexes]
          self.exemplar_sets.append(exemplar_set)

  def construct_exemplar_set(self, X, y, m):
    """
    X only contains elements of a single label y
    """
    with torch.no_grad():
      self.old_parent.eval()

      # Compute class mean of X
      loader = DataLoader(X,batch_size=self.BATCH_SIZE, shuffle=True, drop_last=False, num_workers = 4)
      phi_X = []
      for images in loader:
        images = images.to(self.DEVICE)
        phi_X_batch = self.old_parent.feature_extractor(images)
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
        phi_p = self.old_parent.feature_extractor(p.to(self.DEVICE))
        sum_taken_exemplars = sum_taken_exemplars + phi_p.to('cpu')
        if self.remove_duplicates:
          X = torch.cat((X[:min_index], X[min_index+1:]), dim = 0)
          phi_X = torch.cat((phi_X[:min_index], phi_X[min_index+1:]), dim = 0)
        del phi_p

      Py = torch.stack(Py)
      self.exemplar_sets.append(Py)

  def test_ncm(self, test_dataset, num_old_classes):
    self.old_parent.eval()
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
      preds = self.classify_NMC(images).to(self.DEVICE)
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
    self.old_parent.eval()
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

      outputs = self.old_parent(images)[:,:self.num_tot_classes]
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

  def train_new_parent(self, train_dataset, num_old_classes):
    """
    Return trained network. This will be trained to classify new classes only
    """
    new_classes = np.unique(train_dataset.targets)
    print(f"Training new parent (genitore 2) on classes {new_classes}")

    # Instantiate new model and perform a normal training
    if num_old_classes == 0:
        new_parent = ResNet18()
    else:
        new_parent = deepcopy(self.old_parent)


    new_parent = new_parent.to(self.DEVICE)

    optimizer = optim.SGD(new_parent.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY, momentum=self.MOMENTUM)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.MILESTONE, gamma=self.GAMMA)

    dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)

    for epoch in range(self.NUM_EPOCHS):
      print(f'(Genitore 2) EPOCH {epoch+1}/{self.NUM_EPOCHS}, LR = {scheduler.get_last_lr()}')

      mean_loss_epoch = 0
      mean_reg_loss = 0
      for images, labels in dataloader:
        images = images.to(self.DEVICE)
        labels = labels.to(self.DEVICE)

        new_parent.train()
        optimizer.zero_grad()

        # Leave the output complete and cut it afterwards
        outputs_new_classes = new_parent(images)[:, num_old_classes:self.num_tot_classes] # non dovrebbe essere --> num_old_classes:self.num_tot_classes
        labels_onehot = nn.functional.one_hot(labels, self.num_tot_classes).type_as(outputs_new_classes)[:, num_old_classes:] # non dovrebbe essere --> num_old_classes:self.num_tot_classes

        classification_loss = self.clf_loss(outputs_new_classes, labels_onehot).sum(dim=1)
        mean_loss_epoch += classification_loss.data.mean()
        # add regularization
        if num_old_classes >= 10:
            fts_new_parent = new_parent.feature_extractor(images)
            fts_old_parent = self.old_parent.feature_extractor(images)
            regularization_loss = 1 - (fts_new_parent*fts_old_parent).sum(dim=1)
            mean_reg_loss += regularization_loss.data.mean()
            # Compute clf loss and backward
            loss = (classification_loss + regularization_loss).mean()
        else:
            loss = classification_loss.mean()


        loss.backward()
        optimizer.step()
        # -- end batch
      scheduler.step()
      print(f"Mean classification loss: {mean_loss_epoch/len(dataloader):.5}")
      print(f"Mean regularization loss: {mean_reg_loss/len(dataloader):.5}")
      # -- end epoch
    return new_parent

  def incremental_train(self, train_dataset, test_dataset):
    labels = train_dataset.targets
    new_classes = np.unique(labels)
    print(f'Arriving new classes {new_classes}')
    self.num_new_classes = len(new_classes)
    num_old_classes = self.num_tot_classes
    self.num_tot_classes = num_old_classes + self.num_new_classes

    new_parent = self.train_new_parent(train_dataset, num_old_classes)

    # Reduce previous exemplars
    m = int(self.K/self.num_tot_classes)
    self.reduce_exemplar_set(m=m)

    # Construct new exemplars
    for label in new_classes:
      bool_idx = (train_dataset.targets == label)
      idx = np.argwhere(bool_idx).flatten()
      print(f'Constructing exemplar set for label {label} (memory: {len(gc.get_objects())})')
      images_of_y = []

      for single_index in idx:
        img, label = train_dataset[single_index]
        images_of_y.append(img)

      images_of_y = torch.stack(images_of_y)
      self.random_construct_exemplar_set(X=images_of_y, y=label, m=m)

    # We now have a new parent and all exemplars. If this is the first batch, we are done
    if num_old_classes == 0:
      self.old_parent = deepcopy(new_parent)
    else:
      # Concatenate current exemplar sets with respective labels
      exemplars_dataset = []
      for label, exemplar_set in enumerate(self.exemplar_sets):
        for exemplar in exemplar_set:
          exemplars_dataset.append((exemplar, label))
      child = self.distill_parents(self.old_parent, new_parent, exemplars_dataset)
      self.old_parent = deepcopy(child)


    if not self.all_data_means:
      self.compute_exemplars_means()
    else:
    # Compute means with all data
      for label in new_classes:
        bool_idx = (train_dataset.targets == label)
        idx = np.argwhere(bool_idx).flatten()
        images_of_y = []

        for single_index in idx:
          img, label = train_dataset[single_index]
          images_of_y.append(img)

        images_of_y = torch.stack(images_of_y)
        self.compute_class_means_with_training(images_of_y)
    print(f'exemplar means: {len(self.class_means)}')
    self.test_ncm(test_dataset, num_old_classes)
    self.test_fc(test_dataset, num_old_classes)

  def distill_parents(self, old_parent, new_parent, dataset):# 0-9, 10-19
    """
    Return child
    """
    print(f"Distilling parents into child (lulwut?)")

    num_old_classes = self.num_tot_classes - self.num_new_classes
    old_parent.eval()
    new_parent.eval()
    child_model = deepcopy(new_parent)

    child_model = child_model.to(self.DEVICE)
    child_model.train(True)

    for param in old_parent.parameters():
        param.requires_grad = False
    for param in new_parent.parameters():
        param.requires_grad = False


    optimizer = optim.SGD(child_model.parameters(), lr=0.9, weight_decay=self.WEIGHT_DECAY, momentum=self.MOMENTUM)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50,70, 90], gamma=0.3)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=False)

    for epoch in range(60):
      print(f'(Distillation) EPOCH {epoch+1}/{self.NUM_EPOCHS}, LR = {scheduler.get_last_lr()}')

      mean_loss_epoch_clf = 0
      mean_loss_epoch_dist = 0
      for images, labels in dataloader:
        images = images.to(self.DEVICE)
        labels = labels.to(self.DEVICE)

        child_model.train()
        optimizer.zero_grad()

        # Take outputs of old parent
        out_old_parent = old_parent(images)[:, :num_old_classes]
        q_old = nn.functional.sigmoid(out_old_parent)
        # Take outputs of new parent
        out_new_parent = new_parent(images)[:, num_old_classes:self.num_tot_classes]
        q_new = nn.functional.sigmoid(out_new_parent)

        target = torch.cat((q_old, q_new), dim=1)
        # Distill in child
        out_child = child_model(images)[:, :self.num_tot_classes]
        dist_loss_contribution = self.dist_loss(out_child, target).sum(dim=1)

        # Add a further classification step
        labels_onehot = nn.functional.one_hot(labels, self.num_tot_classes).type_as(out_child)
        clf_loss_contribution = self.clf_loss(out_child, labels_onehot).sum(dim=1)


        loss = (dist_loss_contribution + clf_loss_contribution).mean()

        mean_loss_epoch_clf += clf_loss_contribution.data.mean().item()
        mean_loss_epoch_dist += dist_loss_contribution.data.mean().item()
        loss.backward()
        optimizer.step()
        # -- end batch
      scheduler.step()
      print(f"Mean clf loss: {mean_loss_epoch_clf/len(dataloader):.5}")
      print(f"Mean dist loss: {mean_loss_epoch_dist/len(dataloader):.5}")
      # -- end epoch
    # -- end training
    return child_model
