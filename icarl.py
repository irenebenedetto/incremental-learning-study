import torch
from torch.utils.data import Subset, DataLoader, TensorDataset
import torch.nn as nn
from torch.backends import cudnn
import torch.optim as optim
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import random
import pandas as pd
import os
import gc
from copy import deepcopy

class iCaRL():
  """
  Implements iCaRL as decribed in *insert paper*
  We attempt to use all tensors instead of lists
  """
  def __init__(self, net, mean_on_all_data=True):
    self.exemplar_sets = dict()
    # Inizializing with parameters from paper ** insert reference **
    self.MOMENTUM = 0.9
    self.LR = 2
    self.BATCH_SIZE = 128
    self.MILESTONE = [49, 63]
    self.WEIGHT_DECAY = 1e-5
    self.GAMMA = 0.2
    self.NUM_EPOCHS = 70 # CHANGE
    self.DEVICE = 'cuda'
    self.mean_on_all_data = mean_on_all_data
    self.class_means = dict()
    # Removing last FC layer and leaving only convolutional feature mapping
    self.net = net
    self.feature_map = deepcopy(self.net)
    self.net = net.to(self.DEVICE)
    self.feature_map.fc = nn.Sequential()
    self.feature_map = self.feature_map.to(self.DEVICE)
    self.first_run = True

    gc.collect()

  def get_params(self):
    params_to_return = {
      'MOMENTUM': self.MOMENTUM,
      'LR': self.LR,
      'BATCH_SIZE': self.BATCH_SIZE,
      'MILESTONE': self.MILESTONE,
      'WEIGHT_DECAY': self.WEIGHT_DECAY,
      'GAMMA': self.GAMMA,
      'NUM_EPOCHS': self.NUM_EPOCHS,
      'DEVICE': self.DEVICE
    }
    return params_to_return

  def set_params(self, params):
    self.MOMENTUM = params['MOMENTUM']
    self.LR = params['LR']
    self.BATCH_SIZE = params['BATCH_SIZE']
    self.MILESTONE = params['MILESTONE']
    self.WEIGHT_DECAY = params['WEIGHT_DECAY']
    self.GAMMA = params['GAMMA']
    self.NUM_EPOCHS = params['NUM_EPOCHS']
    gc.collect()


  def classify(self, X):
    """
    Performs a classification using NCM.

    Params:
      x: input tensor(s)
    Return:
      Image labels
    """
    gc.collect()
    # We move cpu copy of feature map to cuda
    cuda_fm = self.feature_map.to(self.DEVICE)
    cuda_fm.train(False)
    class_means = []

    if self.mean_on_all_data:
      class_means = self.class_means
    else:
      class_means = dict()
      for label, exemplar_set in enumerate(self.exemplar_sets.values()):
        class_mean = self.compute_class_mean(exemplar_set)
        class_mean.data = class_mean.data / class_mean.data.norm()
        class_means[label] = class_mean

    # NCM -----
    list_means = torch.stack(list(class_means.values()))
    list_labels = list(class_means.keys())

    # Take label that produces minium distance between mean and transformed x. Grouping the distances into a tensor
    labels = []
    for mapped_img in cuda_fm(X):

      # Broadcast mapped img and take norm on dimension 1 (dimension 0 is the "rows")
      mapped_img.data = mapped_img.data/mapped_img.data.norm()
      mapped_img = mapped_img.to('cpu')
      distances = torch.norm(mapped_img - list_means, dim=1)
      argmin = torch.argmin(distances).item()
      y = list_labels[argmin]
      labels.append(y)
    # Return the labels as LongTensor; clean fm from cuda
    del cuda_fm
    torch.cuda.empty_cache()
    gc.collect()
    return torch.stack(labels).type(torch.long).to(self.DEVICE)


  def incremental_train(self, train_dataloader,test_dataloader, K):
    gc.collect()
    labels = []
    images = []
    for imgs, lbls in train_dataloader:
      images.append(imgs)
      labels.append(lbls)

    labels = torch.cat(labels)
    X = torch.cat(images)
    new_classes = torch.unique(labels)
    print(f'Arriving new classes {new_classes}')

    t = self.update_representation(X = X, labels = labels)
    m = int(K/t)

    self.reduce_exemplar_set(m=m)

    gc.collect()
    label_subsets = []
    for label in new_classes:
      bool_idx = (labels.numpy() == label.item())
      idx = np.argwhere(bool_idx).flatten()
      print(f'class: {label.item()} memory: {len(gc.get_objects())}')
      self.construct_exemplar_set(X=X[idx], m=m, label=label.item())
      gc.collect()
      del bool_idx
      del idx

    self.test(test_dataloader)


  def compute_class_mean(self, X):
    """
    Params:
      X: images of a single class label (tensor of tensors)

    Returns:
      Mean of elements of X after having been mapped into feature space
    """
    cuda_fm = self.feature_map.to(self.DEVICE)
    # Computing mapped X list (using explicit for because of CUDA :/)
    mapped_X = []
    for x in X:
        cuda_x = x.unsqueeze(0).to(self.DEVICE)
        cuda_mapped_x = cuda_fm(cuda_x)
        mapped_x = cuda_mapped_x.to('cpu')
        del cuda_mapped_x
        mapped_X.append(mapped_x)
        del cuda_x
        
        torch.cuda.empty_cache()

    mapped_X = torch.cat(mapped_X)
    class_mean = mapped_X.mean(dim = 0)
    del cuda_fm
    return class_mean

  def split_classes(self, X, labels):
    """
    Split classes X into list of lists accordig to labels.

    Parameters:
      X: images to split (tensor of tensors)
      labels: corresponding labels, in order

    Returns:
      Dictionary containing:
        - label as key
        - Tensor of images of the corresponding label as value
    """
    unique_labels = torch.unique(labels)

    split_X = dict()

    for label in unique_labels:
      bool_idx = (labels.numpy() == label.item())
      idx = np.argwhere(bool_idx).flatten()
      split_X[label] = X[idx]

    return split_X

  def update_representation(self, X, labels):
    """
    The function update the parameters of the network computing the class loss and the
    distillation loss

    Params:
      X: training images of new classes s, ..., t
      labels: the damn associated labels

    Return:
      La bici di Bibbona

    """
    from random import shuffle

    if self.mean_on_all_data:
      # Construct mean of class using ALL data available
      split_X = self.split_classes(X, labels)
      print('Computing class mean over the full training set...')
      for label, grouped_X in split_X.items():
        self.class_means[label] = self.compute_class_mean(grouped_X)

    if self.first_run:
        num_new_classes = self.train_first_run(X, labels)
        gc.collect()
        return num_new_classes
    cuda_net = self.net.to(self.DEVICE)


    # Io dico che D serve (that's to say: construction of D)
    num_new_classes = torch.unique(labels).size(0)
    # Store in one single tensor all list of tensors of exemplars... wait what?
    # I mean, store in a single "tensor list" all exemplars. We don't care about the labels for now
    past_exemplars = torch.cat([exemplar_set for exemplar_set in self.exemplar_sets.values()])
    # Infer number of old classes from q, cuz fuck the law // UPDATE: acutally that was a shit idea
    num_old_classes = len(self.exemplar_sets)
    # Buckle up: construct D as a dataset. First, retrieve the old labels of the exemplars
    label_list = []
    for i, exemplar_set in enumerate(self.exemplar_sets.values()):
      label_list += [i] * exemplar_set.size(0)
    old_labels = torch.Tensor(label_list).type(torch.long)
    # You can also do it in one line, thank me for sparing you from this pain:
    # old_labels = torch.cat([torch.Tensor([i] * exemplar_set.size(0)) for i, exemplar in enumerate(self.exemplar_sets)]).type(torch.long)
    # Turn this crap in a dataset and then a dataloader
    X_old_and_new = torch.cat((X, past_exemplars))
    y_old_and_new = torch.cat((labels, old_labels))
    D = TensorDataset(X_old_and_new, y_old_and_new)
    D_dataloader = DataLoader(D, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
    # We now majestically compute q for each batch and store it as a list of triples
    X_y_q_loader = []
    # Again copy cpu net to cuda and work on that
    for images, labels in D_dataloader:
      q = cuda_net(images.to(self.DEVICE))
      q = torch.sigmoid(q).data
      #q = q.to('cpu') # Bringing it back to cpu in a futile attempt to save memory
      X_y_q_loader.append((images, labels, q))
    # adding 10 new neuron to the last fc layer
    if num_old_classes > 0:
      old_weights = cuda_net.fc.weight.data
      old_bias = cuda_net.fc.bias.data
      cuda_net.fc = nn.Linear(64, num_old_classes + num_new_classes).to(self.DEVICE)
      cuda_net.fc.weight.data[:num_old_classes] = old_weights
      cuda_net.fc.bias.data[:num_old_classes] = old_bias
    else:
      cuda_net.fc = nn.Linear(64, num_new_classes).to(self.DEVICE)

    # Initalize optimizer, stepsizer, coser varier
    # We actually have to optimize everything this time
    params_to_optimize = cuda_net.parameters()
    optimizer = optim.SGD(params_to_optimize, lr=self.LR,  momentum=self.MOMENTUM, weight_decay=self.WEIGHT_DECAY)
    # Clf loss and distillation loss are the same criterion with different inputs
    dist_criterion = nn.BCELoss()
    class_criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.MILESTONE, gamma=self.GAMMA)
    cudnn.benchmark
    # ...whatever i'm too tired, bye <- left for historical interest
    # Begin training this tank of a net
    for epoch in range(self.NUM_EPOCHS):
      mean_clf_loss = 0
      mean_dist_loss = 0
      print(f"STARTING EPOCH {epoch +1}/{self.NUM_EPOCHS}")
      # We use the loader with the qs, but we shuffle it before each training epoch
      #shuffle(X_y_q_loader)
      for images, labels, q in X_y_q_loader:
        images = images.to(self.DEVICE)
        labels = labels.to(self.DEVICE)
        #q = q.to(self.DEVICE)
        # Old outputs are stored, we modify the current network
        cuda_net.train(True)
        optimizer.zero_grad()
        outputs = cuda_net(images, norm_features=False)
        labels_one_hot = nn.functional.one_hot(labels, num_old_classes + num_new_classes).type_as(outputs)
        g = torch.sigmoid(outputs)

        dist_loss = dist_criterion(g[:, :num_old_classes], q) #/ num_old_classes
        class_loss = class_criterion(outputs[:, num_old_classes:], labels_one_hot[:, num_old_classes:])

        mean_clf_loss += class_loss.item()
        mean_dist_loss += dist_loss.item()

        loss = dist_loss + class_loss
        loss.backward()
        optimizer.step()
        #q = q.to('cpu')
        del images
        del labels
        del outputs
        torch.cuda.empty_cache()
        # --- end batch

      mean_clf_loss = mean_clf_loss/len(X_y_q_loader)
      mean_dist_loss = mean_dist_loss/len(X_y_q_loader)
      print(f"Mean classification loss per batch: {mean_clf_loss:.4}")
      print(f"Mean distillation loss per batch: {mean_dist_loss:.4}")
      scheduler.step()
      # --- end epoch

    # Return a copy of the traied netweeek, but bring it back to cpu first
    self.net = cuda_net
    del cuda_net
    torch.cuda.empty_cache()
    self.feature_map = deepcopy(self.net)
    self.feature_map.fc = nn.Sequential()
    return num_old_classes + num_new_classes


  def train_first_run(self, X, labels):
    """
    The function performs the training on the first step of training

    Params:
      X: tensor of images
    Return:
      the netwotk with the weight updated

    """
    # Again work on a cuda copy and save it to cpu afterwards, yo!
    cuda_net = self.net.to(self.DEVICE)

    num_new_classes = torch.unique(labels).size(0) # always 10 but panino is insistent ...

    params_to_optimize = cuda_net.parameters()
    optimizer = optim.SGD(params_to_optimize, lr=self.LR,  momentum=self.MOMENTUM, weight_decay=self.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.MILESTONE, gamma=self.GAMMA)

    # Defining a new dataloader from given data
    train_dataset = [(img, lb) for img, lb in zip(X, labels)]
    # alternative
    # train_dataset = TensorDataset(X, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)
    print(f'First run, new classes {num_new_classes}')
    for epoch in range(self.NUM_EPOCHS):
      print(f"Starting epoch {epoch +1}/{self.NUM_EPOCHS}")
      mean_loss = 0
      for images, labels in train_dataloader:
        cuda_net.train(True)
        optimizer.zero_grad()
        images = images.to(self.DEVICE)
        outputs = cuda_net(images, norm_features=False)

        labels_one_hot = nn.functional.one_hot(labels, num_new_classes).type_as(outputs).to(self.DEVICE)
        loss = criterion(outputs, labels_one_hot)
        mean_loss += loss.item()

        loss.backward()
        optimizer.step()

        del images
        del outputs
        del labels_one_hot
        torch.cuda.empty_cache()

      mean_loss = mean_loss/len(train_dataloader)
      print(f"Mean classification loss per batch: {mean_loss:.4}")
      scheduler.step()

    self.first_run = False

    self.net = cuda_net
    del cuda_net
    torch.cuda.empty_cache()
    self.feature_map = deepcopy(self.net)
    self.feature_map.fc = nn.Sequential()
    gc.collect()
    print()
    return num_new_classes


  def reduce_exemplar_set(self, m):
    """
    The function reduces the number of images for each exampler set at m

    Params:
      m: number of elements that has to be collected
    Return:
      the list of exemplar_sets updated

    """
    for i, exemplar_set in self.exemplar_sets.items():
      self.exemplar_sets[i] = exemplar_set[:m]
    return self.exemplar_sets


  def construct_exemplar_set(self, X, m, label):
    """
    Construct the new exemplar set from X, where the images contained belongs to the same
    label

    Params:
      subset: subset of images with a certain label
      m: number of elements that has to be collected
    Return:
      the list of exemplar_sets updated with the new set

    """
    mapped_x_list = []
    # Taking the first label is ok, they are all the same
    # It is CRUCIAL not to shuffle here! (we needed dataloader to save cuda memory)
    for x in X:
      x = torch.unsqueeze(x, 0).to(self.DEVICE)
      mapped_x = self.feature_map(x).to('cpu')
      mapped_x_list.append(mapped_x)
      del x
      del mapped_x
      torch.cuda.empty_cache()
    # Instantiating tensor of mapped images and normal images
    mapped_X = torch.cat(mapped_x_list) # mapped_X is a list of images in tensor form
    #del cuda_feature_map
    torch.cuda.empty_cache()
    # Get size of mapped images (basically 512)
    # [ self.feature_map(X)[0].size()[0] = 512]
    size_mapped_images = mapped_X[0].size()[0]
    # Compute mean of the samples of the new class
    class_mean = torch.mean(mapped_X, dim=0)
    class_mean.data = class_mean.data / class_mean.norm()
    new_exemplar_set = []
    new_examplar_map = []
    # Accumulates sum of exemplars
    sum_mapped_exemplars = torch.zeros([size_mapped_images],  dtype=torch.float)
    for k in range(1, m+1):
      distances = torch.Tensor([torch.norm(class_mean - (1/k)*((phi_x + sum_mapped_exemplars).data / (phi_x + sum_mapped_exemplars).data.norm())) for phi_x in mapped_X])
      min_index = torch.argmin(distances)
      # Assuming subset is in the same order (which it should)
      min_img = X[min_index]
      sum_mapped_exemplars += mapped_X[min_index]
      new_exemplar_set.append(min_img)
      sum_mapped_exemplars.data = sum_mapped_exemplars.data / sum_mapped_exemplars.norm()
      X = torch.cat((X[:min_index], X[min_index+1:]), dim = 0)
      mapped_X = torch.cat((mapped_X[:min_index], mapped_X[min_index+1:]), dim = 0)

    new_exemplar_set = torch.stack(new_exemplar_set)

    del X
    del mapped_X
    del sum_mapped_exemplars
    # Append newly constructed set to list of examplar sets

    self.exemplar_sets[label] = new_exemplar_set
    return


  def test(self, test_dataloader):
    self.net.train(False)

    running_corrects_classify = 0
    running_corrects_fc = 0
    tot_loss = 0
    for images, labels in test_dataloader:
      # print(f"Test labels: {np.unique(labels.numpy())}")
      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)

      # Forward Pass
      outputs = self.net(images)
      # Get predictions
      preds_classify = self.classify(images)
      _, preds_fc = torch.max(outputs.data, 1)

      # Update Corrects
      running_corrects_classify += torch.sum(preds_classify == labels.data).data.item()
      running_corrects_fc += torch.sum(preds_fc == labels.data).data.item()

    # Calculate Accuracy and mean loss
    accuracy_classify = running_corrects_classify / len(test_dataloader.dataset)
    accuracy_fc = running_corrects_fc / len(test_dataloader.dataset)


    print(f'\033[94mAccuracy on test set classify :{accuracy_classify}\x1b[0m')
    print(f'\033[94mAccuracy on test set fc :{accuracy_fc}\x1b[0m')
