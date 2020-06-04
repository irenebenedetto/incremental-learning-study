import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset, DataLoader, TensorDataset, Dataset
from torch.backends import cudnn
import torch.optim as optim
from torchvision.transforms import Compose
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
import numpy as np
from matplotlib import cm
import random
import pandas as pd
import os
import torch.nn as nn



class SubCIFAR100(Dataset):
  """
  Fa due cose:
  - prende determiante labeles da cifrar100
  - le mappa come indicato
  - terza cosa giusto per rompere le palle
  """
  def __init__(self, labels_to_retrieve, label_mapping, train=True, transform=None, root='./CIFAR100', download=True):
    self.cifar100 = CIFAR100(root=root, train=train, download=download)
    self.transform = transform
    # Create a dict containing map[original label] = mapped label
    mapping = {original_label: mapped_label for original_label, mapped_label in zip(labels_to_retrieve, label_mapping)}

    # Retrieve indices of desired labels
    bool_idx = np.zeros(len(self.cifar100), dtype=bool)
    images = []
    labels = []
    for label in labels_to_retrieve:
      bool_idx += (self.cifar100.targets == label)
    # Turn bool mask into numeric labels
    indices = np.argwhere(bool_idx).flatten()
    for index in indices:
      img, label = self.cifar100[index]
      images.append(transforms.ToTensor()(img))
      labels.append(label)
    # Turn the lists into marvellous tensors
    self.data = torch.stack(images)
    self.targets = np.array([mapping[label] for label in labels], dtype='long')

  def __len__(self):
    """
    I wonder what this does.
    """
    return self.data.size(0)

  def pil_loader(self, tensor_img):
    f = transforms.ToPILImage()(tensor_img)
    return f.convert('RGB')

  def __getitem__(self, index):
    '''
    getitem should access an element through its index
    Args:
        index (int): Index
    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    '''
    image = self.pil_loader(self.data[index])
    label = self.targets[index]

    if self.transform is not None:
        image = self.transform(image)

    return image, label

class MergeDataset():
  """
  Class that merges two datasets into one.

  Returns:
    Due bici di Bibbona
  """
  def __init__(self, ds1, ds2, augment1=False, augment2=False):
    self.ds1 = ds1
    self.ds2 = ds2
    self.augment1 = augment1
    self.augment2 = augment2
    self.transform = transforms.Compose([
                                      transforms.ToPILImage(),
                                      transforms.RandomCrop(32, padding=4),# Crops a random squares of the image
                                      transforms.RandomHorizontalFlip(p=0.5), # Randomly flip the image with probability of 0.5
                                      transforms.ColorJitter(brightness=0.24705882352941178),
                                      transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                                      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d815
                                      ])

  def __len__(self):
    return len(self.ds1) + len(self.ds2)

  def __getitem__(self, index):
    if index < len(self.ds1):
      img, lb = self.ds1[index]
      if self.augment1:
        img = self.transform(img)
    else: # in this czse index >= len(ds1)
      img, lb = self.ds2[index - len(self.ds1)]
      if self.augment2:
        img = self.transform(img)
    return img, lb

class CIFARFactory():
  """
  Factory class for retrieving re-mapped SubCifar100 datasets. Can accidentally destroy the world.
  """

  def __init__(self, seed=1993, shuffle_classes=True, split_size=10):
    self.CIFAR100_LABELS_LIST = [
      'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
      'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
      'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
      'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
      'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
      'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
      'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
      'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
      'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
      'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
      'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
      'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
      'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
      'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
      'worm'
    ]
    self.downloaded_dataset = False
    np.random.seed(seed=seed)
    self.original_labels = np.arange(100)
    if shuffle_classes == True:
      np.random.shuffle(self.original_labels)

    self.mapped_labels = np.arange(100)
    # Also geenrating classes in ranges 0 to N
    self.zero_N_labels = self.mapped_labels % split_size
    self.human_readable_label = {mapped_label: self.CIFAR100_LABELS_LIST[original_label] for mapped_label, original_label  in zip(self.mapped_labels, self.original_labels)}

    # Splitting batch of classes in subgroups
    self.original_sets = np.split(self.original_labels, split_size)
    self.mapped_sets = np.split(self.mapped_labels, split_size)
    self.zero_N_sets = np.split(self.zero_N_labels, split_size)

    self.train_transform = transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),# Crops a random squares of the image
                                      transforms.RandomHorizontalFlip(p=0.5), # Randomly flip the image with probability of 0.5
                                      transforms.ColorJitter(brightness=0.24705882352941178),
                                      transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                                      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                                      ])

    self.no_transform = transforms.Compose([
                                      transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                                      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                                    ])

  def get_train_dataset(self, step, zero_N=False, augmentation=True):
    if zero_N:
      mapping = self.zero_N_sets[step]
    else:
      mapping = self.mapped_sets[step]

    if augmentation:
      transform = self.train_transform
    else:
      transform = self.no_transform

    dataset = SubCIFAR100(self.original_sets[step], mapping, train=True, transform=transform, download=(not self.downloaded_dataset))
    self.downloaded_dataset = True
    return dataset

  def get_test_dataset(self, step):
    labels_to_retrieve = np.concatenate([self.original_sets[i] for i in range(step+1)])
    mapping = np.concatenate([self.mapped_sets[i] for i in range(step+1)])
    dataset = SubCIFAR100(labels_to_retrieve, mapping, train=False, transform=self.no_transform, download=(not self.downloaded_dataset))
    self.downloaded_dataset = True
    return dataset
