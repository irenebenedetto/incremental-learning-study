from MLDL.nets.eg_resnet import EG_Resnet
import torch
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
from torch.backends import cudnn
import torch.optim as optim
from torchvision.transforms import Compose
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import random
import os
from copy import deepcopy

def generate_images_with_network(self, label, n_new_images, X):
    """
    The function try to generate new images from a set of existing one, by 
    using a network with 3*32*32 output neurons

    Params:
    - X: set of images to replicate, they belong to the same class label
    - n: number of images to generate
    """
    print(f'generating +{n_new_images} new images of label {label}')
    net = EG_Resnet()
    net = net.cuda()
    l1_loss = nn.L1Loss()

    NUM_EPOCHS = 30

    BATCH_SIZE = 5
    LR = 0.01
    WEIGHT_DECAY = 5e-5

    MOMENTUM = 0.9
    GAMMA = 0.5
    MILESTONE = [int(NUM_EPOCHS/5), int(NUM_EPOCHS/3), int(NUM_EPOCHS/2)]

    # creating a dataset of random images
    ds_random_images = torch.randn(size=X.size())

    optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONE, gamma=GAMMA)
    
    # start straining to generate new images from X
    for epoch in range(NUM_EPOCHS):
        for random_image, x in zip(ds_random_images, X):
            net.train(True)
            random_image = random_image.cuda()
            output = net(random_image.unsqueeze(0)).squeeze()
            
            x = x.view(X.size()[1]*X.size()[2]*X.size()[3]).cuda()
            loss = l1_loss(output, x)
            # visualizing the evolution of an image
            #o = output.view(3, 32, 32)
            #plt.title(f'epoch {epoch}/{NUM_EPOCHS}, loss {loss.item()}')
            #plt.imshow(transforms.ToPILImage()(o.cpu()))
            #plt.show()

            loss.backward()
            optimizer.step()
            # --- end batch
        scheduler.step()

    # generating a new set of  n images initialized randomly
      
    with torch.no_grad():
        ds_new_random_images = torch.randn(size=(n_new_images, X.size()[1], X.size()[2], X.size()[3]))
        new_images = []
        net.eval()

        # feedforward the images to the network 
        for random_images in ds_new_random_images:
            random_images = random_images.cuda()
            outputs = net(random_images.unsqueeze(0)).squeeze().cpu()

            outputs = outputs.view(X.size()[1], X.size()[2], X.size()[3])
            new_images.append(outputs)

        new_images = torch.stack(new_images, dim=0)

        return new_images


def generate_new_image(self, label, n_new_images, X):
    print(f'generating +{n_new_images} new images of label {label}')
    mean_of_X = X.mean(dim=0)
    std_of_X = X.std(dim=0)
    new_images = []
      
    for i in range(n_new_images):
        factor = np.random.random()*np.random.randint(-1, 2) 
        new_image = mean_of_X + factor*std_of_X
        new_images.append(new_image)

    return torch.stack(new_images)


def generate_exemplar_max_activation(self, label, n_new_images, X):
    """
     The function generates new exemplars training on a random images and maximizing 
     the activation of the neuron that correspons to the label 
     Params:
        - label: the class label of the new images to generate
        - n_new_images: the number of new exemplars to generare
        - X: not used, only to match with the other functions
     Returns:
        the new images set for the label
    """
    new_images = []
    n_iter=200
    print(f'generating +{n_new_images} new images of label {label}')
    self.net.eval()
    copy_net = deepcopy(self.net)

    # starting iteration to generate the new image
    while len(new_images) < n_new_images:
        random_img = torch.randn(size = ([1, 3, 32, 32])).cuda()
        random_img.requires_grad = True
        # optimizing the random image with sgd
        optimizer = optim.SGD([random_img], lr=0.2+np.random.rand()*0.3, weight_decay=5e-5, momentum=0)

        for i in range(n_iter):
            optimizer.zero_grad()
            output = copy_net(random_img).squeeze()
            # we want to maximize the activation of the neuron that corresponds to the right label
            loss = 100000 - output[label]
            
            loss.backward()
            optimizer.step()

        if i >= 50 and i%5 == 0:
            
            new_images.append(deepcopy(random_img.squeeze().data))

    new_images = torch.stack(new_images)
    return new_images[:n_new_images]    
