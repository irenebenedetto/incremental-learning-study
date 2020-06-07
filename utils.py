from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm 

from sklearn.manifold import TSNE
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import seaborn as sns

# DISPLAY 
def show_image_label(img, label):
  """
  Print an image (taken in tensor form) and its human readable label.

  Returns:
    A '68 fender stratocaster
  """
  print(f"Showing an image of label {human_readable_label[label]}")
  plt.imshow(transforms.ToPILImage()(img), interpolation="bicubic")
  plt.show()


def new_heat_matrix():
  lenx = 10
  leny = 100
  data = np.zeros((leny, lenx))
  return data

def show_heat_matrix(data):
  fig, ax = plt.subplots(figsize=(15,9))
  ax = sns.heatmap(data, linewidth=0.2,cmap='Reds')
  plt.show()

def update_heat_matrix(data, preds, num_batch, normalization):
  #during test phase
  preds.cpu()
  for el in preds:
    preds = el.item()
    if (normalization == False) or (num_batch == 0):
      data[preds,num_batch] = data[preds,num_batch]+1
    else:
      data[preds,num_batch] = data[preds,num_batch]+1/(num_batch)
        
#CONFUSION MATRIX
def update_confusion_matrix(matrix, preds, datas):
  for pred, data in zip(preds,datas):
    matrix[data.item(),pred.item()] = matrix[data.item(),pred.item()]+1
        
def new_confusion_matrix(lenx=100, leny=100):
  matrix = np.zeros((leny, lenx))
  return matrix

def show_confusion_matrix(matrix):
  fig, ax = plt.subplots(figsize=(15,9))
  ax = sns.heatmap(matrix, linewidth=0.2,cmap='Reds')
  plt.show()

def save(list_to_save, name_file):
  with open(name_file + '.txt', 'w') as outfile:
    outfile.write(str(list_to_save) + '\n')
  
def plot_metrics(x, y, name, xlabel, ylabel, title):
  """
    The function plots the metrics.

    Params:
      - x: independent variable to plot over the x axis
      - y: lists of metrics to plot on y axis
      - name: legend label for each y_i plotted
      - label: name of the y axis
  """
  
  color = cm.coolwarm(np.linspace(0,1,len(y)))
  plt.figure(figsize=(15, 10))
  for yi, namei, c in zip(y, name, color):
    plt.plot(x, yi, marker='o', color=c, markersize=10, linewidth=4, label = namei)

  plt.legend()
  plt.title(title, loc='center', fontsize=16, fontweight=0, color='black')
  plt.xlabel("Number of classes")
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.grid()
  plt.show()


def scatter_images(x, colors, human_readable_label):

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    RS = 123

    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(15, 10))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.grid()

    ax.axis('off')
    ax.axis('tight')
    

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, human_readable_label[i], fontsize=15)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def create_tsne(net, human_readable_label):
    """
    The function plots the t-sne representation for the exemplar set  

    Ex.
        human_readable_label = cifar100.human_readable_label
        create_tsne(icarl, human_readable_label)

    Params:
      net: the model chosen
      human_readable_label: the names of the label assigned to each image
    Return:
      t-sne representation of image in 2 dimensions
    """
    with torch.no_grad():
        for i, exemplar_set in enumerate(net.exemplar_sets):
            dim = exemplar_set.size()[0]
            fts_exemplar = []
            if i == 0:
                all_images = []
            for exemplar in  exemplar_set:
                ft_map = icarl.net.feature_extractor(exemplar.to("cuda").unsqueeze(0)).squeeze().cpu()
                fts_exemplar.append(ft_map)

            fts_exemplar = torch.stack(fts_exemplar)

            if i == 0:
                all_images = fts_exemplar
                all_labels = np.full((dim), i)
            else:
                all_images = torch.cat((all_images, fts_exemplar), 0)
                all_labels = np.concatenate((all_labels, np.full((dim), i)))
            
              
        #Now I Have all_images and all_labels, I can start the reduce phase
        fashion_tsne = TSNE().fit_transform(all_images.cpu().detach().numpy())
        #Plot
        scatter_images(fashion_tsne, all_labels, human_readable_label)
