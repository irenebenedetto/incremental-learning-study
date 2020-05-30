from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm 
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
  plt.figure(figsize=(10, 5))
  for yi, namei, c in zip(y, name, color):
    plt.plot(x, yi, marker='o', color=c, markersize=10, linewidth=4, label = namei)

  plt.legend()
  plt.title(title, loc='center', fontsize=16, fontweight=0, color='black')
  plt.xlabel("Number of classes")
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.grid()
  plt.show()
    
  



