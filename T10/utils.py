## This function checks the accuracy of the prediction
from torchsummary import summary
import torch
import matplotlib.pyplot as plt
#from model import model as m

from torchsummary import summary
import yaml
from pprint import pprint
import random
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms


def display_images(images,labels,num,classes):
  fig = plt.figure(figsize=(25, 4))

  # We plot 4 images from our train_dataset
  for idx in np.arange(num):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx])) #converting to numpy array as plt needs it.
    ax.set_title(classes[labels[idx].item()])


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def display_model_summary(model,input_structure=(1,28,28)):
  summary(model, input_size=input_structure)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def calculate_mean_std(dataset):
    if dataset == 'CIFAR10':
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        mean = train_set.data.mean(axis=(0,1,2))/255
        std = train_set.data.std(axis=(0,1,2))/255
        return (mean), (std)
# We need to convert the images to numpy arrays as tensors are not compatible with matplotlib.
def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy() # This process will happen in normal cpu.
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image
# We need to convert the images to numpy arrays as tensors are not compatible with matplotlib.
def im_convert_numpy(image):
  #image = tensor.cpu().clone().detach().numpy() # This process will happen in normal cpu.
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image

def find_misclassified_images(num_of_images,test_loader,device,model):
  count = 0
  fig = plt.figure(figsize=(25, 4))
  # Evaluate the model on the test dataset
  misclassified_images = []
  misclassified_labels = []
  true_labels = []
  ## Collect 15 miss-classified images
  while (count < num_of_images):
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    labels = labels.to(device)
    output = model(images)
    _, preds = torch.max(output, 1)
    for idx in range(4):
      if preds[idx] !=labels[idx] and count < 15:
        count +=1
        misclassified_images.append(images[idx].cpu().detach().numpy())
        misclassified_labels.append(preds[idx].cpu().detach().numpy())
        true_labels.append(labels[idx].cpu().detach().numpy())
      else:
        break
  return misclassified_images,misclassified_labels,true_labels

def display_missclassfied_images(missclassified_images,classes):
  #### Displaying those images
  fig = plt.figure(figsize=(10, 4))
  for idx in range(len(missclassified_images)):
      ax = fig.add_subplot(3, 5, idx+1, xticks=[], yticks=[])
      plt.imshow(im_convert_numpy(missclassified_images[idx]))
      ax.set_title("{} ({})".format(str(classes[missclassified_images[idx]]), str(classes[true_labels[idx]])), color=("red"))
