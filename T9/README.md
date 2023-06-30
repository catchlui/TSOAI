# Image Classification Application for CIFAR10 dataset

This application is designed for image classification using a convolutional neural network. The images used for classification have a size of 32x32 pixels with 3 channel are from the CIFAR10 dataset.There are 10 classes to classify from plane,car,bird,cat,deer,dog,frog,horse,ship,truck.
# Below are the constraints
  - Architecture should be in the format of  C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) 
  - total RF must be more than 44
  - one of the layers must use Depthwise Separable Convolution
  - one of the layers must use Dilated Convolution
  - use GAP (compulsory):- add FC after GAP to target #of classes (optional)
  - use albumentation library and apply:
  - horizontal flip
  - shiftScaleRotate
  - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
  - achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

# Results 
  - Achieved more than 85% accuracy with less than 200 K parameters(195508)
  - Number of Epochs to achieve is less than 175  

## Files

- `model.py`: This file contains the architecture of the convolutional neural network model used for image classification. It defines the structure of the model and the forward propagation method . Class Name of the CNN model is `NetDilated`
- `SP- S9_v1.ipynb`: This Jupyter Notebook contains the main code to run the application for the batch normalizatoin . It demonstrates how to import the model architecture and training class from `model.py`
- `dataset.py` : This module has a class that loads the cifar10 data and apply necessary transformation. The DataTransformation class helps to apply the necessary image augmentation using albumentation 
- `utils.py`: This file contains all the utility funtions


## Additional Notes


  ### Wrongly Classified Predictation 
  ![](img/wrongly_classified_BN.png)

  ### Accuracy Graph
  ![](img/accuracy_graph_BN.png)

  ### Loss Graph
  ![](img/BN_loss.png)


