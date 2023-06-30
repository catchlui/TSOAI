# Image Classification Application for CIFAR10 dataset

This application is designed for image classification using a convolutional neural network. The images used for classification have a size of 32x32 pixels with 3 channel are from the CIFAR10 dataset.There are 10 classes to classify from plane,car,bird,cat,deer,dog,frog,horse,ship,truck.
# Below are the constraints
  - has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) 
  - total RF must be more than 44
  - one of the layers must use Depthwise Separable Convolution
  - one of the layers must use Dilated Convolution
  - use GAP (compulsory):- add FC after GAP to target #of classes (optional)
  - use albumentation library and apply:
  - horizontal flip
  - shiftScaleRotate
  - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
  - achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

## Files

- `model.py`: This file contains the architecture of the convolutional neural network model used for image classification. It defines the structure of the model and the forward propagation method.
- `SP- S8_v3_BatchNorm.ipynb`: This Jupyter Notebook contains the main code to run the application for the batch normalizatoin . It demonstrates how to import the model architecture and training class from `model.py`
- `SP- S8_v2_groupNorm.ipynb` : This jupyter Notebook contains the main code and graphs for Group Normalization
- `SP- S8_v3_layerNorm.ipynb`: This jupyter Notebook contains the main code and graphs for Layer Normalization

## Instructions

To use this application, follow these steps:

1. Install the required dependencies (PyTorch ).
2. Import the `model.py` file to access the model architecture,training, testing
3. Run the code in `SP- S8_v3_BatchNorm.ipynb` to train and test the model on your dataset with Batch Normalization Architecture

## Additional Notes

- what is your code all about ?
-   I have three models and three jupyter note books depicting Batch Normalization,Group Normalization and Layer Normalization
  ## Batch Normalization
-  Name of the file
-    `SP- S8_v3_BatchNorm.ipynb` -
-    Name of the model architecture NetBatchNorm present in the `model.py`
-    I was able to achieve more than 70% accuracy in 10 epoch with 47876 parameters
-    Without Batch Normalization the model was struggling to achieve same accuracy
-    11th Epoch : training loss: 0.2039, acc 71.31%  validation loss: 0.1946, validation acc 72.91% 
-    Added One cycle LR to change the learning rate and i found it to be very effective

  ### Wrongly Classified Predictation 
  ![](img/wrongly_classified_BN.png)

  ### Accuracy Graph
  ![](img/accuracy_graph_BN.png)

  ### Loss Graph
  ![](img/BN_loss.png)

  ### Advantages of Batch Normalization
    - Improved training speed: BN normalizes the activations within each mini-batch, reducing the internal covariate shift problem and allowing for faster convergence during training.
    - Increased stability and generalization: BN adds a regularization effect by reducing the dependence on specific weights and biases, making the model more robust and reducing overfitting.
    - Allows for higher learning rates: BN helps stabilize the gradient flow in the network, allowing for the use of larger learning rates without causing the gradients to explode.


  ### 
   ## Group Normalization
-   Name of the File
-   `SP- S8_v2_groupNorm.ipynb`
-   Name of the model architecture NetGroupNorm present in the `model.py`
-    I was able to achieve more than 70% accuracy in 10 epoch with 47876 parameters
-    In 10th Epoch training loss: 0.1968, acc 72.22% validation loss: 0.1859, validation acc 73.99%

  ### Advantages of Group Normalization
    Reduced batch size dependency: GN divides the channels into groups and normalizes the activations within each group, making it less sensitive to the batch size used during training.
    Effective for small batch sizes: GN can perform well even with small batch sizes, making it suitable for scenarios where memory constraints limit the batch size.
    Captures inter-channel correlations: GN considers the spatial dimensions of the input data and captures inter-channel correlations, making it suitable for CNNs.
      
   ### 
   ## Layer Normalization
    Name of the File
-   `SP- S8_v3_layerNorm.ipynb`
-   Name of the model architecture NetGroupNorm present in the `model.py`
-    I was able to achieve more than 70% accuracy in 10 epoch with 47876 parameters
-    In 20th Epoch training training loss: 0.1650, acc 76.80% validation loss: 0.1591, validation acc 77.93%
-    Observation : When i added layer norm, the number of parameters gone extremely high
-    Made elementwise_affine=False to decrease the number of parameters for layer norm function

  ### Advantages of Layer Normalization
      Invariant to batch size: LN normalizes the activations across each layer independently, making it suitable for scenarios with varying batch sizes or when batch statistics are unreliable.
    Stable performance: LN performs consistently well across different batch sizes and training scenarios.
    Suitable for recurrent neural networks (RNNs and LLMs): LN is particularly effective in RNNs due to its ability to handle variable-length sequences.
   

