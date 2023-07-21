# Image Classification Application for MNIST dataset

This application is designed for image classification using a convolutional neural network. The images used for classification have a size of 28x28 pixels with 1 channel are from the MNIST dataset.There are 10 classes to classify from 0 to 9.

# Requirement for this assigment
  Write a customLinks to an external site. ResNet architecture for CIFAR10 that has the following architecture:
  PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
  Layer1 -
  X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
  R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
  Add(X, R1)
  Layer 2 -
  Conv 3x3 [256k]
  MaxPooling2D
  BN
  ReLU
  Layer 3 -
  X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
  R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
  Add(X, R2)
  MaxPooling with Kernel Size 4
  FC Layer 
  SoftMax
  Uses One Cycle Policy such that:
  Total Epochs = 24
  Max at Epoch = 5
  LRMIN = FIND
  LRMAX = FIND
  NO Annihilation
  Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
  Batch size = 512
  Use ADAM, and CrossEntropyLoss
  Target Accuracy: 90%
  

## Files

- `model.py`: This file contains the architecture of the convolutional neural network model used for image classification. It defines the structure of the model and the forward propagation method.
- `utils.py`: This file contains utility functions for training, testing, and calculating accuracy. It includes functions such as `train`, `test`, and `get_accuracy`, which are used for training the model, testing it on unseen data, and calculating the accuracy, respectively.
- `dataset.py` , has the dataset related class and the data transformation related class
- `S5_S10_v1.ipynb`: This Jupyter Notebook contains the main code to run the application. It demonstrates how to import the model architecture from `model.py` and use the utility functions from `utility.py` to train and test the model.

## Instructions

To use this application, follow these steps:

1. Install the required dependencies (PyTorch, tqdm, ).
2. Import the `model.py` file to access the model architecture.
3. Import the utility functions from `utils.py` to perform training, testing, and accuracy calculations.
4. import the dataset from the `dataset.py`
5. Run the code in `S5_S10_v1.ipynb` to train and test the model on your dataset.

## Additional Notes

- Make sure your dataset is prepared and formatted correctly before running the application.
- Adjust the hyperparameters and model architecture in `model.py` and the training/testing procedures `utils.py` and main code in `S5_Snehashis.ipynb` as per your specific requirements.
- For more detailed information, refer to the code comments within each files.

