# Image Classification Application for MNIST dataset

This application is designed for image classification using a convolutional neural network. The images used for classification have a size of 28x28 pixels with 1 channel are from the MNIST dataset.There are 10 classes to classify from 0 to 9.This application has 4 conv layers with two fully connected layers.
This is a very rudimentary architecture.One should not follow this architecture.

## Files

- `model.py`: This file contains the architecture of the convolutional neural network model used for image classification. It defines the structure of the model and the forward propagation method.
- `utility.py`: This file contains utility functions for training, testing, and calculating accuracy. It includes functions such as `train`, `test`, and `get_accuracy`, which are used for training the model, testing it on unseen data, and calculating the accuracy, respectively.
- `Session_5_Snehashis.ipynb`: This Jupyter Notebook contains the main code to run the application. It demonstrates how to import the model architecture from `model.py` and use the utility functions from `utility.py` to train and test the model.

## Instructions

To use this application, follow these steps:

1. Install the required dependencies (PyTorch, tqdm, ).
2. Import the `model.py` file to access the model architecture.
3. Import the utility functions from `utility.py` to perform training, testing, and accuracy calculations.
4. Run the code in `S5_SP.ipynb` to train and test the model on your dataset.

## Additional Notes

- Make sure your dataset is prepared and formatted correctly before running the application.
- Adjust the hyperparameters and model architecture in `model.py` and the training/testing procedures `utility.py` and main code in `Session_5_Snehashis.ipynb` as per your specific requirements.
- For more detailed information, refer to the code comments within each files.
-
