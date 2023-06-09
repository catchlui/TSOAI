
# Image Classification Application

This application is designed for image classification using a convolutional neural network. The images used for classification have a size of 28x28 pixels with 1 channel are from the MNIST dataset.There are 10 classes to classify.

The plan was to bring the test accuracy to more than 99.4 percent within 20 K parameters and 20 epoch. The solution crosses 99.4 in 15th epoch and the final accuracy is 99.44.
To achieve this i need to redesign the model architecture and also added rotational transformation to it. I did not use padding as when i visualize the image i did not see any portion of the images in edge  
## Files

- `model.py`: This file contains the architecture of the convolutional neural network model used for image classification. It defines the structure of the model and the forward propagation method.
You can instantiate the Model by importing the Net Class from the model module.
`from model import Net`
This also includes training and test functions which takes care of the training and testing of the model. More information about the parameters of these functions have been provided in the descriptions of the functions 
 
- `utility.py`: This file contains utility functions on the accuracy of the prediction`GetCorrectPredCount`and `display_model_summary` which are used displaying the model summary. You 
- `SP_S6.ipynb`: This Jupyter Notebook contains the main code to run the application. It imports model architecture from `model.py` and use the utility functions from `utility.py` to train and test the model. It also displays the train and test loss through graph 

## Instructions

To use this application, follow these steps:

1. Install the required dependencies (PyTorch, tqdm, etc.).
2. Import the `model.py` file to access the model architecture.
3. Import the utility functions from `utils.py` to perform training, testing, and accuracy calculations.
4. Run the code in `SP_S6.ipynb` to train and test the model on your dataset.

## Additional Notes

- Make sure your dataset is prepared and formatted correctly before running the application.
- Adjust the hyperparameters and model architecture in `model.py` and the training/testing procedures in `SP_S6.ipynb` as per your specific requirements.
- For more detailed information, refer to the code comments within each file.
