

#  Image Classification Application

This application is designed for image classification using a convolutional neural network. The images used for classification have a size of 28x28 pixels with 1 channel are from the MNIST dataset.There are 10 classes to classify.

The plan was to bring the test accuracy to more than 99.4 percent within 8 K parameters and 15 epoch. The solution crosses 99.4 in 12th epoch and the final accuracy is consistantly crossed 99.40 percent accuracy
To achieve this i need to redesign the model architecture and also added rotational transformation to it. I did not use padding as when i visualize the image i did not see any portion of the images in edge  
## Files

- `model.py`: This file contains the architecture of the convolutional neural network model used for image classification. It defines the structure of the model and the forward propagation method.
You can instantiate the Model by importing the Net Class from the model module.
`from model import Net`
This also includes training and test functions which takes care of the training and testing of the model. More information about the parameters of these functions have been provided in the descriptions of the functions 
 
- `utility.py`: This file contains utility functions on the accuracy of the prediction`GetCorrectPredCount`and `display_model_summary` which are used displaying the model summary. You 
- `SP_S7_skl_v2_final_model.ipynb`: This Jupyter Notebook contains the main code to run the application. It imports model architecture from `model.py` and use the utility functions from `utility.py` to train and test the model. It also displays the train and test loss through graph.The final model achieves the target accuracy.
- `SP_S7_skl_initial_model.ipynb`: This Jupyter Notebook contains the model which has the initial skeleton model.
- `SP_S7_skl_v2_intermediate_model.ipynb` : This jupyter Notebook contains an intermediate model
- `TSOAI_assignment7.docx`: This document has the analysis and screen shots of each of the experiments

## Instructions

To use this application, follow these steps:

1. Install the required dependencies (PyTorch, tqdm, etc.).
2. Import the `model.py` file to access the model architecture.
3. Import the utility functions from `utils.py` to perform training, testing, and accuracy calculations.
4. Run the code using the respective jupyter notebook  to train and test the model on your dataset.

## Additional Notes

- Make sure your dataset is prepared and formatted correctly before running the application.
- For more detailed information, refer to the code comments within each file.

## Intial Skeleton Model

Target:
Create a skeleton model which trains with MNIST data

Results:
 Number of Parameters 15408

 Best Training Accuracy 99.54

 Best Test Accuracy 98.62

 Number of Epoch 15

 Model Name Net2


Analysis:

 Model has more capacity to learn
 It’s over fitting model
 
## Intermediate Model

Target:

 Improve the Accuracy of the model
 We will try to do some image augmentation
 Reduce Overfitting in the later stage
Number of Parameters 6422

Best Training Accuracy 99.13

Best Test Accuracy 99.28

Number of Epoch 15

Model Name Net4

Analysis:

 Maintained the model size
 There is no more over-fitting. Test Accuracy is always greater than training accuracy

## Final Model

Target

Improved Accuracy of the  accuracy and take it beyond 99.40 percent in last few epochs
 Let’s change the LR using scheduler
 After 6th epoch we are reducing by factor of gamma
Number of Parameters 6422

Best Training Accuracy 99.50

Best Test Accuracy 99.40

Number of Epoch 15

Model Name Net6

Analysis : 
The network is constantly hitting 99.40 with some amount of over-fitting..

Submitting the model as it's meeting the objective
