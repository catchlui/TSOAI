from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import GetCorrectPredCount


import torch
import torch.nn as nn

class ResNet(nn.Module):

  def __init__(self):
    super(ResNet, self).__init__()

    # Block 1: Input preprocessing
    self.prep = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )

    # Block 2: First convolutional layer and pooling
    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )

    # Residual Block 1
    self.res1 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
    )

    # Block 3: Second convolutional layer and pooling
    self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(256),
        nn.ReLU()
    )

    # Block 4: Third convolutional layer and pooling
    self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )

    # Residual Block 2
    self.res2 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
    )

    # Max pooling layer
    self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4)

    # Fully connected layer
    self.fc1 = nn.Linear(512, 10, bias=False)

  def forward(self, x):
    out = self.prep(x)        # Block 1 (Input to Output): (3, 64, 32, 32) -> (64, 32, 32)
    out = self.layer1(out)    # Block 2 (Input to Output): (64, 32, 32) -> (128, 16, 16)
    res1 = self.res1(out)     # Residual Block 1 (Input to Output): (128, 16, 16) -> (128, 16, 16)
    out = res1 + out          # Residual connection
    out = self.layer2(out)    # Block 3 (Input to Output): (128, 16, 16) -> (256, 8, 8)
    out = self.layer3(out)    # Block 4 (Input to Output): (256, 8, 8) -> (512, 4, 4)
    res2 = self.res2(out)     # Residual Block 2 (Input to Output): (512, 4, 4) -> (512, 4, 4)
    out = res2 + out          # Residual connection
    out = self.maxpool1(out)  # Max pooling: (512, 4, 4) -> (512, 1, 1)
    out = out.view(out.size(0), -1)  # Flatten the output for fully connected layer
    out = self.fc1(out)      # Fully connected layer: (512) -> (10)

    return out





class NetDilated(nn.Module):
  def __init__(self,dropout=0.05):
    super(NetDilated,self).__init__()
    ## Convolution Block1
    self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias = False),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 128, 3, padding=1, bias = False), # Input: 32x32x32 | Output: 32x32x64 | RF: 5x5
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout)
        )
        
    ## Transition Block1
    self.trans1 = nn.Sequential(
            nn.Conv2d(128, 64,1, stride=2), # Input: 32x32x64 | Output: 16x16x32 | RF: 5x5
            nn.ReLU(),
        )

    ## Convolution Block2
    self.conv2 =  nn.Sequential(
            nn.Conv2d(64, 64, 3,  padding=1, bias = False), # Input: 16x16x32 | Output: 16x16x32 | RF: 9x9
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),

            ## Depthwise Seperable Convolution1
            nn.Conv2d(64,64, 3,  padding=1,groups=64 ,bias = False),  # Input: 16x16x32 | Output: 16x16x32 | RF: 13x13
            nn.Conv2d(64, 64, 1, padding=1, bias = False),   # Input: 16x16x32 | Output: 18x18x64 | RF: 13x13
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),
        )
        
    #Transition Block2
    self.trans2 = nn.Sequential(

            nn.Conv2d(64, 32, 1, stride=2), # Input: 18x18x32 | Output: 9x9x64 | RF: 13x13
            nn.ReLU()
        )

    #Convolution Block3
    self.conv3 = nn.Sequential(
            
            ## Dilation Block
            nn.Conv2d(32, 64, 3,  padding=1, bias = False,dilation=2), # Input: 9x9x64 | Output: 7x7x64 | RF: 29x29
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 64, 3,  padding=1, bias = False),  # Input: 7x7x64| Output: 7x7x64 | RF: 45x45
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),
        )

    #Transition Block3
    self.trans3 = nn.Sequential(

            nn.Conv2d(64, 32, 1, stride=2), # Input: 7x7x64| Output: 4x4x16 | RF: 61x61
            nn.ReLU()
        )

    #Convolution Block4        
    self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias = False), # Input: 4x4x16 | Output: 4x4x32 | RF: 93x93
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout),

            ## Depthwise seperable Convolution2
            nn.Conv2d(32,32, 3,  padding=1,groups=32 ,bias = False),# Input: 4x4x16 | Output: 4x4x32 | RF: 125x125
            nn.Conv2d(32, 10, 1, padding=1, bias = False),          # Input: 4x4x32| Output: 6x6x10 | RF: 125x125
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout2d(dropout),
        )

    ## Output Block
    self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        ) 


  def forward(self, x):

    x = self.conv1(x)
    x = self.trans1(x)

    x = self.conv2(x) 
    x = self.trans2(x) 

    x = self.conv3(x) 
    x = self.trans3(x)

    x = self.conv4(x)
    x = self.gap(x)

    x = x.view(-1,10)
    return x

class NetLayerNorm(nn.Module):
    def __init__(self):
        super(NetLayerNorm, self).__init__()
        # Input Block
        # Input channels: 3, Output channels: 20, Receptive Field: 3x3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm((20, 32, 32), elementwise_affine=False)
        )

        # Convolution Block 1
        # Input channels: 20, Output channels: 20, Receptive Field: 5x5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm((20, 32, 32), elementwise_affine=False)
        )

        # Convolution Block 1
        # Input channels: 20, Output channels: 20, Receptive Field: 7x7
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.LayerNorm((20, 32, 32), elementwise_affine=False)
        )

        # Transition Block 1
        # Max Pooling: Input channels: 20, Output channels: 20, Receptive Field: 8x8
        self.pool1 = nn.MaxPool2d(2, 2)
        # Convolution: Input channels: 20, Output channels: 32, Receptive Field: 8x8
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm((32, 32, 32), elementwise_affine=False)
        )

        # Convolution Block 2
        # Input channels: 32, Output channels: 32, Receptive Field: 12x12
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm((32, 16, 16), elementwise_affine=False)
        )
        # Input channels: 32, Output channels: 32, Receptive Field: 16x16
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.LayerNorm((32, 16, 16), elementwise_affine=False)
        )
        # Input channels: 32, Output channels: 32, Receptive Field: 16x16
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.LayerNorm((32, 16, 16), elementwise_affine=False)
        )

        # Transition Block 2
        # Input channels: 32, Output channels: 24, Receptive Field: 16x16
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=24, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.GroupNorm(8, 24)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # Max Pooling: Input channels: 24, Output channels: 24, Receptive Field: 18x18

        # Convolution Block 3
        # Input channels: 24, Output channels: 24, Receptive Field: 22x22
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.LayerNorm((24, 8, 8), elementwise_affine=False)
        )

        # Convolution Block 4
        # Input channels: 24, Output channels: 24, Receptive Field: 26x26
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.AdaptiveAvgPool2d(1)
        )

        # Output Block
        # Input channels: 24, Output channels: 10, Receptive Field: 26x26
        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )


    def forward(self, x):
        # passing the data through network

        x = self.convblock1(x)
        short_cut = x
        x = self.convblock2(x)
        x = short_cut + self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        #x = self.convblock11(x)
        x = self.convblock12(x)
        # Changing the shape to pass to the softmax
        x = x.view(-1, 10)
        return x

class NetGroupNorm(nn.Module):
    def __init__(self):
        super(NetGroupNorm, self).__init__()
       # Input Block
        # Input channels: 3, Output channels: 20, Receptive Field: 3x3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(5, 20)
        )  # output_size = | RF 3

        # Convolution Block 1
        # Input channels: 20, Output channels: 20, Receptive Field: 5x5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(5, 20)
        )  # output_size = 12, 24, 24 | RF 5

        # Convolution Block 1
        # Input channels: 20, Output channels: 20, Receptive Field: 7x7
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.GroupNorm(5, 20)
        )  # output_size = 22 | RF 7

        # Transition Block 1
        # Max Pooling: Input channels: 20, Output channels: 20, Receptive Field: 8x8
        self.pool1 = nn.MaxPool2d(2, 2)
        # Convolution: Input channels: 20, Output channels: 32, Receptive Field: 8x8
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.GroupNorm(8, 32)
        )  # output_size = 11 | RF 8

        # Convolution Block 2
        # Input channels: 32, Output channels: 32, Receptive Field: 12x12
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(8, 32)
        )  # output_size = 9 | RF 12
        # Input channels: 32, Output channels: 32, Receptive Field: 16x16
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.GroupNorm(8, 32)
        )  # output_size = 7 | RF 16
        # Input channels: 32, Output channels: 32, Receptive Field: 16x16
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.GroupNorm(8, 32)
        )

        # Transition Block 2
        # Input channels: 32, Output channels: 24, Receptive Field: 16x16
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=24, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.GroupNorm(8, 24)
        )  # output_size = 11 | RF 18
        # Max Pooling: Input channels: 24, Output channels: 24, Receptive Field: 18x18
        self.pool2 = nn.MaxPool2d(2, 2)
        # Input channels: 24, Output channels: 24, Receptive Field: 22x22
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.GroupNorm(8, 24)
        )
        # Input channels: 24, Output channels: 24, Receptive Field: 26x26
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.AdaptiveAvgPool2d(1)
        )

        # Output Block
        # Input channels: 24, Output channels: 10, Receptive Field: 26x26
        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # output_size = 10x1x1


    def forward(self, x):
        # passing the data through network

        x = self.convblock1(x)
        short_cut = x
        x = self.convblock2(x)
        x = short_cut + self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        
        x = self.convblock12(x)
        # Changing the shape to pass to the softmax
        x = x.view(-1, 10)
        return x #F.log_softmax(x, dim=-1)


class NetBatchNorm(nn.Module):
    def __init__(self):
        super(NetBatchNorm, self).__init__()
        # Input channels: 3, Output channels: 20, Receptive Field: 3x3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20)
        )  

        # Convolution Block 1
        # Input channels: 20, Output channels: 20, Receptive Field: 5x5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20)
        )  
        # Convolution Block 1
        # Input channels: 20, Output channels: 20, Receptive Field: 7x7
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(20)
        )  

        # Transition Block 1
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 11
        # Input channels: 20, Output channels: 32, Receptive Field: 8x8
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(32)
        )  

        # Convolution Block 2
        # Input channels: 32, Output channels: 32, Receptive Field: 12x12
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )  
        # Input channels: 32, Output channels: 32, Receptive Field: 16x16
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(32)
        )  
        # Input channels: 32, Output channels: 32, Receptive Field: 16x16
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(32)
        )

        # Transition Block 2
        # Input channels: 32, Output channels: 24, Receptive Field: 8x8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=24, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(24)
        )  
        self.pool2 = nn.MaxPool2d(2, 2) 

        # Input channels: 24, Output channels: 24, Receptive Field: 26x26
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(24)
        )
        # Input channels: 24, Output channels: 24, Receptive Field: 26x26
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.AdaptiveAvgPool2d(1)
        )

        # Output Block
        # Input channels: 24, Output channels: 10, Receptive Field: 26x26
        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # output_size = 10x1x1


    def forward(self, x):
        # passing the data through network
        
        x = self.convblock1(x)
        short_cut = x
        x = self.convblock2(x)
        x = short_cut + self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x =  self.convblock7(x)
        x = self.pool2(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.convblock12(x)
        # Changing the shape to pass to the softmax
        x = x.view(-1, 10)
        return x 



## class to train & validate the model
class Train:
  def __init__(self,model,device,criterion,optimizer,scheduler,train_loader,test_loader,num_epochs,batch_size,train_mode=True):
    super(Train,self).__init__()
    self.model =model
    self.device = device
    self.criterion = criterion
    self.optimizer = optimizer 
    self.scheduler = scheduler
    self.train_mode = train_mode
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.batch_size = batch_size
    self.epochs = num_epochs
    self.running_loss_history = []
    self.running_corrects_history=[]
    self.val_running_loss_history = []
    self.val_running_corrects_history = []
    self.step_to_lr = []
  
  def __call__(self):

    for e in range(self.epochs): # training our model, put input according to every batch.

      running_loss = 0.0
      running_corrects = 0.0
      val_running_loss = 0.0
      val_running_corrects = 0.0
      train_processed =0
      test_processed = 0

      for inputs, labels in self.train_loader:
        inputs = inputs.to(self.device) # input to device as our model is running in mentioned device.
        labels = labels.to(self.device)
        outputs = self.model(inputs) # every batch of 100 images are put as an input.
        loss = self.criterion(outputs, labels) # Calc loss after each batch i/p by comparing it to actual labels.

        self.optimizer.zero_grad() #setting the initial gradient to 0
        loss.backward() # backpropagating the loss
        self.optimizer.step() # updating the weights and bias values for every single step.

        _, preds = torch.max(outputs, 1) # taking the highest value of prediction.
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data) # calculating te accuracy by taking the sum of all the correct predictions in a batch.
        train_processed += len(inputs)
        self.step_to_lr.append(self.scheduler.get_last_lr())
        #print("LR Value for the step",self.scheduler.get_last_lr())
        self.scheduler.step()
        


      with torch.no_grad(): # we do not need gradient for validation.
        for val_inputs, val_labels in self.test_loader:
          val_inputs = val_inputs.to(self.device)
          val_labels = val_labels.to(self.device)
          val_outputs = self.model(val_inputs)
          val_loss = self.criterion(val_outputs, val_labels)

          _, val_preds = torch.max(val_outputs, 1)
          val_running_loss += val_loss.item()
          val_running_corrects += torch.sum(val_preds == val_labels.data)
          test_processed += len(val_inputs)


        epoch_loss = running_loss/train_processed # loss per epoch
        epoch_acc = running_corrects.float()/ train_processed # accuracy per epoch
        self.running_loss_history.append(epoch_loss) # appending for displaying
        self.running_corrects_history.append(epoch_acc)
        val_epoch_loss = val_running_loss/test_processed
        self.val_epoch_acc = val_running_corrects.float()/ test_processed
        self.val_running_loss_history.append(val_epoch_loss)
        self.val_running_corrects_history.append(self.val_epoch_acc)
        print('epoch :', (e+1))
        print('training loss: {:.4f}, acc {:.2%} '.format(epoch_loss, epoch_acc.item()))
        print('validation loss: {:.4f}, validation acc {:.2%} '.format(val_epoch_loss, self.val_epoch_acc.item()))
   