from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import GetCorrectPredCount

## Skeleton Model
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.ReLU() NEVER!
        ) # output_size = 1 7x7x10 | 7x7x10x10 | 1x1x10

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # output_size = 26 | RF 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12)
        ) # output_size = 24 |  RF 5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(.05),
            nn.BatchNorm2d(12)
        ) # output_size = 22 | RF 7
  

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.BatchNorm2d(12)
        ) # output_size = 11 |RF 8

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12)
        ) # output_size = 9 | RF 12
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(.05),
            nn.BatchNorm2d(12)
        ) # output_size = 7 | RF 16

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 5 | RF 18
       

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            #nn.ReLU()
            #nn.AdaptiveAvgPool2d(1)
        ) # output_size = 3 | RF 26
       
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        #x = self.convblock8(x)
        #x = self.convblock9(x)
        
        #x = m(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # output_size = 10,26,26 | RF 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12)
        ) # output_size = 12,24,24 | RF 5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(.05),
            nn.BatchNorm2d(12)
        ) # output_size = 22 | RF 7
  

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.BatchNorm2d(12)
        ) # output_size = 11 | RF 8

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12)
        ) # output_size = 9 | RF 12
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(.05),
            nn.BatchNorm2d(12)
        ) # output_size = 7 | RF 16

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 3 | RF 18
       

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(3, 3), padding=0, bias=False), # outputsize = 1| RF 26
            #nn.ReLU()
            nn.AdaptiveAvgPool2d(1)
        ) # output_size = 10 * 1 * 1
       

    def forward(self, x):
        # passing the data through network
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        # Changing the shape to pass to the softmax
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



def train(model, device, train_loader, optimizer, epoch):
  # model set for training
    model.train()
    # model set for training
    pbar = tqdm(train_loader)
    # model set for training
    train_loss = 0
    correct = 0
    processed = 0
  
    # for each batch get the batch idx,data and target
    for batch_idx, (data, target) in enumerate(pbar):

      # for each batch get the batch idx,data and target
      data, target = data.to(device), target.to(device)
      # flush accumulated grads
      optimizer.zero_grad()
      # Predict
      output = model(data)
      # Calculate loss
      loss = F.nll_loss(output, target)
      # accumulate loss
      train_loss+=loss.item()
      # Backpropagation
      loss.backward()
      # applies optimizer to optimize the weights
      optimizer.step()
     
      # find out the current train accuracy
      correct += GetCorrectPredCount(output, target)
      processed += len(data)
      pbar.set_description(desc= f'Loss={loss.item()} Accuracy={100*correct/processed:0.2f}')
    train_acc=100*correct/processed
    train_loss=train_loss/len(train_loader)
    
    return train_acc,train_loss
     
        

def test(model, device, test_loader):
    #set the model eval
    model.eval()
    # set the test counter
    test_loss = 0
    # set the accuracy
    correct = 0
    # set the system not calculate grad
    with torch.no_grad():
        # loop through batch
        for data, target in test_loader:
            # load the data to device
            data, target = data.to(device), target.to(device)
            # predict the target
            output = model(data)
            # calculate the loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # find the predicted class
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # find out the accuracy
            correct += pred.eq(target.view_as(pred)).sum().item()
    # test loss
    test_loss /= len(test_loader.dataset)
    # test accuracy
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_acc,test_loss
