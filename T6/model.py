from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import GetCorrectPredCount

class Net(nn.Module):
    def __init__(self):
      # initialize the superclass constructor
        super(Net, self).__init__()
       
        
        self.conv1_block = nn.Sequential(
            # Input channel: 1, Output channel: 8, Kernel size: 3x3
            nn.Conv2d(1, 8, 3),
            # Batch Normalization
            nn.BatchNorm2d(8),
            # relu activation
            nn.ReLU(),
            # Regularization
            nn.Dropout(0.05),
            # Input channel: 8, Output channel: 16, Kernel size: 3x3
            nn.Conv2d(8, 16, 3, padding=0),
            # batch normalization 
            nn.BatchNorm2d(16),
            #activation
            nn.ReLU(),
            # regularization 
            nn.Dropout(0.05),
            # # Input channel: 16, Output channel: 16, Kernel size: 3x3
            nn.Conv2d(16, 16, 3, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05),
            # max Pooling
            nn.MaxPool2d(2, 2),
            
        )
        # Second block
        self.conv2_block = nn.Sequential(
            # Input channel: 16, Output channel: 16, Kernel size: 3x3
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05),
            # Input channel: 16, Output channel: 16, Kernel size: 3x3 
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05),
            # Input channel: 16, Output channel: 16, Kernel size: 3x3
            nn.Conv2d(16,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05),
            # Input channel: 16, Output channel: 16, Kernel size: 3x3
            nn.Conv2d(16,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(16,16,3),
           
          
        )
        #Global adaptive avg pool
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # fully connected to give the logit
        self.fc1 = nn.Linear(16,10)



    def forward(self, x):
        x = self.conv1_block(x) # 28|26|24|22|11   RF 3|5|7|8
        x = self.conv2_block(x) # 9|7|5|3|1        RF 12|16|20|24|28
        x = self.global_avg_pool(x)
        # flatten it to feed to fc  
        x = x.view(-1, 16)
        x = self.fc1(x)
        # apply log softmax
        return F.log_softmax(x,dim=1)

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
      pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
      # find out the current train accuracy
      correct += GetCorrectPredCount(output, target)
      processed += len(data)
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
