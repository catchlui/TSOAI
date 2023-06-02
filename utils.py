import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


## This function checks the accuracy of the prediction
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


# dictionary to store the test incorrect prediction
test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}
# train function takes model , device, train loader, optimizer and a loss function
def train(model, device, train_loader, optimizer, criterion):
  # model set for training
  model.train()
  # load the train loader with tqdm to track the progress
  pbar = tqdm(train_loader)
  # initiatize loss data 
  train_loss = 0
  correct = 0
  processed = 0
  # for each batch get the batch idx,data and target
  for batch_idx, (data, target) in enumerate(pbar):
    # move the data to device cuda or CPU
    data, target = data.to(device), target.to(device)
    # flush earlier grads
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    # accumulate lost
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    # applies optimizer to optimize the weights
    optimizer.step()
    # find out the current train accuracy
    correct += GetCorrectPredCount(pred, target)
    # keep track of process data, as the last batch may not have full number of records
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
  # log the train loss and accuracy
  train_acc=100*correct/processed
  train_loss=train_loss/len(train_loader)
  return train_acc,train_loss
# this is for testing the performance of the trained model
# model,device , testloader and loss 
def test(model, device, test_loader, criterion):
   # put the model into eval mode
    model.eval()
   #initializes the loss
    test_loss = 0
    correct = 0
    # ensure no grads are calculated for test 
    with torch.no_grad():
        # for the test loader batch
        for batch_idx, (data, target) in enumerate(test_loader):
           # move to device
            data, target = data.to(device), target.to(device)
            # result of the trained model
            output = model(data)
            #test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            # calculate loss
            loss = criterion(output,target)
            # accumulate loss to calculate overall loss
            test_loss += loss.item() 

            correct += GetCorrectPredCount(output, target)

    # over all test loss and accuracy
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    
    # print information also returns the train and test loss,accuracy
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_acc,test_loss