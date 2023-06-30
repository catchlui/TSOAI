import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
import albumentations 
import albumentations as A
import albumentations.pytorch
from albumentations.pytorch import ToTensorV2



class Cifar10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
            label = torch.tensor(label)

        return image, label


class DataTransformation:
  def __init__(self,mean,std):
    super(DataTransformation,self).__init__()
    self.train = A.Compose([A.HorizontalFlip(p=.3),
                                  A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=.3),
                                  A.CoarseDropout(max_holes=1,min_holes = 1, max_height=16, max_width=16, 
                                  p=.5,fill_value=tuple([x * 255.0 for x in mean]),
                                  min_height=16, min_width=16),
                                  A.ColorJitter(p=0.25,brightness=0.3, contrast=0.3, saturation=0.30, hue=0.2),
                                  A.ToGray(p=0.15),
                                  A.Normalize(mean=mean, std=std,always_apply=True),
                                  ToTensorV2()
                                ])
    self.test = A.Compose([A.Normalize(mean=mean, std=std, always_apply=True),
                                 ToTensorV2()])
    

  









