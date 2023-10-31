# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:15:33 2023

@author: 97091
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 21:45:25 2023

@author: 97091
This is the example of spitial hand-object interaction classifaiction using VGG16
Data sets includes:
    1: Spitial Image
    2. Corresponding 6 labels 



"""

from torch.utils.data import Dataset
import pandas as pd
import os
import torchvision
from PIL import Image
import torch
from torch.utils.data import DataLoader
import cv2
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data 
import pandas as pd
import numpy as np
from collections import OrderedDict

from torch import nn
from torch import optim
class CustomeDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        y_label = torch.tensor(float(self.annotations.iloc[index, 1]))

        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)

transform = transforms.Compose(
        [
            transforms.Resize((140, 140)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    
    
dataset=CustomeDataset('.\image', 'train_csv1.csv',transform)   
train_set, validation_set = torch.utils.data.random_split(dataset,[180,2]) 
train_loader=DataLoader(train_set,shuffle=True,batch_size=16)
validation_loader=DataLoader(validation_set,shuffle=True,batch_size=32)

   


model = models.vgg16(init_weights = True)
for param in model.parameters():
    param.required_grad = False
hidden_layers = [4096, 2048]
input_size = 25088    
output_size=4



classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, hidden_layers[0])),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p = 0.3)),
                            ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p = 0.3)),
                            ('fc3', nn.Linear(hidden_layers[1], 6)),
                            ('output', nn.LogSoftmax(dim = 1))                        
]))


model.classifier = classifier    
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
criterion = nn.NLLLoss()
def cal_accuracy(model, dataloader):
    validation_loss = 0
    accuracy = 0
    for i, (inputs,labels) in enumerate(dataloader):
                optimizer.zero_grad()
                labels = labels.type(torch.LongTensor)
                inputs, labels = inputs.to('cuda') , labels.to('cuda')
                model.to('cuda')
                with torch.no_grad():    
                    outputs = model.forward(inputs)
                     
                    validation_loss = criterion(outputs,labels)
                    ps = torch.exp(outputs).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
    validation_loss = validation_loss / len(dataloader)
    accuracy = accuracy /len(dataloader)
    
    return validation_loss, accuracy

def train(model, image_trainloader, image_valloader, epochs, print_every, criterion, optimizer):
        epochs = epochs
        print_every = print_every
        steps = 0

        # change to cuda
        model.to('cuda')

        for e in range(200):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(image_trainloader):
                steps += 1
                labels = labels.type(torch.LongTensor)
                inputs, labels = inputs.to('cuda') , labels.to('cuda')

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()
                    val_loss, val_ac = cal_accuracy(model, image_valloader)
                    train_loss, train_ac = cal_accuracy(model, image_trainloader)
                    print("Epoch: {}/{}... | ".format(e+1, epochs),
                          "Loss: {:.4f} | ".format(running_loss/print_every),
                          "val_ac {:.4f} | ".format(val_ac),
                          "train_ac {:.4f}".format(train_ac))

                    running_loss = 0
train(model, train_loader, validation_loader, 10, 10, criterion, optimizer)
torch.save(model.state_dict(), 'model.pt')