# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:47:35 2023

@author: 97091

This is an example of fine_tunning the S+T action detector model 
It has better convergence with less training time
Input: spitial hand-object interaction with correspoding motions
Output: Action labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from torch.utils.data import Dataset
import pandas as pd
import os
import torchvision
from PIL import Image
from torch.utils.data import DataLoader


class CNN_Spatio(nn.Module): #pre-define the VGG model, final output layer not included
    def __init__(self, *,h_1=4096,h_2=2048,emd_size=128):
        super(CNN_Spatio, self).__init__()
        self.model=models.vgg16(init_weights = False)
        hidden_layers = [4096, 2048]
        



        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, hidden_layers[0])),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p = 0.3)),
                            ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p = 0.3)),
                            # ('fc3', nn.Linear(hidden_layers[1], 128))
                                                    
                            ]))
        self.fc3= nn.Linear(hidden_layers[1], 128)
        self.model.classifier=classifier
        model_dict=self.model.state_dict()
        pretrained_dict=torch.load('model.pt')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(pretrained_dict)
    def forward(self,x):
        features=self.model.forward(x)
        
        features=self.fc3(features)
        return features

class MLSTMfcn(nn.Module):#pre-define the LSTM_FCN model, final output layer not included
    def __init__(self, *, num_features,
                 num_lstm_out=128, num_lstm_layers=1, 
                 conv1_nf=128, conv2_nf=256, conv3_nf=128,
                 lstm_drop_p=0.8, fc_drop_p=0.3):
        super(MLSTMfcn, self).__init__()

        
        
        self.num_features = num_features

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=self.num_features, 
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)
        
        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

       

        self.relu = nn.ReLU()
       
        self.convDrop = nn.Dropout(self.fc_drop_p)

       
    
    def forward(self, x):
        ''' input x should be in size [B,T,F], where 
            B = Batch size
            T = Time samples
            F = features
        '''
       
        x1, (ht,ct) = self.lstm(x)
        # x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True, 
                                                  # padding_value=0.0)
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2,1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        # # x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        # x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2,2)
        
        x_all = torch.cat((x1,x2),dim=1)
        # print(x2.shape)
        
        # x_out = F.log_softmax(x_out, dim=1)

        return x_all


class LSTM_Temporal(nn.Module):#pre-load the VGG model, with modefied final output embedding
    def __init__(self, embed_size, num_features
                  
                 ):
        super(LSTM_Temporal, self).__init__()

        self.model=MLSTMfcn(num_features=2)
        model_dict=self.model.state_dict()
        pretrained_dict=torch.load('motion.pt')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(pretrained_dict)

        self.fc = nn.Linear(256, embed_size)
    
    def forward(self, x):
        ''' input x should be in size [B,T,F], where 
            B = Batch size
            T = Time samples
            F = features
        '''
        x_all=self.model(x)
       
        x_out=self.fc(x_all)
       

        return x_out
class S_T(nn.Module):
    def __init__(self,num_features,h_1,h_2,emd_size,output):
        super(S_T, self).__init__()
        self.LSTM_Temporal=LSTM_Temporal(emd_size, num_features)
        self.CNN_Spatio=CNN_Spatio() #pre-load VGG model with modefied final output embedding
        self.fc = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(emd_size+emd_size, 128)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p = 0.3)),
                            ('fc2', nn.Linear(128, 64)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p = 0.3)),
                            ('fc3', nn.Linear(64, output)),
                            ('output', nn.LogSoftmax(dim = 1))                       
                             ]))
    def forward(self,x2,x1):
        x_1=self.LSTM_Temporal(x1)
        x_2=self.CNN_Spatio(x2)
        x_all = torch.cat((x_1,x_2),dim=1)
        x=self.fc(x_all)
        return x
        
    

model=S_T(2,4096,2048,128,18)        
m_=torch.load('data.pt')

class CustomeDataset(Dataset): #This is the customized dataset for storing motions,images and labels
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform
        self.motion=m_

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        y_label = torch.tensor(float(self.annotations.iloc[index, 1]))
        motion=self.motion[self.annotations.iloc[index, 2]]
        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label,motion)
transform = transforms.Compose(
        [
            transforms.Resize((140, 140)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    
    
dataset=CustomeDataset('.\image', 'train_csv.csv',transform)
train_set, validation_set = torch.utils.data.random_split(dataset,[int(len(dataset)*0.9),len(dataset)-int(len(dataset)*0.9)]) 
train_loader=DataLoader(train_set,shuffle=True,batch_size=32)
validation_loader=DataLoader(validation_set,shuffle=True,batch_size=32)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
criterion = nn.NLLLoss()
def cal_accuracy(model, dataloader):
    validation_loss = 0
    accuracy = 0
    for i, (inputs,labels,motions) in enumerate(dataloader):
                optimizer.zero_grad()
                labels = labels.type(torch.LongTensor)
                inputs, labels,motions = inputs.to('cuda') , labels.to('cuda'),motions.to('cuda')
                model.to('cuda')
                with torch.no_grad():    
                    outputs = model.forward(inputs,motions)
                     
                    validation_loss = criterion(outputs,labels)
                    ps = torch.exp(outputs).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
    validation_loss = validation_loss / len(dataloader)
    accuracy = accuracy /len(dataloader)
    
    return validation_loss, accuracy
def train(model, image_trainloader, image_valloader, epochs, print_every, criterion, optimizer):
        epochs = 10000
        print_every = print_every
        steps = 0
        

        # change to cuda
        model.to('cuda')

        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels,motions) in enumerate(image_trainloader):
                model.train()
                steps += 1
                labels = labels.type(torch.LongTensor)
                inputs, labels,motions = inputs.to('cuda') , labels.to('cuda'),motions.to('cuda')

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs,motions)
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
                    if train_ac==1 and val_ac==1:
                        torch.save(model.state_dict(), 'model1.pt')
                    

                    running_loss = 0
train(model, train_loader, validation_loader, 10, 10, criterion, optimizer)
torch.save(model.state_dict(), 'model1.pt')
