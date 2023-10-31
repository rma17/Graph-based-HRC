# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 22:30:15 2023

@author: 97091
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
class MLSTMfcn(nn.Module):
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


class LSTM_Temporal(nn.Module):
    def __init__(self, embed_size, num_features
                  
                 ):
        super(LSTM_Temporal, self).__init__()

        self.model=MLSTMfcn(num_features=2)
      

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
class CNN_Spatio(nn.Module):
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
        
    def forward(self,x):
        features=self.model.forward(x)
        
        features=self.fc3(features)
        return features
class S_T(nn.Module):
    def __init__(self,num_features,h_1,h_2,emd_size,output):
        super(S_T, self).__init__()
        self.LSTM_Temporal=LSTM_Temporal(emd_size, num_features)
        self.CNN_Spatio=CNN_Spatio()
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
        
    
if __name__=='__main__':
    model=S_T(2,4096,2048,128,17)        
    m_=torch.load('data.pt')
    
    class CustomeDataset(Dataset):
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
    model.load_state_dict(torch.load('model1.pt'))
    z=torch.load('label.pt')
    model.eval()
    count=0
    model.to('cuda')
    image=Image.open('test.jpg').convert("RGB")
    tra=dataset[858][2]
    image=transform(image)
    image=image.to('cuda')
    
    tra=tra.to('cuda')
    output=model(image.float().unsqueeze(0),tra.float().unsqueeze(0))
    print(torch.argmax(output,dim=1))
    for image,label,tra in dataset:
     
    
      image=image.to('cuda')
    
      tra=tra.to('cuda')
     
    
      output=model(image.float().unsqueeze(0),tra.float().unsqueeze(0))
        
      if torch.argmax(output,dim=1)==label:
          count=count+1
       
    