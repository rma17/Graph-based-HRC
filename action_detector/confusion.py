# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 23:48:21 2023

@author: ruidong
Confusion Matrix script
"""

from sklearn.metrics import confusion_matrix
import seaborn as sns
from CNN_LSTM import S_T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from torch.utils.data import Dataset
import pandas as pd
import os

from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt 
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

loader=DataLoader(dataset,shuffle=True,batch_size=1)











model=S_T(2,4096,2048,128,18) 
model.load_state_dict(torch.load('model1.pt'))
model.eval()
model.to('cuda')
y_test=[]
y_pred=[]
for i,(inputs,labels,motions) in enumerate(loader):
    
    labels = labels.type(torch.LongTensor)
    inputs, labels,motions = inputs.to('cuda') , labels.to('cuda'),motions.to('cuda')
    output=model(inputs,motions)
    y_pred.append(torch.argmax(output,dim=1).detach().cpu().numpy()[0])
    y_test.append(labels.detach().cpu().numpy()[0])
    
    
    
    












cm = confusion_matrix(y_test, y_pred)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
target_names=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)