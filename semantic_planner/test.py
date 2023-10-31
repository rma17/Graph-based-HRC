# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:36:15 2023

@author: ruidong

This is the simulation test script for the semantic planner GNN_LSTM

Testing data includes:
    1. edge2.csv  testing graph's edges
    2. graph1.csv testing graphs
    3. index1.csv  testing indexes
    4. des2.txt  ground truth 
"""

import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import torch.nn.functional as F
import math
from HRC_Encoder1 import GNN
from HRC_Decoder import DecoderRNN
datas = pd.read_csv('./Samentic_Data/edge2.csv').values
index=[i for i in range(0,datas.shape[0]+1,2)]
edges=[]
for i in range(0,len(index)-1):
    temp=datas[index[i]:index[i+1]]
    edges.append(temp)
edge=[]    
for k in range(0,len(edges)):
    a=[]
    for e in edges[k]:
        x=[item for item in e if math.isnan(item)==False]
        a.append(x)
    edge.append(a)
def process_csv(path,path1,edges):
    
    
    
    
     datas = pd.read_csv(path).values
     
     indi= pd.read_csv(path1).values


     a={i:list(np.where(indi==i)[0]) for i in np.unique(indi)}
     BATCH=[]
     for k in a.keys():
      
      x=torch.tensor(datas[a[k],:],dtype=torch.float)
      
      
      
      
      edge_index=torch.tensor(edges[k],dtype=torch.long)
     
     
      
      data=Data(x=x,edge_index=edge_index)
      BATCH.append(data)
    
     return BATCH

dataset=process_csv('./Samentic_Data/graph1.csv', './Samentic_Data/index1.csv', edge)
Captions = open('./Samentic_Data/des2.txt')
Caption = Captions.read().split("\n")
Encoder = GNN(hidden_channels=128)
Decoder = DecoderRNN(131,100,11)
Encoder.load_state_dict(torch.load('encoder.pt'))
Decoder.load_state_dict(torch.load('decoder.pt'))
count=0
def trans_label(sequence):
   
    to_idx= {"0":0,"Gate": 1, "Ball": 2, "Globe": 3,'One':4,'Two':5,
                                     'Three':6,'Four':7,'Five':8,'Six':9,'Finished':10}
    idx = [to_idx[w] for w in sequence]
    return idx
for i in range(len(dataset)):
    data=dataset[i]
    ca=Caption[i]
    batch=[0]*data.x.size()[0]
    batch=torch.tensor(batch)
    out,state=Encoder(data.x.float(),data.edge_index.long(),batch)
    out=torch.argmax(out)
    out=F.one_hot(out,num_classes=3)
    state=torch.argmax(state)
    embedding=Encoder.node_embeedings(data.x.float(),data.edge_index.long(), state.cpu().detach().numpy()+6)
    output=Decoder.Predict(embedding.unsqueeze(0), out.unsqueeze(0))
    output=list(filter(lambda x: x != 0, output))
    if output==trans_label(ca.split()):
        count=count+1
print(count/len(dataset))
     
    