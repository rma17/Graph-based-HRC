# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 23:17:56 2022

@author: 97091
"""

import torch
# from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from torch_geometric.nn import GraphConv
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.nn import ReLU
from torch.nn import Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,GNNExplainer
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import KFold
import math
# datas = pd.read_csv('./Samentic_Data/edge.csv').values

# index=[i for i in range(0,datas.shape[0]+1,2)]
# edges=[]
# for i in range(0,len(index)-1):
#     temp=datas[index[i]:index[i+1]]
#     edges.append(temp)
# edge=[]    
# for k in range(0,len(edges)):
#     a=[]
#     for e in edges[k]:
#         x=[item for item in e if math.isnan(item)==False]
#         a.append(x)
#     edge.append(a)

def process_csv(path,path1,path2,edges,path3):
    
    
    
    
     datas = pd.read_csv(path).values
     
     indi= pd.read_csv(path1).values
     
     label=pd.read_csv(path2).values
     
     state=pd.read_csv(path3).values
     

     


     a={i:list(np.where(indi==i)[0]) for i in np.unique(indi)}
     BATCH=[]
     for k in a.keys():
      
      x=torch.tensor(datas[a[k],:],dtype=torch.float)
      
      
      
      
      edge_index=torch.tensor(edges[k],dtype=torch.long)
      l=torch.tensor(label[k],dtype=torch.long)
      
      s=torch.tensor(state[k],dtype=torch.long)
     
      
      data=Data(x=x,y=l,edge_index=edge_index,s=s)
      BATCH.append(data)
    
     return BATCH
# dataset=process_csv('./Samentic_Data/graph.csv', './Samentic_Data/index.csv', './Samentic_Data/label.csv',edge,'./Samentic_Data/state1.csv')

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 =GraphConv(2, hidden_channels) 
        self.conv2 =GraphConv(hidden_channels, hidden_channels)  
        self.conv3 =GraphConv(hidden_channels, hidden_channels) 
        
        self.importance=Linear(hidden_channels,3)
        
        self.state=Linear(hidden_channels+3,3)
        # self.soft_max=torch.nn.softmax(dim=1)
        
       
        

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
       

        x = global_mean_pool(x, batch)

        
        importance=self.importance(x)
        
        state=self.state(torch.cat((x,importance.float()),dim=1))
        
        
            
           
        # im=torch.nn.functional.softmax(importance)
        # im=torch.nn.functional.normalize(importance, p=2.0, dim=1, eps=1e-12, out=None)
        
        
        
        
       
        
        
        
        
        
        
        return importance,state
    def node_embeedings(self,x,edge_index,obj1):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        return x[obj1]
# model = GNN(hidden_channels=128)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# criterion1=torch.nn.CrossEntropyLoss()    


# def test(loader):
#     model.eval()
#     correct=0
#     correct1=0
#     loader=DataLoader(loader, batch_size=6)
   
#     for data in loader:
       
#         out,out1=model(data.x,data.edge_index,data.batch)
        
#         pred1=out.argmax(dim=1)
#         pred2=out1.argmax(dim=1)
#         correct+=int((pred2==data.s).sum())
#         correct1+=int((pred1==data.y).sum())
   
  
#     return correct1/len(loader.dataset),correct/len(loader.dataset)
# def save(loader):
#     model.eval()
#     temp=[]
#     loader=DataLoader(loader, batch_size=1)
   
#     for data in loader:
#           for i in range(6,9,1):
          
#             embeding=model.node_embeedings(data.x, data.edge_index, i)
#             temp.append(list(embeding.cpu().detach().numpy()))
#     torch.save(torch.tensor(temp), 'data.pt')   
#     return torch.tensor(temp)   
        
        
# def train_model(datasets):
#     model.train()
#     datas=DataLoader(datasets, batch_size=16, shuffle=True)
#     for _ in range(3000):
#       for data in datas:
#         out,out1=model(data.x,data.edge_index,data.batch)
       
#         loss1=criterion1(out,data.y)
       
#         loss2=criterion1(out1,data.s)
#         loss=(loss1+loss2)
#         loss.backward()
     
#         optimizer.step()
#         optimizer.zero_grad()
      
# train_model(dataset)
# acc=test(dataset)
# data=save(dataset)

# torch.save(model.state_dict(), 'encoder.pt')












