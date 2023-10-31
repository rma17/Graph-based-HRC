# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:01:54 2022
@author: Ruidong

This is the LSTM part of the paper: graph-based semantic planning for adaptive 
human-robot-collaboration in assemble-to-order scenarios

The simulated training data includes:
data.pt    saved node embeddings from HRC_Encoder
des.txt    ground truth contextual plan for each node embedding
score.csv  ground truth goals for each node

LSTM produce the contextual plan by decoding the node embedding 
 

"""
from __future__ import print_function, division




import torch
import pandas as pd

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import read_image
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F
device =  torch.device('cpu')




    
    
    
class CustomDataset(Dataset): #This is to build a customized torch dataset for embeding-to-sequence 
    def __init__(self):
        Captions = open("./Samentic_Data/des.txt")

        Caption = Captions.read().split("\n")
        label=pd.read_csv("./Samentic_Data/score.csv")
        self.captions = Caption
        self.inputs = torch.load('data.pt')
        self.label=label.values
        self.word_to_idx=self.voc()
        self.tag_to_idx= {"0":0,"Gate": 1, "Ball": 2, "Globe": 3,'One':4,'Two':5,
                                     'Three':6,'Four':7,'Five':8,'Six':9,'Finished':10} 
        self.idx_to_word= {1:"Gate", 2:"Ball", 3:"Globe",4:'One',5:'Two',
                                     6:'Three',7:'Four',8:'Five',9:'Six',10:'Finished'} 
        
    def voc(self):
        word_to_ix={}
        for sent in self.inputs:
         for word in sent:
          if word not in word_to_ix:  # word has not been assigned an index yet
           word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
        return word_to_ix
        
    def perpare_input(self, sequence,to_ix):
          
          sequence=list(sequence)
          
          idx = [to_ix[w] for w in sequence]
            
          return idx
    def perpare_label(self, sequence,to_ix):
          
          sequence+=['0']*(4-len(sequence))
         
          
          idx = [to_ix[w] for w in sequence]
            
          return idx
    def transelate(self,output):
        # output=list(output.cpu().detach().numpy())
        
        output=list(filter(lambda x: x != 0, output))
        
        output=[self.idx_to_word[i] for i in output]
        return output
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        
        
        inputs=self.inputs[idx]
       
        inputs=inputs.unsqueeze(dim=0)
        
        captions = self.captions[idx]
        
       
        captions=self.perpare_label(captions.split(),self.tag_to_idx)
        
        
        captions=torch.tensor(captions).long()
        # captions=captions.unsqueeze(dim=0)
        # captions=F.one_hot(captions,num_classes=10)
        
        
        label=self.label[idx]
        
        label=torch.tensor(label).long()
        label=F.one_hot(label,num_classes=3)
        
        
        sample={'inputs':inputs, 'captions':captions,'label':label}
        
        return sample
data=CustomDataset()

class DecoderRNN(nn.Module):#Main decoder function
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super( DecoderRNN , self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size 
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding( self.vocab_size , self.embed_size )
        self.lstm  = nn.LSTM(    input_size  =  self.embed_size , 
                             hidden_size = self.hidden_size,
                             num_layers  = self.num_layers ,
                             batch_first = True 
                             )
        self.fc = nn.Linear( self.hidden_size , self.vocab_size  )
        

    def init_hidden( self, batch_size ):
      return ( torch.zeros( self.num_layers , batch_size , self.hidden_size  ),
      torch.zeros( self.num_layers , batch_size , self.hidden_size  ))
    
    def forward(self, features, captions,label):#Training            
      captions = captions[:,:-1]      
      self.batch_size = features.shape[0]
      self.hidden = self.init_hidden( self.batch_size )
      embeds = self.word_embedding( captions )
      
      features=torch.cat((features,label.float()),dim = -1)
      inputs = torch.cat( ( features , embeds ) , dim =1  )      
      lstm_out , self.hidden = self.lstm(inputs , self.hidden)      
      outputs = self.fc( lstm_out )      
      return outputs

    def Predict(self, inputs,label):#Testing        
        final_output = []
        batch_size = inputs.shape[0]         
        hidden = self.init_hidden(batch_size) 
        inputs=torch.cat((inputs,label.float()),dim = -1)
        inputs=inputs.unsqueeze(1)
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.fc(lstm_out)  
            outputs = outputs.squeeze(1) 
            _, max_idx = torch.max(outputs, dim=1) 
            final_output.append(max_idx.cpu().numpy()[0].item())             
            if (len(final_output) >=4):
                break
            
            inputs = self.word_embedding(max_idx) 
            inputs = inputs.unsqueeze(1) 
                    
        return final_output  
decoder=DecoderRNN(131,100,11)
loader=DataLoader(data,batch_size=15,shuffle=True)
criterion = nn.CrossEntropyLoss()
lr = 0.001
all_params = list(decoder.parameters() )
optimizer = torch.optim.Adam( params  = all_params , lr = lr  )

# for _ in range(1000):#Training loop
#     for datas in loader:
#         decoder.zero_grad()
#         output=decoder(datas['inputs'],datas['captions'],datas['label'])
#         loss = criterion( output.view(-1, 11) , datas['captions'].view(-1) )
#         loss.backward()
#         optimizer.step()
#     print(loss)
# decoder.eval()
# output=decoder.Predict(data[32]['inputs'], data[32]['label'])
# print(data.transelate(output),data.transelate(list(data[32]['captions'].cpu().detach().numpy())))
# torch.save(decoder.state_dict(), 'decoder.pt')