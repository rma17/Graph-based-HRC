# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:53:02 2023

@author: 97091
This is the real-time semantic planning code working with ROS and realsense
It need the trained model for GNN and LSTM for planning

"""

from HRC_Encoder1 import GNN
from HRC_Decoder import DecoderRNN
import torch
import torch.nn.functional as F
import numpy as np
Encoder = GNN(hidden_channels=128)
Decoder = DecoderRNN(131,100,11)
Encoder.load_state_dict(torch.load('encoder.pt'))
Decoder.load_state_dict(torch.load('decoder.pt'))
Encoder.eval()
Decoder.eval()
Encoder.to('cpu')
Decoder.to('cpu')
class HRC():
    def __init__(self):
        self.graph=[[1., 0.],
                    [2., 0.],
                    [3., 0.],
                    [4., 0.],
                    [5., 0.],
                    [6., 0.],
                    [0., 1.],
                    [0., 2.],
                    [0., 3.]]
        self.label_to_edge={'0':[0,6],'1':[1,6],'2':[2,6],'3':[3,6],'4':[4,6],'5':[5,6],
                            '6':[0,7],'7':[1,7],'8':[2,7],'9':[3,7],'10':[4,7],'11':[5,7],
                            '12':[0,8],'13':[3,8],'14':[5,8],'15':None,'16':None}
        self.output_to_word= {1:"Obj1", 2:"Obj2", 3:"Obj3",4:'1',5:'2',
                                     6:'3',7:'4',8:'5',9:'6',10:'Finished'}
        self.label_to_part={'0': 0,'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 1,
                            '7': 1, '8': 1,'9':1,'10': 1,'11': 1,'12': 2, '13': 2,
                            '14': 2,'15': 3,'16':None}
        
        self.count={'0':0,'1':0,'2':0}
       
        self.edge=[[],[]]
    def update_graph(self,robot_plan):
        l=[k for k,v in self.label_to_edge.items() if v==robot_plan]
        
        self.creat_graph(l)
    def initlize(self):
        self.graph=[[1., 0.],
                    [2., 0.],
                    [3., 0.],
                    [4., 0.],
                    [5., 0.],
                    [6., 0.],
                    [0., 1.],
                    [0., 2.],
                    [0., 3.]]
        self.label_to_edge={'0':[0,6],'1':[1,6],'2':[2,6],'3':[3,6],'4':[4,6],'5':[5,6],
                            '6':[0,7],'7':[1,7],'8':[2,7],'9':[3,7],'10':[4,7],'11':[5,7],
                            '12':[0,8],'13':[3,8],'14':[5,8],'15':None,'16':None}
        self.output_to_word= {1:"Obj1", 2:"Obj2", 3:"Obj3",4:'1',5:'2',
                                     6:'3',7:'4',8:'5',9:'6',10:'Finished'}
        self.label_to_part={'0': 0,'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 1,
                            '7': 1, '8': 1,'9':1,'10': 1,'11': 1,'12': 2, '13': 2,
                            '14': 2,'15': 3,'16':None}
        
        self.count={'0':0,'1':0,'2':0}
       
        self.edge=[[],[]]
    def creat_graph(self,label):
        if label[0]!=15 and label[0]!=16:
            label=set(label)
            temp=[]
            temp=[self.label_to_edge[str(l)] for l in label if self.label_to_edge[str(l)] is not None]
            
            for t in temp:
                
                 self.edge[0].append(t[0])
                 self.edge[1].append(t[1])
            for l in label:
                if self.label_to_part[str(l)] is not None:
                 self.count[str(self.label_to_part[str(l)])]=self.count[str(self.label_to_part[str(l)])]+1
    def translate(self,output):
            word=[self.output_to_word[o] for o in output]
            return word
    def semantic_planning(self):
             edge_index=self.edge
             edge_index=torch.tensor(edge_index).to('cpu')
             x=torch.tensor(self.graph).to('cpu')
             batch=[0]*x.size()[0]
             batch=torch.tensor(batch).to('cpu')
             out,state=Encoder(x.float(),edge_index.long(),batch)
             out1=torch.argmax(out)
             out=F.one_hot(out1,num_classes=3)
             state=torch.argmax(state)
             embedding=Encoder.node_embeedings(x.float(),edge_index.long(), state.cpu().detach().numpy()+6)
             output=Decoder.Predict(embedding.unsqueeze(0), out.unsqueeze(0))
             output=list(filter(lambda x: x != 0, output))
             word=self.translate(output)
             
             word1=[]
             for i in range(3):
                 embedding1=Encoder.node_embeedings(x.float(),edge_index.long(), i+6)
                 output1=Decoder.Predict(embedding1.unsqueeze(0), out.unsqueeze(0))
                 output1=list(filter(lambda x: x != 0, output1))
                 word1.append(self.translate(output1))
             
             complete=[wor[1] for wor in word1]
             
             if (np.array(complete)=='Finished').all():
                 self.initlize()
                 
             return int(out1.detach().cpu().numpy()),word,output[0]-1,self.count[str(output[0]-1)],output[1]-4,word1
   
if __name__=='__main__':
    model=HRC()
    model.creat_graph([9])
    model.creat_graph([10])
    
    goal,des,obj,loc,obj_goal,des1=model.semantic_planning()
       
    model.update_graph([obj_goal,obj+6])  
    goal,des,obj1,loc1,obj_goal1,des1=model.semantic_planning() 
    model.update_graph([obj_goal1,obj1+6])  
    goal,des,obj1,loc1,obj_goal,des1=model.semantic_planning()    
    