# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 21:37:34 2023

@author: ruidong

This an example of building the Agumented data set for spitial-temporal dataset
finding the crops with their corresponding motions 
Have 17 labels in total (fetch not included)
"""

import pandas as pd
import os
import torch
import numpy as np
import random


train_df = pd.DataFrame(columns=["img_path","label","motion"])

dic={str(i)+"_":i   for i in range(19)}
dic1={'0_': 0,
 '1_': 1,
 '2_': 2,
 '3_': 3,
 '4_': 4,
 '5_': 5,
 '6_': 0,
 '7_': 1,
 '8_': 2,
 '9_':3,
 '10_': 4,
 '11_': 5,
 '12_': 0,
 '13_': 3,
 '14_': 5,
 '15_': 6,
 '16_':7,
 '17_':8}

dic2={'0_': 0,
 '1_': 0,
 '2_': 0,
 '3_': 0,
 '4_': 0,
 '5_': 0,
 '6_': 1,
 '7_': 1,
 '8_': 1,
 '9_':1,
 '10_': 1,
 '11_': 1,
 '12_': 2,
 '13_': 2,
 '14_': 2,
 '15_': 3,
 '16_':4,
 '17_':5}

z=torch.load('label.pt')
z_=z.numpy()
df = pd.read_csv('train_csv2.csv')
df.head()
z1=df.values

def create_csv(i,i_):
   motion=[]
   img=[]
   label=[]
  
   c=np.where(z1[:,0]==dic2[i_])[0]
   temp=z1[c,:]
   index=np.where(temp[:,2]==dic1[i_])
   ind1=temp[index,1][0]
   
   # ind1=random.sample(list(ind1), k=3)
   for x in range(len(ind1)):
           motion.append(ind1[x])
           img.append(i)
           label.append(dic[i_]) 
   return motion,img,label
    
    
# train_df["img_path"] = os.listdir("D:\Seq-Seq\hand_motion\motion\data11\image")
M=[]
I=[]
L=[]
for idx, i in enumerate(os.listdir(".\image")):
    if "0_" in i and not "10" in i:
      motion,img,label=create_csv(i,"0_")
      M=M+motion
      I=I+img
      L=L+label
       
    if "1_" in i and not "11" in i:
       motion,img,label=create_csv(i,"1_")
       M=M+motion
       I=I+img
       L=L+label
    if "2_" in i and not "12" in i:
      motion,img,label=create_csv(i,"2_")
      M=M+motion
      I=I+img
      L=L+label
    if "3_" in i and not "13_" in i:
      motion,img,label=create_csv(i,"3_")
      M=M+motion
      I=I+img
      L=L+label
    if "4_" in i and not "14_" in i:
      motion,img,label=create_csv(i,"4_")
      M=M+motion
      I=I+img
      L=L+label
    if "5_" in i and not "15_" in i:
      motion,img,label=create_csv(i,"5_")
      M=M+motion
      I=I+img
      L=L+label
    if "6_" in i and not "16_" in i:
      motion,img,label=create_csv(i,"6_")
      M=M+motion
      I=I+img
      L=L+label
    if "7_" in i and not "17_" in i:
      motion,img,label=create_csv(i,"7_")
      M=M+motion
      I=I+img
      L=L+label
    if "8_" in i and not "18_" in i:
      motion,img,label=create_csv(i,"8_")
      M=M+motion
      I=I+img
      L=L+label
    if "9_" in i:
      motion,img,label=create_csv(i,"9_")
      M=M+motion
      I=I+img
      L=L+label
    if "10_" in i:
      motion,img,label=create_csv(i,"10_")
      M=M+motion
      I=I+img
      L=L+label
    if "11_" in i:
      motion,img,label=create_csv(i,"11_")
      M=M+motion
      I=I+img
      L=L+label
    if "12_" in i:
      motion,img,label=create_csv(i,"12_")
      M=M+motion
      I=I+img
      L=L+label
    if "13_" in i:
      motion,img,label=create_csv(i,"13_")
      M=M+motion
      I=I+img
      L=L+label
    if "14_" in i:
      motion,img,label=create_csv(i,"14_")
      M=M+motion
      I=I+img
      L=L+label
    if "15_" in i:
      motion,img,label=create_csv(i,"15_")
      M=M+motion
      I=I+img
      L=L+label
    if "16_" in i:
      motion,img,label=create_csv(i,"16_")
      M=M+motion
      I=I+img
      L=L+label
    if "17_" in i:
      motion,img,label=create_csv(i,"17_")
      M=M+motion
      I=I+img
      L=L+label

    
train_df["img_path"]=I
train_df["label"]=L
train_df["motion"]=M
train_df.to_csv (r'train_csv.csv', index = False, header=True)