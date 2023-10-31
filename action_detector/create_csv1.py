# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:52:31 2023

@author: 97091

This is an example of buidling dataset for spitial hand-object interaction
Five labels stands for different interatction status


"""



import pandas as pd
import os
import torch
import numpy as np
import random
device = ("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.DataFrame(columns=["img_path","label"])

dic={str(i)+"_":i   for i in range(19)}
dic1={'0_': 0,
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


def create_csv(i,i_):
   motion=[]
   img=[]
   label=[]
   m_l=dic1[i_]
   ind1=np.where(z_==m_l)[0]
   # ind1=random.sample(list(ind1), k=3)
   for x in range(len([0])):
           motion.append(ind1[x])
           img.append(i)
           label.append(dic1[i_]) 
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

train_df.to_csv (r'train_csv1.csv', index = False, header=True)