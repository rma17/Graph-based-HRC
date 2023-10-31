# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:01:41 2022

@author: 97091

This is an example of collecting hand motion dataset using mediapipe
More info on mediapipe palm detection can found :
    https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
also a train_csv2 for indexing the motions with the interacted objects
"""

import cv2
import mediapipe as mp
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hands = mp.solutions.hands
train_df = pd.DataFrame(columns=["object","index","motion"])
path='./motion_data'
path_list=os.listdir(path)
path_list=path_list
label=[]
points=[]
point_color = (0, 0, 255) # BGR
length=[]
count=0
objs={"0":0,"1":1,"2":2,"3":3,"4":4,"5":5}
ob=[]
index=[]

for filename in path_list:
    
    p=os.path.join(path,filename)
    p1=os.listdir(p)
    
    for p_ in p1:
        center=[]
        
        label.append(int(filename[-1])-1)
        # label.append(int(filename[-1])-1)
        lists=os.path.join(path,filename,p_)
        index.append(count)
        ob.append(objs[p_[0]])
        
        lists=os.listdir(lists)
        
        f=[]
        for files in lists:
         if os.path.splitext(files)[1] == '.jpg':
            f.append(int(os.path.splitext(files)[0]))
        f.sort()
        f1=[ str(f_)+'.jpg' for f_ in f]
        IMAGE_FILES = f1
        with mp_hands.Hands(
          static_image_mode=True,
          max_num_hands=2,
          min_detection_confidence=0.5) as hands:
          for idx, file in enumerate(IMAGE_FILES):
              fil=os.path.join(path,filename,p_,file)
              image = cv2.imread(fil)
              
        # Convert the BGR image to RGB before processing.
              results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
              # if not results.multi_hand_landmarks:
              #      continue
              image_height, image_width, _ = image.shape
              raw_height, raw_width = image.shape[:2]
              hand=results.hand_rects[0]
              annotated_image = image.copy()
              x=hand.x_center*raw_width
              y=hand.y_center*raw_height
              h=(0.25*raw_height)/2
              w=(0.22*raw_width)/2
        
              # cv2.rectangle(annotated_image,(int(x-h),int(y-w)),(int(x+h),int(y+w)),point_color)
              hand=results.hand_rects[0]
              center.append([hand.x_center,hand.y_center])
        
          length.append(len(center))  
          count=count+1
             
        points.append(center)
    


import torch
max_length=0
for p in points:
    temp=len(p)
    if temp>max_length:
        max_length=temp
def pad_tensor(t):
     t = torch.tensor(t)
     padding = max_length - t.size()[0]
     t = torch.cat((t, torch.zeros(padding,2)),0)
     return t
i=0

for p in points:
    print(i)
    if i==0:
     t=pad_tensor(p)
     temp=t
    
    elif i==1:
       t=pad_tensor(p)
       temp=torch.stack((temp,t))
       
       
    else:
       t=pad_tensor(p)
       temp=torch.cat((temp,t.unsqueeze(0)),0)
    i=i+1
train_df["object"]=ob
train_df["index"]=index
train_df["motion"]=label
train_df.to_csv (r'train_csv2.csv', index = False, header=True)
label=torch.tensor(label)
length=torch.tensor(length)
torch.save(label,'label.pt')
torch.save(length,'length.pt')
torch.save(temp,'data.pt')
