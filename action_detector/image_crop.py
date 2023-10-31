# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:12:14 2023

@author: ruidong

This is an example of building hand-region crops using mediapipe
More info on mediapipe palm detection can found :
    https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

"""

import cv2
import mediapipe as mp
import os
import matplotlib.pyplot as plt
import numpy as np
import re
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hands = mp.solutions.hands

path='./label'
path_list=os.listdir(path)
path_list=path_list
label=[]

point_color = (0, 0, 255) # BGR
length=[]
for filename in path_list:
    
    p=os.path.join(path,filename)
    p1=os.listdir(p)
    points=[]
    count=0
    for p_ in p1:
        center=[]
        
        
        
        lists=os.path.join(path,filename,p_)
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
              fil=os.path.join(path,filename,p_,IMAGE_FILES[-1])
              image = cv2.imread(fil) 
              results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
              raw_height, raw_width = image.shape[:2]
              hand=results.hand_rects[0]
              annotated_image = image.copy()
              x=hand.x_center*raw_width
              y=hand.y_center*raw_height
              h=(0.22*raw_height)/2
              w=(0.22*raw_width)/2
              cropImg=annotated_image[int(y-70):int(y+70),int(x-70):int(x+60)]
              cropImg=cv2.resize(cropImg, [140,140])
              p22=os.path.join('.\image', re.findall(r"\d+",filename)[-1]+'_'+str(count)+'.jpg')
              cv2.imwrite(p22,cropImg)
              count=count+1