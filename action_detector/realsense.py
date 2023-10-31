# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:20:25 2022

@author: Ruidong

This is an example code for recording the human assembly as images using realsense camera 
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
# Configure depth and color streams
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hands = mp.solutions.hands
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
# config.enable_record_to_file('object_detection.bag')

# Start streaming
pipeline.start(config)

e1 = cv2.getTickCount()
obj='gate'
import os
path = "globe_6_4"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created!")
t1=0
count=0
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        if not os.path.exists(os.path.join("D:\Seq-Seq\Cropped1",path,str(count))):
            os.makedirs(os.path.join("D:\Seq-Seq\Cropped1",path,str(count)))
            print('created')
        frames = pipeline.wait_for_frames()
        
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        
        color_image = np.asanyarray(color_frame.get_data())

        with mp_hands.Hands(
        static_image_mode=True,
         max_num_hands=2,
         min_detection_confidence=0.5) as hands:
         
          
          image = color_image
          
          results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
          if not results.multi_hand_landmarks:
               pass

        # Show images
          else:
              cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
              name=str(t1)+'.jpg'
              cv2.imwrite(os.path.join("D:\Seq-Seq\Cropped1",path,str(count),name), image)
              cv2.imshow('RealSense', image)
              cv2.waitKey(1)
              t1=t1+1
              if t1==45:
                  count=count+1
                  t1=0
              
              
          
         
          
       

finally:

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()