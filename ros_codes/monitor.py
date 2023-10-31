#!/usr/bin/env python3

"""
Created on Wed Feb 22 15:50:20 2023

@author: ruidong


This is the real-time detection and a designed UI code working with ROS and realsense
It need the trained model S_T for action detection 
"""

import rospy
from std_msgs.msg import Float64,Bool,Float64MultiArray 
import numpy as np
import numpy as np
import cv2
import time
from LSTM_CNN import S_T
import torch
from torchvision import transforms
# Configure depth and color streams
from PIL import Image
import mediapipe as mp
from plan import HRC
import pyrealsense2 as rs
import time
pipeline = rs.pipeline()

# #Create a config并配置要流​​式传输的管道
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

profile = pipeline.start(config)
planner=HRC()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hands = mp.solutions.hands

# config.enable_record_to_file('object_detection.bag')
transform = transforms.Compose(
            [
                transforms.Resize((140, 140)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
# Start streaming

model=S_T(2,4096,2048,128,17) 
model.load_state_dict(torch.load('model1.pt'))
model.eval()
model.to('cuda')
class monitor():
    def __init__(self):
        self.model=model
        self.planner=HRC()
        self.pub=rospy.Publisher('/label', Float64, queue_size=10)
        self.L=[]
        self.t1=0
        self.out=None
        self.word1=None
        self.word2=None
        self.word3=None
        self.word=[None,None]
        self.l=None
        self.center=[]
        self.detect=True
        self.pub1=rospy.Publisher('/pos', Float64MultiArray,queue_size=10)
        rospy.Subscriber("/detection", Bool, self.callback)
        
    def callback(self,data):
         self.detect=data.data
    def detection(self,image):
        with mp_hands.Hands(
         static_image_mode=True,

         max_num_hands=2,
         min_detection_confidence=0.5) as hands:
         
          
          
         
          results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
          if not results.multi_hand_landmarks:
              if self.L != [] and self.L[-1]!=15:
                   self.l=self.L[-1]
                   planner.creat_graph([self.l])
                   self.L=[]
                   self.out,_,_,_,_,des1=planner.semantic_planning() 
                   self.word1=' '.join(des1[0][1:len(des1[0])])
                   self.word2=' '.join(des1[1][1:len(des1[1])])
                   self.word3=' '.join(des1[2][1:len(des1[2])])
                  
              self.center=[]
              self.t1=0
              # pass

        # Show images
          else:
             
              raw_height, raw_width = image.shape[:2]
              
              hand=results.hand_rects[0]
              self.center.append([hand.x_center,hand.y_center])
              
              
              self.t1=self.t1+1
              x=hand.x_center*raw_width
              y=hand.y_center*raw_height
            
              image.flags.writeable = True
              
            
              if self.detect:
               if self.t1>45 and self.t1%30==0:
                 
                 
                  raw_height, raw_width = image.shape[:2]
                  hand=results.hand_rects[0]
                  annotated_image = image.copy()
                  x=hand.x_center*raw_width
                  y=hand.y_center*raw_height
                  
                  cropImg1=annotated_image[int(y-70):int(y+70),int(x-70):int(x+60)]
                  cv2.imwrite('test.jpg',cropImg1)
                  cropImg1=Image.open('test.jpg').convert("RGB")
                  
                  tra=self.center[-45:]
                  im_tensor=transform(cropImg1)
                  tra_tensor=torch.tensor(tra)
                  im_tensor,tra_tensor=im_tensor.to('cuda'),tra_tensor.to('cuda')
                  output=model(im_tensor.float().unsqueeze(0),tra_tensor.float().unsqueeze(0))
                  label=torch.argmax(output,dim=1).detach().cpu().numpy()[0]
                  
                         
                  self.L.append(label)
                  if label==16:
                      self.l=label
                      out,self.word,obj,loc,obj_goal,_=planner.semantic_planning()
                      if self.word[1]=='Finished':
                          self.word[0]=None
                      
                      
                      planner.update_graph([obj_goal,obj+6])
                      # self.pub.publish(label)
                      data_to_send = Float64MultiArray()
                      data_to_send.data=[label,obj,loc,obj_goal]
                      
                      self.pub1.publish(data_to_send)
                      print(obj,loc,obj_goal)
                      time.sleep(1)
                      self.t1=0
                      self.center=[]
                      self.L=[]
                      # self.pub.publish(None)
                      data_to_send = Float64MultiArray()
                      data_to_send.data=[0,obj,loc,obj_goal]
                      self.pub1.publish(data_to_send)
                     
                      
                      
                          
              else:
                   self.l=None
                   self.t1=0
                   self.center=[]
                   self.L=[]
                  
                 
                  
                  
                  
              #    C.append(center)
        
              
        
       
    def capture(self): 
        # cap = cv2.VideoCapture('test37.mp4')
        # while(cap.isOpened()):
        while True:
            frames = pipeline.wait_for_frames()
            # ret, color_frame = cap.read()
        
        
        

        # Convert images to numpy arrays
        
            # image = color_frame

        

        
            color_frame = frames.get_color_frame()
            image= np.asanyarray(color_frame.get_data())

            if  not color_frame:
                continue
            if self.l!=None:
                self.out,_,_,_,_,des1=planner.semantic_planning() 
                self.word1=' '.join(des1[0][1:len(des1[0])])
                self.word2=' '.join(des1[1][1:len(des1[1])])
                self.word3=' '.join(des1[2][1:len(des1[2])])

            
            
            left = np.full([480, 320, 3], 255, dtype=np.uint8)
            im=np.hstack((image,left))
            cv2.putText(
                        img = im,
                        text = 'frame_count:',
                        org = (645, 60),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        color = (255,0,0),
                        thickness = 3
                       )
            cv2.putText(
                        img = im,
                        text = 'action:',
                        org = (645, 110),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        color = (255,0,0),
                        thickness = 3
                       )
            cv2.putText(
                        img = im,
                        text = 'Goal:',
                        org = (645, 150),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        color = (255,0,0),
                        thickness = 3
                       )
            cv2.putText(
                        img = im,
                        text = 'Obj1:',
                        org = (645, 200),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        color = (255,0,0),
                        thickness = 3
                       )
            cv2.putText(
                        img = im,
                        text = 'Obj2:',
                        org = (645, 250),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        color = (255,0,0),
                        thickness = 3
                       )
            cv2.putText(
                       img = im,
                       text = 'Obj3:',
                       org = (645, 300),
                       fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale = 1,
                       color = (255,0,0),
                       thickness = 3
                      ) 
            
            cv2.putText(
                       img = im,
                       text = 'Next Obj:',
                       org = (645, 400),
                       fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale = 1,
                       color = (0,255,0),
                       thickness = 3
                      ) 
            
            cv2.putText(
                       img = im,
                       text = 'Goal Pose:',
                       org = (645, 450),
                       fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale = 1,
                       color = (0,255,0),
                       thickness = 3
                      ) 
            self.detection(image)
            
            cv2.putText(
                     img = im,
                     text = str(self.t1),
                     org = (900, 60),
                     fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                     fontScale = 1,
                     color = (255,0,0),
                     thickness = 3
                    ) 
            
            
            cv2.putText(
                          img = im,
                          text = str(self.l),
                          org = (850, 110),
                          fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale = 1,
                          color = (255,0,0),
                          thickness = 3
                         )  
            
            cv2.putText(
                           img = im,
                           text = str(self.out),
                           org = (830, 150),
                           fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale = 1,
                           color = (255,0,0),
                           thickness = 3
                          )
            cv2.putText(
                           img = im,
                           text = str(self.word1),
                           org = (800, 200),
                           fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale = 1,
                           color = (255,0,0),
                           thickness = 3
                          )
            cv2.putText(
                           img = im,
                           text = str(self.word2),
                           org = (800, 250),
                           fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale = 1,
                           color = (255,0,0),
                           thickness = 3
                          )
            cv2.putText(
                          img = im,
                          text = str(self.word3),
                          org = (800, 300),
                          fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale = 1,
                          color = (255,0,0),
                          thickness = 3
                         )
            cv2.putText(
                           img = im,
                           text = str(self.word[0]),
                           org = (830, 400),
                           fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale = 1,
                           color = (255,0,0),
                           thickness = 3
                          )
            cv2.putText(
                           img = im,
                           text = str(self.word[1]),
                           org = (830, 450),
                           fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale = 1,
                           color = (255,0,0),
                           thickness = 3
                          )
            
            cv2.imshow('RealSense', im)
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)  
            if cv2.waitKey(5) & 0xFF == 27:
              cv2.destroyWindow('P')
              break
        
        


if __name__ == '__main__':
    rospy.init_node("detector_manager_node",anonymous=True)
    m=monitor()
    try:
        m.capture()
        
    except rospy.ROSInterruptException:
        pass