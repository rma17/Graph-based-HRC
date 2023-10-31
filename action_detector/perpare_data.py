# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:20:22 2023

@author: Ruidong
This is an example script for perparing the training data step by step
"""
import os
cmd='python track.py'  # get hand motion
os.system(cmd)
cmd='python image_crop.py' #get hands crop
os.system(cmd)
cmd='python create_csv1.py' #create image dataset
os.system(cmd)
cmd='python motion.py'  #train the motion model
os.system(cmd)
cmd='python create_csv.py' #create spitial-temporal dataset
os.system(cmd)
