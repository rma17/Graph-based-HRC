This the training codes for hand_centric action detector

Prerequisite: 

pytorch, CUDA 
pyrealsense2
MediaPipe

Data sets are in the google drive
Pipeline:
perpare_data.py will process the data step by step
!!! Note that the file root in the scripts should be changed !!!
1. Detailed Data perpare:
run track.py for collecting motion data 
run image_crop.py for collecting hand_centric crops
run create_csv.py and create_csv1.py to build the csv file for creating datasets

!!! For fine tune apporach !!!
2. run vgg.py and motion.py 
run fine_tune.py to get the final model

!!! Alternative Training from scracth !!!
3. run S+T.py

Also note that realsense.py is an example code for recording videos using RealSense