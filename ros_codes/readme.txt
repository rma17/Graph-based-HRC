Codes for running the whole framework in real-world
Prerequisite:
Ubuntu 20.04 ROS Noetic
CUDA 
UR robot driver: https://github.com/UniversalRobots/Universal_Robots_ROS_Driver
pyrealsense2
torch_geometric
torch
!!! Please note the origin experiments used Robotiq gripper for grasping, it should be changed accordingly !!!

Pipeline:
launch the UR robot driver and its MoveIt (plz see: https://github.com/UniversalRobots/Universal_Robots_ROS_Driver)
run monitor.py
run HRC.py


The main logic behind the codes:
1. monitor.py will monitor the single hand motions
2. Plan.py will update the Graph and produce plan according to the monitor.py
3. If screw action is detected, the monitor.py will send paln as ros meesage to HRC.py, and it will be deactivated
4. HRC.py will control the robot.
5. If the robot is finished, activate monitor.py again. 
