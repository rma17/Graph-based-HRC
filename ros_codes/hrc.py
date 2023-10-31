#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:45:05 2023

@author: ruidong

This is the code for controlling the robot with ROS
Note that the robotiq_gripper is for grasping the object, it should be replaced if robotiq is not used

"""
from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint
import rospy, sys
import geometry_msgs.msg
from geometry_msgs.msg import Vector3, Quaternion, Transform, Pose, Point, Twist, Accel,PoseStamped
import rospy
from std_msgs.msg import Bool,Float64,Float64MultiArray
import numpy as np
import math
import moveit_commander
import robotiq_gripper
import time

class Robot():
    def __init__(self,arm):
        self.pub = rospy.Publisher('/detection', Bool, queue_size=10)
        
        
        
        self.sub1=rospy.Subscriber('/pos', Float64MultiArray,self.callback1)
        
        
        self.l=None
        self.object=None
        
        
        
        self.rob_pos={'0':[[-0.1869938693857149, -0.43548035852576794,0.3964312823590514],  #gate
                     [-0.11566636642804348, -0.43297106755173703,  0.3986265315802757],  #gate
                     [-0.04909498720969213, -0.4244449720121072, 0.3985847025585979]],
                  '1':[[-0.19005319025366238, -0.5085498969451805,  0.4175673149709414], #ball
                      [-0.12659871593103783, -0.5111594854958961, 0.41785946538178247]],
                  '2': [[-0.04493774380408388, -0.5138836726201285, 0.45232252169337996]]}  
        self.move_pos=[[-0.4708315755715853, -0.5998260320099791, 0.382],
                  [-0.531, -0.5998260320099791, 0.382],
                  [-0.5972171713634173,-0.5998260320099791, 0.382],
                  [-0.670450899258899, -0.5998260320099791, 0.382],
                  [-0.7364469914077167, -0.5998260320099791, 0.382],
                  [-0.79, -0.5998260320099791,  0.382]]
        self.final_height={'0':0.365,'1':0.375,'2':0.415}
        
        #This should be repalced
        self.gripper = robotiq_gripper.RobotiqGripper()
        print("Connecting to gripper...")
        ip = "192.168.0.131"
        self.gripper.connect(ip, 63352)
        self.gripper.activate()
        print("Activating gripper...")
        ###########################
        self.arm=arm
    def callback(self,data):
        self.l=data.data
        
    def callback1(self,data):
        da=data.data
        print(da)
        self.l=da[0]
        self.object=da[1:4]
       
        
    def publish(self,detection):
        self.pub.publish(detection)
    def move(self,angle):
        target_pose = PoseStamped()
        target_pose.header.frame_id = reference_frame
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x=float(angle[0])
        target_pose.pose.position.y=float(angle[1])
        target_pose.pose.position.z=float(angle[2])
        target_pose.pose.orientation.x = 0
        target_pose.pose.orientation.y = 1
        target_pose.pose.orientation.z = 0
        target_pose.pose.orientation.w = 0
        
        self.arm.set_joint_value_target(target_pose,True)
        plan_=self.arm.plan()
        if type(plan_) is tuple:
            # noetic
            success, plan, planning_time, error_code = plan_
        self.arm.execute(plan) 
    def reset(self):
        
         pos1=np.array([85.27,-41.81,-124.34,-99.20,89.99,0])*math.pi/180
         print(pos1)
         self.arm.set_joint_value_target(pos1)
         self.arm.go()
    def executation(self,obj,num,goal):#main control function to control the robot
        pos1=self.rob_pos[str(int(obj))][int(num)]
        self.gripper.move_and_wait_for_pos(70, 255, 255)
        self.move(pos1)
        
       
        
        
        pos1[2]=pos1[2]-0.04
        self.move(pos1)
        self.gripper.move_and_wait_for_pos(150, 255, 255)
        pos1[2]=pos1[2]+0.18
        self.move(pos1)
        
        grab_pos=self.move_pos[int(goal)]
        grab_pos[2]=grab_pos[2]+0.18
        self.move(grab_pos)
        
        grab_pos[2]=self.final_height[str(int(obj))]
        
        
        self.move(grab_pos)
        
        
        
        
        self.gripper.move_and_wait_for_pos(70, 255, 255)
        grab_pos[2]=0.51
        self.move(grab_pos)
        self.reset()
if __name__ == "__main__":
    
    
    nh=rospy.init_node('agent',anonymous=True)
   
    moveit_commander.roscpp_initialize(sys.argv)
    arm=moveit_commander.MoveGroupCommander('manipulator')
    reference_frame = 'base_link'
    arm.set_pose_reference_frame(reference_frame)
    arm.set_goal_joint_tolerance(0.001)
    arm.set_max_acceleration_scaling_factor(0.03)
    arm.set_max_velocity_scaling_factor(0.03)
    arm.set_planer_id = "RRTkConfigDefault"
    arm.set_planning_time(50)
    r=Robot(arm)
    r.reset()
    try:
        while not rospy.is_shutdown():
           
            if r.l==16:
                r.publish(False)
                
                
                r.executation(r.object[0], r.object[1], r.object[2])
                
               
                
                
                
                r.publish(True)
                print('True')
            elif r.l is None:
                r.publish(True)
            
        
    except rospy.ROSInterruptException:
        pass
        