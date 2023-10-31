# Robot-ROM
Hi John and Ryan:
Here are the codes for ROMAN paper. It contains training codes for the framework, real world ROS codes, and action datasets. Be aware that:
1. First of all, the experimental setup in York may varies, we have a camera mounted on the top at Sheffield Lab. Please refer to my paper .
   
2. The gripper we used in Sheffield is robotiq-85, but I think is different from York's. However, I have set up the gripper in York and there is code to control it in our group laptop. It should be easy to integrate with my ROS codes ( I have comment at this part).
   
3. Most importantly, the kinematics and dynamic setting of the UR5 can be different from UR10 we used in Sheffield Lab. So in my ROS codes, you should define different objects position according to your specific robot setting if you want to reuse it. But the usage of MoveIt should be the same.
 
4. Datasets and models are in the shared google drive
