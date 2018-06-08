***Deep Learning Visual Odometry using robot_localization module for integration of R and t (local frame) with IMU and GPS data (global frame).***

Requires as a third-party software: [robot localization](https://github.com/cra-ros-pkg/robot_localization).

To run the code:
```
git clone https://github.com/ramonastef28/ProjectCS231n.git
cd Proj231n
catkin_make
rosrun loc_VO rtvo 
roslaunch robot_localization ekf_localization.launch 
```

***It requires to subscribe to the following ROS topics:***
```
/sensors/navsat/fix 
/imu/data
/camera/rawimage
```
and will publish the following topics:
```
/vo/mono/odom   - odometry message 
/vo/mono/image  - image message with detected features

to run the deepvo:
```
python deepvo.py

```
 
