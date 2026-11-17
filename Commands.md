
# Relevant commands (yet)

## Robot-Startup
```shell
source ~/turtlebot3_ws/turtlebot-mapping/robot-env.sh
#or
source ~/turtlebot3_ws/turtlebot-mapping/cyclo-robot-env.sh # sets peer in cyclonedds too
```

## Scripts on Robot

- Basic launch (for teleop, cartographer, etc.)
```
ros2 launch turtlebot3_bringup robot.launch.py
```

- Launch the camera node (occupies the shell, so do in second shell)
```shell
ros2 run camera_ros camera_node 
# or with arguments
ros2 run camera_ros camera_node --ros-args -p format:=RGB888 -p width:=640 -p height:=480 # had format mismatch w/o it
```

## Scripts on Remote

### Source env:
```shell
source ~/turtlebot3_ws/turtlebot-mapping/remote-env.sh <ROBOT-IP>

# Since it has a static IP in Lab:
source ~/turtlebot3_ws/turtlebot-mapping/remote-env.sh 192.168.0.147
```

### Run dualsense-teleoperation package:

- VM-side
```shell
ros2 run turtlebot3_teleop_dualsense teleop_dualsense
```

- on Own PC (where the controller is connected)
```shell
python3 run dualsense_sender.py <VM-IP> --port 9999 
```

### View the topics + visualiser + etc.
```shell
rqt
```