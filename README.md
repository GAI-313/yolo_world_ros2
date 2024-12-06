# yolo_world_ros2
## install
Please execute the following command to build the interface package so that you can connect to this package's interface locally.
```bash
colcon build --symlink-install --packages-select yolo_world_msgs yolo_world_srvs
```
For the first time, the following command will build the container and run the model.
```bash
docker compose up yolo_world_ros2
```

## edit
 Edit the remapping variable in src/yolo_world_ros2/launch/bringup_yolo_world.launch.py so that the required topics are subscribed.<br>
 After editing the launch file, run the above command again; since the src directory and beyond are mounted in the container, there is no need to rebuild the container.
