#!/usr/bin/env python3

from rclpy.node import Node
import rclpy

from yolo_world_ros2_api.yolo_world_ros2 import YoloWorldRos2


rclpy.init()
node = Node("sample")

yolo_world = YoloWorldRos2(node)

yolo_world.execute(True)
yolo_world.set_classes(
    classes=["bag"],
    conf=0.2
)
pose = yolo_world.get_object_poses()
print(pose)