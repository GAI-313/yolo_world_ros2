#!/usr/bin/env python3
from rclpy.node import Node
import rclpy

from yolo_world_interfaces.srv import SetClasses
from std_srvs.srv import SetBool


class YoloWorldRos2():
    def __init__(self, node:Node, timeout=5.0):
        self._node = node

        self._set_classes_cli = self._node.create_client(SetClasses, '/yolo_world/classes')
        self._set_execute_cli = self._node.create_client(SetBool, '/yolo_world/execute')

        while not self._set_classes_cli.wait_for_service(timeout_sec=timeout) or not self._set_execute_cli.wait_for_service(timeout_sec=timeout):
            self._node.get_logger().fatal('yolo_world_ros2 node is not running.')
            raise RuntimeError('yolo_world_ros2 node is not running. are you bringup yolo_world_ros2 ?')
