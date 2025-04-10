#!/usr/bin/env python3
from rclpy.node import Node
import rclpy

from geometry_msgs.msg import PoseArray
from yolo_world_interfaces.srv import SetClasses
from std_srvs.srv import SetBool


class YoloWorldRos2():

    POSES: PoseArray = None

    def __init__(self, node:Node, timeout=5.0):
        self._node = node

        self._sub_pose = self._node.create_subscription(PoseArray, '/yolo_world/pose/pose_3d', self._pose_cb, 10)

        self._set_classes_cli = self._node.create_client(SetClasses, '/yolo_world/classes')
        self._set_execute_cli = self._node.create_client(SetBool, '/yolo_world/execute')

        while not self._set_classes_cli.wait_for_service(timeout_sec=timeout) or not self._set_execute_cli.wait_for_service(timeout_sec=timeout):
            self._node.get_logger().fatal('yolo_world_ros2 node is not running.')
            raise RuntimeError('yolo_world_ros2 node is not running. are you bringup yolo_world_ros2 ?')
    

    def _pose_cb(self, msg: PoseArray):
        self.POSES = msg

    def execute(self, running: bool) -> bool:
        """YoLo World 起動メソッド

        Args:
            running (bool): 物体検出を開始する：True。物体検出を停止：False

        Returns:
            bool: 実行結果。True：成功
        """

        req = SetBool.Request()
        req.data = running

        future = self._set_execute_cli.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)

        res: SetBool.Response = future.result()

        self._node.get_logger().info(res.message)

        return res.success
    

    def set_classes(self, classes=[], conf: float=0.5) -> bool:
        """_summary_

        Args:
            classes (list): クラス一覧. Defaults to [].
            conf (float): 最低信頼度. Defaults to 0.5.

        Returns:
            bool: 実行結果。True：成功
        """

        req = SetClasses.Request()
        req.classes = classes
        req.conf = conf

        future = self._set_classes_cli.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)

        res: SetClasses.Response = future.result()

        return res.success


    def get_object_poses(self) -> PoseArray:
        self.POSES: PoseArray = None

        while self.POSES == None:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            print(self.POSES)
        
        return self.POSES