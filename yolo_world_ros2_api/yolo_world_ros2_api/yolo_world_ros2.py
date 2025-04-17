#!/usr/bin/env python3
from rclpy.node import Node
import rclpy

from geometry_msgs.msg import PoseArray, PoseStamped
from yolo_world_interfaces.srv import SetClasses
from std_srvs.srv import SetBool

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from tf2_geometry_msgs import do_transform_pose

import time
import traceback


class YoloWorldRos2():

    POSES: PoseArray = None

    def __init__(self, node:Node, tf_buffer:Buffer=None, timeout=5.0):
        self._node = node

        # create tf buffer
        self._tf_buffer = tf_buffer
        if self._tf_buffer is None:
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self._node)

        # sub
        self._sub_pose = self._node.create_subscription(PoseArray, '/yolo_world/pose/pose_3d', self._pose_cb, 10)

        # service clients
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


    def get_object_poses(self, 
                        source_frame: str = None, 
                        target_frame: str = None, 
                        timeout_sec: float = 1.0) -> PoseArray:
        """
        検出した物体の姿勢を取得し、必要に応じて座標変換を行う
        
        Args:
            source_frame (str): 変換元フレーム（Noneの場合はYOLOの出力フレームを使用）
            target_frame (str): 変換先フレーム（Noneの場合は変換しない）
            timeout_sec (float): TF変換のタイムアウト時間（秒）
            
        Returns:
            PoseArray: 変換後の姿勢配列（失敗時はNone）
        """
        # 最新の検出結果を取得
        start_time = time.time()
        while self.POSES is None and (time.time() - start_time) < timeout_sec:
            rclpy.spin_once(self._node, timeout_sec=0.1)
        
        if self.POSES is None:
            self._node.get_logger().warn("No object poses received within timeout period")
            return None

        # フレーム変換が不要な場合
        if target_frame is None:
            return self.POSES

        # ソースフレームの決定
        if source_frame is None:
            source_frame = self.POSES.header.frame_id

        # 姿勢変換を実行
        transformed_poses = PoseArray()
        transformed_poses.header.stamp = self.POSES.header.stamp
        transformed_poses.header.frame_id = target_frame
        
        try:
            # TF変換を取得
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_sec))
            
            # 全姿勢を変換
            for pose in self.POSES.poses:
                # 直接Poseオブジェクトを渡す
                transformed_pose = do_transform_pose(pose, transform)
                
                # 変換結果を追加
                transformed_poses.poses.append(transformed_pose)
                
            return transformed_poses
            
        except TransformException as ex:
            self._node.get_logger().error(
                f"Failed to transform poses from '{source_frame}' "
                f"to '{target_frame}': {str(ex)}")
            return None
        except Exception as ex:
            self._node.get_logger().error(f"Unexpected error in pose transformation: {str(ex)}")
            traceback.print_exc()
            return None
