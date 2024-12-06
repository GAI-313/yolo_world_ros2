#!/usr/bin/env python3
from ultralytics import YOLOWorld

from yolo_world_msgs.msg import *
from yolo_world_srvs.srv import *
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError

from rclpy.node import Node
import rclpy
import message_filters

import numpy as np
import cv2


class YoloWorldRos2Main(Node):
    def __init__(self) ->None:
        super().__init__('yolo_world_ros2_main')

        self.classes = ['chair']
        self.predict = 0.1

        self.bbox_publisher = self.create_publisher(BbPixelPoseArray, 'bbox_array', 10)
        self.objectimages_publisher = self.create_publisher(ObjectImageArray, 'object/images', 10)
        self.image_publisher = self.create_publisher(Image, 'detect_image', 10)
        self.image_sub = self.create_subscription(Image, 'image_raw', self._image_cb, 10)
        self.bridge = CvBridge()

        self.get_logger().info('YoLo-World ROS2 setup ...')
        self.model = YOLOWorld('/home/user/yolov8l-world.pt')
        self.model.set_classes(self.classes)
        self.get_logger().info('YoLo-World ROS2 Ready !')

        self.execute_srv = self.create_service(SetBool, 'execute', self._srv_cb)
        self.setclass_srv = self.create_service(SetClasses, 'set_class', self._setclass)
        self.setpred_srv = self.create_service(SetPredict, 'set_predict', self._setpred)
        self.execute_flag = False
    
    def _srv_cb(self, req: SetBool.Request, res: SetBool.Response) ->SetBool.Response:
        try:
            self.execute_flag = req.data
            res.success = req.data

            if req.data:
                res.message = 'YoLo-World Start!'
            else:
                res.message = 'YoLo-World Stop'
                cv2.destroyWindow('detect image')
        except cv2.error:
            pass
        finally:
            return res
    
    def _setclass(self, req: SetClasses.Request, res: SetClasses.Response) ->SetClasses.Response:
        self.classes = req.classes
        self.model.set_classes(self.classes)
        self.get_logger().info(f'Set Detect classes: {self.classes}')

        _execute_flag = self.execute_flag
        self.execute_flag = False

        del self.model

        self.get_logger().info('YoLo-World ROS2 setup ...')
        self.model = YOLOWorld('/home/user/yolov8l-world.pt')
        self.model.set_classes(self.classes)
        self.get_logger().info('YoLo-World ROS2 Ready !')

        if _execute_flag:
            self.execute_flag = True

        return res

    def _setpred(self, req: SetPredict.Request, res: SetPredict.Response) ->SetPredict.Response:
        self.predict = req.predict
        self.get_logger().info(f'Set Detect predict: {self.predict}')
        return res
    
    def _image_cb(self, msg: Image) ->None:
        if self.execute_flag:
            try:
                image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except CvBridgeError as e:
                self.get_logger().error(e)
            
            results = self.model.predict(
                source=image,
                conf=self.predict,
                iou=0.1,
                max_det=10,
                verbose=False
            )

            results = results[0].cpu()
            result_image = results.plot()

            cv2.waitKey(1)
            cv2.imshow('detect image', result_image)

            bounding_box_list = []
            object_image_list = []
            for bbox, cls, pred in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                obj_image = ObjectImage()
                bp = BbPixelPose()
                bp.pose1.x, bp.pose1.y, bp.pose2.x, bp.pose2.y = map(int, bbox)
                if bp.pose1.x != bp.pose2.x and bp.pose1.y != bp.pose2.y:
                    bp.class_name = results.names.get(int(cls))
                    bp.id = int(cls)
                    bp.predict = pred.item()
                    bounding_box_list.append(bp)
                    
                    obj_image.image = self.bridge.cv2_to_imgmsg(image[bp.pose1.y : bp.pose2.y, bp.pose1.x : bp.pose2.x], "bgr8")
                    obj_image.image.header = msg.header
                    obj_image.class_name = bp.class_name 
                    obj_image.id = bp.id
                    obj_image.predict = bp.predict
                    object_image_list.append(obj_image)
            
            bpa = BbPixelPoseArray()
            bpa.header = msg.header
            bpa.poses = bounding_box_list

            oia = ObjectImageArray()
            oia.objects = object_image_list
            self.objectimages_publisher.publish(oia)

            image_msg = self.bridge.cv2_to_imgmsg(result_image, encoding="bgr8")
            image_msg.header = msg.header

            self.bbox_publisher.publish(bpa)
            self.image_publisher.publish(image_msg)


class PoseTransformer(Node):
    def __init__(self) ->None:
        super().__init__('pose_transformer')

        self.bridge = CvBridge()
        self.depth_image_units_divisor = 1000.0
        #self.depth_image_units_divisor = 1.0

        self.depth_info = message_filters.Subscriber(self, CameraInfo, 'depth/camera_info')
        self.depth_image = message_filters.Subscriber(self, Image, 'depth/image_raw')
        self.bbox_array = message_filters.Subscriber(self, BbPixelPoseArray, 'bbox_array')

        self.syncronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_info, self.depth_image, self.bbox_array), 10, 0.5
        )
        self.syncronizer.registerCallback(self._cb)

        self.poses_publisher = self.create_publisher(ObjectPoseArray, 'object/poses', 10)
        self.posemarkers_publisher = self.create_publisher(MarkerArray, 'object/pose_markers', 10)
    
    def _cb(self, camerainfo: CameraInfo, depthimage: Image, bboxarray: BbPixelPoseArray) -> None:
        depth_image = self.bridge.imgmsg_to_cv2(depthimage, "32FC1")
        fx = camerainfo.k[0]
        fy = camerainfo.k[4]
        cx = camerainfo.k[2]
        cy = camerainfo.k[5]

        obj_pose = ObjectPose()
        obj_poses = []
        obj_marker = Marker()
        obj_markers = []
        i = 0
        for bbox in bboxarray.poses:
            obj_center = [
                bbox.pose1.x + (bbox.pose2.x - bbox.pose1.x) // 2,
                bbox.pose1.y + (bbox.pose2.y - bbox.pose1.y) // 2,
            ]

            d = depth_image[int(obj_center[1]), int(obj_center[0])] / self.depth_image_units_divisor
            #self.get_logger().info(str(d))
            if d > 0:
                x = (obj_center[0] - cx) * d / fx
                y = (obj_center[1] - cy) * d / fy
                z = d

                obj_pose.pose.x = x
                obj_pose.pose.y = y 
                obj_pose.pose.z = z
                obj_pose.class_name = bbox.class_name
                obj_pose.id = bbox.id
                obj_pose.predict = bbox.predict
                obj_poses.append(obj_pose)

                obj_marker.header = bboxarray.header
                obj_marker.ns = bbox.class_name
                obj_marker.id = i
                obj_marker.action = Marker.ADD
                obj_marker.type = Marker.CUBE
                obj_marker.pose.position = obj_pose.pose
                obj_marker.scale.x = 0.1
                obj_marker.scale.y = 0.1
                obj_marker.scale.z = 0.1
                obj_marker.color.r = 1.0
                obj_marker.color.g = 1.0
                obj_marker.color.b = 0.0
                obj_marker.color.a = 0.8
                obj_marker.lifetime = rclpy.duration.Duration(seconds=0.25).to_msg()
                obj_markers.append(obj_marker)

                i += 1


        obj_poses_array = ObjectPoseArray()
        obj_poses_array.poses = obj_poses
        obj_poses_array.header = bboxarray.header
        self.poses_publisher.publish(obj_poses_array)

        marker_array = MarkerArray()
        marker_array.markers = obj_markers
        self.posemarkers_publisher.publish(marker_array)


def main():
    rclpy.init()
    node = YoloWorldRos2Main()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        node.destroy_node()

def pose_transformer():
    rclpy.init()
    node = PoseTransformer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        node.destroy_node()

if __name__ == '__main__':
    main()
