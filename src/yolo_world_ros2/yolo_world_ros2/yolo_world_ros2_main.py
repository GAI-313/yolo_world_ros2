#!/usr/bin/env python3
from ultralytics import YOLOWorld, SAM
from ultralytics import settings

import rclpy.subscription
from yolo_world_msgs.msg import *
from yolo_world_srvs.srv import *
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError

from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.exceptions import InvalidParameterTypeException, ParameterNotDeclaredException
from rclpy.node import Node
import rclpy
import message_filters

import random
import numpy as np
import cv2


class YoloWorldRos2Main(Node):
    def __init__(self) ->None:
        super().__init__('yolo_world')

        is_empty = False

        self.declare_parameter('default_predict', 0.1)
        #
        try:
            self.declare_parameter('default_classes', ['person', 'table', 'chair', 'phone'])
        except InvalidParameterTypeException:
            self.declare_parameter('default_classes', '')
            self.get_logger().warn('Default classes is empty.')
            is_empty = True
        self.declare_parameter('default_use_sam', False)
        self.declare_parameter('default_execute', False)

        self.predict = self.get_parameter('default_predict').get_parameter_value().double_value
        # 
        if not is_empty: self.classes = self.get_parameter('default_classes').get_parameter_value().string_array_value
        if is_empty: self.classes = self.get_parameter('default_classes').get_parameter_value(); self.classes = []
        self.use_sam = self.get_parameter('default_use_sam').get_parameter_value().bool_value
        self.execute_flag = self.get_parameter('default_execute').get_parameter_value().bool_value

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10
        )

        # output log default settings
        self.get_logger().info(f"""\n
Default predict: {self.predict}
Default classes: {self.classes}
SAM: {self.use_sam}
Execute: {self.execute_flag}
        \n""")

        self.pub_objectinfo = self.create_publisher(ObjectInfo2DArray, 'object/image/info', qos_profile)
        self.pub_detectimage = self.create_publisher(Image, 'detect_image', 10)
        #self.pub_testimage = self.create_publisher(Image, 'test_image', 10)
        self.sub_image = self.create_subscription(Image, 'image_raw', self._image_cb, 10)
        self.bridge = CvBridge()

        self.get_logger().info('YoLo-World ROS2 setup ...')
        settings.update({'weights_dir':'/'})                # set include directory of weight data.
        settings.reset()                                    # load sertting to module
        self.yw_model = YOLOWorld('/yolov8l-world.pt')
        self.sam_model = SAM('/sam_b.pt')
        if len(self.classes) > 0: self.yw_model.set_classes(self.classes)
        self.get_logger().info('YoLo-World ROS2 Ready !')

        self.srv_execute = self.create_service(SetBool, 'execute', self._set_exec)
        self.srv_setclass = self.create_service(SetClasses, 'set_classes', self._setclass)
        self.srv_setpred = self.create_service(SetPredict, 'set_predict', self._setpred)
        self.srv_usesam = self.create_service(SetBool, 'use_sam', self._setsam)


    def _set_exec(self, req: SetBool.Request, res: SetBool.Response) ->SetBool.Response:
        try:
            self.execute_flag = req.data
            res.success = req.data

            if req.data and self.classes:
                res.message = 'YoLo-World Start!'
            elif req.data and not self.classes:
                res.message = 'classes is empty. please call set_classes before. default detect is activate.'
                self.get_logger().warn(res.message)
            else:
                res.message = 'YoLo-World Stop'
                cv2.destroyWindow('detect image')
        except cv2.error:
            pass
        finally:
            return res

    def _setclass(self, req: SetClasses.Request, res: SetClasses.Response) ->SetClasses.Response:
        self.classes = req.classes
        self.yw_model.set_classes(self.classes)
        self.get_logger().info(f'Set Detect classes: {self.classes}')

        _execute_flag = self.execute_flag
        self.execute_flag = False

        del self.yw_model

        self.get_logger().info('YoLo-World ROS2 setup ...')
        self.yw_model = YOLOWorld('/yolov8l-world.pt')
        if self.classes: self.yw_model.set_classes(self.classes)
        self.get_logger().info('YoLo-World ROS2 Ready !')

        if _execute_flag:
            self.execute_flag = True

        return res

    def _setpred(self, req: SetPredict.Request, res: SetPredict.Response) ->SetPredict.Response:
        self.predict = req.predict
        self.get_logger().info(f'Set Detect predict: {self.predict}')
        _execute_flag = self.execute_flag
        self.execute_flag = False

        del self.yw_model

        self.get_logger().info('YoLo-World ROS2 setup ...')
        self.yw_model = YOLOWorld('/home/user/yolov8l-world.pt')
        self.yw_model.set_classes(self.classes)
        self.get_logger().info('YoLo-World ROS2 Ready !')

        if _execute_flag:
            self.execute_flag = True

        return res

    def _setsam(self, req: SetBool.Request, res: SetBool.Response) ->SetBool.Response:
        try:
            self.use_sam = req.data
            res.success = req.data

            if req.data:
                res.message = 'SAM activate'
                self.sam_model = SAM('/sam_b.pt')
            else:
                res.message = 'SAM deactivate'
                del self.sam_model
                cv2.destroyWindow('detect image sam')

        except cv2.error:
            pass
        finally:
            return res

    def object_info_loader(self, bbox, cls, pred, results, image, msg):
        object_info = ObjectInfo2D()
        object_info.name = results.names.get(int(cls))
        object_info.id = int(cls)
        object_info.predict = pred.item()

        x1, y1, x2, y2 = map(int, bbox)
        object_info.roi.x_offset = x1
        object_info.roi.y_offset = y1
        object_info.roi.width = x2 - x1
        object_info.roi.height = y2 - y1

        object_info.bboximage = self.bridge.cv2_to_imgmsg(image[y1 : y2, x1 : x2], "bgr8")
        object_info.bboximage.header = msg.header

        return object_info

    def _image_cb(self, msg: Image) ->None:
        if self.execute_flag:
            try:
                image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except CvBridgeError as e:
                self.get_logger().error(e)

            results = self.yw_model.predict(
                source=image,
                conf=self.predict,
                iou=0.1,
                max_det=10,
                verbose=False
            )

            results = results[0].cpu()
            result_image = results.plot()
            result_image_msg = self.bridge.cv2_to_imgmsg(result_image, encoding="bgr8")
            result_image_msg.header = msg.header

            cv2.waitKey(1)
            #cv2.imshow('detect image', result_image)
            ##
            object_info_array = ObjectInfo2DArray()
            object_info_array.header = msg.header
            if self.use_sam and len(results.boxes.xyxy) > 0:
                result_sum = self.sam_model(image, bboxes=results.boxes.xyxy, verbose=False)[0].cpu()
                for bbox, cls, pred, mask in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf, result_sum.masks.data.cpu().numpy()):
                    object_info = self.object_info_loader(bbox, cls, pred, results, image, msg)
                    object_info.maskimage = self.bridge.cv2_to_imgmsg((mask > 0).astype(np.uint8) * 255, "mono8")

                    object_info_array.info.append(object_info)
            
            elif not self.use_sam and len(results.boxes.xyxy) > 0:
                for bbox, cls, pred in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                    object_info = self.object_info_loader(bbox, cls, pred, results, image, msg)
                    object_info_array.info.append(object_info)

            
            # topic publish
            self.pub_detectimage.publish(result_image_msg)
            self.pub_objectinfo.publish(object_info_array)
            #self.pub_testimage.publish(object_info.maskimage)


class PoseTransformer(Node):
    def __init__(self) ->None:
        super().__init__('pose_transformer')

        self.bridge = CvBridge()
        self.object_colors = {}

        self.declare_parameter('depth_image_units_divisor', 1000.0)
        self.depth_image_units_divisor = self.get_parameter('depth_image_units_divisor').get_parameter_value().double_value

        self.depth_info = message_filters.Subscriber(self, CameraInfo, 'depth/camera_info')
        self.depth_image = message_filters.Subscriber(self, Image, 'depth/image_raw')
        self.object_info = message_filters.Subscriber(self, ObjectInfo2DArray, 'object/image/info')

        self.syncronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_info, self.depth_image, self.object_info), 100, 0.5
        )
        self.syncronizer.registerCallback(self._cb)

        self.pub_imagecamerainfo = self.create_publisher(CameraInfo, 'camera_info', 10)
        self.pub_poses = self.create_publisher(ObjectInfo3DArray, 'object/pose/info', 10)
        self.pub_posemarkers = self.create_publisher(MarkerArray, 'object/pose/markers', 10)
    
    def get_color_for_object(self, object_id):
        if object_id not in self.object_colors:
            # Generate a random color and store it
            self.object_colors[object_id] = (
                random.random(),  # Red
                random.random(),  # Green
                random.random()   # Blue
            )
        return self.object_colors[object_id]

    
    def add_object_pose_marker(self, object_3d_info:ObjectInfo3D, object_info_array:ObjectInfo2DArray, object_info:ObjectInfo2D, size, i):
        R, G, B = self.get_color_for_object(object_3d_info.id)

        object_marker = Marker()
        object_marker.action = Marker.ADD
        object_marker.type = Marker.CUBE
        object_marker.header = object_info_array.header
        object_marker.id = i
        object_marker.ns = f"{object_info.name}_{i}"
        object_marker.pose.position = object_3d_info.pose.position
        object_marker.scale.x = size[0]
        object_marker.scale.y = size[1]
        object_marker.scale.z = size[2]
        object_marker.color.r = R
        object_marker.color.g = G
        object_marker.color.b = B
        object_marker.color.a = 0.8
        object_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()

        return object_marker

    def add_object_text_marker(self, object_3d_info:ObjectInfo3D, object_info_array:ObjectInfo2DArray, object_info:ObjectInfo2D, i):
        object_marker = Marker()
        object_marker.action = Marker.ADD
        object_marker.type = Marker.TEXT_VIEW_FACING
        object_marker.header = object_info_array.header
        object_marker.id = i
        object_marker.ns = f"text_{object_info.name}_{i}"
        object_marker.text = f"{object_info.name}"
        object_marker.pose.position = object_3d_info.pose.position
        object_marker.scale.x = 0.1
        object_marker.scale.y = 0.1
        object_marker.scale.z = 0.1
        object_marker.color.r = 1.0
        object_marker.color.g = 1.0
        object_marker.color.b = 1.0
        object_marker.color.a = 1.0
        object_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()

        return object_marker

    def get_pointclouds(self, mask, depth_image, camerainfo):
        center = []
        size = []

        masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)
        valid_pixels = np.where(mask > 0)
        u_coords = valid_pixels[1]
        v_coords = valid_pixels[0]
        depths = masked_depth[v_coords, u_coords] / self.depth_image_units_divisor

        mean_depth = np.mean(depths)
        std_depth = np.std(depths)
        filtered_indices = np.where((depths > mean_depth - std_depth) & (depths < mean_depth + std_depth))
        u_coords = u_coords[filtered_indices]
        v_coords = v_coords[filtered_indices]
        depths = depths[filtered_indices]

        points = []
        for u, v, d in zip(u_coords, v_coords, depths):
            if d > 0:
                x = (u - camerainfo[2]) * d / camerainfo[0]
                y = (v - camerainfo[3]) * d / camerainfo[1]
                z = d
                points.append([x, y, z])
        
        if points:
            points = np.array(points)
            min_point = np.min(points, axis=0)
            max_point = np.max(points, axis=0)

            center = (min_point + max_point) / 2.0
            size = max_point - min_point
        
        return center, size



    def _cb(self, camerainfo: CameraInfo, depthimage: Image, object_info: ObjectInfo2DArray) -> None:
        depth_image = self.bridge.imgmsg_to_cv2(depthimage, "32FC1")
        fx = camerainfo.k[0]
        fy = camerainfo.k[4]
        cx = camerainfo.k[2]
        cy = camerainfo.k[5]

        object_marker_array = MarkerArray()
        object_3d_info_array = ObjectInfo3DArray()
        object_3d_info_array.header = object_info.header
        info:ObjectInfo2D
        i = 0
        for info in object_info.info:
            try:
                object_3d_info = ObjectInfo3D()
                object_3d_info.id = info.id
                object_3d_info.name = info.name
                object_3d_info.predict = info.predict

                x, y = 0, 0
                size = [0.1, 0.1, 0.1]

                # from segmentation mask image
                if info.maskimage.width and info.maskimage.height:
                    mask = self.bridge.imgmsg_to_cv2(info.maskimage, "mono8")
                    center, size = self.get_pointclouds(mask, depth_image, [fx, fy, cx, cy])

                    if len(center) == 3:
                        object_3d_info.pose.position.x = center[0]
                        object_3d_info.pose.position.y = center[1]
                        object_3d_info.pose.position.z = center[2]

                
                # from bbox image
                if not any([x,y]):
                    x = info.roi.x_offset + info.roi.width // 2
                    y = info.roi.y_offset + info.roi.height // 2

                    d = depth_image[int(x), int(y)] / self.depth_image_units_divisor
                    if d > 0:
                        object_3d_info.pose.position.x = (x - cx) * d / fx
                        object_3d_info.pose.position.y = (y - cy) * d / fy
                        object_3d_info.pose.position.z = d

                
                #if d < 1.0: print(d)
                
                object_3d_info_array.info.append(object_3d_info)

                object_marker = self.add_object_pose_marker(object_3d_info, object_info, info, size, i)
                object_marker_array.markers.append(object_marker)
                i += 1
                object_marker = self.add_object_text_marker(object_3d_info, object_info, info, i)
                object_marker_array.markers.append(object_marker)
                i += 1
                                                   
            except IndexError as e:
                #print(e)
                pass
                
        
        # publish
        self.pub_poses.publish(object_3d_info_array)
        self.pub_posemarkers.publish(object_marker_array)
        self.pub_imagecamerainfo.publish(camerainfo)


def pose_transformer(args=None):
    rclpy.init(args=args)
    node = PoseTransformer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        node.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloWorldRos2Main()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        node.destroy_node()

if __name__ == '__main__':
    main()
