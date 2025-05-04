#!/usr/bin/env python3
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.node import Node
import rclpy

from message_filters import ApproximateTimeSynchronizer, Subscriber
from yolo_world_interfaces.msg import RoiPoseArray, PoseStampedArray
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA

from cv_bridge import CvBridge
import numpy as np
import random
import uuid


class PoseTransformer(Node):
    def __init__(self):
        super().__init__('pose_transformer')
        
        # パラメータ設定
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('marker_lifetime', 0.1)
        self.declare_parameter('depth_sample_size', 5)
        self.declare_parameter('min_detection_confidence', 0.03)

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # サブスクライバーの設定
        pose_sub = Subscriber(self, RoiPoseArray, '/yolo_world/pose/pose_2d')
        depth_sub = Subscriber(self, Image, 'depth_image_raw')
        depth_info_sub = Subscriber(self, CameraInfo, 'depth_camerainfo')
        color_info_sub = Subscriber(self, CameraInfo, 'color_camerainfo')
        
        # 時間同期フィルタ
        self.ts = ApproximateTimeSynchronizer(
            [pose_sub, depth_sub, depth_info_sub, color_info_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        
        # パブリッシャーの設定
        self.marker_pub = self.create_publisher(MarkerArray, '/yolo_world/markers', 10)
        self.pose_pub = self.create_publisher(PoseStampedArray, '/yolo_world/pose/pose_3d', 10)
        
        # その他の初期化
        self.bridge = CvBridge()
        self.depth_camera_matrix = None
        self.color_camera_matrix = None
        self.depth_dist_coeffs = None
        self.color_dist_coeffs = None
        self.marker_id = 0
        self.object_colors = {}
        self.last_marker_ids = set()
        self.object_instances = {}

    def sync_callback(self, pose_msg, depth_msg, depth_info_msg, color_info_msg):
        try:
            # カメラパラメータ初期化
            if self.depth_camera_matrix is None or self.color_camera_matrix is None:
                self.depth_camera_matrix = np.array(depth_info_msg.k).reshape(3, 3)
                self.color_camera_matrix = np.array(color_info_msg.k).reshape(3, 3)
                self.depth_dist_coeffs = np.array(depth_info_msg.d)
                self.color_dist_coeffs = np.array(color_info_msg.d)
                self.get_logger().info("カメラパラメータを初期化しました")
            
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            depth_scale = self.get_parameter('depth_scale').value
            sample_size = self.get_parameter('depth_sample_size').value
            min_conf = self.get_parameter('min_detection_confidence').value
            
            marker_array = MarkerArray()
            current_marker_ids = set()
            current_frame_instances = {}
            pose_stamped_list = []
            
            # 古いマーカーを削除
            for marker_id in self.last_marker_ids:
                marker = Marker()
                marker.header = pose_msg.header
                marker.id = marker_id
                marker.action = Marker.DELETE
                marker_array.markers.append(marker)
            
            for idx, roi_pose in enumerate(pose_msg.poses):
                if roi_pose.conf < min_conf:
                    continue
                
                # カラー座標→深度座標
                u_depth, v_depth = self._align_color_to_depth(
                    roi_pose.x, roi_pose.y,
                    self.color_camera_matrix,
                    self.depth_camera_matrix
                )
                
                # 深度取得
                depth = self._get_median_depth(
                    depth_image, 
                    int(u_depth), 
                    int(v_depth), 
                    sample_size
                ) * depth_scale
                
                if depth <= 0:
                    continue
                
                # 3D座標計算
                x = (u_depth - self.depth_camera_matrix[0, 2]) * depth / self.depth_camera_matrix[0, 0]
                y = (v_depth - self.depth_camera_matrix[1, 2]) * depth / self.depth_camera_matrix[1, 1]
                z = depth
                
                # インスタンス管理
                instance_id = self._find_closest_instance(roi_pose.class_name, x, y, z)
                if instance_id is None:
                    instance_id = f"{roi_pose.class_name}_{uuid.uuid4().hex[:4]}"
                current_frame_instances[instance_id] = (x, y, z, roi_pose)
                
                # Marker 作成
                color = self._get_object_color(instance_id)
                # 中心点
                self.marker_id += 1
                current_marker_ids.add(self.marker_id)
                marker = Marker()
                marker.header = pose_msg.header
                marker.ns = instance_id
                marker.id = self.marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.lifetime = rclpy.duration.Duration(
                    seconds=self.get_parameter('marker_lifetime').value).to_msg()
                marker.pose.position = Point(x=x, y=y, z=z)
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
                marker_array.markers.append(marker)
                
                # テキスト
                self.marker_id += 1
                current_marker_ids.add(self.marker_id)
                text_marker = Marker()
                text_marker.header = pose_msg.header
                text_marker.ns = f"{instance_id}_text"
                text_marker.id = self.marker_id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.lifetime = rclpy.duration.Duration(
                    seconds=self.get_parameter('marker_lifetime').value).to_msg()
                text_marker.pose.position = Point(x=x, y=y, z=z+0.15)
                text_marker.scale.z = 0.1
                text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                text_marker.text = f"{roi_pose.class_name}\nconf: {roi_pose.conf:.2f}"
                marker_array.markers.append(text_marker)
                
                # PoseStamped 生成
                ps = PoseStamped()
                ps.header.stamp = depth_msg.header.stamp
                ps.header.frame_id = roi_pose.class_name
                ps.pose.position.x = x
                ps.pose.position.y = y
                ps.pose.position.z = z
                ps.pose.orientation.w = 1.0
                pose_stamped_list.append(ps)

            # 状態更新
            self.object_instances = current_frame_instances
            self.last_marker_ids = current_marker_ids
            
            # マーカーと姿勢情報をパブリッシュ
            if marker_array.markers:
                self.marker_pub.publish(marker_array)
            
            if pose_stamped_list:
                pose_array_msg = PoseStampedArray()
                pose_array_msg.header = depth_msg.header
                pose_array_msg.posestampedarray = pose_stamped_list
                self.pose_pub.publish(pose_array_msg)

        except Exception as e:
            self.get_logger().error(f"処理エラー: {str(e)}")

    def _align_color_to_depth(self, u_color, v_color, color_matrix, depth_matrix):
        scale_x = depth_matrix[0, 0] / color_matrix[0, 0]
        scale_y = depth_matrix[1, 1] / color_matrix[1, 1]
        u_depth = (u_color - color_matrix[0, 2]) * scale_x + depth_matrix[0, 2]
        v_depth = (v_color - color_matrix[1, 2]) * scale_y + depth_matrix[1, 2]
        return u_depth, v_depth

    def _get_median_depth(self, depth_image, u, v, size):
        h, w = depth_image.shape
        half = size // 2
        u_min = max(0, u-half)
        u_max = min(w, u+half+1)
        v_min = max(0, v-half)
        v_max = min(h, v+half+1)
        roi = depth_image[v_min:v_max, u_min:u_max]
        valid_depths = roi[roi > 0]
        return float(np.median(valid_depths)) if valid_depths.size > 0 else 0.0

    def _find_closest_instance(self, class_name, x, y, z, max_distance=0.5):
        closest_id = None
        min_distance = float('inf')
        for instance_id, (prev_x, prev_y, prev_z, _) in self.object_instances.items():
            if not instance_id.startswith(class_name):
                continue
            dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2 + (z - prev_z)**2)
            if dist < max_distance and dist < min_distance:
                min_distance = dist
                closest_id = instance_id
        return closest_id

    def _get_object_color(self, instance_id):
        if instance_id not in self.object_colors:
            self.object_colors[instance_id] = (
                random.uniform(0.2, 1.0),
                random.uniform(0.2, 1.0),
                random.uniform(0.2, 1.0)
            )
        return self.object_colors[instance_id]


def main(args=None):
    rclpy.init(args=args)
    node = PoseTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
