#!/usr/bin/env python3
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.node import Node
import rclpy

from yolo_world_interfaces.msg import RoiPoseArray, RoiPose
from yolo_world_interfaces.srv import SetClasses
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool

from cv_bridge import CvBridge
import cv2

from ultralytics import YOLO


class YoloWorldRos2(Node):
    def __init__(self):
        super().__init__('yolo_world_ros2')

        self.declare_parameter('auto_bringup', True)
        self.declare_parameter('yolo_model', 'yolov8m-world.pt')

        self.execute = self.get_parameter('auto_bringup').get_parameter_value().bool_array_value
        self.yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        
        self.detect_conf = 0.5

        # YOLO-Worldモデルの初期化
        if self.execute:
            self.model = self.bringup_model()
        
        # OpenCVブリッジ
        self.bridge = CvBridge()

        # QoS
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # 画像サブスクライバー
        self.subscription = self.create_subscription(
            Image,
            '/color_image_raw',
            self.image_callback,
            qos_profile
        )
        
        # RoiPoseArrayパブリッシャー
        self.publisher_ = self.create_publisher(
            RoiPoseArray,
            '/yolo_world/pose/pose_2d',
            10
        )

        # サービス
        self.srv_execute = self.create_service(
            SetBool,
            '/yolo_world/execute',
            self._cb_execute_manager
        )
        self.srv_classes = self.create_service(
            SetClasses,
            '/yolo_world/classes',
            self._cb_classes_manager
        )
        
        # OpenCVウィンドウの設定
        cv2.namedWindow("YOLO-World Detection", cv2.WINDOW_NORMAL)
        
        self.get_logger().info("YoloWorldRos2 node has been initialized")
    

    def bringup_model(self):
        self.get_logger().info("""
LOAD MODEL : %s
MINIMUM_CONF : %f                               
        """%(self.yolo_model, self.detect_conf))

        return YOLO(self.yolo_model)
    

    def _cb_execute_manager(self, req:SetBool.Request, res:SetBool.Response):
        if req.data:
            self.model = self.bringup_model()
            self.execute = True
            res.message = "Load %s"%self.yolo_model
        
        else:
            self.execute = False
            del self.model
            res.message = 'Stop detection'
        
        res.success = True
        return res
    

    def _cb_classes_manager(self, req:SetClasses.Request, res:SetClasses.Response):
        classes = req.classes
        self.detect_conf = req.conf
        try:
            if self.execute:

                self.get_logger().info(f"""
SET CLASS ...
{classes}
CONF: {self.detect_conf}
                """)

                if not any(classes):
                    self.get_logger().warn('Not declare classes. detect default classes')
                    del self.model
                    self.model = self.bringup_model()
                
                else:
                    self.model.set_classes(classes) 
                
                res.success = True
            
            else:
                self.get_logger().warn('Model is not running. Please execute model.')
                res.success = False
        
        except RuntimeError:
            self.get_logger().error(f'Invalid classes. {classes} is invalid type.')
            res.success = False

        return res
        

    def image_callback(self, msg:Image):
        if self.execute:
            try:
                # ROS Image -> OpenCV Image
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                # YOLO-Worldで推論
                results = self.model.predict(cv_image, verbose=False, conf=self.detect_conf)
                
                # 検出結果の描画
                annotated_frame = results[0].plot()
                cv2.imshow("YOLO-World Detection", annotated_frame)
                cv2.waitKey(1)
                
                # 検出結果のパブリッシュ
                roi_pose_array = RoiPoseArray()
                roi_pose_array.header = msg.header
                
                for box in results[0].boxes:
                    roi_pose = RoiPose()
                    # バウンディングボックスの中心座標を計算
                    x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                    y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                    
                    roi_pose.x = x_center
                    roi_pose.y = y_center
                    roi_pose.conf = float(box.conf)
                    roi_pose.class_name = results[0].names[int(box.cls)]
                    
                    roi_pose_array.poses.append(roi_pose)

                self.publisher_.publish(roi_pose_array)
                
            except Exception as e:
                self.get_logger().error(f"Error in image_callback: {str(e)}")

    def __del__(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = YoloWorldRos2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()