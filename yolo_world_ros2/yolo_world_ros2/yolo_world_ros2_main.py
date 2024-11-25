#!/usr/bin/env python3
from ultralytics import YOLOWorld

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from rclpy.node import Node
import rclpy

import cv2


class YoloWorldRos2Main(Node):
    def __init__(self) ->None:
        super().__init__('yolo_world_ros2_main')

        self.classes = ['black_cube']

        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self._image_cb, 10)
        self.bridge = CvBridge()

        self.model = YOLOWorld('yolov8l-world.pt')
        self.model.set_classes(self.classes)
    
    def _image_cb(self, msg: Image) ->None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(e)
        
        results = self.model.predict(
            source=image,
            conf=0.01,
            iou=0.1,
            max_det=10
        )

        image = results[0].plot()

        cv2.waitKey(1)
        cv2.imshow('detect image', image)


def main():
    rclpy.init()
    node = YoloWorldRos2Main()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        node.destroy_node()

if __name__ == '__main__':
    main()