#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    namespace = 'yolo_world'

    remappings = [
        ('image_raw', '/camera/camera/color/image_raw'),
        ('depth/camera_info', '/camera/camera/aligned_depth_to_color/camera_info'),
        ('depth/image_raw', '/camera/camera/aligned_depth_to_color/image_raw')
    ]

    yolo_world_ros2_main = Node(
        package='yolo_world_ros2',
        executable='yolo_world_ros2_main',
        namespace=namespace,
        output='screen',
        remappings=remappings
    )
    pose_transformer = Node(
        package='yolo_world_ros2',
        executable='pose_transformer',
        namespace=namespace,
        output='screen',
        remappings=remappings
    )

    ld.add_action(yolo_world_ros2_main)
    ld.add_action(pose_transformer)

    return ld
