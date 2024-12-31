#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

# default values for debug
DEFAULT_PREDICT = 0.1
DEFAULT_CLASSES = ''
DEFAULT_USESAM = False
DEFAULT_EXECUTE = False

def generate_launch_description():
    ld = LaunchDescription()


    namespace = 'yolo_world'


    default_predict = LaunchConfiguration('predict')
    default_classes = LaunchConfiguration('classes')
    default_use_sam = LaunchConfiguration('use_sam')
    default_execute = LaunchConfiguration('execute')


    remappings = [
        ('image_raw', '/camera/rgb/image_raw'),
        ('depth/camera_info', '/camera/depth/camera_info'),
        ('depth/image_raw', '/camera/depth/image_raw')
    ]
    remappings = [
        ('image_raw', '/camera/camera/color/image_raw'),
        ('depth/camera_info', '/camera/camera/aligned_depth_to_color/camera_info'),
        ('depth/image_raw', '/camera/camera/aligned_depth_to_color/image_raw')
    ]


    declare_predict = DeclareLaunchArgument(
        'predict', default_value=str(DEFAULT_PREDICT),
        description='Defines the default minimum confidence threshold for object detection.'
    )
    declare_classes = DeclareLaunchArgument(
        'classes', default_value=str(DEFAULT_CLASSES),
        description='Specifies the default list of object names to detect. If an empty string is provided, YoLo will search for standard detectable objects.'
    )
    declare_use_sam = DeclareLaunchArgument(
        'use_sam', default_value=str(DEFAULT_USESAM),
        description='Enables SAM to obtain segmentation information for detected objects by default.'
    )
    declare_execute = DeclareLaunchArgument(
        'execute', default_value=str(DEFAULT_EXECUTE),
        description='Enables object detection by default. If set to False, object detection can be activated via a service.'
    )

    ld.add_action(declare_predict)
    ld.add_action(declare_classes)
    ld.add_action(declare_use_sam)
    ld.add_action(declare_execute)


    yolo_world_ros2_main = Node(
        package='yolo_world_ros2',
        executable='yolo_world_ros2_main',
        name='yolo_world',
        parameters=[
            {'default_predict':default_predict},
            {'default_classes':default_classes},
            {'default_use_sam':default_use_sam},
            {'default_execute':default_execute}
        ],
        namespace=namespace,
        output='screen',
        emulate_tty=True,
        remappings=remappings
    )
    pose_transformer = Node(
        package='yolo_world_ros2',
        executable='pose_transformer',
        name='pose_transformer',
        namespace=namespace,
        output='screen',
        emulate_tty=True,
        remappings=remappings
    )


    ld.add_action(yolo_world_ros2_main)
    ld.add_action(pose_transformer)


    return ld
