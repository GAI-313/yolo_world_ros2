import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'yolo_world_ros2'

data_files = []
data_files.append(
    ("share/ament_index/resource_index/packages", ["resource/" + package_name])
)
data_files.append(("share/" + package_name, ["package.xml"]))


def package_files(directory, data_files):
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            data_files.append(
                (
                    "share/" + package_name + "/" + path,
                    glob(path + "/**/*.*", recursive=True),
                )
            )
    return data_files

# Add directories
data_files = package_files("launch", data_files)

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gai',
    maintainer_email='nakatogawagai@gmail.com',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_world_ros2_main = yolo_world_ros2.yolo_world_ros2_main:main',
            'pose_transformer = yolo_world_ros2.yolo_world_ros2_main:pose_transformer',
            'seg_pose_transformer = yolo_world_ros2.yolo_world_ros2_main:seg_pose_transformer'
        ],
    },
)
