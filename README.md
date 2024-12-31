# yolo_world_ros2

## Overview

This package provides a Docker container for implementing YOLO world.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/GAI-313/yolo_world_ros2.git
   ```
   It is recommended to place this repository directly within the host's workspace. The package `yolo_world_ros2` contained in this repository is designed to operate within a Docker container, and builds on the host will be ignored due to the presence of the COLCON_IGNORE file. Additionally, by installing the `object_detect_interfaces` package mentioned later, you will be able to access the resources provided by `yolo_world_ros2`. 

2. Navigate to the `src` directory of the repository and clone the following repository:

   ```bash
   git clone https://github.com/GAI-313/object_detect_interfaces.git
   ```

### Prerequisites

Ensure that the following are installed:

- Docker
- Docker Compose
- NVIDIA Container Toolkit

This package requires a computer equipped with an NVIDIA GPU that supports CUDA.

## Building and Running the Container

To build and run the container, use the following command:

```bash
docker compose up yolo_world_ros2
```

## License

This project is licensed under the MIT License.
