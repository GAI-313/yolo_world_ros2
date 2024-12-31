# Copyright (c) 2024 Gai Nakatogawa
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

FROM gai313/ros:humble

WORKDIR /
RUN wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-world.pt &&\
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_b.pt

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /colcon_ws

RUn apt-get update &&\
    apt-get install -y ros-humble-cv-bridge

COPY ./src ./src
RUN . /opt/ros/humble/setup.bash &&\
    colcon build --symlink-install

COPY ./entrypoint.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh

ENTRYPOINT ["/tmp/entrypoint.sh"]
CMD ["terminator"]
