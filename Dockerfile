# Copyright (c) 2024 Gai Nakatogawa
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 as base

LABEL maintainer="Gai Nakatogawa <nakatogawagai@gmail.com>"
LABEL nakatogawalaboratory.vendor="Gai Nakatogawa"\
      nakatogawalaboratory.version="1.0.0"\
      nakatogawalaboratory.released="July 19 2024"

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG ja_JP.UTF-8
ENV TZ=Asia/Tokyo

RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
    locales \
    software-properties-common \
    tzdata &&\
    locale-gen ja_JP ja_JP.UTF-8 &&\
    update-locale LC_ALL=ja_JP.UTF-8 LANG=ja_JP.UTF-8 &&\
    add-apt-repository universe &&\
    apt-get update; apt-get install -y \ 
    build-essential cmake g++ iproute2 gnupg gnupg1 gnupg2 libcanberra-gtk* \
    python3-pip python3-tk \
    git wget curl nano htop gnupg2 lsb-release \
    x11-utils x11-apps xauth \
    software-properties-common gdb valgrind sudo \
    terminator

    # ROS2 install
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg &&\
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null &&\
    apt-get update ; apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    python3-rosdep2 &&\
    pip install setuptools==58.2.0 &&\
    . /opt/ros/humble/setup.sh &&\
    rm /etc/ros/rosdep/sources.list.d/20-default.list &&\
    rosdep init

ARG UID=1000
ARG GID=1000
ARG USER_NAME=user
ARG GROUP_NAME=user
ARG PASSWORD=password
RUN groupadd -g $GID $GROUP_NAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USER_NAME && \
    echo $USER_NAME:$PASSWORD | chpasswd && \
    echo "$USER_NAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${USER_NAME}

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /home/${USER_NAME}/colcon_ws/src
RUN cd ~ && wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-world.pt
COPY ./src .

WORKDIR /home/${USER_NAME}/colcon_ws
RUN . /opt/ros/humble/setup.bash &&\
    colcon build --symlink-install
RUN echo "source ~/colcon_ws/install/setup.bash" >> /home/user/.bashrc

COPY ./entrypoint.sh /tmp/entrypoint.sh
RUN sudo chmod +x /tmp/entrypoint.sh

ENTRYPOINT ["/tmp/entrypoint.sh"]
