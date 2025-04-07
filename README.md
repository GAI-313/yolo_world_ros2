# yolo-world-ros2

## Install

- ROS2 Humble をインストール
- Nvidia Driver & CUDA をインストール

---

1. `pip` `venv` をインストール
    ```bash
    sudo apt install -y python3-pip python3-venv
    ```

2. `yolo_world_ros2` をダウンロード
    ```bash
    cd ~/colcon_ws/src
    https://github.com/GAI-313/yolo_world_ros2.git
    ```

3. パッケージをビルド
    ```bash
    cd ~/colcon_ws
    source /opt/ros/humble/setup.bash
    ```
    ```bash
    # yolo_world_ros2 のみビルド
    colcon build --symlink-install --packages-up-to yolo_world_ros2
    ```
    ```bash
    source ~/colcon_ws/install/setup.bash
    ```

4. `venv` 環境を作成
    　ここでは `YOLO` という名前の仮想環境を `~` 内に作成する。
    ```bash
    python3 -m venv YOLO
    ```
    仮想環境起動
    ```bash
    source ~/YOLO/bin/activate
    ```

5. 依存関係のインストール
    　`CvBridge` は `"numpy<2"` しか対応していない。
    ```bash
    pip install ultralytics "numpy<2"
    ```

## Running

> [!WARNING]
> venv 環境下で実行することを推奨します。

- 起動
    ```bash
    ros2 launch yolo_world_ros2 yolo_world_ros2_launch.py \
    color_image:=<RGB camera topic> \
    depth_image:=<Depth camera topic> \
    depth_camerainfo:=<Depth camera info> \
    color_camerainfo:=<RGB camera info>
    ```

- プロセスの開始をリクエスト
    ```bash
    ros2 service call /yolo_world/execute std_srvs/srv/SetBool "data: True"
    ```
    デフォルトでは通常の YoLo 物体検出が実行されます。

- 検出対象の指定<br>
    　例：[椅子、携帯、タブレット、ペン]を 25% の精度で検出する。
    ```bash
    ros2 service call /yolo_world/classes yolo_world_interfaces/srv/SetClasses \
    "{classes: [chair, phone, tablet, pencil], conf: 0.25}"
    ```
