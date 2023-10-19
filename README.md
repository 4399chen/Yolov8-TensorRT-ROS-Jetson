# Yolov8-TensorRT-ROS-Jetson
---

> version:3.0
> 
> Dong chen

## TODO


- ~~ROS_image_topic input~~
- segment
- ~~detect~~

## News

- `2023-10-07` `v3` can use file and ROS topic input
- `2023-10-07` `v2` just for ROS image topic input
- `2023-10-07` put the detect part of [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT) into ROS noetic 

## Hardware and software

- Nvidia Jetson Orin NX
- Ubuntu 20.04 ( Jetpack 5.1.2 )
- ROS noetic

## Prepare the environment

### Update CMake

The default CMake version in jetson is too low, so you can not install onnxsim by pip.

```powershell
sudo apt remove cmake
```

Download the source `cmake-3.27.7.tar.gz` from [https://cmake.org/download/](https://cmake.org/download/), and unzip it.

```powershell
# check the environment
./bootstrap

# build
sudo make

# install
sudo make install

# test
cmake --version
```

### Prepare the environment like YOLOv8-TensorRT

1. Install follow CUDA official website.CUDA
   
   🚀 RECOMMENDED >= 11.4CUDA

2. Install follow TensorRT official website.TensorRT

    🚀 RECOMMENDED >= 8.4TensorRT

3. Install python requirements.

    ```
    pip install -r requirements.txt
    ```

4. Install ultralytics package for ONNX export or TensorRT API building.

    ```
    pip install ultralytics
    ```

5. Prepare your own PyTorch weight such as `yolov8s.pt` or ` yolov8s-seg.pt`.

## How to run

1. Build ONNX

```
python3 export-det.py --weights yolov8s.pt --sim
```

2. Build engine

```
# Using trtexec tools for export engine
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s.onnx \
--saveEngine=yolov8s.engine
```

3. catkin_make

```
cd ~/catkin_ws
catkin_make
```

```
# rosrun yolov8 yolov8 /path/engine /path/data or /path/data/image or /path/data/video
rosrun yolov8 yolov8 /home/orin/Downloads/yolov8_tensorrt_jetson_detect/engine/yolov8n.engine /home/orin/Downloads/yolov8_tensorrt_jetson_detect/data/bus.jpg

# rosrun yolov8 yolov8 /path/engine /ROStopic
rosrun yolov8 yolov8 /home/orin/catkin_ws/src/yolov8/engine/yolov8n.engine /usb_cam/image_raw
```

> input type:
> - video
>   - mp4
>   - avi
>   - m4v
>   - mpeg
>   - mov
>   - mkv
> - image
>   - jpg
>   - jpeg
>   - png


## Tips

In this package, you can not use the OpenCV which is in the ROS, so this part of CMakeLists.txt is very important.

```txt
file(GLOB_RECURSE OLD_OPENCV "/usr/lib/aarch64-linux-gnu/libopencv*")
list(REMOVE_ITEM catkin_LIBRARIES ${OLD_OPENCV})
```
