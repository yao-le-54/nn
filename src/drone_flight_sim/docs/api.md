# API 参考

## 键盘控制 API

```python
from keyboard_control import KeyboardController, print_control_help

# 打印控制说明
print_control_help()

# 创建键盘控制器并启动
controller = KeyboardController(drone)
controller.start()
```

## 相机控制 API

```python
# 创建无人机控制器
drone = DroneController()

# 设置图片保存目录（可选，默认保存到 drone_images 文件夹）
drone.set_output_dir("my_photos")

# 拍摄 RGB 彩色图像
drone.capture_image()

# 指定文件名保存
drone.capture_image(filename="my_photo.png")

# 拍摄并显示预览窗口
drone.capture_image(show_preview=True)

# 拍摄深度图像（伪彩色）
drone.capture_depth_image()

# 拍摄分割图像
drone.capture_segmentation_image()

# 同时拍摄 RGB + 深度 + 分割三种图像
drone.capture_all_cameras()

# 显示无人机状态
drone.get_telemetry()
```

## 航点规划 API

```python
from flight_path import FlightPath

# 使用正方形路径
waypoints = FlightPath.square_path(size=15, height=-3)

# 使用矩形路径
waypoints = FlightPath.rectangle_path(width=20, length=10, altitude=-3)

# 使用自定义路径
waypoints = [(5, 0, -3), (5, -5, -3), (0, -5, -3), (0, 0, -3)]
```
