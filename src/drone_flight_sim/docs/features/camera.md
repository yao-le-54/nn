# 相机拍照功能

## 键盘快捷键

| 按键 | 功能 | 说明 |
|------|------|------|
| **P** | RGB拍照 | 拍摄彩色图像，自动保存 |
| **T** | 全景拍照 | 一次性拍摄 RGB + 深度 + 分割三种图像 |
| **N** | 深度图像 | 拍摄深度图（伪彩色：蓝=近，红=远） |
| **B** | 实时预览 | 打开相机预览窗口，实时查看无人机视角 |

## 拍照功能详情

### RGB拍照（P键）
拍摄无人机视角的彩色照片，自动保存到 `drone_images/` 目录，文件名包含时间戳和位置信息。

### 全景拍照（T键）
同时获取 RGB 彩色图、深度图、分割图三种图像，适合需要完整数据的场景。

### 深度图像（N键）
拍摄深度图，使用伪彩色显示（JET色彩表：蓝色表示近，红色表示远），可用于测距和避障。

### 实时预览（B键）
打开相机实时预览窗口，可以直观看到无人机视角，适合探索环境时使用。

## 图片保存

- 保存位置：`drone_images/` 目录（自动创建）
- RGB图像：`rgb_时间戳_X_Y_n序号.png`
- 深度图像：`depth_时间戳_X_Y.png`
- 分割图像：`seg_时间戳_X_Y.png`
- 全景图像：`all_时间戳_X_Y_rgb/depth/seg.png`

## 代码示例

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
```
