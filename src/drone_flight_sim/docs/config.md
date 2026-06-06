# 配置说明

在 `config.py` 中可以修改以下参数：

## 配置参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TAKEOFF_HEIGHT` | -5 | 起飞高度（米） |
| `FLIGHT_VELOCITY` | 3 | 飞行速度（米/秒） |
| `MAX_FLIGHT_TIME` | 60 | 最大飞行时间（秒） |
| `COLLISION_COOLDOWN` | 1.0 | 碰撞冷却时间（秒） |
| `RGB_CAMERA_NAME` | "0" | RGB 相机名称 |
| `KEYBOARD_VELOCITY` | 2 | 键盘控制默认速度（米/秒） |
| `KEYBOARD_STEP` | 2 | 键盘控制位移步长（米） |

## 示例配置

```python
# config.py

# 起飞高度
TAKEOFF_HEIGHT = -5

# 飞行速度
FLIGHT_VELOCITY = 3

# 最大飞行时间（秒）
MAX_FLIGHT_TIME = 60

# 碰撞冷却时间（秒）
COLLISION_COOLDOWN = 1.0

# RGB 相机名称
RGB_CAMERA_NAME = "0"

# 键盘控制速度（米/秒）
KEYBOARD_VELOCITY = 2

# 键盘控制位移步长（米）
KEYBOARD_STEP = 2
```
