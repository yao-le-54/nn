# 航点规划

## 预设航点

系统预设 11 个航点，覆盖更大飞行区域，飞行高度 5 米。

## API 使用

```python
from flight_path import FlightPath

# 正方形路径
waypoints = FlightPath.square_path(size=15, height=-3)

# 矩形路径
waypoints = FlightPath.rectangle_path(width=20, length=10, altitude=-3)

# 三角形路径
waypoints = FlightPath.triangle_path(size=15, height=-5)

# 自定义路径
waypoints = [(5, 0, -3), (5, -5, -3), (0, -5, -3), (0, 0, -3)]
```

## 碰撞检测

- 实时监测碰撞事件
- 自动过滤地面/道路接触
- 碰撞后自动恢复（最多 3 次后退避障）
- 恢复失败后切换手动接管
