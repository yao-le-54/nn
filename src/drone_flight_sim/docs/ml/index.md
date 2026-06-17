# 机器学习

本节介绍碰撞检测相关的机器学习模块。

## 数据采集

项目提供两种数据采集方式：

### 自动采集（推荐）

```bash
python auto_collision_collector.py
```

飞行模式：

| 模式 | 说明 | 碰撞率 |
|------|------|--------|
| 1. 螺旋飞行 | 螺旋向外扩大飞行 | 中 |
| 2. 随机飞行 | 飞向随机目标点 | 中 |
| 3. 折线飞行 | 高速直线折返 | 高 |
| 4. 撞墙模式 | 专门朝障碍物飞行 | 最高 |

安全样本(label=0)：飞行中定期采集。危险样本(label=1)：碰撞时自动标注。

### 手动采集

```bash
python collision_data_collector.py
```

| 按键 | 功能 |
|------|------|
| 0 | 设置安全标签 |
| 1 | 设置危险标签 |
| C | 采集当前样本 |

数据保存：`collision_dataset/depth/` + `collision_dataset/labels.csv`

## 碰撞预测模型

CNN 对深度图像进行二分类（安全 vs 危险碰撞风险）：

```bash
python train_collision_model.py
```

### 模型架构

```
输入: 深度图像 (64x64 灰度)
    ↓
Conv2D(1→16) + BN + ReLU + MaxPool
    ↓
Conv2D(16→32) + BN + ReLU + MaxPool
    ↓
Conv2D(32→64) + BN + ReLU + MaxPool
    ↓
Conv2D(64→128) + BN + ReLU + MaxPool
    ↓
Flatten → Dense(256) → Dropout → Dense(64) → Dense(1)
    ↓
输出: 碰撞风险概率 [0~1]
```

### 性能

- 数据集：`collision_dataset/`
- 样本数：301（安全: 278, 危险: 23）
- 测试准确率：91.80%
- 模型文件：`collision_model.pth`
