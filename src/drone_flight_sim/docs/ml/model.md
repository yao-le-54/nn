# 碰撞预测模型

使用 CNN 对深度图像进行二分类（安全 vs 危险碰撞风险）。

## 模型训练

```bash
python train_collision_model.py
```

## 模型评估

```bash
python train_collision_model.py --eval
```

## 模型文件

训练好的模型保存在：`collision_model.pth`

## 模型架构

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

## 训练数据

- 数据集：`collision_dataset/`
- 样本数：301（安全: 278, 危险: 23）
- 数据增强：过采样平衡类别
- 测试准确率：91.80%
