"""批量归一化 (Batch Normalization) 实现

批量归一化是一种训练技巧，可以：
1. 加速神经网络训练收敛速度
2. 允许使用更大的学习率
3. 减少对参数初始化的依赖
4. 起到一定的正则化作用

核心思想：
在每一层的输入上进行标准化，使其均值为 0,方差为 1,
然后通过可学习的缩放参数 γ 和偏移参数 β 恢复表达能力。

公式:BN(x) = γ * (x - μ) / √(σ² + ε) + β
"""

import torch
from torch import nn
from d2l import torch as d2l


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """批量归一化函数
    
    参数：
        X: 输入数据，形状为 (batch_size, num_features) 或 (batch_size, channels, height, width)
        gamma: 缩放参数（可学习）
        beta: 偏移参数（可学习）
        moving_mean: 移动平均均值（用于推理）
        moving_var: 移动平均方差（用于推理）
        eps: 防止除零的小常数（通常为 1e-5)
        momentum: 移动平均的动量参数（通常为 0.9)
    
    返回：
        Y: 归一化后的输出
        moving_mean: 更新后的移动平均均值
        moving_var: 更新后的移动平均方差
    """
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 【预测模式】如果是在预测模式下，直接使用训练时累积的移动平均均值和方差
        # 这样可以保证推理时的稳定性，不受单个批次统计量的影响
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 【训练模式】
        assert len(X.shape) in (2, 4)  # 确保输入是全连接层(2维)或卷积层(4维)
        if len(X.shape) == 2:
            # 【全连接层】形状: (batch_size, num_features)
            # 在 batch_size 维度(dim=0)上计算每个特征的均值和方差
            mean = X.mean(dim=0)  # 输出形状: (num_features,)
            var = ((X - mean) ** 2).mean(dim=0)  # 输出形状: (num_features,)
        else:
            # 【卷积层】形状: (batch_size, channels, height, width)
            # 在 batch_size(0)、height(2)、width(3) 维度上计算每个通道的均值和方差
            # 保持通道维度，这样可以对每个通道独立进行归一化
            # keepdim=True 保持维度以便后续广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)  # 输出形状: (1, channels, 1, 1)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)  # 输出形状: (1, channels, 1, 1)
        # 训练模式下，用当前批次的均值和方差做标准化
        # X_hat = (X - μ_batch) / √(σ²_batch + ε)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        
        # 更新移动平均的均值和方差（用于推理阶段）
        # 使用指数移动平均：new = momentum * old + (1 - momentum) * current
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    
    # 缩放和偏移：Y = γ * X_hat + β
    # γ 和 β 是可学习参数，让模型能够恢复原始表达能力
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    """批量归一化层
    
    封装批量归一化逻辑，使其可以作为 nn.Module 嵌入到网络中。
    
    参数：
        num_features: 特征数量
            - 对于全连接层：输出神经元数量
            - 对于卷积层：输出通道数
        num_dims: 输入张量的维度数
            - 2: 全连接层，形状 (batch_size, num_features)
            - 4: 卷积层，形状 (batch_size, channels, height, width)
    """
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        # 根据输入维度确定参数形状
        if num_dims == 2:
            shape = (1, num_features)  # 全连接层：(1, 特征数)
        else:
            shape = (1, num_features, 1, 1)  # 卷积层：(1, 通道数, 1, 1)
        
        # 【可学习参数】参与梯度下降和反向传播
        self.gamma = nn.Parameter(torch.ones(shape))   # 缩放参数 γ，初始化为 1
        self.beta = nn.Parameter(torch.zeros(shape))   # 偏移参数 β，初始化为 0
        
        # 【非可学习参数】用于推理阶段的移动平均统计量
        self.moving_mean = torch.zeros(shape)  # 移动平均均值，初始化为 0
        self.moving_var = torch.ones(shape)    # 移动平均方差，初始化为 1

    def forward(self, X):
        """前向传播
        
        参数：
            X: 输入数据
        
        返回：
            Y: 归一化后的输出
        """
        # 设备兼容性处理：如果输入在 GPU 上，将统计量也移动到 GPU
        # 这确保了在 CPU/GPU 之间切换时的正确性
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        
        # 调用批量归一化函数，并保存更新后的移动平均统计量
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
    

# 构建 LeNet 网络并在每个激活函数前添加批量归一化
# 网络结构：卷积层 → BN → 激活 → 池化 → ... → 全连接层 → BN → 激活 → 输出
net = nn.Sequential(
    # 第一个卷积块：1x28x28 → 6x24x24 → BN → ReLU → 6x12x12
    nn.Conv2d(1, 6, kernel_size=5),      # 卷积层：1 → 6 通道
    BatchNorm(6, num_dims=4),            # 批量归一化（卷积层，4维）
    nn.ReLU(),                         # ReLU 激活函数
    nn.AvgPool2d(kernel_size=2, stride=2),  # 平均池化：降采样
    
    # 第二个卷积块：6x12x12 → 16x8x8 → BN → ReLU → 16x4x4
    nn.Conv2d(6, 16, kernel_size=5),     # 卷积层：6 → 16 通道
    BatchNorm(16, num_dims=4),           # 批量归一化（卷积层，4维）
    nn.ReLU(),                         # ReLU 激活函数
    nn.AvgPool2d(kernel_size=2, stride=2),  # 平均池化：降采样
    nn.Flatten(),                         # 展平：16x4x4 → 256
    
    # 第一个全连接层：256 → 120 → BN → ReLU
    nn.Linear(16*4*4, 120),              # 全连接层
    BatchNorm(120, num_dims=2),          # 批量归一化（全连接层，2维）
    nn.ReLU(),                         # ReLU 激活函数
    
    # 第二个全连接层：120 → 84 → BN → ReLU
    nn.Linear(120, 84),                  # 全连接层
    BatchNorm(84, num_dims=2),           # 批量归一化（全连接层，2维）
    nn.ReLU(),                         # ReLU 激活函数
    
    # 输出层：84 → 10（10个类别）
    nn.Linear(84, 10))                   # 输出层，无激活函数（使用交叉熵损失时）

# 训练超参数设置
lr, num_epochs, batch_size = 1.0, 10, 256  # 学习率=1.0（BN允许使用更大的学习率）

# 加载 Fashion-MNIST 数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 开始训练（自动使用 GPU 如果可用）
# BN 的优势：可以看到收敛速度明显加快，训练更稳定
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

# 显示训练过程的损失和准确率曲线
d2l.plt.show()