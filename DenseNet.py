"""DenseNet (稠密连接网络) 实现

DenseNet的核心思想: 每一层都与前面所有层在通道维度上连接
优点: 缓解梯度消失、加强特征传播、鼓励特征复用、减少参数数量
"""
import torch
from torch import nn
from d2l import torch as d2l


def conv_block(input_channels, num_channels):
    """卷积块: BN -> ReLU -> 3x3卷积
    
    Args:
        input_channels: 输入通道数
        num_channels: 输出通道数 (增长率)
    
    Returns:
        包含批量归一化、ReLU激活和卷积的序列模块
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    """稠密块: DenseNet的核心模块
    
    每个卷积块的输出都会与之前所有层的输出在通道维度上拼接
    如果输入是x0, 经过n层后输出为 [x0, x1, x2, ..., xn]
    """
    def __init__(self, num_convs, input_channels, num_channels):
        """初始化稠密块
        
        Args:
            num_convs: 卷积层数量
            input_channels: 输入通道数
            num_channels: 每层的增长率 (growth rate), 即每层新增的通道数
        """
        super(DenseBlock, self).__init__()
        layer = []
        # 构建多个卷积块，每个块的输入通道数会累积增加
        for i in range(num_convs):
            # 第i层的输入通道数 = 初始输入 + 前面i层各自贡献的num_channels
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        """前向传播: 依次通过各卷积块, 并拼接特征
        
        Args:
            X: 输入张量, shape: (batch_size, input_channels, height, width)
        
        Returns:
            拼接后的输出, shape: (batch_size, input_channels + num_convs * num_channels, height, width)
        """
        for blk in self.net:
            Y = blk(X)  # 当前块的输出
            # 连接通道维度上每个块的输入和输出 (DenseNet的关键操作)
            X = torch.cat((X, Y), dim=1)  # 在通道维度 (dim=1) 上拼接
        return X

def transition_block(input_channels, num_channels):
    """过渡层: 用于连接两个稠密块
    
    作用:
    1. 通过1x1卷积减少通道数, 降低模型复杂度
    2. 通过2x2平均池化减半特征图尺寸
    
    Args:
        input_channels: 输入通道数
        num_channels: 输出通道数 (通常是输入的一半)
    
    Returns:
        包含BN、ReLU、1x1卷积和平均池化的序列模块
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),  # 1x1卷积降维
        nn.AvgPool2d(kernel_size=2, stride=2))  # 2x2池化, 尺寸减半

# ========== DenseNet-121 网络结构构建 ==========

# 第一个模块: 初始特征提取 (类似ResNet的stem)
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 7x7卷积, 步幅2, 输出64通道
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 3x3最大池化, 步幅2

# 构建4个稠密块和3个过渡层 (DenseNet-121配置)
num_channels, growth_rate = 64, 32  # 当前通道数和增长率 (每层新增32个通道)
num_convs_in_dense_blocks = [4, 4, 4, 4]  # 每个稠密块包含的卷积层数: 4+4+4+4=16层
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    # 添加稠密块
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数 = 输入通道数 + 层数 × 增长率
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层, 使通道数量减半 (最后一个稠密块后不加)
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2  # 更新通道数为减半后的值

# 完整的DenseNet网络
net = nn.Sequential(
    b1, *blks,  # 初始模块 + 4个稠密块 + 3个过渡层
    nn.BatchNorm2d(num_channels), nn.ReLU(),  # 最后的归一化和激活
    nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化, 输出1x1
    nn.Flatten(),  # 展平为一维向量
    nn.Linear(num_channels, 10))  # 全连接层, 输出10类 (Fashion-MNIST)

# ========== 训练配置和执行 ==========
lr, num_epochs, batch_size = 0.1, 10, 256  # 学习率0.1, 训练10轮, 批量大小256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)  # 加载Fashion-MNIST, 调整为96x96
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())  # 在GPU上训练
d2l.plt.show()  # 显示训练过程的损失和准确率曲线