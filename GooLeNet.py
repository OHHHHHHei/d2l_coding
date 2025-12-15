"""GoogLeNet (Inception v1) 实现

GoogLeNet 在 2014 年 ILSVRC 竞赛中获得冠军，引入了 Inception 模块。
核心思想：在同一层使用不同大小的卷积核并行处理，然后合并结果。
这样可以捕获不同尺度的特征，同时通过 1x1 卷积降低计算复杂度。
"""

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Inception(nn.Module):
    """Inception 模块
    
    Inception 模块包含 4 条并行路径：
    1. 1x1 卷积：捕获点信息
    2. 1x1 卷积 → 3x3 卷积：捕获中等尺度特征
    3. 1x1 卷积 → 5x5 卷积：捕获大尺度特征
    4. 3x3 最大池化 → 1x1 卷积：保留重要信息
    
    1x1 卷积的作用：降维，减少参数量和计算量
    
    参数：
        in_channels: 输入通道数
        c1: 路径1的输出通道数
        c2: 路径2的输出通道数 (c2[0]=1x1卷积输出, c2[1]=3x3卷积输出)
        c3: 路径3的输出通道数 (c3[0]=1x1卷积输出, c3[1]=5x5卷积输出)
        c4: 路径4的输出通道数
    """
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        # 路径1：直接 1x1 卷积
        p1 = F.relu(self.p1_1(x))
        # 路径2：1x1 降维后接 3x3 卷积
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        # 路径3：1x1 降维后接 5x5 卷积
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        # 路径4：3x3 最大池化后接 1x1 卷积
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出（拼接 4 条路径的结果）
        return torch.cat((p1, p2, p3, p4), dim=1)
    
# 第一个模块 b1：初始特征提取
# 输入: 1x96x96 → 输出: 64x24x24
# 使用较大的 7x7 卷积核快速降低空间尺寸，提取低级特征
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 1x96x96 → 64x48x48
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 64x48x48 → 64x24x24

# 第二个模块 b2：进一步特征提取
# 输入: 64x24x24 → 输出: 192x12x12
# 1x1 卷积降维 + 3x3 卷积提取特征
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),  # 64x24x24 → 64x24x24 (降维)
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 64x24x24 → 192x24x24
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 192x24x24 → 192x12x12

# 第三个模块 b3：第一组 Inception 模块
# 输入: 192x12x12 → 输出: 480x6x6
# 两个 Inception 模块逐步增加通道数
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),  # 192 → 256 通道 (64+128+32+32)
                   Inception(256, 128, (128, 192), (32, 96), 64),  # 256 → 480 通道 (128+192+96+64)
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 480x12x12 → 480x6x6

# 第四个模块 b4：第二组 Inception 模块（网络主体）
# 输入: 480x6x6 → 输出: 832x3x3
# 5 个 Inception 模块，逐步提取更抽象的特征
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),  # 480 → 512 通道
                   Inception(512, 160, (112, 224), (24, 64), 64),  # 512 → 512 通道
                   Inception(512, 128, (128, 256), (24, 64), 64),  # 512 → 512 通道
                   Inception(512, 112, (144, 288), (32, 64), 64),  # 512 → 528 通道
                   Inception(528, 256, (160, 320), (32, 128), 128),  # 528 → 832 通道
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 832x6x6 → 832x3x3

# 第五个模块 b5：最后的 Inception 模块 + 全局平均池化
# 输入: 832x3x3 → 输出: 1024 (展平后的向量)
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),  # 832 → 832 通道
                   Inception(832, 384, (192, 384), (48, 128), 128),  # 832 → 1024 通道 (384+384+128+128)
                   nn.AdaptiveAvgPool2d((1,1)),  # 全局平均池化: 1024x3x3 → 1024x1x1
                   nn.Flatten())  # 展平: 1024x1x1 → 1024

# 组装完整的 GoogLeNet 网络
# 输入: 1x96x96 → 输出: 10 (10 个类别的概率)
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

# 测试网络：打印每一层的输出形状
X = torch.rand(size=(1, 1, 96, 96))  # 创建一个随机输入 (批量大小=1, 通道=1, 高=96, 宽=96)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# 训练超参数设置
lr, num_epochs, batch_size = 0.1, 10, 128

# 加载 Fashion-MNIST 数据集并调整图像大小为 96x96
# (原始 Fashion-MNIST 是 28x28，需要放大以适配 GoogLeNet)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

# 开始训练
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

# 显示训练过程的损失和准确率曲线
d2l.plt.show()