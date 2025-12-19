# ========================================
# ResNet (残差网络) 实现
# 核心思想：通过残差连接解决深层网络的梯度消失问题
# ========================================

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):  #@save
    """残差块 (Residual Block)
    
    实现公式:F(X) = ReLU(Conv2(ReLU(BN(Conv1(X)))) + X)
    其中 X 是输入, 如果输入输出通道数或尺寸不同, 则通过1x1卷积调整X
    
    参数:
        input_channels: 输入通道数
        num_channels: 输出通道数
        use_1x1conv: 是否使用1x1卷积调整输入X的维度
        strides: 卷积步幅，用于降采样
    """
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()

        # 第一个3x3卷积层，可能带步幅实现降采样
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        
        # 第二个3x3卷积层，保持尺寸不变
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        
        # 如果需要，使用1x1卷积调整输入X的通道数和尺寸，以便与F(X)相加
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        # 批量归一化层，用于加速训练和提高稳定性
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        """前向传播：实现残差连接"""

        # 主路径：Conv1 -> BN1 -> ReLU -> Conv2 -> BN2
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        # 跳跃连接：如果需要，调整输入X的维度
        if self.conv3:
            X = self.conv3(X)

        # 残差连接：将输入X与主路径输出Y相加
        Y += X

        # 最后应用ReLU激活函数
        return F.relu(Y)
    

# ========================================
# ResNet-18 网络结构定义
# ========================================

# 第一个模块（b1）：初始特征提取
# - 7x7大卷积核提取低层特征
# - 步幅为2，将输入从224x224降到112x112（若输入为96x96则变为48x48）
# - 最大池化进一步降采样到56x56（或24x24）
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    """构建ResNet的一个阶段, 包含多个残差块
    
    参数:
        input_channels: 输入通道数
        num_channels: 输出通道数
        num_residuals: 该阶段包含的残差块数量
        first_block: 是否为第一个残差阶段（第一个阶段不降采样）
    
    返回:
        包含num_residuals个残差块的列表
    """
    blk = []
    for i in range(num_residuals):
        # 每个阶段的第一个残差块（除了第一阶段）需要：
        # 1. 使用1x1卷积调整通道数
        # 2. 使用步幅2进行降采样（高宽减半）
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            # 其他残差块保持通道数和尺寸不变
            blk.append(Residual(num_channels, num_channels))
    return blk

# 第二个模块（b2）：第一个残差阶段
# - 输入输出都是64通道，不降采样（first_block=True）
# - 包含2个残差块
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))

# 第三个模块（b3）：第二个残差阶段
# - 通道数从64增加到128，第一个残差块将尺寸减半
# - 包含2个残差块
b3 = nn.Sequential(*resnet_block(64, 128, 2))

# 第四个模块（b4）：第三个残差阶段
# - 通道数从128增加到256，第一个残差块将尺寸减半
# - 包含2个残差块
b4 = nn.Sequential(*resnet_block(128, 256, 2))

# 第五个模块（b5）：第四个残差阶段
# - 通道数从256增加到512，第一个残差块将尺寸减半
# - 包含2个残差块
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# 完整的ResNet-18网络
# 结构：b1 -> b2 -> b3 -> b4 -> b5 -> 全局平均池化 -> 全连接层
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),  # 全局平均池化：将任意尺寸特征图变为1x1
                    nn.Flatten(),  # 展平：(batch_size, 512, 1, 1) -> (batch_size, 512)
                    nn.Linear(512, 10))  # 全连接层：输出10个类别的预测

# ========================================
# 训练配置和执行
# ========================================

# 超参数设置
lr, num_epochs, batch_size = 0.05, 10, 256  # 学习率、训练轮数、批量大小

# 加载Fashion-MNIST数据集
# resize=96：将图像从28x28调整到96x96以适配ResNet的深度结构
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

# 训练模型
# - 使用GPU加速（如果可用）
# - train_ch6会自动处理训练循环、损失计算、参数更新和可视化
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

# 显示训练过程中的损失和准确率曲线
d2l.plt.show()