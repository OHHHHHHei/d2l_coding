import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    # 添加num_convs个卷积层，每个卷积层后接ReLU激活函数
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1)) # 保持高和宽不变
        layers.append(nn.ReLU()) # 激活函数
        in_channels = out_channels # 更新输入通道数为当前输出通道数
    # 添加一个最大池化层，池化窗口为2x2，步幅
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers) # 将层列表转换为一个顺序容器

# VGG网络的卷积层架构
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1 # 输入通道数为1（灰度图像）
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels # 当前块的输出通道 = 下一个块的输入通道

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)

# 创建一个随机输入样本：Batch=1, Channel=1, H=224, W=224
X = torch.randn(size=(1, 1, 224, 224))
# 逐层查看每个块的输出形状
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

ratio = 4
# 列表推导式：将所有输出通道数除以 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]

# 实例化缩小版网络
net = vgg(small_conv_arch)

# 训练超参数
lr, num_epochs, batch_size = 0.05, 10, 128

# 加载数据：注意 resize=224，强行把 Fashion-MNIST(28x28) 放大到 224x224
# 这是为了配合 VGG 的 5 次下采样，否则特征图早就变成 1x1 甚至报错了
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
# 训练模型
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

d2l.plt.show()