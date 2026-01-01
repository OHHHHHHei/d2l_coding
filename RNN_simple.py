"""
循环神经网络 (RNN) - 使用PyTorch内置层实现
=====================================
本文件演示如何使用PyTorch的内置RNN层构建语言模型。
主要内容：
1. 使用 nn.RNN 创建循环层
2. 使用 RNNModel 封装完整模型
3. 训练字符级语言模型
4. 文本生成

数据集：《时间机器》(The Time Machine) 文本数据
"""

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
# 从工具模块导入数据加载、模型类和训练函数
from d2l_utils import load_data_time_machine, RNNModel, predict_ch8, train_ch8

# ==================== 数据准备 ====================

# 批量大小：每批处理 32 个序列
# 时间步数：每个序列包含 35 个字符
batch_size, num_steps = 32, 35

# 加载时间机器数据集
# train_iter: 数据迭代器，每次返回 (X, Y) 批次
#   - X 形状: (batch_size, num_steps) 输入序列
#   - Y 形状: (batch_size, num_steps) 目标序列（X 向后移动一位）
# vocab: 词汇表对象，提供字符到索引的映射
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# ==================== 创建RNN层 ====================

# 隐藏单元数量：控制RNN的表达能力
num_hiddens = 256

# 创建单层RNN
# 参数：
#   - len(vocab): 输入特征维度（词汇表大小，用于one-hot编码）
#   - num_hiddens: 隐藏单元数量
# 默认配置：
#   - num_layers=1: 单层RNN
#   - nonlinearity='tanh': 使用tanh激活函数
#   - batch_first=False: 输入形状为 (seq_len, batch, input_size)
rnn_layer = nn.RNN(len(vocab), num_hiddens)

# ==================== 构建完整RNN模型 ====================

# RNNModel类已移动到d2l_utils.py中
# 该类在RNN层基础上添加：
# 1. 输入的one-hot编码转换
# 2. 输出的全连接层（将隐藏状态映射到词汇表大小）
# 3. 状态初始化方法

# 选择计算设备（GPU优先，否则CPU）
device = d2l.try_gpu()

# 创建RNN模型实例
# 参数：
#   - rnn_layer: 之前创建的nn.RNN层
#   - vocab_size: 词汇表大小（输出维度）
net = RNNModel(rnn_layer, vocab_size=len(vocab))

# 将模型移动到指定设备（GPU或CPU）
net = net.to(device)

# ==================== 测试文本生成 ====================

# 使用未训练的模型生成文本（预期输出为随机字符）
# 参数：
#   - 'time traveller': 前缀字符串，用于初始化隐藏状态
#   - 10: 生成10个新字符
#   - net: RNN模型
#   - vocab: 词汇表
#   - device: 计算设备
predict_ch8('time traveller', 10, net, vocab, device)

# ==================== 训练模型 ====================

# 训练超参数
num_epochs, lr = 500, 1  # 训练500轮，学习率为1

# 开始训练
# train_ch8 函数会：
# 1. 每10个epoch打印生成的文本样本
# 2. 绘制困惑度（perplexity）曲线
# 3. 在训练结束时显示最终生成效果
# 
# 困惑度说明：
#   - 困惑度 = exp(交叉熵损失)
#   - 越低越好（接近1表示模型非常确定）
#   - 可以理解为模型在预测时"困惑"于多少个选项
train_ch8(net, train_iter, vocab, lr, num_epochs, device)

# 显示训练过程中生成的图表
d2l.plt.show()

"""
训练效果说明：
1. 初期（epoch 1-50）：
   - 困惑度很高（>100）
   - 生成的文本是随机字符

2. 中期（epoch 50-200）：
   - 困惑度逐渐下降
   - 开始学会英文单词的拼写
   - 能生成简单的词组

3. 后期（epoch 200-500）：
   - 困惑度稳定在较低水平
   - 生成的句子具有语法结构
   - 风格接近原文《时间机器》

对比 RNN_from_scratch.py：
- 本文件使用 PyTorch 内置的 nn.RNN 层
- RNN_from_scratch.py 从零实现所有细节
- 两种方法的训练效果应该相似
"""