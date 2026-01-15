# 导入必要的库
import torch
from torch import nn
from d2l import torch as d2l
from d2l_utils import load_data_time_machine, RNNModelScratch, train_ch8

# 设置批量大小和时间步数
# batch_size: 每个批次的样本数量
# num_steps: 每个序列的时间步长度
batch_size, num_steps = 32, 35
# 加载时间机器数据集，返回数据迭代器和词汇表
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

def get_lstm_params(vocab_size, num_hiddens, device):
    """
    初始化LSTM模型的所有参数
    
    参数:
        vocab_size: 词汇表大小
        num_hiddens: 隐藏单元数量
        device: 计算设备(CPU或GPU)
    
    返回:
        包含所有LSTM参数的列表
    """
    # 输入和输出的特征维度都等于词汇表大小(one-hot编码)
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        """生成服从正态分布的随机张量，标准差为0.01"""
        return torch.randn(size=shape, device=device)*0.01

    def three():
        """生成一组门控单元的参数：输入权重、隐藏状态权重、偏置"""
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    # 为所有参数启用梯度计算
    for param in params:
        param.requires_grad_(True)
    return params

def init_lstm_state(batch_size, num_hiddens, device):
    """
    初始化LSTM的隐藏状态
    
    参数:
        batch_size: 批量大小
        num_hiddens: 隐藏单元数量
        device: 计算设备(CPU或GPU)
    
    返回:
        (H, C)元组，分别是隐藏状态和记忆单元状态，初始值都为零
    """
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    """
    LSTM前向传播函数
    
    参数:
        inputs: 输入序列，形状为(时间步数, 批量大小, 词汇表大小)
        state: 初始状态(H, C)，H是隐藏状态，C是记忆单元状态
        params: 模型参数列表
    
    返回:
        outputs: 所有时间步的输出
        (H, C): 最终的隐藏状态和记忆单元状态
    """
    # 解包所有参数
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    # 解包初始状态
    (H, C) = state
    outputs = []
    # 遍历每个时间步
    for X in inputs:
        # 输入门：控制新信息的输入量
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        # 遗忘门：控制遗忘旧记忆的程度
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        # 输出门：控制输出信息的量
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        # 候选记忆单元：新的候选信息
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        # 更新记忆单元：遗忘旧记忆 + 添加新记忆
        C = F * C + I * C_tilda
        # 更新隐藏状态：基于当前记忆单元和输出门
        H = O * torch.tanh(C)
        # 计算当前时间步的输出
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    # 连接所有时间步的输出，返回输出和最终状态
    return torch.cat(outputs, dim=0), (H, C)

# 设置模型超参数
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
# 训练轮数和学习率
num_epochs, lr = 500, 1
# 创建从零开始实现的LSTM模型
# 参数：词汇表大小、隐藏单元数、设备、参数初始化函数、状态初始化函数、前向传播函数
model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
# 训练模型
train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# 显示训练过程的可视化结果
d2l.plt.show()