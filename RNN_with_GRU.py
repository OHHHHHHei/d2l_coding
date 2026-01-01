# 导入必要的库
import torch
from torch import nn
from d2l import torch as d2l
from d2l_utils import load_data_time_machine, RNNModelScratch, train_ch8

# 设置批量大小和时间步数
# batch_size: 每批处理的样本数量
# num_steps: 每个序列的时间步长度
batch_size, num_steps = 32, 35
# 加载时间机器数据集，获取训练迭代器和词汇表
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

def get_params(vocab_size, num_hiddens, device):
    """
    初始化 GRU 模型的参数
    
    GRU (门控循环单元) 包含三个门控机制：
    1. 更新门 (Update Gate): 控制前一时刻隐状态信息保留多少
    2. 重置门 (Reset Gate): 控制忽略多少前一时刻的隐状态
    3. 候选隐状态: 当前时刻的新信息
    
    参数:
        vocab_size: 词汇表大小（输入和输出的维度）
        num_hiddens: 隐藏层单元数量
        device: 运行设备（CPU 或 GPU）
    
    返回:
        params: 包含所有模型参数的列表
    """
    # 输入和输出的特征维度都等于词汇表大小（one-hot 编码）
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        """生成服从正态分布的随机权重矩阵，标准差为 0.01"""
        return torch.randn(size=shape, device=device)*0.01

    def three():
        """为每个门生成一组参数：输入权重、隐状态权重、偏置"""
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 更新门参数：决定保留多少旧信息
    # Z_t = sigmoid(X_t @ W_xz + H_{t-1} @ W_hz + b_z)
    W_xz, W_hz, b_z = three()  # 更新门参数
    
    # 重置门参数：决定丢弃多少旧信息
    # R_t = sigmoid(X_t @ W_xr + H_{t-1} @ W_hr + b_r)
    W_xr, W_hr, b_r = three()  # 重置门参数
    
    # 候选隐状态参数：计算当前时刻的候选信息
    # H_tilda_t = tanh(X_t @ W_xh + (R_t * H_{t-1}) @ W_hh + b_h)
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    
    # 输出层参数：将隐状态映射到输出空间
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    
    # 将所有参数收集到列表中
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    # 为所有参数附加梯度，使其可以进行反向传播训练
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    """
    初始化 GRU 的隐状态
    
    参数:
        batch_size: 批量大小
        num_hiddens: 隐藏单元数量
        device: 运行设备
    
    返回:
        包含初始隐状态的元组（全零矩阵）
        注意：返回元组是为了保持与其他 RNN 变体（如 LSTM）的接口一致
    """
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    """
    GRU 的前向传播函数
    
    GRU 工作流程：
    1. 计算更新门 Z：决定保留多少前一时刻的隐状态
    2. 计算重置门 R：决定在计算候选隐状态时使用多少前一时刻的信息
    3. 计算候选隐状态 H_tilda：融合当前输入和重置后的历史信息
    4. 更新隐状态 H：通过更新门在旧隐状态和候选隐状态之间插值
    5. 计算输出 Y：基于当前隐状态
    
    参数:
        inputs: 输入序列 (时间步数, 批量大小, 词汇表大小)
        state: 前一时刻的隐状态
        params: 模型参数列表
    
    返回:
        outputs: 所有时间步的输出拼接结果
        (H,): 最后一个时间步的隐状态（元组形式）
    """
    # 解包所有参数
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    # 从元组中提取隐状态
    H, = state
    outputs = []
    
    # 遍历每个时间步的输入
    for X in inputs:
        # 更新门：Z_t = sigmoid(X_t @ W_xz + H_{t-1} @ W_hz + b_z)
        # 值域 [0,1]，接近 1 表示保留更多旧信息，接近 0 表示使用更多新信息
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        
        # 重置门：R_t = sigmoid(X_t @ W_xr + H_{t-1} @ W_hr + b_r)
        # 值域 [0,1]，接近 0 表示忽略历史信息，接近 1 表示保留历史信息
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        
        # 候选隐状态：H_tilda_t = tanh(X_t @ W_xh + (R_t * H_{t-1}) @ W_hh + b_h)
        # 通过重置门控制使用多少前一时刻的隐状态信息
        # tanh 激活函数将值压缩到 [-1, 1] 范围
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        
        # 更新隐状态：H_t = Z_t * H_{t-1} + (1 - Z_t) * H_tilda_t
        # 这是更新门在旧隐状态和候选隐状态之间的加权平均
        # Z 接近 1：保留旧状态；Z 接近 0：使用新候选状态
        H = Z * H + (1 - Z) * H_tilda
        
        # 计算输出：Y_t = H_t @ W_hq + b_q
        Y = H @ W_hq + b_q
        outputs.append(Y)
    
    # 将所有时间步的输出在第 0 维拼接，返回最终的隐状态
    return torch.cat(outputs, dim=0), (H,)

# 设置模型超参数
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
# num_epochs: 训练轮数，lr: 学习率
num_epochs, lr = 500, 1

# 创建 GRU 模型
# 传入词汇表大小、隐藏单元数、设备、参数初始化函数、状态初始化函数、前向传播函数
model = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)

# 训练模型
# 使用第 8 章的训练函数，自动处理训练循环、困惑度计算和可视化
train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# 显示训练过程中的损失曲线图
d2l.plt.show()