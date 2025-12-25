"""循环神经网络 (RNN) 从零实现

本文件实现了一个基础的 RNN 模型，用于字符级语言建模任务。
主要内容：
1. RNN 参数初始化和前向传播
2. 从零构建 RNN 模型类
3. 文本预测与生成
4. 梯度裁剪防止梯度爆炸
5. 完整的训练流程

数据集：《时间机器》(The Time Machine) 文本数据
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
# 从自定义工具模块导入数据加载函数
from d2l_utils import load_data_time_machine

# ==================== 数据准备 ====================

# 批量大小：每次训练使用 32 个序列
# 时间步数：每个序列包含 35 个字符
batch_size, num_steps = 32, 35

# 加载时间机器数据集
# train_iter: 数据迭代器，每次返回一批 (X, Y) 序列对
# vocab: 词表对象，包含字符到索引的映射
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# ==================== 独热编码示例 ====================

# 演示：将索引 [0, 2] 转换为独热向量
# 结果形状：(2, vocab_size)，每行是一个独热向量
F.one_hot(torch.tensor([0, 2]), len(vocab))

# 演示：批量独热编码
# X 形状：(2, 5) 表示 2 个样本，每个样本 5 个时间步
X = torch.arange(10).reshape((2, 5))
# 转置后形状：(5, 2)，然后独热编码成 (5, 2, 28)
F.one_hot(X.T, 28).shape

# ==================== RNN 参数初始化 ====================

def get_params(vocab_size, num_hiddens, device):
    """初始化 RNN 模型参数
    
    RNN 的核心公式:H_t = tanh(X_t @ W_xh + H_{t-1} @ W_hh + b_h)
                   O_t = H_t @ W_hq + b_q
    
    Args:
        vocab_size: 词表大小（输入和输出的特征维度）
        num_hiddens: 隐藏单元数量
        device: 计算设备 (CPU 或 GPU)
        
    Returns:
        list: [W_xh, W_hh, b_h, W_hq, b_q] 所有模型参数
    """
    # 输入和输出维度都等于词表大小（字符级模型）
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        """生成正态分布的小随机数，标准差 0.01"""
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))    # 输入到隐藏：(vocab_size, num_hiddens)
    W_hh = normal((num_hiddens, num_hiddens))   # 隐藏到隐藏：(num_hiddens, num_hiddens)
    b_h = torch.zeros(num_hiddens, device=device)  # 隐藏层偏置：(num_hiddens,)
    
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))   # 隐藏到输出：(num_hiddens, vocab_size)
    b_q = torch.zeros(num_outputs, device=device)  # 输出层偏置：(vocab_size,)
    
    # 将所有参数收集到列表中
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    
    # 为所有参数附加梯度（启用自动微分）
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    """初始化 RNN 隐藏状态
    
    在序列开始时，需要一个初始隐藏状态 H_0。
    这里简单地初始化为全零向量。
    
    Args:
        batch_size: 批量大小
        num_hiddens: 隐藏单元数量
        device: 计算设备
        
    Returns:
        tuple: 包含一个形状为 (batch_size, num_hiddens) 的零张量
               返回元组是为了与 LSTM/GRU 等多状态模型保持接口一致
    """
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    """RNN 前向传播函数
    
    按时间步展开 RNN, 计算所有时间步的输出。
    核心计算：
        H_t = tanh(X_t @ W_xh + H_{t-1} @ W_hh + b_h)  # 更新隐藏状态
        Y_t = H_t @ W_hq + b_q                         # 计算输出
    
    Args:
        inputs: 输入序列，形状 (时间步数, 批量大小, 词表大小)
        state: 初始隐藏状态，元组包含 H_0
        params: 模型参数 [W_xh, W_hh, b_h, W_hq, b_q]
        
    Returns:
        tuple: (outputs, new_state)
            - outputs: 所有时间步的输出，形状 (时间步数 * 批量大小, 词表大小)
            - new_state: 最终隐藏状态 (H_T,)
    """
    # 解包参数
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state  # 从元组中提取隐藏状态
    outputs = []
    
    # 遍历每个时间步
    # X 的形状：(批量大小, 词表大小) - 当前时间步的输入（已独热编码）
    for X in inputs:
        # 计算新的隐藏状态：H_t = tanh(X_t W_xh + H_{t-1} W_hh + b_h)
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # 计算输出：Y_t = H_t W_hq + b_q
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    
    # 将所有时间步的输出拼接成一个张量
    # 拼接后形状：(时间步数 * 批量大小, 词表大小)
    return torch.cat(outputs, dim=0), (H,)

# ==================== RNN 模型类 ====================

class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型
    
    这是一个通用的 RNN 模型封装类，支持不同的 RNN 变体（vanilla RNN, LSTM, GRU）。
    通过传入不同的 forward_fn 和 init_state 函数来实现不同类型的 RNN。
    """
    
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        """初始化 RNN 模型
        
        Args:
            vocab_size: 词表大小
            num_hiddens: 隐藏单元数量
            device: 计算设备
            get_params: 参数初始化函数
            init_state: 状态初始化函数
            forward_fn: 前向传播函数（rnn/lstm/gru）
        """
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        # 初始化模型参数
        self.params = get_params(vocab_size, num_hiddens, device)
        # 保存初始化和前向传播函数的引用
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """前向传播（使模型可以像函数一样调用）
        
        Args:
            X: 输入索引，形状 (批量大小, 时间步数)
            state: 隐藏状态
            
        Returns:
            tuple: (输出, 新状态)
        """
        # 将输入索引转换为独热编码
        # X.T: (时间步数, 批量大小)
        # one_hot 后: (时间步数, 批量大小, 词表大小)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """返回初始隐藏状态
        
        Args:
            batch_size: 批量大小
            device: 计算设备
            
        Returns:
            初始隐藏状态（通常是零向量）
        """
        return self.init_state(batch_size, self.num_hiddens, device)
    
# ==================== 模型实例化与测试 ====================

# 设置隐藏单元数量为 512
num_hiddens = 512

# 创建 RNN 模型实例
# - len(vocab): 词表大小作为输入/输出维度
# - num_hiddens: 隐藏层维度
# - d2l.try_gpu(): 尝试使用 GPU，如果不可用则使用 CPU
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)

# 初始化隐藏状态
state = net.begin_state(X.shape[0], d2l.try_gpu())

# 前向传播测试
# 输入 X 形状：(批量大小, 时间步数)
# 输出 Y 形状：(批量大小 * 时间步数, 词表大小)
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape

# ==================== 文本预测与生成 ====================

def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """基于前缀生成新字符（文本生成函数）
    
    使用训练好的 RNN 模型进行文本生成。
    过程分为两个阶段：
    1. 预热期：用 prefix 中的字符更新隐藏状态
    2. 生成期：根据当前状态预测下一个字符，并继续生成
    
    Args:
        prefix: 前缀字符串，用于"预热"模型
        num_preds: 要生成的新字符数量
        net: RNN 模型
        vocab: 词表对象
        device: 计算设备
        
    Returns:
        str: 包含前缀和生成字符的完整字符串
    """
    # 初始化隐藏状态（批量大小为 1，因为只生成一个序列）
    state = net.begin_state(batch_size=1, device=device)
    
    # outputs 存储生成的字符索引，从前缀的第一个字符开始
    outputs = [vocab[prefix[0]]]
    
    # 辅助函数：将最后一个输出索引转换为模型输入格式
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    
    # 预热期：用前缀中的字符逐步更新隐藏状态
    # 这样模型能够"理解"前缀的上下文信息
    for y in prefix[1:]:  
        _, state = net(get_input(), state)  # 更新状态但不使用输出
        outputs.append(vocab[y])  # 将前缀字符添加到输出
    
    # 生成期：预测接下来的 num_preds 个字符
    for _ in range(num_preds):  
        y, state = net(get_input(), state)  # 前向传播获得预测
        # 选择概率最高的字符（贪心解码）
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    
    # 将索引序列转换回字符串
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# 测试预测函数（模型未训练，输出是随机的）
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())

# ==================== 梯度裁剪 ====================

def grad_clipping(net, theta):  #@save
    """梯度裁剪（解决 RNN 训练中的梯度爆炸问题）
    
    RNN 在训练过程中容易出现梯度爆炸（梯度值变得非常大）。
    梯度裁剪通过限制梯度的 L2 范数来缓解这个问题。
    
    原理：
        如果 ||g|| > θ，则 g = θ * g / ||g||
        即：将梯度缩放到最大范数为 θ
    
    Args:
        net: 神经网络模型
        theta: 梯度裁剪阈值
    """
    # 获取所有需要梯度的参数
    if isinstance(net, nn.Module):
        # PyTorch 内置模型
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        # 自定义模型（如 RNNModelScratch）
        params = net.params
    
    # 计算所有参数梯度的 L2 范数：||g|| = sqrt(sum(g_i^2))
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    
    # 如果梯度范数超过阈值，进行裁剪
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm  # 缩放梯度

# ==================== 训练函数 ====================

#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期
    
    完成一个 epoch 的训练，处理所有批次的数据。
    
    Args:
        net: RNN 模型
        train_iter: 训练数据迭代器
        loss: 损失函数
        updater: 优化器或自定义更新函数
        device: 计算设备
        use_random_iter: 是否使用随机采样（影响隐藏状态的处理）
        
    Returns:
        tuple: (困惑度, 训练速度)
            - 困惑度 = exp(平均损失)，衡量模型预测的不确定性
            - 训练速度：每秒处理的词元数量
    """
    state, timer = None, d2l.Timer()  # 初始化状态和计时器
    metric = d2l.Accumulator(2)  # 累加器：[总损失, 词元总数]
    
    # 遍历所有批次
    for X, Y in train_iter:
        # 隐藏状态的初始化和分离
        if state is None or use_random_iter:
            # 情况1：第一次迭代 或 使用随机采样
            # 随机采样时每个批次是独立的，需要重新初始化状态
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 情况2：使用顺序采样
            # 需要从计算图中分离隐藏状态，避免反向传播到太久远的时间步
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # 针对 nn.GRU：state 是张量
                state.detach_()
            else:
                # 针对 nn.LSTM 或自定义模型：state 是元组
                for s in state:
                    s.detach_()
        
        # 准备标签：转置并展平
        # Y: (batch_size, num_steps) -> Y.T: (num_steps, batch_size) 
        # -> reshape(-1): (batch_size * num_steps,)
        y = Y.T.reshape(-1)
        
        # 将数据移到指定设备
        X, y = X.to(device), y.to(device)
        
        # 前向传播
        y_hat, state = net(X, state)
        
        # 计算损失（交叉熵）
        l = loss(y_hat, y.long()).mean()
        
        # 反向传播和参数更新
        if isinstance(updater, torch.optim.Optimizer):
            # 使用 PyTorch 优化器
            updater.zero_grad()      # 梯度清零
            l.backward()             # 反向传播
            grad_clipping(net, 1)    # 梯度裁剪
            updater.step()           # 更新参数
        else:
            # 使用自定义的 SGD 更新函数
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)    # batch_size=1 因为损失已经取了平均
        
        # 累加损失和词元数
        metric.add(l * y.numel(), y.numel())
    
    # 计算困惑度和训练速度
    # 困惑度 = exp(平均损失)，越低越好
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """RNN 模型完整训练流程
    
    训练 RNN 模型，并定期显示生成的文本和困惑度曲线。
    
    Args:
        net: RNN 模型
        train_iter: 训练数据迭代器
        vocab: 词表对象
        lr: 学习率
        num_epochs: 训练轮数
        device: 计算设备
        use_random_iter: 是否使用随机采样（默认 False，使用顺序采样）
    """
    # 定义交叉熵损失函数（用于分类任务）
    loss = nn.CrossEntropyLoss()
    
    # 创建动画器，用于实时绘制训练曲线
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    
    # 初始化优化器
    if isinstance(net, nn.Module):
        # PyTorch 内置模型：使用 SGD 优化器
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        # 自定义模型：使用自定义 SGD 函数
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    
    # 定义预测函数（生成 50 个字符）
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练一个 epoch
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        
        # 每 10 个 epoch 打印生成的文本并更新图表
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))  # 展示生成效果
            animator.add(epoch + 1, [ppl])    # 添加困惑度数据点
    
    # 训练结束，打印最终统计信息
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    # 展示最终的文本生成效果
    print(predict('time traveller'))
    print(predict('traveller'))

# ==================== 训练执行 ====================

# 设置超参数
num_epochs, lr = 500, 1  # 训练 500 轮，学习率为 1

# 实验 1：使用顺序采样训练
print("\n===== 实验 1: 顺序采样训练 =====")
print("顺序采样：相邻批次在原始序列中是连续的，保留时序依赖")
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())

# 实验 2：使用随机采样训练
print("\n===== 实验 2: 随机采样训练 =====")
print("随机采样：批次之间独立，每次随机选择起始位置")
# 重新初始化模型（避免使用已训练的参数）
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)  # 启用随机采样

# 显示所有图表
d2l.plt.show()

"""
训练说明：
1. 困惑度（Perplexity）：衡量模型的预测质量
   - 困惑度 = exp(交叉熵损失)
   - 值越低越好（1 是完美预测）
   - 相当于模型平均在多少个字符中"困惑"

2. 顺序采样 vs 随机采样：
   - 顺序采样：保留长期依赖，训练稳定，但计算成本高
   - 随机采样：打破依赖，训练更快，但可能损失长程信息

3. 预期效果：
   - 初期：生成随机字符
   - 中期：学会单词拼写和简单语法
   - 后期：生成类似原文风格的句子
"""

