"""序列模型 - 时间序列预测

演示如何使用简单的MLP进行时间序列预测:
1. 单步预测 (one-step prediction): 使用真实历史数据预测下一个时刻
2. 多步预测 (multistep prediction): 使用自己的预测结果作为输入继续预测
3. k步预测: 展示不同预测步长的效果
"""
import torch
from torch import nn
from d2l import torch as d2l

# ========== 生成时间序列数据 ==========
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)  # 时间步: 1到1000
# 生成带噪声的正弦波: sin(0.01*t) + 噪声, 噪声服从N(0, 0.04)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
# 绘制原始时间序列数据
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

# ========== 构造训练数据: 使用过去tau个时间步预测下一个时间步 ==========
tau = 4  # 时间窗口大小, 使用过去4个观测值来预测未来
# 构造特征矩阵: shape (T-tau, tau)
# 每一行是一个样本, 包含连续tau个时间步的观测值
features = torch.zeros((T - tau, tau))
for i in range(tau):
    # 第i列: 从时间步i开始到T-tau+i的所有观测值
    features[:, i] = x[i: T - tau + i]
# 标签: 对应特征的下一个时间步的值
# features[i] = [x[i], x[i+1], x[i+2], x[i+3]], labels[i] = x[i+4]
labels = x[tau:].reshape((-1, 1))

# ========== 准备训练数据加载器 ==========
batch_size, n_train = 16, 600  # 批量大小16, 使用前600个样本训练, 剩余样本用于测试
# 只有前n_train个样本用于训练, 后面的样本用于测试模型的泛化能力
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

# ========== 定义模型和训练函数 ==========

# 初始化网络权重的函数
def init_weights(m):
    """使用Xavier均匀分布初始化线性层权重"""
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机 (MLP)
def get_net():
    """构建一个两层MLP: 输入4维 -> 隐藏层10维 -> 输出1维
    
    网络结构:
    - 输入层: 4个特征 (过去4个时间步的观测值)
    - 隐藏层: 10个神经元, ReLU激活
    - 输出层: 1个神经元 (预测下一个时间步的值)
    """
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)  # 应用Xavier初始化
    return net

# 平方损失。注意: MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')  # reduction='none'返回每个样本的损失

def train(net, train_iter, loss, epochs, lr):
    """训练神经网络
    
    Args:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        loss: 损失函数
        epochs: 训练轮数
        lr: 学习率
    """
    trainer = torch.optim.Adam(net.parameters(), lr)  # 使用Adam优化器
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()  # 梯度清零
            l = loss(net(X), y)  # 计算损失
            l.sum().backward()  # 反向传播
            trainer.step()  # 更新参数
        # 每轮结束后打印训练损失
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

# ========== 训练模型 ==========
net = get_net()  # 创建网络
train(net, train_iter, loss, 5, 0.01)  # 训练5轮, 学习率0.01

# ========== 单步预测 (One-step Prediction) ==========
# 使用真实的历史数据作为输入, 预测下一个时间步
# 这是理想情况, 因为每次预测都有准确的历史数据
onestep_preds = net(features)  # 对所有样本进行单步预测
# 绘制真实数据和单步预测结果的对比
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))

# ========== 多步预测 (Multistep Prediction) ==========
# 使用模型自己的预测结果作为输入继续预测, 更接近实际应用场景
# 预测误差会累积, 导致长期预测效果较差
multistep_preds = torch.zeros(T)  # 初始化预测结果
multistep_preds[: n_train + tau] = x[: n_train + tau]  # 使用真实的训练数据作为初始值
# 从训练集结束后开始进行迭代预测
for i in range(n_train + tau, T):
    # 使用前tau个预测值 (可能包含之前预测的值) 来预测当前时刻
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

# 绘制真实数据、单步预测和多步预测的对比
# 可以看到多步预测在测试集上的误差明显大于单步预测
d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))

# ========== k步预测 (k-step Ahead Prediction) ==========
# 展示不同预测步长 (1步、4步、16步、64步) 的预测效果
max_steps = 64  # 最大预测步数

# 构造包含多步预测的特征矩阵
# shape: (T - tau - max_steps + 1, tau + max_steps)
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))

# 前tau列: 真实观测值, 作为预测的起点
# 列i (i<tau) 是来自x的观测, 其时间步从 (i) 到 (i+T-tau-max_steps+1)
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 后max_steps列: 从1步到max_steps步的预测结果
# 列i (i>=tau) 是来自 (i-tau+1) 步的预测, 其时间步从 (i) 到 (i+T-tau-max_steps+1)
for i in range(tau, tau + max_steps):
    # 使用前tau列 (包含真实观测和之前的预测) 来预测当前列
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

# 绘制不同步长的预测结果
# 可以看到: 预测步长越大, 预测误差越大, 曲线越平滑 (失去细节)
steps = (1, 4, 16, 64)  # 选择1步、4步、16步、64步预测进行对比
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
d2l.plt.show()  # 显示所有图表