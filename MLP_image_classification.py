import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) # 使用Fashion-MNIST数据集

num_inputs, num_outputs, num_hiddens = 784, 10, 256 # 输入层、输出层和隐藏层的神经元数量
# 初始化第一层权重
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
# 初始化第一层偏置
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# 初始化第二层权重
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
# 初始化第二层偏置
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

# 定义多层感知机模型
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
# 定义前向传播
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

params = [W1, b1, W2, b2] # 将所有参数放入列表中

loss = nn.CrossEntropyLoss(reduction='none')
# 训练模型
num_epochs, lr = 10, 0.1
# 使用随机梯度下降优化器
updater = torch.optim.SGD(params, lr=lr)
# 开始训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.plt.show()
# 预测结果
d2l.predict_ch3(net, test_iter)
d2l.plt.show()