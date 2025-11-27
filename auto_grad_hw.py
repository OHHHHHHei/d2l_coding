import torch
import matplotlib.pyplot as plt
import numpy as np

# 1. 创建 x 轴上的点，并设置 requires_grad=True
# 我们创建从 0 到 2*pi 的 1000 个点
x = torch.linspace(0.0, 2 * np.pi, 1000, requires_grad=True)

# 2. 计算 f(x) = sin(x)
y = torch.sin(x)

# 3. 反向传播
# 因为 y 是一个向量，我们对 y 求和（sum）得到一个标量，然后再反向传播
y.sum().backward()

# 4. 此时 x.grad 中就存储了 y (即 sin(x)) 对 x 在每个点上的导数
# x.grad 的值理论上应该等于 cos(x)

# 5. 绘图
plt.figure(figsize=(8, 4))
# 绘制 f(x) = sin(x)
# .detach() 是为了将张量从计算图中分离出来，使其不带梯度
plt.plot(x.detach().numpy(), y.detach().numpy(), label='f(x) = sin(x)')

# 绘制 df(x)/dx，数据来源于 x.grad
# 我们没有手动计算 cos(x)，而是让 PyTorch 自动计算
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='df(x)/dx (autograd)')

plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('f(x) = sin(x) 及其自动微分的导数')
plt.show()