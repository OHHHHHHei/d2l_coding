import torch

def f_control(x):
    # 这个函数会根据 x 的值选择不同的计算路径
    if x > 1:
        y = x ** 3  # 路径 1: 立方
    else:
        y = x ** 2  # 路径 2: 平方
    return y

# --- 案例 1：x <= 1 (走平方路径) ---
x1 = torch.tensor(0.5, requires_grad=True)
y1 = f_control(x1)  # y1 = 0.5^2 = 0.25
y1.backward()       # 求导 L = y1

# --- 案例 2：x > 1 (走立方路径) ---
x2 = torch.tensor(3.0, requires_grad=True)
y2 = f_control(x2)  # y2 = 3.0^3 = 27.0
y2.backward()       # 求导 L = y2

print(f"案例 1 (x=0.5, y=x^2): y={y1.item()}, grad={x1.grad.item()}")
print(f"案例 2 (x=3.0, y=x^3): y={y2.item()}, grad={x2.grad.item()}")