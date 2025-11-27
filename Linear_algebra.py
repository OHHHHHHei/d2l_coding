import torch

x = torch.arange(24, dtype=torch.float).reshape(2, 3, 4)
print(x)
norm = torch.linalg.norm(x)
print(norm)