import torch
import numpy as np

a_numpy = np.random.randn(5, 3)
b_numpy = np.ones((3, 4), dtype=float)

print(a_numpy)
print(b_numpy)

a_torch = torch.from_numpy(a_numpy)
b_torch = torch.from_numpy(b_numpy)

print(a_torch)
print(b_torch)

mul = torch.mm(a_torch, b_torch)
print(mul)
