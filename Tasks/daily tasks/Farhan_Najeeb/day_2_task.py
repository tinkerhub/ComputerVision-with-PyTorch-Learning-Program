import torch
import numpy as np


x = np.ones((5, 3))
print(x)
y = np.random.randn(3, 4)
print(y)
x = torch.from_numpy(x)
y = torch.from_numpy(y)
print(torch.mm(x, y))
