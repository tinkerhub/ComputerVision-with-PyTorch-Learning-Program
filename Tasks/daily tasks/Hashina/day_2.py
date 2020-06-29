import torch
import numpy as np
x = np.random.rand(5, 3)
y = np.random.rand(3, 4)
xt = torch.from_numpy(x)
yt = torch.from_numpy(y)
mul = torch.matmul(xt, yt)
print(mul)
