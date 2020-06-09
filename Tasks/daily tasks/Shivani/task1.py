import numpy as np
import torch
x = np.random.rand(5,3)
print(x)
y = np.random.rand(3,4)
print(y)
xt = torch.from_numpy(x)
print(xt)
yt = torch.from_numpy(y)
print(yt)
result = torch.mm(xt,yt)
print(result)
