import numpy as np
import torch


a = np.empty((5, 3))
b = np.empty((3, 4))
a = torch.from_numpy(a)
b = torch.from_numpy(b)
c = torch.mm(a, b)
print(c)
