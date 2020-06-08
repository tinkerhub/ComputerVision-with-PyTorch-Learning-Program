from __future__ import print_function
import torch
import numpy as np
a = np.random.rand(5,3)
b = np.random.rand(3,4)
c = torch.from_numpy(a)
d = torch.from_numpy(b)
print(c)
print(d)
e = torch.mm(c,d)
print(e)
