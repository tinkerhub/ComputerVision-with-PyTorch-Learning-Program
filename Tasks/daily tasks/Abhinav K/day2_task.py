import numpy as np
import torch

a=np.random.rand(5,3)
b=np.random.rand(3,4)
a2=torch.from_numpy(a)
b2=torch.from_numpy(b)

c=torch.mm(a2,b2)
print(c)
