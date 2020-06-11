import numpy as np
import torch

a = np.random.randint(10, size=(5, 3))
b = np.random.randint(10, size=(3, 4))
a_t = torch.from_numpy(a)
b_t = torch.from_numpy(b)

c_t = torch.matmul(a_t, b_t)
print(c_t)
