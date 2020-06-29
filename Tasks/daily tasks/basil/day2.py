import torch
import numpy as np
a=np.random.rand(5,3)
b=np.random.rand(3,4)
tensor_a=torch.from_numpy(a)
tensor_b=torch.from_numpy(b)
print(torch.mm(tensor_a,tensor_b))
