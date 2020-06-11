import numpy as np
import torch

x = np.random.rand(5,3)     #created numpy array of shape 5,3 with random values
y = np.random.rand(3,2)     #created numpy array of shape 3,2 with random values
a = torch.from_numpy(x)     #converted numpy array into torch tensor
b = torch.from_numpy(y)
z = torch.matmul(a,b)       #multiplied the 2 torch tensors
print(z)
