import numpy as np
import torch

#creating numpy arrays
x = np.zeros([5, 3])
y = np.ones([3, 4])

#converting to tensors
x_ = torch.from_numpy(x)
y = torch.from_numpy(y)

#Multiplying the tensors
z = torch.matmul(x_, y_)
print(z)
