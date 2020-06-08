import torch
import numpy as np

#Create two numpy arrays of size 5x3 and 3x4
a=np.random.rand(5,3)
b=np.random.rand(3,4)
#Convert them into torch tensors
a_t = torch.from_numpy(a)
b_t = torch.from_numpy(b)
#Multiply the two tensors and print the result
result=torch.matmul(a_t,b_t)
print(result)
