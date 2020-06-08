import numpy as np
import torch

arr1=np.random.rand(5,3)
arr2=np.random.rand(3,4)

t1=torch.from_numpy(arr1)
t2=torch.from_numpy(arr2)

print(torch.matmul(t1,t2))

