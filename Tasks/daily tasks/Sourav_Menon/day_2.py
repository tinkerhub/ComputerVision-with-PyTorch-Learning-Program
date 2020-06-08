import torch
import numpy as np

#creating random numpy array of(5,3)
array_1=np.random.rand(5,3)

#creating random array numpy array of (3,4)
array_2=np.random.rand(3,4)

#converting arrays into tensors
arra1_1=torch.from_numpy(array_1)
array_2=torch.from_numpy(array_2)

#multiplying the both tensors... and printing the product

print(torch.matmul(array_1,array_2))


