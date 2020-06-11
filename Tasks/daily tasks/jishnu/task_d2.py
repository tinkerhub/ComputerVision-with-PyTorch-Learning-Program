import numpy as np
import torch

np.random.seed(0)

#creating two random numpy arrays
arr_a = np.random.randn(5,3)
arr_b = np.random.randn(3,4)

#converting numpy array to torch tensors
tensor_a = torch.tensor(arr_a)
tensor_b = torch.tensor(arr_b)

print(tensor_a.matmul(tensor_b))
