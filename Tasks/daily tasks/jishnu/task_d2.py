import numpy as np
import torch

np.random.seed(0)

array_a = np.random.randn(5,3)
array_b = np.random.randn(3,4)

tensor_a = torch.tensor(array_a)
tensor_b = torch.tensor(array_b)

print(tensor_a.matmul(tensor_b))