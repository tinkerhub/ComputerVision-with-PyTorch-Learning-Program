# Done on day 2

import torch
import numpy as np

arr1 = np.random.randn(5, 3)
arr2 = np.random.randn(3, 4)

arr1_t = torch.tensor(arr1)
arr2_t = torch.tensor(arr2)

print(torch.matmul(arr1_t, arr2_t))