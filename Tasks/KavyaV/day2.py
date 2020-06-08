import torch
import numpy as np

np_arr1 = np.arange(15).reshape(5,3)
np_arr2 = np.arange(12).reshape(3,4)
tensor1 = torch.from_numpy(np_arr1)
tensor2 = torch.from_numpy(np_arr2)
print(torch.matmul(tensor1,tensor2))
