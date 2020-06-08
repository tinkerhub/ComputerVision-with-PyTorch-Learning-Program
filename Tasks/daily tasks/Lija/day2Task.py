

#Create two numpy arrays of size 5x3 and 3x4.
#Convert them into torch tensors.
#Multiply the two tensors and print the result.

import torch
import numpy as np

array1=np.random.rand(5,3)
array2=np.random.rand(3,4)

tensor1=torch.from_numpy(array1)
tensor2=torch.from_numpy(array2)
z=tensor1@tensor2














