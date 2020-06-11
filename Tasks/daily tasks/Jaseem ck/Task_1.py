import torch as th
import numpy as np

#numpy of size 5*3
a = np.arange(15).reshape(5,3)
#numpy of size 3*4
b = np.arange(12).reshape(3,4)

#Converting numpy arrays into tensors
a_torch = th.from_numpy(a)
b_torch = th.from_numpy(b)

#Multiply the tensors to generate the product
product = th.mm(a_torch, b_torch)

print(product)
