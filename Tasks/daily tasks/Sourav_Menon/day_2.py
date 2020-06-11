import numpy
import torch

#creating numpy array of dim-(5,3)
array_1=numpy.random.rand(5,3)

#creating numpy array of dim-(3,4)
array_2=numpy.random.rand(3,4)

#converting arrays into torch tensor
a=torch.from_numpy(array_1)
b=torch.from_numpy(array_2)

#printing the prouduct of 2 torch tensors

print(torch.matmul(a,b))

