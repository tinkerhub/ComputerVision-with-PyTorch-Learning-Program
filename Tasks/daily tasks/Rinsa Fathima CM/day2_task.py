import numpy as np
import torch

#creating numpy arrays
a=np.random.randint(15,size=(5,3))
b=np.random.randint(5,size=(3,4))

#converting numpy arrays to torch tensors
c=torch.from_numpy(a)
d=torch.from_numpy(b)

#multiplying torch tensors
product=torch.mm(c,d)
print(product)
