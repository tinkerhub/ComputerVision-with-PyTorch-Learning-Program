from __future__ import print_function
import numpy as np

import torch

p=np.random.random((5,3))
q=np.random.random((3,4))
x=torch.from_numpy(p)
y=torch.from_numpy(q)

product=torch.matmul(x,y)
print("The product of 5x3 and 3x4 tensor are  >>  \n ......\n",product)
