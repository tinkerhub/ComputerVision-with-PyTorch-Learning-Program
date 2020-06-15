from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

import numpy as np
x = np.random.random((5,3))
y = np.random.random((3,4))
a= torch.from_numpy(x)
b = torch.from_numpy(y)

t=torch.mm(a,b)

print(t)
