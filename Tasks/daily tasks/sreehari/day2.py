import torch
import numpy as np
a=np.ones((5,3))
b=np.ones((3,4))
TTa=torch.from_numpy(a)
TTb=torch.from_numpy(b)
print (torch.mm(TTa,TTb))