import torch
import numpy as np 

a =np.random.randint(5,size = (5,3))
b =np.random.randint(5,size = (3,4))
a_tensor = torch.from_numpy(a)
b_tensor = torch.from_numpy(b)
result = torch.matmul(a_tensor,b_tensor)
print(result)
