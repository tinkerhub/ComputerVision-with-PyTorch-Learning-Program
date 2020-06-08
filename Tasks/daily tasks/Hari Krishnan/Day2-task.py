import numpy as np
import torch

# Creating numpy array of the specified type
a = np.empty((5, 3))
b = np.empty((3, 4))

# Converting numpy array to Torch tensors
a = torch.from_numpy(a)
b = torch.from_numpy(b)

# Multiplying two tensors
c = torch.mm(a, b)

print(c)
