#Day2
# Playing around with tensors

# 1. Create two numpy arrays of size 5x3 and 3x4.
# 2. Convert them into torch tensors.
# 3. Multiply the two tensors and print the result.


import numpy as np
import torch as tp

a= np.random.rand(5, 3) 
b=np.random.rand(3, 4) 
a=tp.tensor(a)
b=tp.tensor(b) 
print(a.size())
print(b.size())
Multipli_tensor=tp.mm(a,b)
print("Multipli_tensor\n",Multipli_tensor)
