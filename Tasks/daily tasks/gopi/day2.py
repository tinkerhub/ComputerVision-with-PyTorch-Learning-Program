import numpy
import torch
x = numpy.ones([5, 3])
y = numpy.ones([3, 4])
x_ = torch.from_numpy(x)
y_ = torch.from_numpy(y)
k = torch.matmul(x_, y_)
