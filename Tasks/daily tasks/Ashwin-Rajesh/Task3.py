import numpy

import torch
import torch.nn.functional as F
import torch.nn as nn

class neural_net(nn.Module):
    # Arguments : input size, output size, hidden layer sizes
    def __init__(self, s_in, s_out, h_1, h_2, h_3):
        super().__init__()
        self.in_layer = nn.Linear(s_in, h_1)
        self.h1_layer = nn.Linear(h_1, h_2)
        self.h2_layer = nn.Linear(h_2, h_3)
        self.out_layer = nn.Linear(h_3, s_out)

    def forward(x):
        x = F.sigmoid(self.in_layer(x))
        x = F.sigmoid(self.h1_layer(x))
        x = F.sigmoid(self.h2_layer(x))
        x = F.sigmoid(self.out_layer(x))
        return x

model = neural_net(10, 1, 100, 50, 10)
print(model)