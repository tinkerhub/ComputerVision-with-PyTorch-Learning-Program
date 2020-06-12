import torch
import torch.nn as nn
class Net(nn.Module):

    def __init__(self,n_in,n_out):
        super(Net, self).__init__()
        self.in_layer   = nn.Linear(n_in, 512) 
        self.h1_layer   = nn.Linear(512, 256)
        self.h2_layer   = nn.Linear(256, 128)
        self.out_layer  = nn.Linear(128, n_out)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.in_layer(x)
        x = self.h1_layer(x)
        x = self.activation(x)
        x = self.h2_layer(x)
        x = self.out_layer(x)
        return x
net = Net(1024,64)
print(net)
