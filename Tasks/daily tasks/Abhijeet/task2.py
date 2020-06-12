import torch.nn as nn

import torch

class Network(nn.Module):
    def __init__(self,N_in,N_out):
        super(Network, self).__init__()
        self.input = nn.Linear(N_in, 500)
        self.fc1 = nn.Linear(500, 120)
        self.fc2 = nn.Linear(120, 84)
        self.Output = nn.Linear(84, N_out)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
    	x=self.input(x)
    	x= self.fc1(x)
    	x = self.sigmoid(x)
    	x=self.fc2(x)
    	x = self.Output(x)
    	return x


net = Network(600,10)
print(net)

