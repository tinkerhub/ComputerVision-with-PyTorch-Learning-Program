import torch.nn as nn
import torch.nn.functional as F
import torch

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.Output = nn.Linear(84, 10)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
    	x=self.conv1(x)
    	x = self.sigmoid(x)
    	x=self.conv2(x)
    	x = self.sigmoid(x)
    	x= self.fc1(x)
    	x = self.sigmoid(x)
    	x=self.fc2(x)
    	x = self.sigmoid(x)
    	x = self.Output(x)
    	return x


net = Network()
print(net)

