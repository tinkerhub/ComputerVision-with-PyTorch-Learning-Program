import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inputSize = 2
        self.outputSize = 1
        self.input = nn.Linear(self.inputSize, 10)
        self.hidden = nn.Linear(10, 15)
        self.output = nn.Linear(15, self.outputSize)
    def forward(self, X):
        X = self.input(X)
        X = nn.Sigmoid(self.hidden(X))
        o = self.output(X)
        return o

net = Net()
print(net)
