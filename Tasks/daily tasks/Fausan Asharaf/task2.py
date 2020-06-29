import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        self.input = nn.Linear(D_in, 500)
        self.sigmoid = nn.Sigmoid()
        self.h1 = nn.Linear(500, 200)
        self.h2 = nn.Linear(200, 100)
        self.output = nn.Linear(100, D_out)

    def forward(self, x):
        x = self.input(x)
        x = self.sigmoid(self.h1(x))
        x = self.h2(x)
        x = self.output(x)
        return x


model = Net(D_in=784, D_out=10)
print(model)
