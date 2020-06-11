import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Linear(784, 392)
        self.hidden1 = nn.Linear(392, 98)
        self.sigmoid = nn.Sigmoid()
        self.hidden2 = nn.Linear(98, 14)
        self.output = nn.Linear(14, 3)

    def forward(self, x):
        x = self.input(x)
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = Net()
print(model)
