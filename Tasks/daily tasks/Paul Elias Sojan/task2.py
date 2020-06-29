import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input = nn.Linear(28*28,100)
        self.fc1 = nn.Linear(100, 200)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(200, 50)
        self.out = nn.Linear(50,10)

    def forward(self,x):
        x=self.input(x)
        x = self.sigmoid(self.fc1(x))
        x=self.sigmoid(self.fc2(x))
        x=self.out(x)

        
n = NeuralNetwork()
print(n)
