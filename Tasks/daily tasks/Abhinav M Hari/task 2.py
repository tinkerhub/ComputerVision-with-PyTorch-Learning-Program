import torch
import torch.nn as nn

class Net(nn.Module):
  def __init__(self):
  super(Net, self). __init__()
  self.layer1 = nn.Linear(120, 84)
  self.layer2 = nn.Linear(84, 10)
  
  def forward(self, x):
    x = self.layer1(x)
    x = torch.sigmoid(self.layer2(x))

net = Net()
input = torch.randn(120)
output = net(input)
print(output)
  
