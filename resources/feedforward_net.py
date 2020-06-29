import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/dummy_model')
x = torch.tensor([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]).float()
y = torch.tensor([[0], [1], [1], [0]]).float()
class Net(nn.Module):
    def __init__(self, inp, out):
        super(Net, self).__init__()
        self.input = nn.Linear(inp, 4)
        self.sigmoid = nn.Sigmoid()
        self.h1 = nn.Linear(4, 8)
        self.h2 = nn.Linear(8, 16)
        self.output = nn.Linear(16, out)

    def forward(self, x):
        x = self.input(x)
        x = self.sigmoid(self.h1(x))
        x = self.h2(x)
        x = self.output(x)
        return x


model = Net(inp=3, out=1)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
#print(model.forward(x))
#print(model.forward(torch.tensor([[0, 0, 1]]).float()))
model.zero_grad()
criterion = nn.MSELoss()
optimr = optim.SGD(model.parameters(), lr=0.001)
for i in range(60000):
    output = model(x)
    target_ = y
    loss = criterion(output, target_)
    print(loss)
    loss.backward()
    optimr.step()
    
print(model(x))
