import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(3, 6)
        self.h2 = nn.Linear(6, 6)
        self.op = nn.Linear(6, 1)
        self.sigmoid = nn.Sigmoid()
    
  
    def forward(self, x):
        x = self.sigmoid(self.h1(x))
        x = self.sigmoid(self.h2(x))
        x = self.op(x)
        return x
     

ip = torch.randn(4, 3)
op = torch.randn(4, 1)

model = Net()
epoch = 10000
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for i in range(epoch):
    out = model(ip)
    loss = criterion(out, op)
    model.zero_grad()
    loss.backward()
    optimizer.step()
  
print("Loss : ", loss)
print("Prediction : ", out)
