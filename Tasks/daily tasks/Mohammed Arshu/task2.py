import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inputSize = 3
        self.outputSize = 1
        self.input = nn.Linear(self.inputSize, 10)
        self.sigmoid = nn.Sigmoid()
        self.hidden = nn.Linear(10, 15)
        self.output = nn.Linear(15, self.outputSize)
    def forward(self, X):
        X = self.sigmoid(self.input(X))
        X = self.sigmoid(self.hidden(X))
        X = self.sigmoid(self.output(X))
        return X

net = Net()
print(net)

X=torch.randn(3,3)
Y=torch.tensor(
    [
        [1.0],
        [1.0],
        [0.0]
    ]
)

net.zero_grad()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for i in range(10000):
    output = net(X)
    loss = criterion(output,Y)
    print(loss)
    loss.backward()
    optimizer.step()

print(net(X))
print(Y)