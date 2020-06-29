import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.input=nn.Linear(400,200)
        self.hidden1=nn.Linear(200,100)
        self.sigmoid=nn.Sigmoid()
        self.hidden2=nn.Linear(100,50)
        self.output=nn.Linear(50,25)

    def forward(self,x):
        x=self.input(x)
        x=self.hidden1(x)
        x=self.sigmoid(x)
        x=self.hidden2(x)
        x=self.output(x)
        return x
model=Net()
print(model)