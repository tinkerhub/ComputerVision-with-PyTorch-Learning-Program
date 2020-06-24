import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root='~/data',
    train=True,
    download=False,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='~/data',
    train=False,
    download=False,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

classes = (
    'plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=0.001
)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # data = (inputs, labels)
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss.item()
        if i % 2000 == 1999:
            print(
                '[%d, %5d] loss: %.3f' %
                (epoch + 1, i+1, running_loss/2000)
            )
            running_loss = 0.0
print("vola")

# 1) Just the normal code
# [1,  2000] loss: 2.303
# [1,  4000] loss: 2.303
# [1,  6000] loss: 2.301
# [1,  8000] loss: 2.299
# [1, 10000] loss: 2.296
# [1, 12000] loss: 2.289
# [2,  2000] loss: 2.257
# [2,  4000] loss: 2.209
# [2,  6000] loss: 2.161
# [2,  8000] loss: 2.085
# [2, 10000] loss: 1.978
# [2, 12000] loss: 1.925
# 82.325 seconds taken

# 2) Adding a sigmoid function at output
# [1,  2000] loss: 2.303
# [1,  4000] loss: 2.302
# [1,  6000] loss: 2.302
# [1,  8000] loss: 2.302
# [1, 10000] loss: 2.301
# [1, 12000] loss: 2.301
# [2,  2000] loss: 2.300
# [2,  4000] loss: 2.300
# [2,  6000] loss: 2.298
# [2,  8000] loss: 2.296
# [2, 10000] loss: 2.292
# [2, 12000] loss: 2.287
# 81.954 seconds taken

# 3) Case 2 with increasing learning rate to 0.01
# [1,  2000] loss: 2.243
# [1,  4000] loss: 2.140
# [1,  6000] loss: 2.084
# [1,  8000] loss: 2.056
# [1, 10000] loss: 2.041
# [1, 12000] loss: 2.021
# [2,  2000] loss: 1.995
# [2,  4000] loss: 1.990
# [2,  6000] loss: 1.969
# [2,  8000] loss: 1.954
# [2, 10000] loss: 1.946
# [2, 12000] loss: 1.937
# 84.034 seconds taken

# 4) Case 1 with learning rate 0.01
# [1,  2000] loss: 1.254
# [1,  4000] loss: 1.220
# [1,  6000] loss: 1.212
# [1,  8000] loss: 1.203
# [1, 10000] loss: 1.212
# [1, 12000] loss: 1.193
# [2,  2000] loss: 1.120
# [2,  4000] loss: 1.120
# [2,  6000] loss: 1.132
# [2,  8000] loss: 1.126
# [2, 10000] loss: 1.129
# [2, 12000] loss: 1.101
# 80.073 seconds taken

# 5) output softmax with lr = 0.01
# [1,  2000] loss: 2.303
# [1,  4000] loss: 2.302
# [1,  6000] loss: 2.302
# [1,  8000] loss: 2.302
# [1, 10000] loss: 2.301
# [1, 12000] loss: 2.293
# [2,  2000] loss: 2.274
# [2,  4000] loss: 2.261
# [2,  6000] loss: 2.239
# [2,  8000] loss: 2.208
# [2, 10000] loss: 2.179
# [2, 12000] loss: 2.151
# 81.663 seconds taken