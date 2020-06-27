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
    root='./data',
    train=True,
    download=False,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
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

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

'''
original code :
epochs=2 , batch_size=4 , lr=0.001 , loss=1.887 , accuracy=32%

changing learning rate:
lr=0.0001 , loss=2.299 , accuracy=10%
lr=0.01 , loss=1.312 , accuracy=54%
lr=0.1 , loss=1.961 , accuracy=24%

changing batch size:
batch_size=2 , loss=1.537 , accuracy=43%
batch_size=1 , loss=1.368 , accuracy=51%
batch_size=8 , loss=2.145 , accuracy=25%

changing number of epochs:
epochs=1 , loss=2.292 , accuracy=15%
epochs=6 , loss=1.395 , accuracy=50%

changing kernel size of conv2d:
kernel_size=3, loss=1.80 , accuracy=35%

changing output channels:
output_channels=(10,20) , loss=1.183 , accuracy=34%
output_channels=(6,10) , loss=1.189 , accuracy=33%

'''