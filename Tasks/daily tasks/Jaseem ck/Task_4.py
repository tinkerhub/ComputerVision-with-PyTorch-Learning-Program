#Task 4
#Run these codes in colab

#------------------------------------------------------------------------------------
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
        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=0.01
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

#------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------

'''
Epochs = 2; Batch_size = 4; lr = 0.001; loss = 1.818; accuracy = 34

Epochs = 2; Batch_size = 10; lr = 0.001; loss = 2.268; accuracy = 22

Epochs = 2; Batch_size = 1; lr = 0.001; loss = 1.392; accuracy = 51

Epochs = 2; Batch_size = 4; lr = 0.01; loss = 1.259; accuracy = 55

Epochs = 2; Batch_size = 4; lr = 0.1; loss = 1.985; accuracy = 22

Epochs = 10; Batch_size = 4; lr = 0.01; loss = 0.848; accuracy = 62

Epochs = 20; Batch_size = 4; lr = 0.01; loss = 0.726; accuracy = 58


---

with adam optimizer:
Epochs = 20; Batch_size = 4; lr = 0.01; loss = 2.307; accuracy = 10



---
Epochs = 2; Batch_size = 4; lr = 0.01; output_channel=24; kernel_size=5; Accuracy = 63

Epochs = 2; Batch_size = 4; lr = 0.01; output_channel=32; kernel_size=5; Accuracy = 59

Epochs = 2; Batch_size = 4; lr = 0.01; output_channel=256; kernel_size=3; Accuracy = 59
'''
