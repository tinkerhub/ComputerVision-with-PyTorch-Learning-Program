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
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


alphas = [0.001, 0.01, 0.1, 1.0]

loss_function = nn.CrossEntropyLoss()


for alpha in alphas:
    net = Net()
    optimizer = optim.SGD(
        net.parameters(),
        lr=alpha)
    print('Alpha = ', alpha)
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
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    print('Accuracy of %.4f: %d %%' % (alpha, 100 * correct / total))

# OBSERVATIONS:

# output_channels=(3, 6, 16), kernel=5, lr=0.001: loss changes from 2.303 to 1.713 and accuracy = 37%
# output_channels=(3, 6, 16), kernel=5, lr=0.01: loss changes from 2.188 to 1.324 and accuracy = 54%
# output_channels=(3, 6, 16), kernel=5, lr=0.1: loss changes from 2.074 to 1.940 and accuracy = 26%
# output_channels=(3, 6, 16), kernel=5, lr=0.001: loss changes from 2.361 to 2.365 and accuracy = 10%

# output_channels=(3, 6, 16), kernel=3, lr=0.001: loss changes from 2.303 to 1.865 and accuracy = 33%
# output_channels=(3, 6, 16), kernel=3, lr=0.01: loss changes from 2.226 to 1.256 and accuracy = 56%
# output_channels=(3, 6, 16), kernel=3, lr=0.1: loss changes from 2.028 to 1.831 and accuracy = 35%
# output_channels=(3, 6, 16), kernel=3, lr=1.0: loss changes from 2.362 to 2.361 and accuracy = 10%

# output_channels=(3, 6, 10), kernel=5, lr=0.001: loss changes from 2.303 to 1.836 and accuracy = 35%
# output_channels=(3, 6, 10), kernel=5, lr=0.01: loss changes from 1.826 to 1.284 and accuracy = 51%
# output_channels=(3, 6, 10), kernel=5, lr=0.1: loss changes from 2.044 to 1.994 and accuracy = 22%
# output_channels=(3, 6, 10), kernel=5, lr=1.0: loss changes from 2.362 to 2.363 and accuracy = 10%

# output_channels=(3, 6, 10), kernel=3, lr=0.001: loss changes from 2.304 to 1.879 and accuracy = 32%
# output_channels=(3, 6, 10), kernel=3, lr=0.01: loss changes from 2.260 to 1.271 and accuracy = 53%
# output_channels=(3, 6, 10), kernel=3, lr=0.1: loss changes from 2.069 to 1.864 and accuracy = 33%
# output_channels=(3, 6, 10), kernel=3, lr=1.0: loss changes from 2.362 to 2.365 and accuracy = 10%

# output_channels=(3, 10, 16), kernel=5, lr=0.001: loss changes from 2.304 to 1.739 and accuracy = 37%
# output_channels=(3, 10, 16), kernel=5, lr=0.01: loss changes from 2.228 to 1.224 and accuracy = 55%
# output_channels=(3, 10, 16), kernel=5, lr=0.1: loss changes from 2.076 to 1.990 and accuracy = 22%
# output_channels=(3, 10, 16), kernel=5, lr=1.0: loss changes from 2.358 to 2.363 and accuracy = 10%

# output_channels=(3, 10, 16), kernel=5, lr=0.001: loss changes from 2.302 to 1.848 and accuracy = 34%
# output_channels=(3, 10, 16), kernel=5, lr=0.01: loss changes from 2.201 to 1.213 and accuracy = 56%
# output_channels=(3, 10, 16), kernel=5, lr=0.1: loss changes from 2.043 to 1.882 and accuracy = 31%
# output_channels=(3, 10, 16), kernel=5, lr=1.0: loss changes from 2.362 to 2.368 and accuracy = 10%

#maxpool = (4, 4)
# output_channels=(3, 6, 16), kernel=3, lr=0.001: loss changes from 2.303 to 2.119 and accuracy = 22%
# output_channels=(3, 6, 16), kernel=3, lr=0.01: loss changes from 2.298 to 1.567 and accuracy = 42%
# output_channels=(3, 6, 16), kernel=3, lr=0.1: loss changes from 2.067 to 1.909 and accuracy = 29%
# output_channels=(3, 6, 16), kernel=3, lr=1.0: loss changes from 2.364 to 2.361 and accuracy = 10%
