import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

if __name__ == '__main__':
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
            self.conv1 = nn.Conv2d(3, 192, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(192, 512, 3)
            self.fc1 = nn.Linear(512 * 6 * 6, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 512 * 6 * 6)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.03

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

"""

With the given code
Accuracy : 37%          Loss : 1.789
With a Learning Rate of 0.0001
Accuracy : 10%          Loss : 2.302
With output channels of Conv layers made 4x.
Accuracy : 40%          Loss : 1.620
With output channels of Conv layers made 4x and learning rate = 0.003
Accuracy : 53%          Loss : 1.258
With output channels of Conv layers made 4x and learning rate = 0.007
Accuracy : 60%          Loss : 1.105
With output channels of Conv layers made 4x and learning rate = 0.01
Accuracy : 24%          Loss : 1.979
With output channels of Conv layers made 8x and learning rate = 0.008
Accuracy : 64%          Loss : 1.021
With output channels of Conv layers made 16x and learning rate = 0.009
Accuracy : 67%          Loss : 0.932
With output channels of Conv layers made 32x and learning rate = 0.009
Accuracy : 68%          Loss : 0.930
With output channels of Conv layers made 32x and learning rate = 0.009 and kernel size (3,3)
Accuracy : 68%          Loss : 0.915
With output channels of Conv layers made 32x and learning rate = 0.009 and kernel size (3,3)
Accuracy : 67%          Loss : 0.950
With output channels of Conv layers made 32x and learning rate = 0.01 and kernel size (3,3)
Accuracy : 68%          Loss : 0.900
With output channels of Conv layers made 32x and learning rate = 0.03 and kernel size (3,3)
Accuracy : 69%          Loss : 0.892
"""
