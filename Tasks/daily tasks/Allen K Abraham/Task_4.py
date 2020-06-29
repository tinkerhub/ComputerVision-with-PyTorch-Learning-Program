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
    lr=0.01
)

for epoch in range(20):
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
        dumb, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('accuracy: %f' % (correct / total))

### ======================== ###

# epoch = 2, batch_size = 4, lr = 0.001 : loss = 1.814, accuracy = 35.5%
# epoch = 5, batch_size = 4, lr = 0.001 : loss = 1.469, accuracy = 46.3%
# epoch = 2, batch_size = 10, lr = 0.001 : loss = 2.275, accuracy = 19.8%
# epoch = 20, batch_size = 4, lr = 0.01 : loss = 0.721, accuracy = 60.8%

### ======================== ###

# Epochs = 2; Batch_size = 4; lr = 0.01; output_channel=24; kernel_size=5; Accuracy = 63
# Epochs = 2; Batch_size = 4; lr = 0.01; output_channel=32; kernel_size=5; Accuracy = 59
# Epochs = 2; Batch_size = 4; lr = 0.01; output_channel=256; kernel_size=3; Accuracy = 59

### ======================== ###
