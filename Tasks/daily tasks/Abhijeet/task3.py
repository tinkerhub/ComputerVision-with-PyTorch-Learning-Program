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

'''
OUTPUT:
 number of output channels=16, kernel size=5, learning rate=0.001 epoch=2
final loss ===========>[2, 12000] loss: 1.811
Test Accuracy of plane: 50% (502/1000)
Test Accuracy of   car: 49% (495/1000)
Test Accuracy of  bird: 15% (157/1000)
Test Accuracy of   cat: 15% (151/1000)
Test Accuracy of  deer: 36% (362/1000)
Test Accuracy of   dog: 22% (222/1000)
Test Accuracy of  frog: 49% (495/1000)
Test Accuracy of horse: 34% (348/1000)
Test Accuracy of  ship: 52% (520/1000)
Test Accuracy of truck: 42% (426/1000)

Test Accuracy (Overall): 36% (3678/10000)

 number of output channels=16, kernel size=5, learning rate=0.01 epoch=2
final loss ===========>[2, 12000] loss: 1.255

Test Accuracy of plane: 60% (609/1000)
Test Accuracy of   car: 71% (719/1000)
Test Accuracy of  bird: 45% (453/1000)
Test Accuracy of   cat: 44% (445/1000)
Test Accuracy of  deer: 46% (469/1000)
Test Accuracy of   dog: 42% (423/1000)
Test Accuracy of  frog: 68% (681/1000)
Test Accuracy of horse: 65% (659/1000)
Test Accuracy of  ship: 66% (660/1000)
Test Accuracy of truck: 34% (344/1000)

Test Accuracy (Overall): 54% (5462/10000)

number of output channels=16, kernel size=3 learning rate=0.01 epoch=2
final loss ===========>[2, 12000] loss: 1.270
Test Accuracy of plane: 63% (632/1000)
Test Accuracy of   car: 74% (747/1000)
Test Accuracy of  bird: 41% (419/1000)
Test Accuracy of   cat: 36% (366/1000)
Test Accuracy of  deer: 61% (610/1000)
Test Accuracy of   dog: 30% (306/1000)
Test Accuracy of  frog: 74% (740/1000)
Test Accuracy of horse: 62% (624/1000)
Test Accuracy of  ship: 62% (621/1000)
Test Accuracy of truck: 49% (491/1000)

Test Accuracy (Overall): 55% (5556/10000)

number of output channels=32, kernel size=3 learning rate=0.01 epoch=2
final loss ===========>[2, 12000] loss: 1.170
Test Accuracy of plane: 64% (644/1000)
Test Accuracy of   car: 67% (673/1000)
Test Accuracy of  bird: 37% (376/1000)
Test Accuracy of   cat: 59% (593/1000)
Test Accuracy of  deer: 60% (603/1000)
Test Accuracy of   dog: 38% (381/1000)
Test Accuracy of  frog: 70% (706/1000)
Test Accuracy of horse: 64% (646/1000)
Test Accuracy of  ship: 70% (707/1000)
Test Accuracy of truck: 72% (728/1000)

Test Accuracy (Overall): 60% (6057/10000)

number of output channels=32, kernel size=3 learning rate=0.01 epoch=4
final loss ===========>[[4, 12000] loss: 0.966

number of output channels=32, kernel size=3 learning rate=0.01 epoch=4 batch 5
final loss ===========>[[6, 10000] loss: 0.799
Test Accuracy of plane: 60% (606/1000)
Test Accuracy of   car: 83% (831/1000)
Test Accuracy of  bird: 49% (498/1000)
Test Accuracy of   cat: 38% (385/1000)
Test Accuracy of  deer: 71% (712/1000)
Test Accuracy of   dog: 69% (690/1000)
Test Accuracy of  frog: 73% (733/1000)
Test Accuracy of horse: 68% (682/1000)
Test Accuracy of  ship: 80% (808/1000)
Test Accuracy of truck: 65% (651/1000)

Test Accuracy (Overall): 65% (6596/10000)

number of output channels=32, kernel size=3 learning rate=0.01 epoch=4 batch 5 with padding 
final loss ===========>[6, 10000] loss: 0.718
Test Accuracy of plane: 68% (689/1000)
Test Accuracy of   car: 64% (645/1000)
Test Accuracy of  bird: 57% (573/1000)
Test Accuracy of   cat: 48% (481/1000)
Test Accuracy of  deer: 58% (588/1000)
Test Accuracy of   dog: 63% (633/1000)
Test Accuracy of  frog: 83% (838/1000)
Test Accuracy of horse: 65% (653/1000)
Test Accuracy of  ship: 82% (829/1000)
Test Accuracy of truck: 75% (755/1000)

Test Accuracy (Overall): 66% (6684/10000)
'''