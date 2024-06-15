import torch.nn as nn
import torch.nn.functional as F

from .prune import PruningModule, MaskedLinear, MaskedConv2d

class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        Conv2d = MaskedConv2d if mask else nn.Conv2d
        self.conv1 = Conv2d(1, 6, kernel_size=(5, 5),stride=1,padding=2)
        self.conv2 = Conv2d(6, 16, kernel_size=(5, 5),stride=1)
        self.fc1 = linear(16*5*5,120)
        self.fc2 = linear(120, 84)
        self.fc3 = linear(84, 10)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv3

        # Fully-connected
        x = x.view(-1, 16*5*5)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x



