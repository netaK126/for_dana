import torch.nn as nn
import torch.nn.functional as F

class FNN_3_10(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FNN_3_50(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN0(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        # call constructor from superclass
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        # define network layers
        self.conv1 = nn.Conv2d(self.k, 3, 4, stride=(4, 4), padding='valid')
        self.conv2 = nn.Conv2d(3, 3, 3, stride=(4, 4), padding='valid')
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(12, 10)
        self.m = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        # define forward pass
        x = x.reshape(-1, self.k, self.w, self.h)
        x = F.relu(self.conv1(x))
        x = self.m(x)
        x = F.relu(self.conv2(x))
        x = self.m(x)
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = self.m(x)
        x = self.fc2(x)
        return x

class CNN1(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        # call constructor from superclass
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        # define network layers
        self.conv1 = nn.Conv2d(self.k, 6, 4, stride=(3, 3), padding='valid')
        self.conv2 = nn.Conv2d(6, 6, 3, stride=(3, 3), padding='valid')
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(54, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        # define forward pass
        x = x.reshape(-1, self.k, self.w, self.h)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN2(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        # call constructor from superclass
        super().__init__()
        # define network layers
        self.conv1 = nn.Conv2d(self.k, 3, 4, stride=(1, 1), padding='valid')
        self.conv2 = nn.Conv2d(3, 3, 3, stride=(3, 3), padding='valid')
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(192, 10)
        self.fc2 = nn.Linear(10, 10)
        self.m = nn.Dropout(p=0.75)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # define forward pass
        x = x.reshape(-1, self.k, self.w, self.h)
        x = F.relu(self.conv1(x))
        x = self.m(x)
        x = F.relu(self.conv2(x))
        x = self.m(x)
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = self.m(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class deep(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=10, hidden_num=2):
        super(deep, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for i in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, 2))
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x#.view(-1)