import torch.nn as nn
import torch.nn.functional as F

import torch.quantization as quan


class Quantized_FNN_3_10(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.quant = quan.QuantStub()
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 10) 
        self.fc2 = nn.Linear(10, 10) 
        self.fc3 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.dequant = quan.DeQuantStub()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.quant(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x
    

class VerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(VerySimpleNet,self).__init__()
        self.linear1 = nn.Linear(28*28, hidden_size_1) 
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

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
    
class FNN_4_10(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class FNN_2_10(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 10)
        self.fc2 = nn.Linear(10, 10)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FNN_5_10(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class FNN_10_10(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)
        self.fc7 = nn.Linear(10, 10)
        self.fc8 = nn.Linear(10, 10)
        self.fc9 = nn.Linear(10, 10)
        self.fc10 = nn.Linear(10, 10)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = self.fc10(x)
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
    
class FNN_3_100(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FNN_5_50(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
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

class CNN3(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        # call constructor from superclass
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        # define network layers
        self.conv1 = nn.Conv2d(self.k, 6, 4, stride=(3, 3), padding='valid')
        self.conv2 = nn.Conv2d(6, 6, 3, stride=(3, 3), padding='valid')
        self.conv3 = nn.Conv2d(6, 6, 3, stride=(3, 3), padding='valid')
        self.conv4 = nn.Conv2d(6, 6, 3, stride=(3, 3), padding='valid')
        self.conv5 = nn.Conv2d(6, 6, 3, stride=(3, 3), padding='valid')
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(54, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        # define forward pass
        x = x.reshape(-1, self.k, self.w, self.h)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x