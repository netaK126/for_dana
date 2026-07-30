import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN_2_10(nn.Module):
    def __init__(self, k=1, w=28, h=28, output_size=10):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 10)
        self.fc2 = nn.Linear(10, output_size)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FNN_3_10(nn.Module):
    def __init__(self, k=1, w=28, h=28, output_size=10):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, output_size)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FNN_5_10(nn.Module):
    def __init__(self, k=1, w=28, h=28, output_size=10):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, output_size)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class FNN_6_10(nn.Module):
    def __init__(self, k=1, w=28, h=28, output_size=10):
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
        self.fc6 = nn.Linear(10, output_size)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class FNN_10_10(nn.Module):
    def __init__(self, k=1, w=28, h=28, output_size=10):
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
        self.fc10 = nn.Linear(10, output_size)
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
    def __init__(self, k=1, w=28, h=28, output_size=10):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, output_size)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FNN_3_100(nn.Module):
    def __init__(self, k=1, w=28, h=28, output_size=10):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, output_size)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FNN_6_100(nn.Module):
    def __init__(self, k=1, w=28, h=28, output_size=10):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, output_size)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class FNN_9_200(nn.Module):
    def __init__(self, k=1, w=28, h=28, output_size=10):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 200)
        self.fc7 = nn.Linear(200, 200)
        self.fc8 = nn.Linear(200, 200)
        self.fc9 = nn.Linear(200, output_size)
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
        x = self.fc9(x)
        return x

class FNN_5_50(nn.Module):
    def __init__(self, k=1, w=28, h=28, output_size=10):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, output_size)
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
    def __init__(self, k=1, w=28, h=28, output_size=10):
        # call constructor from superclass
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        # define network layers
        self.conv1 = nn.Conv2d(self.k, 3, 4, stride=(4, 4), padding='valid')
        self.conv2 = nn.Conv2d(3, 3, 3, stride=(4, 4), padding='valid')
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(12, output_size)
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
    def __init__(self, k=1, w=28, h=28, output_size=10):
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
        self.fc2 = nn.Linear(10, output_size)

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
    def __init__(self, k=1, w=28, h=28, output_size=10):
        # call constructor from superclass
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        # define network layers
        self.conv1 = nn.Conv2d(self.k, 3, 4, stride=(1, 1), padding='valid')
        self.conv2 = nn.Conv2d(3, 3, 3, stride=(3, 3), padding='valid')
        self.flatten1 = nn.Flatten()
        # Flattened conv-output size derived from the input geometry so the net
        # adapts across datasets: 28x28x1 -> 192 (MNIST/Fashion-MNIST),
        # 32x32x3 -> 243 (CIFAR-10). valid padding: out = (in - kernel)//stride + 1.
        w1 = (self.w - 4) // 1 + 1
        h1 = (self.h - 4) // 1 + 1
        w2 = (w1 - 3) // 3 + 1
        h2 = (h1 - 3) // 3 + 1
        flatten_num = 3 * w2 * h2
        self.fc1 = nn.Linear(flatten_num, 10)
        self.fc2 = nn.Linear(10, output_size)
        self.m = nn.Dropout(p=0.75)

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

class CNN3(nn.Module):
    def __init__(self, k=1, w=28, h=28, output_size=10):
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
        self.fc3 = nn.Linear(10, output_size)

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


class FNN_4_10(nn.Module):
    def __init__(self, k=1, w=28, h=28, output_size=10):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, output_size)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ── Pretrained tabular benchmark net (HAR) ─────────────────────────────────
# Used only under --internet_nets_benchmarks. Weights are the .pth emitted by
# utils/nnet_to_pickle.py (state_dict keys fc1.weight/bias, ...). Defaults bake
# in the true input/output sizes because load_model() in the hyper-attack
# instantiates these with no arguments.
class FNN_HAR(nn.Module):
    # HAR: 561 inputs -> one hidden ReLU layer of 500 -> 6 classes.
    def __init__(self, k=1, w=561, h=1, output_size=6):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 500)
        self.fc2 = nn.Linear(500, output_size)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN5(nn.Module):
    # Larger sibling of CNN2: a 2-CONV + 2-FC net with 10 channels, sized so the
    # CIFAR-10 (32x32x3) instance has ~8500-9000 hidden ReLU neurons (bigger than
    # the 7386-neuron conv3/cnn4). conv1 (4x4, stride 1) -> 29x29x10 = 8410,
    # conv2 (4x4, stride 4) -> 7x7x10 = 490, fc1 -> 10; total 8910 hidden neurons.
    def __init__(self, k=1, w=28, h=28, output_size=10):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.conv1 = nn.Conv2d(self.k, 10, 4, stride=(1, 1), padding='valid')
        self.conv2 = nn.Conv2d(10, 10, 4, stride=(4, 4), padding='valid')
        self.flatten1 = nn.Flatten()
        # Flattened conv-output size derived from the input geometry so the net
        # adapts across datasets (valid padding: out = (in - kernel)//stride + 1):
        # 32x32 -> 490 (CIFAR-10), 28x28 -> 360 (MNIST/Fashion-MNIST).
        w1 = (self.w - 4) // 1 + 1
        h1 = (self.h - 4) // 1 + 1
        w2 = (w1 - 4) // 4 + 1
        h2 = (h1 - 4) // 4 + 1
        flatten_num = 10 * w2 * h2
        self.fc1 = nn.Linear(flatten_num, 10)
        self.fc2 = nn.Linear(10, output_size)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
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


class CNN4(nn.Module):
    # conv2-style net widened to ~5522 ReLU neurons (2 CONV + 2 FC).
    # conv1 stride 1, conv2 stride 3, then fc1 -> 10. The flatten dim after the
    # two convs depends on the input geometry, so fc1's in_features is derived
    # from (k, w, h) rather than hardcoded: MNIST 1x28x28 -> conv1 8x25x25=5000,
    # conv2 8x8x8=512; CIFAR-10 3x32x32 -> conv1 8x29x29=6728, conv2 8x9x9=648.
    def __init__(self, k=1, w=28, h=28, output_size=10):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.conv1 = nn.Conv2d(self.k, 8, 4, stride=(1, 1), padding='valid')
        self.conv2 = nn.Conv2d(8, 8, 3, stride=(3, 3), padding='valid')
        self.flatten1 = nn.Flatten()
        with torch.no_grad():
            _flatten_num = self.conv2(self.conv1(
                torch.zeros(1, self.k, self.w, self.h))).numel()
        self.fc1 = nn.Linear(_flatten_num, 10)  # 8*8*8=512 (MNIST) / 8*9*9=648 (CIFAR-10)
        self.fc2 = nn.Linear(10, output_size)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k, self.w, self.h)
        x = self.m(F.relu(self.conv1(x)))
        x = self.m(F.relu(self.conv2(x)))
        x = self.flatten1(x)
        x = self.m(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x