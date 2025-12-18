import torch.nn as nn

# includes models for neural networks.

class mlp(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=256, output_dim=10):
        super(mlp, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))

class deep(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=5, hidden_num=2):
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

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(in_channels=256, out_channels=10, kernel_size=5))
    def forward(self, x):
        return self.layers(x).view(x.size(0), -1)

def binary_loss(outputs, labels):
    # calculates binary loss.
    # outputs: outputs of the network.
    # labels: true label for each output.
    # return: loss function.
    return (1 + (-labels.float() * outputs).exp()).log().mean()

