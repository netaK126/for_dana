import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os
from models import *
import pickle
import numpy as np

# Make torch deterministic
_ = torch.manual_seed(0)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load the MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# Create a dataloader for the training
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

# Load the MNIST test set
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

# Define the device
device = "cpu"

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
model_type = "3x10"
if model_type == "3x10":
    model = FNN_3_10()
elif model_type == "3x50":
    model = FNN_3_50()
elif model_type == "3x100":
    model = FNN_3_100()
elif model_type == "5x50":
    model = FNN_5_50()
elif model_type == "5x10":
    model = FNN_5_10()
elif model_type == "10x10":
    model = FNN_10_10()
elif model_type == "cnn0":
    model = CNN0()
elif model_type == "cnn1":
    model = CNN1()
elif model_type == "cnn2":
    model = CNN2()
elif model_type == "cnn3":
    model = CNN3()
net = model.to(device)
def train(train_loader, net, epochs=5, total_iterations_limit=None):
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    total_iterations = 0

    for epoch in range(epochs):
        net.train()

        loss_sum = 0
        num_iterations = 0

        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = net(x.view(-1, 28*28))
            loss = cross_el(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return
            
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp_delme.p")
    print('Size (KB):', os.path.getsize("temp_delme.p")/1e3)
    os.remove('temp_delme.p')

MODEL_FILENAME = '/root/Downloads/vaghgar_for_quantisized_network/models/3x10/model.pth'

if Path(MODEL_FILENAME).exists():
    net.load_state_dict(torch.load(MODEL_FILENAME))
    print('Loaded model from disk')
else:
    # train(train_loader, net, epochs=1)
    # # Save the model to disk
    # torch.save(net.state_dict(), MODEL_FILENAME)
    print("ERROR - MODEL PATH DON'T EXISTS!!")
    exit(1)


def test(model: nn.Module, total_iterations: int = None):
    correct = 0
    total = 0

    iterations = 0

    model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = model(x.view(-1, 784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct +=1
                total +=1
            iterations += 1
            if total_iterations is not None and iterations >= total_iterations:
                break
    print(f'Accuracy: {round(correct/total, 3)}')

print('Size of the model before quantization')
print_size_of_model(net)
print(f'Accuracy of the model before quantization: ')
test(net)


net_quantized = Quantized_FNN_3_10().to(device)
# Copy weights from unquantized model
net_quantized.load_state_dict(net.state_dict())
net_quantized.eval()

net_quantized.qconfig = torch.ao.quantization.default_qconfig
net_quantized = torch.ao.quantization.prepare(net_quantized) # Insert observers
net_quantized

print_size_of_model(net_quantized)
print_size_of_model(net)
# exit()

print(f'Check statistics of the various layers')
net_quantized

net_quantized = torch.ao.quantization.convert(net_quantized)
print("fc1 input/output scale:", net_quantized.fc1.scale)
print("fc2 input/output scale:", net_quantized.fc2.scale)
print("fc3 input/output scale:", net_quantized.fc3.scale)
net_quantized
print('Weights after quantization')
print(torch.int_repr(net_quantized.fc1.weight()))
print('Original weights: ')
print(net.fc1.weight)
print('')
print(f'Dequantized weights: ')
print(torch.dequantize(net_quantized.fc1.weight()))
print('')
print('Size of the model after quantization')
print_size_of_model(net_quantized)
print('Testing the model after quantization')
test(net_quantized)



print("fc1 weight scale:", net_quantized.fc1.weight().q_scale())
print("fc2 weight scale:", net_quantized.fc2.weight().q_scale())
print("fc3 weight scale:", net_quantized.fc3.weight().q_scale())
# Collect parameters
params = {}
for name, module in net_quantized.named_modules():
    if hasattr(module, "weight") and callable(module.weight):
        try:
            params[f"{name}.weight_scale"] = module.weight().q_scale()
            params[f"{name}.weight_zero_point"] = module.weight().q_zero_point()
        except Exception as e:
            print(f"Skipping {name} weight: {e}")

# --- Biases (stored as float, no scale needed) ---
for name, param in net_quantized.named_parameters():
    params[name] = param.detach().cpu().numpy()

# --- Activations scales/zero points ---
# Need to look at pre-convert version with observers
net_prepared = torch.ao.quantization.prepare(net_quantized, inplace=False)
dummy_input = torch.randn(1, 1, 28, 28)  # MNIST shape
net_prepared(dummy_input)

for name, module in net_prepared.named_modules():
    if hasattr(module, "activation_post_process"):
        try:
            scale, zp = module.activation_post_process.calculate_qparams()
            params[f"{name}.act_scale"] = float(scale)
            params[f"{name}.act_zero_point"] = int(zp)
        except Exception:
            pass

# Save everything
save_path = Path(MODEL_FILENAME).parent / "model_quantized_Sm_params.pkl"
with open(save_path, "wb") as f:
    pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved quantization params to {save_path}")

model_quan_path = Path(MODEL_FILENAME).parent / "model_quantized.p"

a = [
    np.transpose(torch.dequantize(net_quantized.fc1.weight()).detach().cpu().numpy()),
    np.transpose(torch.dequantize(net_quantized.fc1.bias()).detach().cpu().numpy()),
    np.transpose(torch.dequantize(net_quantized.fc2.weight()).detach().cpu().numpy()),
    np.transpose(torch.dequantize(net_quantized.fc2.bias()).detach().cpu().numpy()),
    np.transpose(torch.dequantize(net_quantized.fc3.weight()).detach().cpu().numpy()),
    np.transpose(torch.dequantize(net_quantized.fc3.bias()).detach().cpu().numpy())
]

with open(Path(MODEL_FILENAME).parent / "model_quantized.p", "wb") as f:
    import pickle
    pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)
    
torch.save(net_quantized.state_dict(), Path(MODEL_FILENAME).parent / "model_quantized.pth")
