import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
from models import *
import pickle
import argparse
import random
from torch.utils.data import DataLoader, Subset
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = dsets.MNIST('.', train=True, download=True, transform=transform)
test_dataset = dsets.MNIST('.', train=False, download=True, transform=transform)

def train_model(model, train_loader, optimizer_type, loss_fn, epochs, batch_size, test_loader,lr):
    print("optimizing new model")
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam( model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD( model.parameters(), lr = lr)
    else:
        assert ("New optimizer has been detected, please expand this if condition to support it.")
    model.to(device)
    # model.train()
    # for _ in range(epochs):
    #     for data, target in train_loader:
    #         data, target = data.to(device), target.to(device)
    #         optimizer.zero_grad()
    #         loss = loss_fn(model(data), target)
    #         loss.backward()
    #         optimizer.step()

    for epoch in range(epochs):
        total_batch = len(train_dataset) // batch_size
        for i, (batch_images, batch_labels) in enumerate(train_loader):

            X = batch_images.view(-1, 1, 28, 28).to(device)
            # x = attacker.attack(model.to('cuda:0') , optimizer, batch_images.view(-1,1,28,28).to('cuda:0') , batch_labels.to('cuda:0') ).to('cuda:0')
            # X = atk(batch_images, batch_labels)

            Y = batch_labels.to(device)
            pre = model(X)
            if loss_type == "L1":
                Y = torch.nn.functional.one_hot(Y, 10).float()
            cost = loss_fn(pre, Y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if (i + 1) % 200 == 0:
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images).to(device)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum()
            print('Test accuracy: %.2f %%' % (100 * float(correct) / total))
    return model

# Ensemble evaluation
def evaluate_ensemble(models, test_loader):
    correct, total = 0, 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            probs = torch.stack([torch.softmax(m(data), dim=1) for m in models]).to(device)
            avg_probs = probs.mean(dim=0).to(device)
            pred = avg_probs.argmax(dim=1).to(device)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct / total


# def create_bootstrap_loader(dataset, batch_size):
#     indices = np.random.choice(len(dataset), size=len(dataset), replace=True)
#     subset = torch.utils.data.Subset(dataset, indices)
#     return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

def create_begging_ensemble(model_class,model_type, optimizer_type, loss_fn, epochs,models_num):
    bagging_models = []
    batch_size_list = [128, 128]
    base_lr=1e-3
    for i in range(models_num):  # 5 models
        set_seed(42)
        lr = base_lr * (1 + 0.0001 * (i - (models_num - 1) / 3)) 
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size_list[i], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_list[i], shuffle=False)
        # torch.manual_seed(i * 17 + 42)  # different seed per model
        # random.seed(i * 17 + 42)
        
        model = model_class()
        # loader = create_bootstrap_loader(train_dataset, batch_size_list[i])
        trained = train_model(model, train_loader, optimizer_type, loss_fn, epochs, batch_size_list[i], test_loader,lr)
        bagging_models.append(copy.deepcopy(trained))

    bagging_acc = evaluate_ensemble(bagging_models, test_loader)
    print(f"Bagging Accuracy: {bagging_acc*100:.2f}%")
    save_model(bagging_models, epochs, model_type, output_dir)



def save_model(model_list, itr, model_type, output_dir):
    model_path = output_dir+"/"+model_type+"_"+str(itr)+"/"
    os.system("mkdir " + model_path)
    with open(os.path.join(model_path, "mlp_ensemble.p"), "wb") as f:
        pickle.dump([m.state_dict() for m in model_list], f)
    torch.save([m.state_dict() for m in model_list], os.path.join(model_path, "mlp_ensemble.pth"))

    for j,m in enumerate(model_list):
        a = []
        for i in m.parameters():
            print((np.transpose(i.cpu().detach().numpy())).shape)
            a.append(np.transpose(i.cpu().detach().numpy()))
        model_name = "mnist"+str(j+1)
        pickle.dump(a, open(model_path + model_name + ".p", "wb"))
        torch.save(m.state_dict(), model_path + model_name + '.pth')

def create_bootstrap_loader(dataset, batch_size):
    # Random bootstrap sample with replacement
    indices = [random.randint(0, len(dataset)-1) for _ in range(len(dataset))]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)

def set_seed(seed=42):
    """Sets the seed for reproducibility across all relevant libraries."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a consistent seed for all models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--model', type=str, default="3x10", help='3x10, 3x50,3x100,3x50 cnn1, cnn2 or cnn3')
    parser.add_argument('--output_dir', type=str, default="/root/Downloads/code_deprecated_active_just_for_models/model/ensemble/", help='output directory')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--loss', type=str, default="Cross", help='Cross, MSE, or L1')
    parser.add_argument('--optimizer', type=str, default="Adam", help='Adam, or SGD')
    parser.add_argument('--ensemble_type', type=str, default="Bagging", help='Bagging/Seed/Boosting')
    parser.add_argument('--ensemble_size', type=int, default=2, help='number of models in ensamble')
    args = parser.parse_args()

    num_epochs = args.epochs
    output_dir = args.output_dir
    model_type = args.model
    loss_type = args.loss
    optimizer_type = args.optimizer

    if model_type == "3x10":
        model_class = FNN_3_10
    elif model_type == "3x50":
        model_class = FNN_3_50
    elif model_type == "3x100":
        model_class = FNN_3_100
    elif model_type == "5x50":
        model_class = FNN_5_50
    elif model_type == "5x10":
        model_class = FNN_5_10
    elif model_type == "10x10":
        model_class = FNN_10_10
    elif model_type == "cnn0":
        model_class = CNN0
    elif model_type == "cnn1":
        model_class = CNN1
    elif model_type == "cnn2":
        model_class = CNN2
    elif model_type == "cnn3":
        model_class = CNN3
    else:
        assert ("New model arch has been detected, please expand models.py and this if condition.")


    if loss_type == "Cross":
        loss_fn = nn.CrossEntropyLoss().to(device)
    elif loss_type == "MSE":
        loss_fn = nn.MSELoss().to(device)
    elif loss_type == "L1":
        loss_fn = nn.L1Loss().to(device)
    else:
        assert ("New loss has been detected, please expand this if condition to support it.")
    print("Loading train set and test set")
    set_seed(42)
    if args.ensemble_type=="Bagging":
        print("creating ensemble")
        create_begging_ensemble(model_class,model_type, optimizer_type, loss_fn, num_epochs, args.ensemble_size)
