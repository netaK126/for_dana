import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
from models import *
import pickle
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor()])

# dataset name -> (torchvision class, channels k, width w, height h)
DATASETS = {
    "mnist":   (dsets.MNIST,        1, 28, 28),
    "cifar10": (dsets.CIFAR10,      3, 32, 32),
    "fmnist":  (dsets.FashionMNIST, 1, 28, 28),
}

def save_model(model, itr,output):
    a = []
    for i in model.parameters():
        print((np.transpose(i.cpu().detach().numpy())).shape)
        a.append(np.transpose(i.cpu().detach().numpy()))
    print("---------------",itr,"-----------------")
    for i in a:
        print(i.shape)
    model_name = "model"
    model_path = os.path.join(output, str(itr)) + "/"
    os.makedirs(model_path, exist_ok=True)
    pickle.dump(a, open(model_path + model_name + ".p", "wb"))
    torch.save(model.state_dict(), model_path + model_name + '.pth')


def save_final(model, save_dir):
    """Save the final-epoch model directly into save_dir (model.p + model.pth)."""
    os.makedirs(save_dir, exist_ok=True)
    a = [np.transpose(p.cpu().detach().numpy()) for p in model.parameters()]
    pickle.dump(a, open(os.path.join(save_dir, "model.p"), "wb"))
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    print("Saved final model to", save_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--model', type=str, default="cnn0", help='3x10, 3x50, cnn1, or cnn2')
    parser.add_argument('--output_dir', type=str, default="./model/", help='output directory')
    parser.add_argument('--dataset', type=str, default="mnist", help='mnist, cifar10, or fmnist')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='if set, save only the FINAL-epoch model to this dir (model.p + model.pth)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--loss', type=str, default="Cross", help='Cross, MSE, or L1')
    parser.add_argument('--optimizer', type=str, default="Adam", help='Adam, or SGD')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--pgd_training', action='store_true',
                        help='Train with PGD adversarial examples (Madry et al. 2018)')
    parser.add_argument('--pgd_epsilon', type=float, default=0.1,
                        help='L-infinity radius for PGD adversarial training')
    parser.add_argument('--pgd_alpha', type=float, default=0.01,
                        help='PGD step size')
    parser.add_argument('--pgd_steps', type=int, default=7,
                        help='Number of PGD iterations per batch')
    args = parser.parse_args()

    def pgd_attack(model, images, labels, epsilon, alpha, num_steps, device):
        adv = images.clone().detach()
        adv = adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)
        adv = torch.clamp(adv, 0.0, 1.0)
        criterion = nn.CrossEntropyLoss()
        for _ in range(num_steps):
            adv.requires_grad_(True)
            outputs = model(adv)
            loss_val = criterion(outputs, labels)
            grad = torch.autograd.grad(loss_val, adv)[0]
            adv = adv.detach() + alpha * grad.sign()
            delta = torch.clamp(adv - images, -epsilon, epsilon)
            adv = torch.clamp(images + delta, 0.0, 1.0)
        return adv.detach()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    batch_size = args.batch_size
    num_epochs = args.epochs
    output_dir = args.output_dir
    model_type = args.model
    loss_type = args.loss
    optimizer_type = args.optimizer

    ds_class, k, w, h = DATASETS[args.dataset]

    MODEL_CLASSES = {
        "3x10": FNN_3_10, "3x50": FNN_3_50, "3x100": FNN_3_100,
        "6x100": FNN_6_100, "9x200": FNN_9_200,
        "cnn0": CNN0, "cnn1": CNN1, "cnn2": CNN2, "cnn5": CNN5,
    }
    if model_type not in MODEL_CLASSES:
        assert ("New model arch has been detected, please expand models.py and this if condition.")
    model = MODEL_CLASSES[model_type](k=k, w=w, h=h)

    if loss_type == "Cross":
        loss = nn.CrossEntropyLoss().to(device)
    elif loss_type == "MSE":
        loss = nn.MSELoss().to(device)
    elif loss_type == "L1":
        loss = nn.L1Loss().to(device)
    else:
        assert ("New loss has been detected, please expand this if condition to support it.")

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam( model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-4)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD( model.parameters(), lr = 0.05)
    else:
        assert ("New optimizer has been detected, please expand this if condition to support it.")
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    print(model)

    # atk = PGD(model, eps=0.2, alpha=0.05, steps=4)
    # attacker = PGM(0.05, 0.25, 5, 0)

    train_ds = ds_class(root='./data/', train=True, transform=transform, download=True)
    test_ds = ds_class(root='./data/', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        total_batch = len(train_ds) // batch_size
        for i, (batch_images, batch_labels) in enumerate(train_loader):

            X = batch_images.view(-1, k, w, h).to(device)
            Y = batch_labels.to(device)
            if args.pgd_training:
                model.eval()
                X = pgd_attack(model, X, Y, args.pgd_epsilon,
                               args.pgd_alpha, args.pgd_steps, device)
                model.train()
            pre = model(X)
            if loss_type == "L1":
                Y = torch.nn.functional.one_hot(Y, 10).float()
            cost = loss(pre, Y)
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

        if args.save_dir is None:
            save_model(model, epoch, output_dir)

    if args.save_dir is not None:
        save_final(model, args.save_dir)
