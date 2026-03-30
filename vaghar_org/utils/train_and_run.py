#!/usr/bin/env python3
import argparse
import os
import pickle
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Import all network definitions from models.py
from models import FNN_3_10, FNN_3_50, FNN_3_100, FNN_4_10, FNN_10_10, CNN0, CNN1, CNN2


def build_model(model_name, k=1, w=28, h=28):
    if model_name == "3x10":
        return FNN_3_10(k=k, w=w, h=h)
    if model_name == "3x50":
        return FNN_3_50(k=k, w=w, h=h)
    if model_name == "3x100":
        return FNN_3_100(k=k, w=w, h=h)
    if model_name == "4x10":
        return FNN_4_10(k=k, w=w, h=h)
    if model_name == "10x10":
        return FNN_10_10(k=k, w=w, h=h)
    if model_name == "cnn0":
        return CNN0(k=k, w=w, h=h)
    if model_name == "cnn1":
        return CNN1(k=k, w=w, h=h)
    if model_name == "cnn2":
        return CNN2(k=k, w=w, h=h)
    raise ValueError(f"Unsupported model: {model_name}")


def parse_model_parameters(model):
    # same order as assumptions in MIPVerify loader (weights / biases in parameter order)
    params = []
    for p in model.parameters():
        arr = p.cpu().detach().numpy()
        arr = np.transpose(arr)
        params.append(arr)
    return params


def save_model_checkpoint(model, checkpoint_dir, itr):
    os.makedirs(checkpoint_dir, exist_ok=True)
    target_dir = os.path.join(checkpoint_dir, str(itr))
    os.makedirs(target_dir, exist_ok=True)
    params_list = parse_model_parameters(model)
    with open(os.path.join(target_dir, "model.p"), "wb") as f:
        pickle.dump(params_list, f)
    torch.save(model.state_dict(), os.path.join(target_dir, "model.pth"))


def train(model_name, output_dir, epochs=20, batch_size=128, loss_type="Cross", optimizer_type="Adam"):
    device = torch.device("cpu")#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = dsets.MNIST(root="./data/", train=True, transform=transform, download=True)
    mnist_test = dsets.MNIST(root="./data/", train=False, transform=transform, download=True)

    model = build_model(model_name).to(device)

    if loss_type == "Cross":
        criterion = nn.CrossEntropyLoss().to(device)
    elif loss_type == "MSE":
        criterion = nn.MSELoss().to(device)
    elif loss_type == "L1":
        criterion = nn.L1Loss().to(device)
    else:
        raise ValueError("Unsupported loss")

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-4)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    else:
        raise ValueError("Unsupported optimizer")

    os.makedirs(output_dir, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if loss_type == "L1":
                labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float()
                loss = criterion(outputs, labels_onehot)
            else:
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1} finished. Test accuracy: {acc:.2f}%")

        save_model_checkpoint(model, output_dir, epoch)

    # return path to highest checkpoint
    return os.path.join(output_dir, str(epochs - 1), "model.p")


def run_vaghar(model_path,
               model_name='4x10',
               output_dir='./results_PerturbationInterval/',
               dataset='mnist',
               perturbation='linf',
               perturbation_size='0.05',
               ctag=1,
               ct='2,3,4,5,6,7,8,9',
               timeout=4000,
               mode='standard',
               use_hyper_attack=True,
               use_perturbed_intervals=False,
               activate_vaghgar_deps=True,
               c_tag_mode=False):

    args = [
        'julia', 'run.jl',
        '--dataset', dataset,
        '--model_name', model_name,
        '--model_path', model_path,
        '--perturbation', perturbation,
        '--perturbation_size', perturbation_size,
        '--ctag', str(ctag),
        '--ct', str(ct),
        '--timout', str(timeout),
        '--output_dir', output_dir,
        '--mode', mode,
    ]

    args += ['--use_hyper_attack', str(use_hyper_attack).lower()]
    args += ['--use_perturbed_intervals', str(use_perturbed_intervals).lower()]
    args += ['--activate_vaghgar_deps', str(activate_vaghgar_deps).lower()]
    args += ['--c_tag_mode', str(c_tag_mode).lower()]

    proc = subprocess.run(args, cwd=os.path.dirname(os.path.abspath(__file__)) + '/..', check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"run.jl failed with exit code {proc.returncode}")

def main():
    parser = argparse.ArgumentParser(description='Train VHAGaR model and run verification')
    parser.add_argument('--model_name', default='3x10',help='Model name: 3x10, 3x50, 4x10, 10x10, cnn0, cnn1, cnn2')
    parser.add_argument('--output_model_dir', default='/root/Downloads/vaghar_org/models_as_in_vaghar/', help='Model checkpoints output directory')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--timout', type=int, default=4000)
    parser.add_argument('--output_result_dir', default='/root/Downloads/vaghar_org/vaghgar_withPerturbedIntervals_pgd_results_itr19/')
    parser.add_argument('--mode', default='standard')
    parser.add_argument('--use_hyper_attack',default=True, action='store_true')
    parser.add_argument('--use_perturbed_intervals', default=True, action='store_true')
    parser.add_argument('--activate_vaghgar_deps',default=True, action='store_true')
    parser.add_argument('--c_tag_mode', action='store_true')
    args = parser.parse_args()
    optimizer_type = "SGD"
    if args.model_name=="3X50":
        optimizer_type="SGD"
    model_folder = os.path.join(args.output_model_dir, args.model_name + '_mnist')
    best_model_path = r"/root/Downloads/vaghar_org/models_as_in_vaghar/3x10_mnist_sgd/19/model.p"
    # best_model_path = train(
    #     args.model_name,
    #     model_folder,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     optimizer_type=optimizer_type,
    # )

    print(f"Trained model saved to {best_model_path}")

    c_tags = [1,2,3,4,5,6,7,8,9,10]
    c_targets = "1,2,3,4,5,6,7,8,9,10"
    perturbations = {"linf": [0.025]}
    for perturbation, sizes in perturbations.items():
        for size in sizes:
            for c_tag_ in c_tags:
                print(f"Running VHAGaR for c_tag={c_tag_}")
                run_vaghar(
                    model_path=best_model_path,
                    model_name=args.model_name,
                    output_dir=args.output_result_dir,
                    dataset='mnist',
                    perturbation=str(perturbation),
                    perturbation_size=str(size),
                    ctag=c_tag_,
                    ct=c_targets,
                    timeout=args.timout,
                    mode=args.mode,
                    use_hyper_attack=args.use_hyper_attack,
                    use_perturbed_intervals=args.use_perturbed_intervals,
                    activate_vaghgar_deps=args.activate_vaghgar_deps,
                    c_tag_mode=args.c_tag_mode,
                )

if __name__ == '__main__':
    main()
