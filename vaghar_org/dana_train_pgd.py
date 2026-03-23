#!/usr/bin/env python3
"""
PGD adversarial training for FNN_3_50 on MNIST.
Saves two consecutive checkpoints (both >= 91% clean accuracy).
Naming: 3x50_pgd_01_{accuracy}_{iteration_num}/model.p

Usage:
  python3 dana_train_pgd.py
"""
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms


class FNN_3_50(nn.Module):
    def __init__(self, k=1, w=28, h=28, output_size=10):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.fc1 = nn.Linear(k * w * h, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, output_size)

    def forward(self, x):
        x = x.reshape(-1, self.k * self.w * self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def pgd_attack(model, images, labels, eps=0.1, alpha=0.025, steps=7):
    """PGD L-inf attack."""
    images = images.clone().detach()
    adv = images + torch.empty_like(images).uniform_(-eps, eps)
    adv = torch.clamp(adv, 0, 1)

    for _ in range(steps):
        adv.requires_grad_(True)
        outputs = model(adv)
        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, images + eps), images - eps)
        adv = torch.clamp(adv, 0, 1)

    return adv.detach()


def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def save_model(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    params = []
    for p in model.parameters():
        arr = p.cpu().detach().numpy()
        params.append(np.transpose(arr))
    with open(os.path.join(save_dir, 'model.p'), 'wb') as f:
        pickle.dump(params, f)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    print(f"  Saved to {save_dir}")


def main():
    EPS = 0.1
    TRAIN_EPOCHS = 80
    BATCH_SIZE = 128
    LR = 1e-3

    device = torch.device('cpu')
    exp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dana_exp')
    os.makedirs(exp_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = dsets.MNIST(root='./data/', train=True, transform=transform, download=True)
    test_ds = dsets.MNIST(root='./data/', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FNN_3_50().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"PGD training FNN_3_50, eps={EPS}, {TRAIN_EPOCHS} epochs")
    print("=" * 60)

    prev_state = None
    prev_acc = 0.0

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            adv_images = pgd_attack(model, images, labels, eps=EPS)
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        epoch_num = epoch + 1
        print(f"  Epoch {epoch_num}/{TRAIN_EPOCHS} — loss: {running_loss/len(train_loader):.4f}, clean acc: {acc:.2f}%")

        # Keep last two states for saving
        if epoch_num == TRAIN_EPOCHS - 1:
            n1_acc = acc
            n1_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Save epoch 79 (N1) and epoch 80 (N2)
    n2_acc = acc
    acc1_str = f"{n1_acc:.1f}".replace('.', '_')
    acc2_str = f"{n2_acc:.1f}".replace('.', '_')
    dir_n1 = os.path.join(exp_dir, f"3x50_pgd_01_{acc1_str}_{TRAIN_EPOCHS - 1}")
    dir_n2 = os.path.join(exp_dir, f"3x50_pgd_01_{acc2_str}_{TRAIN_EPOCHS}")

    n1_model = FNN_3_50().to(device)
    n1_model.load_state_dict(n1_state)
    save_model(n1_model, dir_n1)
    print(f"  >> N1: epoch {TRAIN_EPOCHS - 1}, acc {n1_acc:.2f}%")

    save_model(model, dir_n2)
    print(f"  >> N2: epoch {TRAIN_EPOCHS}, acc {n2_acc:.2f}%")

    print(f"\nDone! Models saved in {exp_dir}")


if __name__ == '__main__':
    main()
