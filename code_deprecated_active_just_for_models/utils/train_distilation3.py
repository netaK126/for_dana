import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from models import *
import pickle
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---  Knowledge Distillation Loss ---

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    Computes the total distillation loss.
    """
    # 1. Hard Target Loss (Standard Cross-Entropy)
    hard_loss = F.cross_entropy(student_logits, labels)

    # 2. Soft Target Loss (KL Divergence)
    # Apply temperature to logits
    soft_teacher_prob = F.softmax(teacher_logits / T, dim=1)
    soft_student_log_prob = F.log_softmax(student_logits / T, dim=1)
    
    # KL Divergence (PyTorch's NLLLoss on log_softmax and softmax)
    distil_loss = F.kl_div(soft_student_log_prob, soft_teacher_prob, reduction='batchmean')
    
    # Scale the distillation loss by T^2 as per Hinton et al.
    scaled_distil_loss = distil_loss * (T * T)

    # 3. Total Loss
    total_loss = alpha * hard_loss + (1 - alpha) * scaled_distil_loss
    return total_loss

# --- 3. Data Setup (Standard MNIST) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --- 4. Training and Distillation Functions ---

def train_model(model, loader, epochs, optimizer, criterion):
    model.train()
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Train Epoch {epoch+1}/{epochs} completed.')

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(loader.dataset)
    print(f'\nTest set: Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def distill_student(student_model, teacher_model, loader, epochs, optimizer, t, alpha):
    student_model.train()
    teacher_model.eval() # Teacher must be in evaluation mode
    for epoch in range(epochs):
        total_loss = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Get logits from both models
            student_logits = student_model(data)
            with torch.no_grad():
                teacher_logits = teacher_model(data)

            # Compute the combined distillation loss
            loss = distillation_loss(student_logits, teacher_logits, target, t, alpha)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader.dataset)
        print(f'Distillation Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}')
        a = [np.transpose(p.cpu().detach().numpy()) for p in student_distilled.parameters()]
        pickle.dump(a, open(output_dir + "GeminiAlgorithm_CEandKLloss_"+"alphaVal"+str(alpha) + "_Tval" + str(t) + "_EpochNum" + str(epoch) + "_model.p", "wb"))
        torch.save(student_distilled.state_dict(), output_dir + "GeminiAlgorithm_CEandKLloss_"+"alphaVal"+str(alpha) + "_Tval" + str(t) + "_EpochNum" + str(epoch) + "_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default="/root/Downloads/code_deprecated_active_just_for_models/models/4x10_GeminiDistilation/")
    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.epochs
    output_dir = args.output_dir
    # Initialize models
    teacher = FNN_4_10().to(device)
    student_distilled = FNN_4_10().to(device)
    # Use a complex optimizer for the teacher
    teacher.load_state_dict(torch.load(r"/root/Downloads/code_deprecated_active_just_for_models/models/4x10/19/model.pth"))
    teacher.eval()
    teacher_acc = evaluate_model(teacher, test_loader)
    # Expected Teacher Accuracy: ~99.3% to 99.5%

    print("--- Distilling Knowledge into Student ---")
    # Hyperparameters for SOTA KD on MNIST
    t_list = [i for i in range(21) if i>=2]
    alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    for t in t_list:
        for alpha in alpha_list:
            distill_optimizer = optim.Adam(student_distilled.parameters(), lr=0.01)
            distill_student(student_distilled, teacher, train_loader, num_epochs, distill_optimizer, t, alpha)
            distilled_acc = evaluate_model(student_distilled, test_loader)
            # Expected Distilled Student Accuracy: ~98.5% to 99.0% 
            print(f"--- Final Results ---")
            print(f"Teacher Accuracy: {teacher_acc:.2f}%")
            print(f"Student Distilled (SOTA) Accuracy: {distilled_acc:.2f}%")