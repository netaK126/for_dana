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
import torch
import torch.nn.functional as F

def compute_conf_margins(teacher_logits, student_logits, focus_on_diffs=True):
    """
    teacher_logits: [B, C]  (no_grad or detached)
    student_logits: [B, C]  (requires_grad=True)
    focus_on_diffs: if True, only keep samples where predictions differ

    Returns:
        teacher_conf: [N]
        student_conf: [N]
        (N = B if focus_on_diffs=False, or #mismatches if True)
    """
    with torch.no_grad():
        t_pred = teacher_logits.argmax(dim=1)        # [B]
        s_pred = student_logits.argmax(dim=1)        # [B]
        if focus_on_diffs:
            diff_mask = (t_pred != s_pred)           # [B]
        else:
            diff_mask = torch.ones_like(t_pred, dtype=torch.bool)

        # subset to selected samples (maybe mismatches only)
        t_logits = teacher_logits[diff_mask]         # [N, C]
        s_logits = student_logits[diff_mask]         # [N, C]
        t_pred_sub = t_pred[diff_mask]               # [N]

    # ---- teacher_conf ----
    t_selected = t_logits.gather(1, t_pred_sub.unsqueeze(1))   # [N, 1]

    t_others = t_logits.clone()
    t_others.scatter_(1, t_pred_sub.unsqueeze(1), float('-inf'))
    t_other_max, _ = t_others.max(dim=1, keepdim=True)         # [N, 1]

    teacher_conf = (t_selected - t_other_max).squeeze(1)       # [N]

    # ---- student_conf (using teacher's indices) ----
    s_selected = s_logits.gather(1, t_pred_sub.unsqueeze(1))   # [N, 1]

    s_others = s_logits.clone()
    s_others.scatter_(1, t_pred_sub.unsqueeze(1), float('-inf'))
    s_other_max, _ = s_others.max(dim=1, keepdim=True)         # [N, 1]

    student_conf = (s_selected - s_other_max).squeeze(1)       # [N]

    return teacher_conf, student_conf



def distillation_loss(student_logits, teacher_logits, labels, T, alpha,conf_lambda,gamma):
    """
    Computes the total distillation loss.
    """
    hard_loss = F.cross_entropy(student_logits, labels)

    soft_teacher_prob = F.softmax(teacher_logits / T, dim=1)
    soft_student_log_prob = F.log_softmax(student_logits / T, dim=1)
    distil_loss = F.kl_div(soft_student_log_prob, soft_teacher_prob, reduction='batchmean')
    scaled_distil_loss = distil_loss * (T * T)

    teacher_conf, student_conf = compute_conf_margins(teacher_logits, student_logits)
    errors = (student_conf - teacher_conf).abs()   # [N]
    if errors.numel() == 0:
        conf_loss = torch.tensor(0.0, device=device)
    else:
        weights = torch.softmax(gamma * errors, dim=0)  # [N], sum to 1
        conf_loss = (weights * (errors ** 2)).sum()


        # 3. Total Loss
    total_loss = alpha * hard_loss + (1 - alpha) * scaled_distil_loss + conf_lambda*conf_loss
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

def distill_student(student_model, teacher_model, loader, epochs, optimizer, t, alpha,conf_lambda,gamma):
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
            loss = distillation_loss(student_logits, teacher_logits, target, t, alpha,conf_lambda,gamma)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader.dataset)
        print(f'Distillation Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}')
        a = [np.transpose(p.cpu().detach().numpy()) for p in student_distilled.parameters()]
        pickle.dump(a, open(output_dir + "GeminiAlgorithm_CEandKLloss_"+"alphaVal"+str(alpha) + "_Tval" + str(t)+"_lambdaConf" + str(conf_lambda) +"_gamma"+str(gamma) + "_EpochNum" + str(epoch) + "_model.p", "wb"))
        torch.save(student_distilled.state_dict(), output_dir + "GeminiAlgorithm_CEandKLloss_"+"alphaVal"+str(alpha) + "_Tval" + str(t)+"_lambdaConf" + str(conf_lambda) +"_gamma"+str(gamma)+ "_EpochNum" + str(epoch) + "_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default="/root/Downloads/code_deprecated_active_just_for_models/models/4x10_GeminiDistilation_checkingWithConfLoss/")
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
    t_list = [i for i in range(21) if i>=11]
    alpha_list = [0.3, 0.4, 0.6, 0.75, 0.9]
    conf_lambda_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
    gamma_values = [1.0, 3.0, 5.0, 8.0]
    for t in t_list:
        for gamma in gamma_values:
            for conf_lambda in conf_lambda_list:
                for alpha in alpha_list:
                    distill_optimizer = optim.Adam(student_distilled.parameters(), lr=1e-3, weight_decay=1e-4)
                    distill_student(student_distilled, teacher, train_loader, num_epochs, distill_optimizer, t, alpha,conf_lambda,gamma)
                    distilled_acc = evaluate_model(student_distilled, test_loader)
                    print(f"--- Final Results ---")
                    print(f"Teacher Accuracy: {teacher_acc:.2f}%")
                    print(f"Student Distilled (SOTA) Accuracy: {distilled_acc:.2f}%")