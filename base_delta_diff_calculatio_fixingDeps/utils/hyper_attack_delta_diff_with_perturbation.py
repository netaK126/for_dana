import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import random_split
import os
from models import *
from tqdm import tqdm


# 1. Load MNIST Dataset
h_dim, w_dim, k_dim = 28, 28, 1
transform = transforms.Compose([transforms.ToTensor()])
trainset = dsets.MNIST(root='./data/', train=True, transform=transform, download=True)
testset = dsets.MNIST(root='./data/',train=False, transform=transform, download=True)


def calculate_conf(all_confidences, data_loader,c_tag):
    for data, target in data_loader:
        data=data.to(device)
        output = model(data) # N(x)
        prediction = torch.argmax(output, dim=1) 
        if prediction.item() != c_tag:
            continue
        prob_correct = output[0][c_tag]
        
        # Get all probabilities except the correct one
        mask = torch.ones(10, dtype=torch.bool)
        mask[c_tag] = False
        prob_others = output[0][mask]
        
        # Apply formula: N(x)[c_tag] - max(N(x)[j]) for j != c_tag
        confidence = prob_correct - torch.max(prob_others)
        all_confidences.append(confidence.item())
    return all_confidences


def get_delta_list(delta_path):
    try:
        with open(delta_path, 'r', encoding='utf-8') as file:
            line_dict = {i: round(float(line.split(',')[3].strip()),2) for i, line in enumerate(file)}
            return line_dict
    except FileNotFoundError:
        return "Error: The file was not found."
    
def get_delta_diff_list(delta_path):
    try:
        with open(delta_path, 'r', encoding='utf-8') as file:
            
            line_dict = {}
            for i, line in enumerate(file):
                line_dict[i]= round(float(line.split(',')[3].split("=")[-1].strip()),2)

            return line_dict
    except FileNotFoundError:
        return "Error: The file was not found."

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='VeGHar Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="mnist", help='dataset')
    parser.add_argument('--model', type=str, default="4x10", help='3x10, 3x50, cnn1, or cnn2')
    parser.add_argument('--model_path1', type=str, default="/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/model_itr17.pth", help='model')
    parser.add_argument('--model_path2', type=str, default="/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/model_itr18.pth", help='model')
    parser.add_argument('--delta_vagar_path1', type=str, default="/root/Downloads/vaghar_org/results/63904073920847_4x10_linf_0.02_ctag0_itr17_cTag1.txt")

    args = parser.parse_args()
    margin = 0.00001

    model_arch = args.model
    model_path1 = args.model_path1
    model_path2 = args.model_path2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = FNN_4_10()
    model1 = model1.to(device)
    model1.load_state_dict(torch.load(model_path1))
    model2 = FNN_4_10()
    model2 = model2.to(device)
    model2.load_state_dict(torch.load(model_path2))

    c_tag_list = [0]#list(range(10))
    c_target = 1

    delta1_vaghar_list = get_delta_list(args.delta_vagar_path1)
    delta_diff_list = get_delta_diff_list(args.delta_diff_path)

    for c_tag in c_tag_list:
    # 3. Evaluate Confidence
        delta_vaghar = delta1_vaghar_list[c_target-1]
        all_confidences = []

        with torch.no_grad():
            calculate_conf(all_confidences,testset,c_tag)
            calculate_conf(all_confidences,trainset,c_tag)

        count_lower_than_upper_bound = sum(1 for x in all_confidences if x < upper_bound)
        count_lower_than_delta_vaghar = sum(1 for x in all_confidences if x < delta_vaghar)

        # חישוב האחוז (כולל הגנה מחלוקה ב-0 אם הרשימה ריקה)
        if len(all_confidences) > 0:
            count_lower_than_upper_bound = (count_lower_than_upper_bound / len(all_confidences)) * 100
            count_lower_than_delta_vaghar = (count_lower_than_delta_vaghar / len(all_confidences)) * 100
        else:
            count_lower_than_upper_bound = 0
            count_lower_than_delta_vaghar = 0

        # 4. Visualization
        indices = list(range(len(all_confidences)))

        plt.figure(figsize=(12, 6))

        # Use a scatter plot for 10,000 points
        # s=1 makes the dots small so they don't overlap too much
        plt.scatter(indices, all_confidences, s=1, alpha=0.5, color='gray')

        # Optional: Add a red line at 0 to show where the model starts failing
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Misclassification Threshold')
        plt.axhline(y=delta_max, color='black', linestyle='--', linewidth=1, label='delta_max')
        plt.axhline(y=upper_bound, color='blue', linestyle='--', linewidth=1, label='delta_vaghar_itr18+delta_diff')
        plt.text(x=0.01,
                 y=upper_bound,
                 s=f'{count_lower_than_upper_bound:.2f}% below upper_bound', 
                color='black',
                fontsize=10,
                fontweight='bold',
                verticalalignment='bottom',
                transform=plt.gca().get_yaxis_transform(),
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2) )
        plt.axhline(y=delta_vaghar, color='green', linestyle='--', linewidth=1, label='delta_vaghar_itr17,c_target='+str(c_target))
        plt.text(x=0.01,
                 y=delta_vaghar-2,
                 s=f'{count_lower_than_delta_vaghar:.2f}% below delta_vaghar', 
                 color='black',
                 fontsize=10,
                 fontweight='bold',
                verticalalignment='bottom',
                transform=plt.gca().get_yaxis_transform(),
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2) )
        

        plt.title('Confidence Score for Each MNIST Sample, c_tag = '+str(c_tag) +", linf=0.02")
        plt.xlabel('Sample Index (0 - 9999)')
        plt.ylabel('Confidence: $N(x)[c_{tag}] - \max_{j \\neq c_{tag}} N(x)[j]$')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.savefig(f'/root/Downloads/code_deprecated_active_just_for_models/utils/confidence_comparison_cTag'+str(c_tag)+'_cTarget='+str(c_target)+'.png', dpi=300, bbox_inches='tight')
        plt.show()