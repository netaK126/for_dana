import argparse
import numpy as np
import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import random_split
import os
from utils.models import *
from tqdm import tqdm
import matplotlib.pyplot as plt

def calc_conf(model,device,set,all_confidences):
    with torch.no_grad():
        for data, target in set:
            data = data.to(device).unsqueeze(0)
            output = model(data) # N(x)
            prob_correct = output[0][target]
        
            # Create mask to find max of other classes
            mask = torch.ones(10, dtype=torch.bool, device=device)
            mask[target] = False
            prob_others = output[0][mask]
            
            # Confidence formula
            confidence = prob_correct - torch.max(prob_others)
            all_confidences.append(confidence.item())
    return all_confidences


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='VeGHar Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="mnist", help='dataset')
    parser.add_argument('--c_tag', type=float, default=0, help='source')
    parser.add_argument('--model', type=str, default="4x10", help='3x10, 3x50, cnn1, or cnn2')
    parser.add_argument('--model_path', type=str, default="/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/model_itr17.pth", help='model')

    args = parser.parse_args()
    model_arch = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path=args.model_path
    if model_arch == "4x10":
        model = FNN_4_10()
    else:
        assert ("New model arch has been detected, please expand models.py and this if condition.")

    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = dsets.MNIST(root='./data/', train=True, transform=transform, download=True)
    test_set = dsets.MNIST(root='./data/',train=False, transform=transform, download=True)
    all_confidences = []

    all_confidences = calc_conf(model,device,train_set,all_confidences)
    all_confidences = calc_conf(model,device,test_set,all_confidences)


    # 4. Visualization
    indices = list(range(len(all_confidences)))

    plt.figure(figsize=(12, 6))

    # Use a scatter plot for 10,000 points
    # s=1 makes the dots small so they don't overlap too much
    plt.scatter(indices, all_confidences, s=1, alpha=0.5, color='blue')

    # Optional: Add a red line at 0 to show where the model starts failing
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Misclassification Threshold')
    plt.axhline(y=15.82, color='green', linestyle='--', linewidth=1, label='delta2_vaghar')
    plt.axhline(y=25.28, color='orange', linestyle='--', linewidth=1, label='delta1_vaghar+delta_diff')
    plt.axhline(y=37.85, color='pink', linestyle='--', linewidth=1, label='delta max')


    plt.title('Confidence Score for Each MNIST Test Sample')
    plt.xlabel('Sample Index')
    plt.ylabel('Confidence: $N(x)[c_{tag}] - \max_{j \\neq c_{tag}} N(x)[j]$')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Save the plot
    plt.savefig(r'/root/Downloads/lucid_delta_diff_with_perturbation/mnist_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.hist(all_confidences, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    # plt.axvline(0, color='red', linestyle='dashed', linewidth=1, label='Misclassification Threshold')
    # plt.title('Distribution of Model Confidence (Margin Score)')
    # plt.xlabel('Confidence: $N(x)[c_{tag}] - \max_{j \\neq c_{tag}} N(x)[j]$')
    # plt.ylabel('Frequency (Number of Samples)')
    # plt.legend()
    # plt.grid(axis='y', alpha=0.3)
    # plt.show()
    # plt.savefig(r'/root/Downloads/lucid_delta_diff_with_perturbation/mnist_confidence_distribution.png', dpi=300, bbox_inches='tight')

