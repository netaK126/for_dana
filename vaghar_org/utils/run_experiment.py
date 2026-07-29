#!/usr/bin/env python3
"""
Shared model registry, dataset config, and save/eval helpers.

Library module used by run_relaxation_sweep.py (Phase 0 training / N2
derivation). The old standalone standard+transfer experiment driver was
removed together with the transfer mode.
"""
import os
import pickle

import numpy as np
import torch
import torchvision.datasets as dsets

from models import (
    FNN_2_10, FNN_3_10, FNN_3_50, FNN_3_100,
    FNN_4_10, FNN_5_10, FNN_5_50, FNN_6_10, FNN_6_100,
    FNN_9_200, FNN_10_10,
    CNN0, CNN1, CNN2, CNN3, CNN4, CNN5,
    FNN_ACAS, FNN_HAR,
)

# ── architecture registry ────────────────────────────────────────────────
# Maps arch name -> (model class, model_name for julia's --model_name)
ARCH_REGISTRY = {
    '2x10':  (FNN_2_10,  '2x10'),
    '3x10':  (FNN_3_10,  '3x10'),
    '3x50':  (FNN_3_50,  '3x50'),
    '3x100': (FNN_3_100, '3x100'),
    '4x10':  (FNN_4_10,  '4x10'),
    '5x10':  (FNN_5_10,  '5x10'),
    '5x50':  (FNN_5_50,  '5x50'),
    '6x10':  (FNN_6_10,  '6x10'),
    '6x100': (FNN_6_100, '6x100'),
    '9x200': (FNN_9_200, '9x200'),
    '10x10': (FNN_10_10, '10x10'),
    'cnn0':  (CNN0,      'cnn0'),
    'cnn1':  (CNN1,      'cnn1'),
    'cnn2':  (CNN2,      'cnn2'),
    'cnn3':  (CNN3,      'cnn3'),
    'cnn4':  (CNN4,      'cnn4'),
    'cnn5':  (CNN5,      'cnn5'),
    # Pretrained tabular benchmark nets (only under --internet_nets_benchmarks).
    'acas':  (FNN_ACAS,  'acas'),
    'har':   (FNN_HAR,   'har'),
}

# ── dataset config ───────────────────────────────────────────────────────
# Maps dataset name -> (torchvision class, channels, width, height, julia dataset name)
DATASET_CONFIG = {
    'mnist':         (dsets.MNIST,        1, 28, 28, 'mnist'),
    'fashion_mnist': (dsets.FashionMNIST, 1, 28, 28, 'fashion_mnist'),
    'cifar10':       (dsets.CIFAR10,      3, 32, 32, 'cifar10'),
    # Pretrained tabular benchmark nets: no torchvision dataset (sentinel None),
    # channels/width/height encode the flat input dim (channels*w*h). Only the
    # julia-name and input-dim fields are used on the internet-nets path; the
    # torchvision slot must never be dereferenced for these.
    'acas':          (None,               1, 5,   1,  'acas'),
    'har':           (None,               1, 561, 1,  'har'),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_JL_DIR = os.path.join(SCRIPT_DIR, '..')


def _get_device(force_cpu=False):
    """Return cuda device if available and compatible, otherwise cpu."""
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        # Test that CUDA actually works (catches driver/arch mismatch)
        torch.zeros(1, device="cuda:0")
        return torch.device("cuda:0")
    except RuntimeError:
        print("  WARNING: CUDA available but not functional (driver/arch mismatch), falling back to CPU")
        return torch.device("cpu")


# ── helpers ──────────────────────────────────────────────────────────────

def save_model(model, save_dir):
    """Save model in both .p (pickle, for MIPVerify) and .pth (PyTorch) formats."""
    os.makedirs(save_dir, exist_ok=True)
    params = []
    for p in model.parameters():
        arr = p.cpu().detach().numpy()
        params.append(np.transpose(arr))
    with open(os.path.join(save_dir, 'model.p'), 'wb') as f:
        pickle.dump(params, f)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    print(f"  Model saved to {save_dir}")


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

