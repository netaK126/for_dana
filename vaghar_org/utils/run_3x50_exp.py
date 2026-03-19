#!/usr/bin/env python3
"""
3x50 experiment pipeline:
  1. Train 3x50 model for 19 epochs, save checkpoints at epoch 18 and 19
  2. Run VHAGaR standard (with perturbed intervals) for both models
  3. Run VHAGaR transfer: N1=itr18, N2=itr19
"""
import argparse
import os
import pickle
import re
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from models import FNN_3_50

# ── paths ────────────────────────────────────────────────────────────────
EXP_DIR = os.path.join(os.path.dirname(__file__), '..', '3x50_exp')
MODEL_18_DIR = os.path.join(EXP_DIR, 'model_18_itr')
MODEL_19_DIR = os.path.join(EXP_DIR, 'model_19_itr')
VAGHAR_18_DIR = os.path.join(EXP_DIR, 'vagharWithPerturbed_3x50_itr18')
VAGHAR_19_DIR = os.path.join(EXP_DIR, 'vagharWithPerturbed_3x50_itr19')
TRANSFER_DIR = os.path.join(EXP_DIR, 'transfer_3x50_N1_is_itr18')
RUN_JL_DIR = os.path.join(os.path.dirname(__file__), '..')

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


def get_data_loaders(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = dsets.MNIST(root='./data/', train=True, transform=transform, download=True)
    test_ds = dsets.MNIST(root='./data/', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_julia(args_list, step_name):
    """Run julia run.jl with given arguments."""
    cmd = ['julia', 'run.jl'] + args_list
    print(f"\n  Running: {' '.join(cmd[:6])}...")
    proc = subprocess.run(cmd, cwd=RUN_JL_DIR)
    if proc.returncode != 0:
        print(f"  WARNING: {step_name} exited with code {proc.returncode}")
    return proc.returncode


# ── step 1: train 3x50 model ────────────────────────────────────────────

def train_3x50(epochs=19, batch_size=128, lr=1e-3):
    """
    Train a 3x50 model for `epochs` epochs.
    Save checkpoints at epoch 18 and epoch 19 (1-indexed).
    """
    print("=" * 60)
    print(f"STEP 1: Training 3x50 model for {epochs} epochs")
    print("=" * 60)
    device = torch.device("cpu")#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FNN_3_50().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_data_loaders(batch_size)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        epoch_num = epoch + 1  # 1-indexed
        print(f"  Epoch {epoch_num}/{epochs} — loss: {running_loss/len(train_loader):.4f}, acc: {acc:.2f}%")

        if epoch_num == 18:
            save_model(model, MODEL_18_DIR)
            print(f"  >> Saved itr18 checkpoint (acc: {acc:.2f}%)")
        if epoch_num == 19:
            save_model(model, MODEL_19_DIR)
            print(f"  >> Saved itr19 checkpoint (acc: {acc:.2f}%)")

    return model


# ── step 2: run VHAGaR standard with perturbed intervals ────────────────

def run_vaghar_standard(model_path, output_dir, ctag, perturbation_size='0.05',
                        ct='1,2,3,4,5,6,7,8,9,10', timeout=1000, perturbation='linf'):
    """Run VHAGaR in standard mode with hyper attack, vaghar deps, and perturbed intervals."""
    args = [
        '--mode', 'standard',
        '--dataset', 'mnist',
        '--model_name', '3x50',
        '--model_path', model_path,
        '--perturbation', perturbation,
        '--perturbation_size', perturbation_size,
        '--ctag', str(ctag),
        '--ct', ct,
        '--timout', str(timeout),
        '--output_dir', output_dir + '/',
        '--c_tag_mode', 'false',
        '--use_hyper_attack', 'true',
        '--activate_vaghgar_deps', 'true',
        '--use_perturbed_intervals', 'true',
    ]
    return run_julia(args, f'VHAGaR standard 3x50 (ctag={ctag})')


def step2_vaghar_standard(perturbation_size, ctag, ct, timeout, perturbation):
    print("=" * 60)
    print("STEP 2: Running VHAGaR standard for 3x50 itr18 and itr19")
    print("=" * 60)

    model_18_path = os.path.join(MODEL_18_DIR, 'model.p')
    model_19_path = os.path.join(MODEL_19_DIR, 'model.p')

    print(f"\n  --- itr18 (ctag={ctag}) ---")
    run_vaghar_standard(model_18_path, VAGHAR_18_DIR, ctag,
                        perturbation_size=perturbation_size, ct=ct,
                        timeout=timeout, perturbation=perturbation)

    print(f"\n  --- itr19 (ctag={ctag}) ---")
    run_vaghar_standard(model_19_path, VAGHAR_19_DIR, ctag,
                        perturbation_size=perturbation_size, ct=ct,
                        timeout=timeout, perturbation=perturbation)


# ── step 3: run VHAGaR transfer (N1=itr18, N2=itr19) ────────────────────

def run_transfer_from_results(vaghar_results_dir, output_dir, timeout, perturbation, ct):
    """
    Iterate over VHAGaR results files for N1 (itr18).
    Each file contains delta_1 values for a specific perturbation_size and c_tag.
    Parse these from the filename and launch a transfer run.
    """
    pattern = re.compile(rf"_{perturbation}_(.*?)_ctag.*cTag(\d+)")

    if not os.path.exists(vaghar_results_dir):
        print(f"  Error: Directory {vaghar_results_dir} not found.")
        return

    n1_path = os.path.join(MODEL_18_DIR, 'model.p')
    n2_path = os.path.join(MODEL_19_DIR, 'model.p')

    for filename in sorted(os.listdir(vaghar_results_dir)):
        if not filename.endswith('.txt'):
            continue
        match = pattern.search(filename)
        if not match:
            continue

        perturbation_size = match.group(1)
        c_tag_n = match.group(2)
        vaghar_results_path = os.path.join(vaghar_results_dir, filename)

        print(f"  Processing: {filename}  (eps={perturbation_size}, ctag={c_tag_n})")

        command = [
            '--mode', 'transfer',
            '--dataset', 'mnist',
            '--model_name', '3x50',
            '--model_path', n1_path,
            '--model_path2', n2_path,
            '--vaghar_results', vaghar_results_path,
            '--perturbation', perturbation,
            '--perturbation_size', perturbation_size,
            '--ctag', c_tag_n,
            '--ct', ct,
            '--timout', str(timeout),
            '--output_dir', output_dir + '/',
            '--c_tag_mode', 'false',
            '--use_hyper_attack', 'true',
            '--activate_vaghgar_deps', 'true',
            '--use_intervals', 'true',
            '--use_perturbed_intervals', 'true',
            '--n2_fewer_binars_encoding', 'true',
        ]
        run_julia(command, f'transfer (ctag={c_tag_n})')


def step3_transfer(timeout, perturbation):
    print("=" * 60)
    print("STEP 3: Running VHAGaR transfer (N1=itr18, N2=itr19)")
    print("=" * 60)
    run_transfer_from_results(VAGHAR_18_DIR, TRANSFER_DIR, timeout, perturbation, ct)


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='3x50 experiment: train itr18/itr19, VHAGaR standard + transfer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=19, help='Total training epochs (saves at 18 and 19)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--perturbation_size', type=str, default='0.05')
    parser.add_argument('--perturbation', type=str, default='linf')
    parser.add_argument('--ct', type=str, default='4,5,6,7', help='Target classes')
    parser.add_argument('--timeout', type=int, default=6000, help='MIP timeout per class pair')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, use existing models')
    parser.add_argument('--skip_vaghar', action='store_true', help='Skip standard VHAGaR, go to transfer')
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    # Step 1: Train
    if not args.skip_training:
        train_3x50(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        print("Skipping training (--skip_training)")

    # Step 2: VHAGaR standard for both models
    if not args.skip_vaghar:
        for ctag in range(1, 3):
            step2_vaghar_standard(args.perturbation_size, ctag, args.ct, args.timeout, args.perturbation)
    else:
        print("Skipping standard VHAGaR (--skip_vaghar)")

    # Step 3: Transfer (N1=itr18, N2=itr19)
    step3_transfer(args.timeout, args.perturbation, args.ct)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  itr18 model:      {MODEL_18_DIR}/model.p")
    print(f"  itr19 model:      {MODEL_19_DIR}/model.p")
    print(f"  VHAGaR itr18:     {VAGHAR_18_DIR}/")
    print(f"  VHAGaR itr19:     {VAGHAR_19_DIR}/")
    print(f"  Transfer results: {TRANSFER_DIR}/")


if __name__ == '__main__':
    main()
