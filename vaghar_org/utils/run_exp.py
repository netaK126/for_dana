#!/usr/bin/env python3
"""
Generic experiment pipeline for any architecture:
  1. Train model until two consecutive epochs reach 92%+ accuracy (max 80 epochs)
  2. Run VHAGaR standard (with perturbed intervals) for both models
  3. Run VHAGaR transfer: N1=itr_i, N2=itr_{i+1}

Usage examples:
  python utils/run_exp.py --arch 6x10
  python utils/run_exp.py --arch cnn0 --perturbation_size 0.03 --timeout 8000
  python utils/run_exp.py --arch 3x50 --skip_training
  python utils/run_exp.py --arch 3x10 --skip_vaghar
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

from models import (
    FNN_3_10, FNN_3_50, FNN_6_10,
    CNN0, CNN1, CNN2, CNN3,
)

# ── architecture registry ────────────────────────────────────────────────
# Maps arch name -> (model class, model_name for julia's --model_name)
ARCH_REGISTRY = {
    '3x10':  (FNN_3_10, '3x10'),
    '3x50':  (FNN_3_50, '3x50'),
    '6x10':  (FNN_6_10, '6x10'),
    'cnn0':  (CNN0,     'cnn0'),
    'cnn1':  (CNN1,     'cnn1'),
    'cnn2':  (CNN2,     'cnn2'),
    'cnn3':  (CNN3,     'cnn3'),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_JL_DIR = os.path.join(SCRIPT_DIR, '..')


def get_exp_dirs(arch, itr_n1, itr_n2, create=False):
    """Return a dict of all experiment directories for the given architecture.
    If create=True, ensure all directories exist."""
    exp_dir = os.path.join(SCRIPT_DIR, '..', f'{arch}_exp')
    dirs = {
        'exp':         exp_dir,
        'model_n1':    os.path.join(exp_dir, f'model_{itr_n1}_itr'),
        'model_n2':    os.path.join(exp_dir, f'model_{itr_n2}_itr'),
        'vaghar_n1':   os.path.join(exp_dir, f'vagharWithPerturbed_{arch}_itr{itr_n1}'),
        'vaghar_n2':   os.path.join(exp_dir, f'vagharWithPerturbed_{arch}_itr{itr_n2}'),
        'transfer':    os.path.join(exp_dir, f'transfer_{arch}_N1_is_itr{itr_n1}'),
    }
    if create:
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
    return dirs


def detect_iterations(arch):
    """Detect itr_n1 and itr_n2 from existing model_*_itr folders in the exp directory.
    Returns (itr_n1, itr_n2) where itr_n2 == itr_n1 + 1."""
    exp_dir = os.path.join(SCRIPT_DIR, '..', f'{arch}_exp')
    if not os.path.exists(exp_dir):
        print(f"ERROR: Experiment directory {exp_dir} not found.")
        sys.exit(1)

    pattern = re.compile(r'^model_(\d+)_itr$')
    itrs = sorted(
        int(m.group(1))
        for name in os.listdir(exp_dir)
        if (m := pattern.match(name)) and os.path.isdir(os.path.join(exp_dir, name))
    )

    # Find consecutive pair
    for i in range(len(itrs) - 1):
        if itrs[i + 1] == itrs[i] + 1:
            print(f"  Detected model checkpoints: itr{itrs[i]} and itr{itrs[i+1]}")
            return itrs[i], itrs[i + 1]

    print(f"ERROR: Could not find two consecutive model_*_itr folders in {exp_dir}.")
    print(f"  Found: {itrs}")
    sys.exit(1)


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


# ── step 1: train ────────────────────────────────────────────────────────

MIN_ACCURACY = 91.0
MAX_EPOCHS = 100


def train_model(arch, batch_size=128, lr=1e-3):
    """
    Train until two consecutive epochs both reach MIN_ACCURACY (92%).
    Save checkpoints for those two consecutive epochs.
    Exit if MAX_EPOCHS (80) is exceeded without finding such a pair.
    Returns (itr_n1, itr_n2) — the 1-indexed epoch numbers of the saved models.
    """
    model_cls, _ = ARCH_REGISTRY[arch]

    print("=" * 60)
    print(f"STEP 1: Training {arch} (until two consecutive epochs >= {MIN_ACCURACY}% acc, max {MAX_EPOCHS})")
    print("=" * 60)

    device = torch.device("cpu")#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_cls().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_data_loaders(batch_size)

    prev_acc = 0.0
    prev_state = None

    for epoch in range(MAX_EPOCHS):
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
        epoch_num = epoch + 1
        print(f"  Epoch {epoch_num}/{MAX_EPOCHS} — loss: {running_loss/len(train_loader):.4f}, acc: {acc:.2f}%")

        if prev_acc >= MIN_ACCURACY and acc >= MIN_ACCURACY:
            itr_n1 = epoch_num - 1
            itr_n2 = epoch_num
            dirs = get_exp_dirs(arch, itr_n1, itr_n2, create=True)
            # Save previous epoch from cached state
            prev_model = model_cls().to(device)
            prev_model.load_state_dict(prev_state)
            save_model(prev_model, dirs['model_n1'])
            print(f"  >> Saved itr{itr_n1} checkpoint (acc: {prev_acc:.2f}%)")
            # Save current epoch
            save_model(model, dirs['model_n2'])
            print(f"  >> Saved itr{itr_n2} checkpoint (acc: {acc:.2f}%)")
            return itr_n1, itr_n2

        prev_acc = acc
        prev_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\n  ERROR: Failed to find two consecutive epochs with >= {MIN_ACCURACY}% accuracy within {MAX_EPOCHS} epochs.")
    sys.exit(1)


# ── step 2: run VHAGaR standard with perturbed intervals ────────────────

def run_vaghar_standard(arch, model_path, output_dir, ctag,
                        perturbation_size='0.05', ct='1,2,3,4,5,6,7,8,9,10',
                        timeout=1000, perturbation='linf'):
    """Run VHAGaR in standard mode with hyper attack, vaghar deps, and perturbed intervals."""
    _, model_name = ARCH_REGISTRY[arch]
    args = [
        '--mode', 'standard',
        '--dataset', 'mnist',
        '--model_name', model_name,
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
    return run_julia(args, f'VHAGaR standard {arch} (ctag={ctag})')


def step2_vaghar_standard(arch, itr_n1, itr_n2, perturbation_size, ctag, ct, timeout, perturbation):
    dirs = get_exp_dirs(arch, itr_n1, itr_n2)
    print("=" * 60)
    print(f"STEP 2: Running VHAGaR standard for {arch} itr{itr_n1} and itr{itr_n2}")
    print("=" * 60)

    model_n1_path = os.path.join(dirs['model_n1'], 'model.p')
    model_n2_path = os.path.join(dirs['model_n2'], 'model.p')

    print(f"\n  --- itr{itr_n1} (ctag={ctag}) ---")
    run_vaghar_standard(arch, model_n1_path, dirs['vaghar_n1'], ctag,
                        perturbation_size=perturbation_size, ct=ct,
                        timeout=timeout, perturbation=perturbation)

    print(f"\n  --- itr{itr_n2} (ctag={ctag}) ---")
    run_vaghar_standard(arch, model_n2_path, dirs['vaghar_n2'], ctag,
                        perturbation_size=perturbation_size, ct=ct,
                        timeout=timeout, perturbation=perturbation)


# ── step 3: run VHAGaR transfer ──────────────────────────────────────────

def run_transfer_from_results(arch, itr_n1, itr_n2, vaghar_results_dir, output_dir, timeout, perturbation, ct, transfer_relaxations):
    """
    Iterate over VHAGaR results files for N1.
    Each file contains delta_1 values for a specific perturbation_size and c_tag.
    Parse these from the filename and launch a transfer run.
    """
    _, model_name = ARCH_REGISTRY[arch]
    dirs = get_exp_dirs(arch, itr_n1, itr_n2)
    pattern = re.compile(rf"_{perturbation}_(.*?)_ctag.*cTag(\d+)")

    if not os.path.exists(vaghar_results_dir):
        print(f"  Error: Directory {vaghar_results_dir} not found.")
        return

    n1_path = os.path.join(dirs['model_n1'], 'model.p')
    n2_path = os.path.join(dirs['model_n2'], 'model.p')

    for filename in sorted(os.listdir(vaghar_results_dir)):
        if not filename.endswith('.txt'):
            continue
        match = pattern.search(filename)
        if not match:
            continue
        if "0.25" not in filename:
            continue

        perturbation_size = match.group(1)
        c_tag_n = match.group(2)
        vaghar_results_path = os.path.join(vaghar_results_dir, filename)

        print(f"  Processing: {filename}  (eps={perturbation_size}, ctag={c_tag_n})")

        command = [
            '--mode', 'transfer',
            '--dataset', 'mnist',
            '--model_name', model_name,
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
            "--use_relaxations", transfer_relaxations
        ]
        run_julia(command, f'transfer {arch} (ctag={c_tag_n})')


def step3_transfer(arch, itr_n1, itr_n2, timeout, perturbation, ct, transfer_relaxations):
    dirs = get_exp_dirs(arch, itr_n1, itr_n2)
    print("=" * 60)
    print(f"STEP 3: Running VHAGaR transfer for {arch} (N1=itr{itr_n1}, N2=itr{itr_n2})")
    print("=" * 60)
    run_transfer_from_results(arch, itr_n1, itr_n2, dirs['vaghar_n1'], dirs['transfer'], timeout, perturbation, ct, transfer_relaxations)


# ── main ─────────────────────────────────────────────────────────────────

def main():
    arch_choices = list(ARCH_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description='Generic experiment: train until consecutive 92%+ epochs, VHAGaR standard + transfer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--arch', type=str, required=True, choices=arch_choices,
                        help=f'Architecture to run: {arch_choices}')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3, help='SGD learning rate')
    parser.add_argument('--perturbation_size', type=str, default='0.25')
    parser.add_argument('--perturbation', type=str, default='linf')
    parser.add_argument('--ct', type=str, default='4,5', help='Target classes')
    parser.add_argument('--timeout', type=int, default=2000, help='MIP timeout per class pair')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, use existing models')
    parser.add_argument('--skip_vaghar', action='store_true', help='Skip standard VHAGaR, go to transfer')
    parser.add_argument('--transfer_relaxations', type=str, default='false', help='running transfer with relaxations or not')
    args = parser.parse_args()

    arch = args.arch

    os.chdir(RUN_JL_DIR)

    # Step 1: Train
    if not args.skip_training:
        itr_n1, itr_n2 = train_model(arch, batch_size=args.batch_size, lr=args.lr)
    else:
        print("Skipping training (--skip_training)")
        itr_n1, itr_n2 = detect_iterations(arch)

    # Step 2: VHAGaR standard for both models
    if not args.skip_vaghar:
        for ctag in range(1, 3):
            step2_vaghar_standard(arch, itr_n1, itr_n2, args.perturbation_size, ctag, args.ct,
                                  args.timeout, args.perturbation)
    else:
        print("Skipping standard VHAGaR (--skip_vaghar)")

    # Step 3: Transfer (N1=itr_n1, N2=itr_n2)
    step3_transfer(arch, itr_n1, itr_n2, args.timeout, args.perturbation, args.ct, args.transfer_relaxations)

    dirs = get_exp_dirs(arch, itr_n1, itr_n2)
    print("\n" + "=" * 60)
    print(f"EXPERIMENT COMPLETE ({arch})")
    print("=" * 60)
    print(f"  itr{itr_n1} model:      {dirs['model_n1']}/model.p")
    print(f"  itr{itr_n2} model:      {dirs['model_n2']}/model.p")
    print(f"  VHAGaR itr{itr_n1}:     {dirs['vaghar_n1']}/")
    print(f"  VHAGaR itr{itr_n2}:     {dirs['vaghar_n2']}/")
    print(f"  Transfer results: {dirs['transfer']}/")


if __name__ == '__main__':
    main()
