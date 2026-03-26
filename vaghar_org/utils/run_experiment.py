#!/usr/bin/env python3
"""
Experiment pipeline with relaxation_threshold sweep.

Supports multiple datasets (MNIST, FashionMNIST, CIFAR10), all architectures,
and sweeps over relaxation_threshold for transfer mode.

Pipeline:
  1. Train model until two consecutive epochs reach target accuracy (if needed)
  2. Run VHAGaR standard with perturbed intervals for both models (if needed)
  3. For each relaxation_threshold:
       Run VHAGaR transfer with --use_relaxations true --relaxation_threshold <val>
     Also run transfer with --use_relaxations false (baseline)

Usage examples:
  python utils/run_experiment.py --dataset mnist --arch 3x10
  python utils/run_experiment.py --dataset mnist --arch cnn1 --perturbations "linf:0.02,0.05;brightness:0.1,0.2"
  python utils/run_experiment.py --dataset cifar10 --arch 3x50 --relaxation_thresholds "0.0,0.25,0.5,1.0"
  python utils/run_experiment.py --dataset mnist --arch 6x10 --skip_training --skip_standard
  python utils/run_experiment.py --dataset mnist --arch 3x10 --dual_seed --seeds "42,137"
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
    FNN_2_10, FNN_3_10, FNN_3_50, FNN_3_100,
    FNN_4_10, FNN_5_10, FNN_5_50, FNN_6_10, FNN_10_10,
    CNN0, CNN1, CNN2, CNN3,
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
    '10x10': (FNN_10_10, '10x10'),
    'cnn0':  (CNN0,      'cnn0'),
    'cnn1':  (CNN1,      'cnn1'),
    'cnn2':  (CNN2,      'cnn2'),
    'cnn3':  (CNN3,      'cnn3'),
}

# ── dataset config ───────────────────────────────────────────────────────
# Maps dataset name -> (torchvision class, channels, width, height, julia dataset name)
DATASET_CONFIG = {
    'mnist':         (dsets.MNIST,        1, 28, 28, 'mnist'),
    'fashion_mnist': (dsets.FashionMNIST, 1, 28, 28, 'fashion_mnist'),
    'cifar10':       (dsets.CIFAR10,      3, 32, 32, 'cifar10'),
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


def parse_perturbations(spec):
    """Parse perturbation spec string into list of (perturbation, size) pairs.

    Format: "type1:params/params;type2:params/params"
    Uses '/' to separate different runs of the same perturbation type.
    Commas within params are preserved (passed directly to Julia).

    Examples:
      "linf:0.02/0.05"
        -> [("linf","0.02"), ("linf","0.05")]
      "patch:1.0,14,14,1/1.0,14,14,3"
        -> [("patch","1.0,14,14,1"), ("patch","1.0,14,14,3")]
      "linf:0.1;patch:1.0,14,14,5"
        -> [("linf","0.1"), ("patch","1.0,14,14,5")]

    Single perturbation shorthand: "linf:0.25" -> [("linf","0.25")]
    """
    pairs = []
    for block in spec.split(';'):
        block = block.strip()
        if not block:
            continue
        if ':' not in block:
            print(f"ERROR: Invalid perturbation spec '{block}'. Expected 'type:params[/params]'")
            sys.exit(1)
        ptype, sizes_str = block.split(':', 1)
        ptype = ptype.strip()
        for s in sizes_str.split('/'):
            s = s.strip()
            if s:
                pairs.append((ptype, s))
    return pairs


def get_exp_dirs(arch, dataset, itr_n1, itr_n2, perturbation=None, perturbation_size=None,
                 create=False, dual_seed=False, epochs=None, model_dirs=None):
    """Return a dict of all experiment directories for the given architecture and dataset.
    Structure: .../arch_exp/perturbation/perturbation_size/...

    When dual_seed=True, itr_n1/itr_n2 are seed values and epochs=(epoch_n1, epoch_n2)
    gives the training epoch for each. Folder names use 'seed{S}_itr{E}' format.

    When model_dirs=(n1_dir, n2_dir), use explicit model directories and derive tags
    from directory basenames by stripping the 'model_' prefix.
    """
    exp_dir = os.path.join(SCRIPT_DIR, '..', 'paper_experiments', dataset, f'{arch}_exp')
    if perturbation and perturbation_size:
        pert_dir = os.path.join(exp_dir, perturbation, f'eps_{perturbation_size}')
    elif perturbation:
        pert_dir = os.path.join(exp_dir, perturbation)
    else:
        pert_dir = exp_dir

    if model_dirs:
        n1_base = os.path.basename(os.path.normpath(model_dirs[0]))
        n2_base = os.path.basename(os.path.normpath(model_dirs[1]))
        tag1 = n1_base.replace('model_', '', 1)
        tag2 = n2_base.replace('model_', '', 1)
        model_n1_name = n1_base
        model_n2_name = n2_base
    elif dual_seed:
        ep1 = epochs[0] if epochs else 0
        ep2 = epochs[1] if epochs else 0
        tag1 = f'seed{itr_n1}_itr{ep1}'
        tag2 = f'seed{itr_n2}_itr{ep2}'
        model_n1_name = f'model_seed{itr_n1}_itr{ep1}'
        model_n2_name = f'model_seed{itr_n2}_itr{ep2}'
    else:
        tag1, tag2 = f'itr{itr_n1}', f'itr{itr_n2}'
        model_n1_name = f'model_{itr_n1}_itr'
        model_n2_name = f'model_{itr_n2}_itr'

    dirs = {
        'exp':              exp_dir,
        'model_n1':         os.path.join(exp_dir, model_n1_name),
        'model_n2':         os.path.join(exp_dir, model_n2_name),
        'vaghar_n1':        os.path.join(pert_dir, f'vagharWithPerturbed_{arch}_{tag1}'),
        'vaghar_n2':        os.path.join(pert_dir, f'vagharWithPerturbed_{arch}_{tag2}'),
        'vaghar_n1_noPI':   os.path.join(pert_dir, f'vagharNoPerturbed_{arch}_{tag1}'),
        'vaghar_n2_noPI':   os.path.join(pert_dir, f'vagharNoPerturbed_{arch}_{tag2}'),
        'transfer':         os.path.join(pert_dir, f'transfer_{arch}_N1_is_{tag1}'),
    }
    if create:
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
    return dirs


def get_transfer_dir(base_dirs, threshold):
    """Return the transfer output dir for a specific relaxation_threshold."""
    base = base_dirs['transfer']
    if threshold is None:
        return base + '_norelax'
    return base + f'_relax{threshold}'


def detect_iterations(arch, dataset):
    """Detect itr_n1 and itr_n2 from existing model_*_itr folders."""
    exp_dir = os.path.join(SCRIPT_DIR, '..', 'paper_experiments', dataset, f'{arch}_exp')
    if not os.path.exists(exp_dir):
        print(f"ERROR: Experiment directory {exp_dir} not found.")
        sys.exit(1)

    pattern = re.compile(r'^model_(\d+)_itr$')
    itrs = sorted(
        int(m.group(1))
        for name in os.listdir(exp_dir)
        if (m := pattern.match(name)) and os.path.isdir(os.path.join(exp_dir, name))
    )

    if len(itrs) >= 2:
        itr_n1, itr_n2 = itrs[0], itrs[1]
        print(f"  Detected model checkpoints: itr{itr_n1} and itr{itr_n2}")
        return itr_n1, itr_n2

    print(f"ERROR: Need at least two model_*_itr folders in {exp_dir}.")
    print(f"  Found: {itrs}")
    sys.exit(1)


def detect_seeds(arch, dataset):
    """Detect seed_n1/seed_n2 and their epochs from existing model_seed*_itr* folders.
    Returns (seed_n1, seed_n2, epoch_n1, epoch_n2)."""
    exp_dir = os.path.join(SCRIPT_DIR, '..', 'paper_experiments', dataset, f'{arch}_exp')
    if not os.path.exists(exp_dir):
        print(f"ERROR: Experiment directory {exp_dir} not found.")
        sys.exit(1)

    pattern = re.compile(r'^model_seed(\d+)_itr(\d+)$')
    entries = sorted(
        (int(m.group(1)), int(m.group(2)))
        for name in os.listdir(exp_dir)
        if (m := pattern.match(name)) and os.path.isdir(os.path.join(exp_dir, name))
    )

    if len(entries) >= 2:
        (seed_n1, ep1), (seed_n2, ep2) = entries[0], entries[1]
        print(f"  Detected dual-seed model checkpoints: seed{seed_n1}_itr{ep1} and seed{seed_n2}_itr{ep2}")
        return seed_n1, seed_n2, ep1, ep2

    print(f"ERROR: Need at least two model_seed*_itr* folders in {exp_dir}.")
    print(f"  Found: {entries}")
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


def evaluate_robust(model, test_loader, device, epsilon, alpha=0.01, num_steps=20):
    """Evaluate accuracy on PGD adversarial examples (robust accuracy)."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_steps, device)
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def get_data_loaders(dataset, batch_size=128):
    """Return train and test loaders for the given dataset."""
    if dataset not in DATASET_CONFIG:
        print(f"ERROR: Unknown dataset '{dataset}'. Choose from: {list(DATASET_CONFIG.keys())}")
        sys.exit(1)

    ds_class, _, _, _, _ = DATASET_CONFIG[dataset]
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = ds_class(root='./data/', train=True, transform=transform, download=True)
    test_ds = ds_class(root='./data/', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_julia(args_list, step_name):
    """Run julia run.jl with given arguments."""
    cmd = ['julia', 'run.jl'] + args_list
    print(f"\n  Running: {' '.join(cmd[:8])}...")
    proc = subprocess.run(cmd, cwd=RUN_JL_DIR)
    if proc.returncode != 0:
        print(f"  WARNING: {step_name} exited with code {proc.returncode}")
    return proc.returncode


# ── PGD adversarial training helpers ──────────────────────────────────────

def pgd_attack(model, images, labels, epsilon, alpha, num_steps, device):
    """Generate PGD adversarial examples (Madry et al. 2018).

    Args:
        model: network in eval mode
        images: clean batch [B, C, H, W] in [0, 1]
        labels: ground-truth labels [B]
        epsilon: L∞ perturbation radius
        alpha: PGD step size
        num_steps: number of PGD iterations
        device: torch device

    Returns:
        adv_images: adversarial batch clamped to [0, 1] and within ε-ball
    """
    adv = images.clone().detach()
    # Random start within ε-ball
    adv = adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, 0.0, 1.0)

    criterion = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        adv.requires_grad_(True)
        outputs = model(adv)
        loss = criterion(outputs, labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        # Project back to ε-ball around original image
        delta = torch.clamp(adv - images, -epsilon, epsilon)
        adv = torch.clamp(images + delta, 0.0, 1.0)

    return adv.detach()


# ── step 1: train ────────────────────────────────────────────────────────

MIN_ACCURACY = 91.0
# PGD adversarial training reduces clean accuracy; use a lower threshold
MIN_ACCURACY_PGD = 80.0


def train_model(arch, dataset, batch_size=128, lr=1e-3, max_epochs=100, itr_gap=1, force_cpu=False,
                pgd_training=False, pgd_epsilon=0.1, pgd_alpha=0.01, pgd_steps=7,
                optimizer_name='sgd'):
    """
    Train until two epochs separated by itr_gap both reach MIN_ACCURACY.
    Save checkpoints for those two epochs (itr_n1 and itr_n2 = itr_n1 + itr_gap).

    When pgd_training=True, uses PGD adversarial training (Madry et al. 2018):
    each batch is replaced by adversarial examples before computing the loss.
    Returns (itr_n1, itr_n2).
    """
    model_cls, _ = ARCH_REGISTRY[arch]
    _, k, w, h, _ = DATASET_CONFIG[dataset]

    mode_str = f"PGD (eps={pgd_epsilon}, alpha={pgd_alpha}, steps={pgd_steps})" if pgd_training else "standard"
    print("=" * 60)
    print(f"STEP 1: Training {arch} on {dataset} [{mode_str}]")
    if pgd_training:
        print(f"  (fixed {max_epochs} epochs, saving last two with gap {itr_gap})")
    else:
        print(f"  (until two epochs {itr_gap} apart >= {MIN_ACCURACY}% acc, max {max_epochs})")
    print("=" * 60)

    device = _get_device(force_cpu)
    print(f"  Using device: {device}")
    model = model_cls(k=k, w=w, h=h).to(device)
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_data_loaders(dataset, batch_size)

    # Keep a history of (epoch_num, accuracy, state_dict) for the last itr_gap+1 epochs
    from collections import deque
    history = deque(maxlen=itr_gap + 1)

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            if pgd_training:
                model.eval()
                images = pgd_attack(model, images, labels, pgd_epsilon, pgd_alpha, pgd_steps, device)
                model.train()

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        epoch_num = epoch + 1
        print(f"  Epoch {epoch_num}/{max_epochs} — loss: {running_loss/len(train_loader):.4f}, acc: {acc:.2f}%")

        history.append((epoch_num, acc, {k_: v.clone() for k_, v in model.state_dict().items()}))

        if pgd_training:
            # PGD training: always train for all max_epochs, save the last two
            continue
        else:
            # Standard training: early-stop when two epochs itr_gap apart both reach MIN_ACCURACY
            if len(history) == itr_gap + 1:
                old_epoch, old_acc, old_state = history[0]
                if old_acc >= MIN_ACCURACY and acc >= MIN_ACCURACY and epoch_num - old_epoch == itr_gap:
                    itr_n1 = old_epoch
                    itr_n2 = epoch_num
                    dirs = get_exp_dirs(arch, dataset, itr_n1, itr_n2, create=True)
                    n1_model = model_cls(k=k, w=w, h=h).to(device)
                    n1_model.load_state_dict(old_state)
                    save_model(n1_model, dirs['model_n1'])
                    print(f"  >> Saved itr{itr_n1} checkpoint (acc: {old_acc:.2f}%)")
                    save_model(model, dirs['model_n2'])
                    print(f"  >> Saved itr{itr_n2} checkpoint (acc: {acc:.2f}%)")
                    return itr_n1, itr_n2

    if pgd_training:
        # Save the last two checkpoints separated by itr_gap
        if len(history) >= itr_gap + 1:
            itr_n1 = max_epochs - itr_gap
            itr_n2 = max_epochs
            old_epoch, old_acc, old_state = history[0]
            dirs = get_exp_dirs(arch, dataset, itr_n1, itr_n2, create=True)
            n1_model = model_cls(k=k, w=w, h=h).to(device)
            n1_model.load_state_dict(old_state)
            save_model(n1_model, dirs['model_n1'])
            print(f"  >> Saved itr{itr_n1} checkpoint (acc: {old_acc:.2f}%)")
            save_model(model, dirs['model_n2'])
            print(f"  >> Saved itr{itr_n2} checkpoint (acc: {acc:.2f}%)")
            return itr_n1, itr_n2

    print(f"\n  ERROR: Failed to find two epochs {itr_gap} apart with >= {MIN_ACCURACY}% accuracy within {max_epochs} epochs.")
    sys.exit(1)


def _train_single_seed(model_cls, k, w, h, seed, train_loader, test_loader, device,
                       lr, max_epochs, min_accuracy, optimizer_name='sgd',
                       pgd_training=False, pgd_epsilon=0.1, pgd_alpha=0.01, pgd_steps=7,
                       fixed_epochs=False):
    """Train one model with a specific seed. Returns (model, final_acc, epoch_num)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_cls(k=k, w=w, h=h).to(device)
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if pgd_training:
                model.eval()
                images = pgd_attack(model, images, labels, pgd_epsilon, pgd_alpha, pgd_steps, device)
                model.train()

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        epoch_num = epoch + 1
        print(f"    Epoch {epoch_num}/{max_epochs} — loss: {running_loss/len(train_loader):.4f}, acc: {acc:.2f}%")

        if pgd_training or fixed_epochs:
            # Always train for all max_epochs
            continue

        if acc >= min_accuracy:
            print(f"    >> Reached {min_accuracy}% accuracy at epoch {epoch_num}")
            return model, acc, epoch_num

    # Completed all epochs
    if pgd_training or fixed_epochs:
        print(f"    >> Completed {max_epochs} epochs (final acc: {acc:.2f}%)")
    else:
        print(f"    WARNING: Did not reach {min_accuracy}% in {max_epochs} epochs (best: {acc:.2f}%)")
    return model, acc, max_epochs


def train_dual_seed(arch, dataset, seeds, batch_size=128, lr=1e-3, max_epochs=200, force_cpu=False,
                    pgd_training=False, pgd_epsilon=0.1, pgd_alpha=0.01, pgd_steps=7,
                    optimizer_name='sgd', fixed_epochs=False):
    """
    Train two independent networks from different random seeds (same arch/hyperparams).
    Returns (seed_n1, seed_n2, epoch_n1, epoch_n2).
    """
    model_cls, _ = ARCH_REGISTRY[arch]
    _, k, w, h, _ = DATASET_CONFIG[dataset]
    seed_n1, seed_n2 = seeds

    min_acc = MIN_ACCURACY_PGD if pgd_training else MIN_ACCURACY
    mode_str = f"PGD (eps={pgd_epsilon}, alpha={pgd_alpha}, steps={pgd_steps})" if pgd_training else "standard"
    print("=" * 60)
    print(f"STEP 1 (dual-seed): Training {arch} on {dataset} [{mode_str}] with seeds {seed_n1} and {seed_n2}")
    print("=" * 60)

    device = _get_device(force_cpu)
    print(f"  Using device: {device}")
    train_loader, test_loader = get_data_loaders(dataset, batch_size)

    print(f"\n  --- Training N1 (seed={seed_n1}) ---")
    model_n1, acc_n1, ep_n1 = _train_single_seed(model_cls, k, w, h, seed_n1,
                                                   train_loader, test_loader, device,
                                                   lr, max_epochs, min_acc,
                                                   optimizer_name=optimizer_name,
                                                   pgd_training=pgd_training,
                                                   pgd_epsilon=pgd_epsilon,
                                                   pgd_alpha=pgd_alpha,
                                                   pgd_steps=pgd_steps,
                                                   fixed_epochs=fixed_epochs)

    print(f"\n  --- Training N2 (seed={seed_n2}) ---")
    model_n2, acc_n2, ep_n2 = _train_single_seed(model_cls, k, w, h, seed_n2,
                                                   train_loader, test_loader, device,
                                                   lr, max_epochs, min_acc,
                                                   optimizer_name=optimizer_name,
                                                   pgd_training=pgd_training,
                                                   pgd_epsilon=pgd_epsilon,
                                                   pgd_alpha=pgd_alpha,
                                                   pgd_steps=pgd_steps,
                                                   fixed_epochs=fixed_epochs)

    print(f"\n  N1 acc: {acc_n1:.2f}% (epoch {ep_n1}), N2 acc: {acc_n2:.2f}% (epoch {ep_n2})")

    # Now that we know the epochs, create dirs and save
    dirs = get_exp_dirs(arch, dataset, seed_n1, seed_n2, dual_seed=True,
                        epochs=(ep_n1, ep_n2), create=True)

    save_model(model_n1, dirs['model_n1'])
    print(f"  >> Saved N1 seed{seed_n1}_itr{ep_n1} (acc: {acc_n1:.2f}%)")

    save_model(model_n2, dirs['model_n2'])
    print(f"  >> Saved N2 seed{seed_n2}_itr{ep_n2} (acc: {acc_n2:.2f}%)")

    return seed_n1, seed_n2, ep_n1, ep_n2


# ── step 2: run VHAGaR standard with perturbed intervals ────────────────

def run_vaghar_standard(arch, dataset, model_path, output_dir, ctag,
                        perturbation_size='0.05', ct='1,2,3,4,5,6,7,8,9,10',
                        timeout=10800, perturbation='linf', force_cpu=False,
                        use_perturbed_intervals=True, optimizing_intervals=None):
    """Run VHAGaR in standard mode with hyper attack, vaghar deps, and optionally perturbed intervals."""
    _, model_name = ARCH_REGISTRY[arch]
    _, _, _, _, julia_dataset = DATASET_CONFIG[dataset]
    args = [
        '--mode', 'standard',
        '--dataset', julia_dataset,
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
        '--use_perturbed_intervals', str(use_perturbed_intervals).lower(),
        '--force_cpu', str(force_cpu).lower(),
    ]
    if optimizing_intervals is not None:
        args += ['--optimizing_intervals', str(optimizing_intervals).lower()]
    pi_label = "with" if use_perturbed_intervals else "without"
    return run_julia(args, f'VHAGaR standard {arch} (ctag={ctag}, {pi_label} perturbed intervals)')


def step2_vaghar_standard(arch, dataset, itr_n1, itr_n2, perturbation, perturbation_size, ctag, ct, timeout, force_cpu=False, dual_seed=False, epochs=None, optimizing_intervals=None, model_dirs=None):
    dirs = get_exp_dirs(arch, dataset, itr_n1, itr_n2, perturbation=perturbation, perturbation_size=perturbation_size, dual_seed=dual_seed, epochs=epochs, model_dirs=model_dirs)
    os.makedirs(dirs['vaghar_n1'], exist_ok=True)
    os.makedirs(dirs['vaghar_n2'], exist_ok=True)
    os.makedirs(dirs['vaghar_n1_noPI'], exist_ok=True)
    os.makedirs(dirs['vaghar_n2_noPI'], exist_ok=True)

    model_n1_path = os.path.join(dirs['model_n1'], 'model.p')
    model_n2_path = os.path.join(dirs['model_n2'], 'model.p')

    if model_dirs:
        tag1 = os.path.basename(os.path.normpath(model_dirs[0])).replace('model_', '', 1)
        tag2 = os.path.basename(os.path.normpath(model_dirs[1])).replace('model_', '', 1)
    elif dual_seed:
        ep1 = epochs[0] if epochs else 0
        ep2 = epochs[1] if epochs else 0
        tag1 = f'seed{itr_n1}_itr{ep1}'
        tag2 = f'seed{itr_n2}_itr{ep2}'
    else:
        tag1 = f'itr{itr_n1}'
        tag2 = f'itr{itr_n2}'

    # Run WITH perturbed intervals
    print("=" * 60)
    print(f"STEP 2a: VHAGaR standard (WITH perturbed intervals) — {arch} on {dataset}, {perturbation} eps={perturbation_size}, {tag1}&{tag2}")
    print("=" * 60)

    print(f"\n  --- {tag1} (ctag={ctag}) ---")
    run_vaghar_standard(arch, dataset, model_n1_path, dirs['vaghar_n1'], ctag,
                        perturbation_size=perturbation_size, ct=ct,
                        timeout=timeout, perturbation=perturbation, force_cpu=force_cpu,
                        use_perturbed_intervals=True, optimizing_intervals=optimizing_intervals)

    print(f"\n  --- {tag2} (ctag={ctag}) ---")
    run_vaghar_standard(arch, dataset, model_n2_path, dirs['vaghar_n2'], ctag,
                        perturbation_size=perturbation_size, ct=ct,
                        timeout=timeout, perturbation=perturbation, force_cpu=force_cpu,
                        use_perturbed_intervals=True, optimizing_intervals=optimizing_intervals)

    # Run WITHOUT perturbed intervals
    print("=" * 60)
    print(f"STEP 2b: VHAGaR standard (WITHOUT perturbed intervals) — {arch} on {dataset}, {perturbation} eps={perturbation_size}, {tag1}&{tag2}")
    print("=" * 60)

    print(f"\n  --- {tag1} (ctag={ctag}) ---")
    run_vaghar_standard(arch, dataset, model_n1_path, dirs['vaghar_n1_noPI'], ctag,
                        perturbation_size=perturbation_size, ct=ct,
                        timeout=timeout, perturbation=perturbation, force_cpu=force_cpu,
                        use_perturbed_intervals=False, optimizing_intervals=optimizing_intervals)

    print(f"\n  --- {tag2} (ctag={ctag}) ---")
    run_vaghar_standard(arch, dataset, model_n2_path, dirs['vaghar_n2_noPI'], ctag,
                        perturbation_size=perturbation_size, ct=ct,
                        timeout=timeout, perturbation=perturbation, force_cpu=force_cpu,
                        use_perturbed_intervals=False, optimizing_intervals=optimizing_intervals)


# ── step 3: run VHAGaR transfer ──────────────────────────────────────────

def run_transfer_from_results(arch, dataset, itr_n1, itr_n2, vaghar_results_dir,
                              output_dir, timeout, perturbation, ct,
                              transfer_relaxations, delta_diff_positive, Threads_num =32,
                              relaxation_threshold=None, force_cpu=False,
                              use_hyper_attack=True, dual_seed=False, epochs=None,
                              optimizing_intervals=None, model_dirs=None):
    """
    Iterate over VHAGaR results files for N1.
    Each file contains delta_1 values for a specific perturbation_size and c_tag.
    Parse these from the filename and launch a transfer run.

    If relaxation_threshold is not None, pass --relaxation_threshold to Julia.
    """
    _, model_name = ARCH_REGISTRY[arch]
    _, _, _, _, julia_dataset = DATASET_CONFIG[dataset]
    dirs = get_exp_dirs(arch, dataset, itr_n1, itr_n2, dual_seed=dual_seed, epochs=epochs, model_dirs=model_dirs)
    pattern = re.compile(rf"_{perturbation}_(.*?)_ctag.*cTag(\d+)")

    if not os.path.exists(vaghar_results_dir):
        print(f"  Warning: Directory {vaghar_results_dir} not found, trying noPI dir...")
        vaghar_results_dir = vaghar_results_dir.replace('vagharWithPerturbed_', 'vagharNoPerturbed_')
        if not os.path.exists(vaghar_results_dir):
            print(f"  Error: Fallback directory {vaghar_results_dir} also not found.")
            return
        print(f"  Using fallback: {vaghar_results_dir}")

    n1_path = os.path.join(dirs['model_n1'], 'model.p')
    n2_path = os.path.join(dirs['model_n2'], 'model.p')

    os.makedirs(output_dir, exist_ok=True)

    # Collect matching files, falling back to noPI dir if none found
    result_files = []
    for filename in sorted(os.listdir(vaghar_results_dir)):
        if not filename.endswith('.txt'):
            continue
        if '_PerturbedIntervals' in filename:
            continue
        match = pattern.search(filename)
        if not match:
            continue
        result_files.append((filename, match))

    if not result_files:
        fallback_dir = vaghar_results_dir.replace('vagharWithPerturbed_', 'vagharNoPerturbed_')
        if fallback_dir != vaghar_results_dir and os.path.exists(fallback_dir):
            print(f"  No matching files in {vaghar_results_dir}, trying {fallback_dir}...")
            for filename in sorted(os.listdir(fallback_dir)):
                if not filename.endswith('.txt'):
                    continue
                match = pattern.search(filename)
                if not match:
                    continue
                result_files.append((filename, match))
            if result_files:
                vaghar_results_dir = fallback_dir
                print(f"  Found {len(result_files)} files in fallback dir")

    if not result_files:
        print(f"  Error: No matching VHAGaR result files found.")
        return

    for filename, match in result_files:

        perturbation_size = match.group(1)
        c_tag_n = match.group(2)
        vaghar_results_path = os.path.join(vaghar_results_dir, filename)

        print(f"  Processing: {filename}  (eps={perturbation_size}, ctag={c_tag_n})")

        command = [
            '--mode', 'transfer',
            '--dataset', julia_dataset,
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
            '--use_hyper_attack', str(use_hyper_attack).lower(),
            '--activate_vaghgar_deps', 'true',
            '--use_intervals', 'true',
            '--use_perturbed_intervals', 'true',
            '--n2_fewer_binars_encoding', 'true',
            '--use_relaxations', transfer_relaxations,
            '--delta_diff_positive', delta_diff_positive,
            '--force_cpu', str(force_cpu).lower(),
            '--Threads_num', str(Threads_num),
        ]
        if relaxation_threshold is not None:
            command += ['--relaxation_threshold', str(relaxation_threshold)]
        if optimizing_intervals is not None:
            command += ['--optimizing_intervals', str(optimizing_intervals).lower()]

        run_julia(command, f'transfer {arch} (ctag={c_tag_n}, relax_thresh={relaxation_threshold})')


def step3_transfer(arch, dataset, itr_n1, itr_n2, timeout, perturbation, perturbation_size, ct,
                   transfer_relaxations, delta_diff_positive,Threads_num=32, relaxation_threshold=None, force_cpu=False,
                   use_hyper_attack=True, dual_seed=False, epochs=None, optimizing_intervals=None, model_dirs=None):
    dirs = get_exp_dirs(arch, dataset, itr_n1, itr_n2, perturbation=perturbation, perturbation_size=perturbation_size, dual_seed=dual_seed, epochs=epochs, model_dirs=model_dirs)
    output_dir = get_transfer_dir(dirs, relaxation_threshold)
    os.makedirs(output_dir, exist_ok=True)

    if model_dirs:
        tag1 = os.path.basename(os.path.normpath(model_dirs[0])).replace('model_', '', 1)
        tag2 = os.path.basename(os.path.normpath(model_dirs[1])).replace('model_', '', 1)
    elif dual_seed:
        ep1 = epochs[0] if epochs else 0
        ep2 = epochs[1] if epochs else 0
        tag1 = f'seed{itr_n1}_itr{ep1}'
        tag2 = f'seed{itr_n2}_itr{ep2}'
    else:
        tag1 = f'itr{itr_n1}'
        tag2 = f'itr{itr_n2}'
    thresh_label = f"threshold={relaxation_threshold}" if relaxation_threshold is not None else "no relaxation"
    print("=" * 60)
    print(f"STEP 3: VHAGaR transfer — {arch} on {dataset}, {perturbation} (N1={tag1}, N2={tag2}, {thresh_label})")
    print("=" * 60)

    run_transfer_from_results(
        arch, dataset, itr_n1, itr_n2,
        dirs['vaghar_n1'], output_dir, timeout, perturbation, ct,
        transfer_relaxations, delta_diff_positive, Threads_num, relaxation_threshold, force_cpu=force_cpu,
        use_hyper_attack=use_hyper_attack, dual_seed=dual_seed, epochs=epochs,
        optimizing_intervals=optimizing_intervals, model_dirs=model_dirs,
    )


# ── main ─────────────────────────────────────────────────────────────────

def main():
    arch_choices = list(ARCH_REGISTRY.keys())
    dataset_choices = list(DATASET_CONFIG.keys())

    parser = argparse.ArgumentParser(
        description='Experiment pipeline: train, VHAGaR standard, transfer with relaxation_threshold sweep',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True, choices=dataset_choices,
                        help=f'Dataset: {dataset_choices}')
    parser.add_argument('--arch', type=str, required=False, default="3x10", choices=arch_choices,
                        help=f'Architecture: {arch_choices}')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='Optimizer to use for training')
    parser.add_argument('--max_epochs', type=int, default=200, help='Max training epochs')
    parser.add_argument('--itr_gap', type=int, default=5,
                        help='Gap between the two saved model checkpoints (e.g. 1=consecutive, 5=five epochs apart)')
    parser.add_argument('--perturbations', type=str, default='linf:0.25',
                        help='Perturbation spec: "type1:size1,size2;type2:size3,size4" '
                             'e.g. "linf:0.02,0.05;brightness:0.1,0.2"')
    parser.add_argument('--ct', type=str, default='4,5,6', help='Target classes')
    parser.add_argument('--timeout', type=int, default=10800, help='MIP timeout per class pair (seconds)')
    parser.add_argument('--transfer_relaxations', type=str, default='true',
                        help='Run transfer with relaxations (true/false)')
    parser.add_argument('--delta_diff_positive', type=str, default='false',
                        help='Force delta_diff > 0 cutoff (true/false)')
    parser.add_argument('--relaxation_thresholds', type=str, default='0.0,0.25,0.5,1.0',
                        help='Comma-separated relaxation_threshold values to sweep')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU-only mode (no GPU). By default, uses CUDA if available.')
    parser.add_argument('--plot_conf', action='store_true',
                        help='Plot confidence values for N2 on the test set, then exit')
    parser.add_argument('--plot_conf_both', action='store_true',
                        help='Plot confidence values for both N1 and N2 on the same figure, then exit')
    parser.add_argument('--optimizing_intervals', type=str, default=None,
                        help='Override optimizing_intervals flag passed to Julia (true/false). '
                             'Default: let Julia use its own default (true).')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, use existing models')
    parser.add_argument('--skip_standard', action='store_true', help='Skip standard VHAGaR')
    parser.add_argument('--skip_transfer', action='store_true', help='Skip transfer VHAGaR')
    parser.add_argument('--skip_hyper_transfer_attack', action='store_true',
                        help='Disable hyper attack (PGD warm-start) in transfer runs')
    parser.add_argument('--dual_seed', action='store_true',
                        help='Train two independent networks from different random seeds instead of '
                             'using checkpoints from the same training run')
    parser.add_argument('--seeds', type=str, default='42,137',
                        help='Comma-separated pair of seeds for --dual_seed mode (e.g. "42,137")')
    parser.add_argument('--fixed_epochs', action='store_true',
                        help='Train for exactly max_epochs instead of early-stopping on accuracy')
    parser.add_argument('--pgd_training', action='store_true',
                        help='Use PGD adversarial training (Madry et al. 2018). '
                             'Recommended for 3x50 to match paper results.')
    parser.add_argument('--pgd_epsilon', type=float, default=0.1,
                        help='PGD training: L∞ perturbation radius')
    parser.add_argument('--pgd_alpha', type=float, default=0.01,
                        help='PGD training: step size per iteration')
    parser.add_argument('--pgd_steps', type=int, default=7,
                        help='PGD training: number of attack iterations per batch')
    parser.add_argument('--model_n1_dir', type=str, default=None,
                        help='Explicit model directory for N1 (e.g. model_seed42_itr20). '
                             'Tags derived from basename. Skips auto-detection.')
    parser.add_argument('--model_n2_dir', type=str, default=None,
                        help='Explicit model directory for N2 (e.g. model_seed42_itr20_sgd_itr1). '
                             'Tags derived from basename. Skips auto-detection.')
    parser.add_argument('--Threads_num', type=int, default=32,
                        help='Number of threads to use')

    args = parser.parse_args()

    arch = args.arch
    dataset = args.dataset
    dual_seed = args.dual_seed

    # Parse seeds
    seed_values = [int(s.strip()) for s in args.seeds.split(',')]
    if dual_seed and len(seed_values) < 2:
        print("ERROR: --dual_seed requires at least two values in --seeds (e.g. --seeds 42,137)")
        sys.exit(1)

    # Parse relaxation thresholds
    thresholds = []
    for val in args.relaxation_thresholds.split(','):
        val = val.strip()
        if val.lower() == 'inf':
            thresholds.append(float('inf'))
        else:
            thresholds.append(float(val))

    # Parse perturbation spec
    perturbation_pairs = parse_perturbations(args.perturbations)
    print(f"\nPerturbation configs ({len(perturbation_pairs)}):")
    for pt, ps in perturbation_pairs:
        print(f"  {pt} eps={ps}")
    if dual_seed:
        print(f"\nDual-seed mode: seeds={seed_values[0]},{seed_values[1]}")
    print()

    os.chdir(RUN_JL_DIR)

    # Explicit model directories mode (--model_n1_dir / --model_n2_dir)
    model_dirs = None
    if args.model_n1_dir and args.model_n2_dir:
        model_dirs = (args.model_n1_dir, args.model_n2_dir)
        itr_n1, itr_n2 = 0, 0  # dummy, not used when model_dirs is set
        epochs = None
        tag1 = os.path.basename(os.path.normpath(args.model_n1_dir)).replace('model_', '', 1)
        tag2 = os.path.basename(os.path.normpath(args.model_n2_dir)).replace('model_', '', 1)
        print(f"\nExplicit model dirs mode: N1={tag1}, N2={tag2}")

    # Step 1: Train (once — training is perturbation-independent)
    epochs = None  # only used in dual_seed mode
    if model_dirs:
        print("Using explicit model directories, skipping training/detection.")
    elif not args.skip_training:
        if dual_seed:
            itr_n1, itr_n2, ep_n1, ep_n2 = train_dual_seed(
                arch, dataset, seeds=(seed_values[0], seed_values[1]),
                batch_size=args.batch_size, lr=args.lr,
                max_epochs=args.max_epochs, force_cpu=args.cpu,
                pgd_training=args.pgd_training, pgd_epsilon=args.pgd_epsilon,
                pgd_alpha=args.pgd_alpha, pgd_steps=args.pgd_steps,
                optimizer_name=args.optimizer, fixed_epochs=args.fixed_epochs)
            epochs = (ep_n1, ep_n2)
        else:
            itr_n1, itr_n2 = train_model(arch, dataset, batch_size=args.batch_size,
                                          lr=args.lr, max_epochs=args.max_epochs,
                                          itr_gap=args.itr_gap, force_cpu=args.cpu,
                                          pgd_training=args.pgd_training,
                                          pgd_epsilon=args.pgd_epsilon,
                                          pgd_alpha=args.pgd_alpha,
                                          pgd_steps=args.pgd_steps,
                                          optimizer_name=args.optimizer)
    else:
        print("Skipping training (--skip_training)")
        if dual_seed:
            itr_n1, itr_n2, ep_n1, ep_n2 = detect_seeds(arch, dataset)
            epochs = (ep_n1, ep_n2)
        else:
            itr_n1, itr_n2 = detect_iterations(arch, dataset)

    # Plot confidence and exit if requested
    if args.plot_conf:
        for ctag_val in [int(c) for c in args.ct.split(',')]:
            for perturbation, perturbation_size in perturbation_pairs:
                plot_confidence(arch, dataset, ctag_val, perturbation, perturbation_size,
                                itr_n1, itr_n2)
        sys.exit(0)

    if args.plot_conf_both:
        for ctag_val in [int(c) for c in args.ct.split(',')]:
            for perturbation, perturbation_size in perturbation_pairs:
                plot_confidence_both(arch, dataset, ctag_val, perturbation, perturbation_size,
                                     itr_n1, itr_n2)
        sys.exit(0)

    # Steps 2 & 3: loop over each (perturbation, size) pair
    for perturbation, perturbation_size in perturbation_pairs:
        print("\n" + "#" * 60)
        print(f"# Perturbation: {perturbation}  eps={perturbation_size}")
        print("#" * 60)

        # Step 2: VHAGaR standard (shared across thresholds, but per perturbation+size)
        if not args.skip_standard:
            for ctag in range(1, 3):
                step2_vaghar_standard(arch, dataset, itr_n1, itr_n2,
                                      perturbation, perturbation_size,
                                      ctag, args.ct, args.timeout, force_cpu=args.cpu,
                                      dual_seed=dual_seed, epochs=epochs,
                                      optimizing_intervals=args.optimizing_intervals,
                                      model_dirs=model_dirs)
        else:
            print("  Skipping standard VHAGaR (--skip_standard)")

        # Step 3: Transfer — baseline (no relaxation) + sweep over thresholds
        if not args.skip_transfer:
            # Pick N1 as the seed with fastest standard NoPerturbed optimization time
            transfer_n1, transfer_n2 = itr_n1, itr_n2
            transfer_epochs = epochs
            transfer_model_dirs = model_dirs
            if dual_seed and not model_dirs:
                transfer_n1, transfer_n2, ep1_t, ep2_t = pick_n1_by_fastest_standard(
                    arch, dataset, itr_n1, itr_n2, perturbation, perturbation_size,
                    dual_seed=True, epochs=epochs)
                transfer_epochs = (ep1_t, ep2_t)

            # Sweep: transfer with relaxations enabled at each threshold
            for thresh in thresholds:
                step3_transfer(arch, dataset, transfer_n1, transfer_n2, args.timeout, perturbation, perturbation_size,
                               args.ct, 'true', args.delta_diff_positive, args.Threads_num, relaxation_threshold=thresh,
                               force_cpu=args.cpu,
                               use_hyper_attack=not args.skip_hyper_transfer_attack,
                               dual_seed=dual_seed, epochs=transfer_epochs,
                               optimizing_intervals=args.optimizing_intervals,
                               model_dirs=transfer_model_dirs)
        else:
            print("  Skipping transfer VHAGaR (--skip_transfer)")

    # Summary
    if dual_seed:
        ep1, ep2 = epochs
        tag1 = f'seed{itr_n1}_itr{ep1}'
        tag2 = f'seed{itr_n2}_itr{ep2}'
    else:
        tag1 = f'itr{itr_n1}'
        tag2 = f'itr{itr_n2}'
    mode_label = "dual-seed" if dual_seed else "itr-gap"
    print("\n" + "=" * 60)
    print(f"EXPERIMENT COMPLETE ({arch} on {dataset}, {mode_label})")
    print("=" * 60)
    dirs_base = get_exp_dirs(arch, dataset, itr_n1, itr_n2, dual_seed=dual_seed, epochs=epochs)
    print(f"  {tag1} model:  {dirs_base['model_n1']}/model.p")
    print(f"  {tag2} model:  {dirs_base['model_n2']}/model.p")
    for perturbation, perturbation_size in perturbation_pairs:
        dirs = get_exp_dirs(arch, dataset, itr_n1, itr_n2, perturbation=perturbation, perturbation_size=perturbation_size, dual_seed=dual_seed, epochs=epochs)
        print(f"\n  [{perturbation} eps={perturbation_size}]")
        print(f"    VHAGaR {tag1} (PI):     {dirs['vaghar_n1']}/")
        print(f"    VHAGaR {tag2} (PI):     {dirs['vaghar_n2']}/")
        print(f"    VHAGaR {tag1} (no PI):  {dirs['vaghar_n1_noPI']}/")
        print(f"    VHAGaR {tag2} (no PI):  {dirs['vaghar_n2_noPI']}/")
        for thresh in thresholds:
            print(f"    Transfer (t={thresh}):  {get_transfer_dir(dirs, thresh)}/")


def get_noPI_optimization_time(arch, dataset, seed, perturbation, perturbation_size,
                               dual_seed=True, epochs=None, other_seed=None, other_epoch=None):
    """Get the total optimization_time from standard NoPerturbed results for a given seed.

    Returns the sum of optimization_time across all result files, or float('inf') if no results found.
    """
    # We need both seeds to construct the directory path (get_exp_dirs requires itr_n1/itr_n2)
    if other_seed is None:
        return float('inf')

    # Try both orderings to find the directory
    for s1, s2, e1, e2 in [(seed, other_seed, epochs[0] if epochs else 0, epochs[1] if epochs else 0),
                            (other_seed, seed, epochs[1] if epochs else 0, epochs[0] if epochs else 0)]:
        ep = (e1, e2) if epochs else None
        dirs = get_exp_dirs(arch, dataset, s1, s2, perturbation=perturbation,
                           perturbation_size=perturbation_size, dual_seed=dual_seed,
                           epochs=ep)
        # Figure out which noPI dir corresponds to our target seed
        if s1 == seed:
            noPI_dir = dirs['vaghar_n1_noPI']
        else:
            noPI_dir = dirs['vaghar_n2_noPI']

        if os.path.exists(noPI_dir) and os.listdir(noPI_dir):
            break
    else:
        return float('inf')

    total_time = 0.0
    found = False
    for filename in os.listdir(noPI_dir):
        if not filename.endswith('.txt'):
            continue
        filepath = os.path.join(noPI_dir, filename)
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = dict(p.split('=') for p in line.split(','))
                if 'optimization_time' in parts:
                    total_time += float(parts['optimization_time'])
                    found = True
    return total_time if found else float('inf')


def pick_n1_by_fastest_standard(arch, dataset, seed_a, seed_b, perturbation, perturbation_size,
                                dual_seed=True, epochs=None):
    """Pick N1 as the seed with lowest standard NoPerturbed optimization time.

    Returns (n1_seed, n2_seed, n1_epoch, n2_epoch) with the faster seed as N1.
    """
    ep_a = epochs[0] if epochs else None
    ep_b = epochs[1] if epochs else None

    time_a = get_noPI_optimization_time(arch, dataset, seed_a, perturbation, perturbation_size,
                                        dual_seed=dual_seed, epochs=epochs,
                                        other_seed=seed_b, other_epoch=ep_b)
    time_b = get_noPI_optimization_time(arch, dataset, seed_b, perturbation, perturbation_size,
                                        dual_seed=dual_seed, epochs=epochs,
                                        other_seed=seed_a, other_epoch=ep_a)

    if time_a <= time_b:
        print(f"  N1=seed{seed_a} (time={time_a:.1f}s) <= N2=seed{seed_b} (time={time_b:.1f}s)")
        return seed_a, seed_b, ep_a, ep_b
    else:
        print(f"  N1=seed{seed_b} (time={time_b:.1f}s) < N2=seed{seed_a} (time={time_a:.1f}s)")
        return seed_b, seed_a, ep_b, ep_a


def parse_vaghar_results(results_dir, ctag, perturbation, field='upper_bound'):
    """Parse VHAGaR result files for a given ctag.

    Returns dict mapping c_target (0-indexed) -> value of the specified field.
    """
    bounds = {}
    pattern = re.compile(rf"_{perturbation}_.*_ctag{ctag - 1}__.*_cTag{ctag}\.txt$")
    if not os.path.exists(results_dir):
        return bounds
    for filename in os.listdir(results_dir):
        if not pattern.search(filename):
            continue
        filepath = os.path.join(results_dir, filename)
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = dict(p.split('=') for p in line.split(','))
                c_target = int(parts['c_target'])
                bounds[c_target] = float(parts[field])
    return bounds


def parse_transfer_results(results_dir, ctag, perturbation):
    """Parse transfer result files for a given ctag.

    Returns dict mapping c_target (0-indexed) -> lower_bound.
    """
    bounds = {}
    # Transfer files use ctag as 1-indexed in the filename
    pattern = re.compile(rf"_transfer_{perturbation}_.*_ctag{ctag}.*\.txt$")
    if not os.path.exists(results_dir):
        return bounds
    for filename in os.listdir(results_dir):
        if not pattern.search(filename):
            continue
        filepath = os.path.join(results_dir, filename)
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = dict(p.split('=') for p in line.split(','))
                c_target = int(parts['c_target'])
                bounds[c_target] = float(parts['lower_bound'])
    return bounds


def plot_confidence(arch, dataset, ctag, perturbation, perturbation_size, itr_n1, itr_n2):
    """Plot confidence per (ctag, c_target) with upper bound from VHAGaR standard for N2."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model_cls, _ = ARCH_REGISTRY[arch]
    _, k, w, h, _ = DATASET_CONFIG[dataset]
    exp_dir = os.path.join(SCRIPT_DIR, '..', 'paper_experiments', dataset, f'{arch}_exp')
    dirs = get_exp_dirs(arch, dataset, itr_n1, itr_n2,
                        perturbation=perturbation, perturbation_size=perturbation_size)

    if not os.path.exists(exp_dir):
        print(f"ERROR: Experiment directory {exp_dir} not found.")
        sys.exit(1)

    # Get upper bounds from N2's VHAGaR standard results (try with PI, fall back to noPI)
    n2_bounds = parse_vaghar_results(dirs['vaghar_n2'], ctag, perturbation)
    if not n2_bounds:
        n2_bounds = parse_vaghar_results(dirs['vaghar_n2_noPI'], ctag, perturbation)

    if not n2_bounds:
        print(f"WARNING: No VHAGaR results found for ctag={ctag}, {perturbation} eps={perturbation_size}")
        return

    # Get lower bounds from N1's VHAGaR standard results
    n1_lower = parse_vaghar_results(dirs['vaghar_n1'], ctag, perturbation, field='lower_bound')
    if not n1_lower:
        n1_lower = parse_vaghar_results(dirs['vaghar_n1_noPI'], ctag, perturbation, field='lower_bound')

    # Find transfer directory with relaxation threshold 0 only
    # Check multiple possible locations and naming patterns
    transfer_base = os.path.basename(dirs['transfer'])  # e.g. transfer_3x10_N1_is_itr50
    pert_dir = os.path.dirname(dirs['transfer'])
    ablation_dir = os.path.join(os.path.dirname(pert_dir), 'ablation_for_T_size')
    candidate_dirs = []
    for parent in [pert_dir, ablation_dir]:
        if not os.path.isdir(parent):
            continue
        for name in os.listdir(parent):
            full = os.path.join(parent, name)
            if not os.path.isdir(full):
                continue
            # Match patterns like _relax0, _relax0.0, _linf_relax0, _linf_relax0.0
            if name.startswith(transfer_base) and re.search(r'_relax0(\.0)?$', name):
                candidate_dirs.append(full)
    transfer_bounds = {}
    for tdir in candidate_dirs:
        transfer_bounds = parse_transfer_results(tdir, ctag, perturbation)
        if transfer_bounds:
            print(f"  Using transfer results from: {tdir}")
            break

    # Load N2 model
    pth_path = os.path.join(dirs['model_n2'], 'model.pth')
    if not os.path.exists(pth_path):
        print(f"ERROR: Model not found at {pth_path}")
        return

    print(f"Plotting confidence for itr{itr_n2}, ctag={ctag}, {perturbation} eps={perturbation_size}")

    _, test_loader = get_data_loaders(dataset, batch_size=256)
    device = torch.device("cpu")
    c = ctag - 1  # 0-indexed class

    model = model_cls(k=k, w=w, h=h).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    # Compute confidence C(N, x, c) = y_c(x) - max_{k != c} y_k(x)
    all_confs = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            score_c = outputs[:, c]
            mask = torch.ones(outputs.size(1), dtype=torch.bool)
            mask[c] = False
            max_other = outputs[:, mask].max(dim=1).values
            conf = score_c - max_other
            all_confs.append(conf.cpu())

    all_confs = torch.cat(all_confs).numpy()
    # Keep only positive confidence values
    all_confs = all_confs[all_confs > 0]
    all_confs = np.round(all_confs, 2)

    # One plot per c_target
    colors = ['green', 'blue', 'purple', 'orange', 'cyan', 'magenta']
    for c_target, upper_bound in sorted(n2_bounds.items()):
        upper_bound = round(upper_bound, 2)
        plt.figure(figsize=(14, 6))
        plt.scatter(range(len(all_confs)), all_confs, alpha=0.8, s=10, label='confidence')
        plt.axhline(y=upper_bound, color='red', linestyle='--', linewidth=1.5,
                     label=f'delta vhagar(N2) = {upper_bound:.2f}')

        # Add delta_vaghar(N1) + transfer line (relax0 only)
        delta_n1 = n1_lower.get(c_target)
        t_val = transfer_bounds.get(c_target)
        if delta_n1 is not None and t_val is not None:
            delta_n1 = round(delta_n1, 2)
            t_val = round(t_val, 2)
            combined = round(delta_n1 + t_val, 2)
            plt.axhline(y=combined, color='green', linestyle='-.', linewidth=1.5,
                        label=f'delta(N1)+transfer = {delta_n1:.2f}+{t_val:.2f} = {combined:.2f}')

        plt.xlabel('Test samples')
        plt.ylabel(f'Confidence C(N, x, class={ctag})')
        plt.title(f'class {ctag} vs {c_target} — {arch} on {dataset}, itr{itr_n2}, '
                   f'{perturbation} eps={perturbation_size}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(
            exp_dir, f'confidence_c{ctag}_ct{c_target}_{perturbation}_{perturbation_size}_itr{itr_n2}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved to {save_path}")


def plot_confidence_both(arch, dataset, ctag, perturbation, perturbation_size, itr_n1, itr_n2):
    """Plot confidence of both N1 and N2 on the same figure with delta_vaghars for both."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model_cls, _ = ARCH_REGISTRY[arch]
    _, k, w, h, _ = DATASET_CONFIG[dataset]
    exp_dir = os.path.join(SCRIPT_DIR, '..', 'paper_experiments', dataset, f'{arch}_exp')
    dirs = get_exp_dirs(arch, dataset, itr_n1, itr_n2,
                        perturbation=perturbation, perturbation_size=perturbation_size)

    if not os.path.exists(exp_dir):
        print(f"ERROR: Experiment directory {exp_dir} not found.")
        sys.exit(1)

    # Get upper bounds from both models' VHAGaR standard results
    n1_bounds = parse_vaghar_results(dirs['vaghar_n1'], ctag, perturbation)
    if not n1_bounds:
        n1_bounds = parse_vaghar_results(dirs['vaghar_n1_noPI'], ctag, perturbation)

    n2_bounds = parse_vaghar_results(dirs['vaghar_n2'], ctag, perturbation)
    if not n2_bounds:
        n2_bounds = parse_vaghar_results(dirs['vaghar_n2_noPI'], ctag, perturbation)

    all_c_targets = sorted(set(list(n1_bounds.keys()) + list(n2_bounds.keys())))
    if not all_c_targets:
        print(f"WARNING: No VHAGaR results found for ctag={ctag}, {perturbation} eps={perturbation_size}")
        return

    # Load test set
    _, test_loader = get_data_loaders(dataset, batch_size=256)
    device = torch.device("cpu")
    c = ctag - 1  # 0-indexed class

    # Compute confidence for both models
    def compute_confs(pth_path):
        model = model_cls(k=k, w=w, h=h).to(device)
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.eval()
        confs = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                score_c = outputs[:, c]
                mask = torch.ones(outputs.size(1), dtype=torch.bool)
                mask[c] = False
                max_other = outputs[:, mask].max(dim=1).values
                conf = score_c - max_other
                confs.append(conf.cpu())
        confs = torch.cat(confs).numpy()
        confs = confs[confs > 0]
        return np.round(confs, 2)

    pth_n1 = os.path.join(dirs['model_n1'], 'model.pth')
    pth_n2 = os.path.join(dirs['model_n2'], 'model.pth')
    if not os.path.exists(pth_n1) or not os.path.exists(pth_n2):
        print(f"ERROR: Model not found at {pth_n1} or {pth_n2}")
        return

    print(f"Plotting confidence (both) for itr{itr_n1} & itr{itr_n2}, ctag={ctag}, "
          f"{perturbation} eps={perturbation_size}")

    confs_n1 = compute_confs(pth_n1)
    confs_n2 = compute_confs(pth_n2)

    # One plot per c_target
    for c_target in all_c_targets:
        plt.figure(figsize=(14, 6))

        # Scatter confidence for both models
        plt.scatter(range(len(confs_n1)), confs_n1, alpha=0.6, s=10,
                    color='blue', label=f'confidence N1 (itr{itr_n1})')
        plt.scatter(range(len(confs_n2)), confs_n2, alpha=0.6, s=10,
                    color='orange', label=f'confidence N2 (itr{itr_n2})')

        # delta_vaghar(N1) upper bound
        ub_n1 = n1_bounds.get(c_target)
        if ub_n1 is not None:
            ub_n1 = round(ub_n1, 2)
            plt.axhline(y=ub_n1, color='blue', linestyle='--', linewidth=1.5,
                        label=f'delta vhagar(N1) = {ub_n1:.2f}')

        # delta_vaghar(N2) upper bound
        ub_n2 = n2_bounds.get(c_target)
        if ub_n2 is not None:
            ub_n2 = round(ub_n2, 2)
            plt.axhline(y=ub_n2, color='red', linestyle='--', linewidth=1.5,
                        label=f'delta vhagar(N2) = {ub_n2:.2f}')

        plt.xlabel('Test samples')
        plt.ylabel(f'Confidence C(N, x, class={ctag})')
        plt.title(f'class {ctag} vs {c_target} — {arch} on {dataset}, '
                   f'itr{itr_n1} & itr{itr_n2}, {perturbation} eps={perturbation_size}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(
            exp_dir, f'confidence_both_c{ctag}_ct{c_target}_{perturbation}_{perturbation_size}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved to {save_path}")


if __name__ == '__main__':
    main()
