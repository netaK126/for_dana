#!/usr/bin/env python3
"""
Distillation experiment pipeline:
  1. Train 6x10 teacher with SGD
  2. Train 3x10 student via knowledge distillation from 6x10 teacher
  3. Run VHAGaR standard for 3x10
  4. Run VHAGaR standard for 6x10
  5. Run VHAGaR transfer_distilation for 3x10 → 6x10
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
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from models import FNN_3_10, FNN_6_10

# ── paths ────────────────────────────────────────────────────────────────
EXP_DIR = os.path.join(os.path.dirname(__file__), '..', 'distilation_exp')
TEACHER_DIR = os.path.join(EXP_DIR, 'model_6x10')
STUDENT_DIR = os.path.join(EXP_DIR, 'model_3x10')
VAGHAR_3x10_DIR = os.path.join(EXP_DIR, 'vaghar_results_3x10')
VAGHAR_6x10_DIR = os.path.join(EXP_DIR, 'vaghar_results_6x10')
VAGHAR_PERT_3x10_DIR = os.path.join(EXP_DIR, 'vagharWithPerturbed_results_3x10')
VAGHAR_PERT_6x10_DIR = os.path.join(EXP_DIR, 'vagharWithPerturbed_results_6x10')
TRANSFER_NO_COMPOSED_DIR = os.path.join(EXP_DIR, 'transferWithoutComposed_3x10_to_6x10')
TRANSFER_COMPOSED_DIR = os.path.join(EXP_DIR, 'transferWithComposed_3x10_to_6x10')
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


# ── step 1: train teacher (6x10) ────────────────────────────────────────

def train_teacher(epochs=30, batch_size=128, lr=0.05):
    print("=" * 60)
    print("STEP 1: Training 6x10 teacher with SGD")
    print("=" * 60)
    device = torch.device("cpu")#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FNN_6_10().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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
        print(f"  Epoch {epoch+1}/{epochs} — loss: {running_loss/len(train_loader):.4f}, acc: {acc:.2f}%")

    save_model(model, TEACHER_DIR)
    print(f"  Teacher accuracy: {acc:.2f}%")
    return model


# ── step 2: distill student (3x10) from teacher ─────────────────────────

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    Combined loss:
      alpha     * KL(softmax(teacher/T), softmax(student/T)) * T^2
      (1-alpha) * CrossEntropy(student, labels)

    Higher alpha → student mimics teacher more closely → smaller delta_diff.
    We use alpha=0.7, T=4 to keep delta_diff positive but relatively small.
    """
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss


def train_student(teacher_model, epochs=30, batch_size=128, lr=0.05, T=4.0, alpha=0.7):
    """
    Knowledge distillation: 3x10 student from 6x10 teacher.

    Tuning rationale for small positive delta_diff:
    - alpha=0.7 gives strong teacher signal → student closely matches teacher confidence
    - T=4 softens logits enough to transfer dark knowledge
    - Same optimizer (SGD) and similar epochs so training dynamics are comparable
    - The student's smaller capacity (2 ReLU layers vs 5) means it cannot fully
      match the teacher, producing a natural positive delta_diff
    """
    print("=" * 60)
    print("STEP 2: Distilling 3x10 student from 6x10 teacher")
    print(f"  T={T}, alpha={alpha}")
    print("=" * 60)
    device = torch.device("cpu")#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    student = FNN_3_10().to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    train_loader, test_loader = get_data_loaders(batch_size)

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(images)
            student_logits = student(images)

            loss = distillation_loss(student_logits, teacher_logits, labels, T, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = evaluate(student, test_loader, device)
        print(f"  Epoch {epoch+1}/{epochs} — loss: {running_loss/len(train_loader):.4f}, acc: {acc:.2f}%")

    save_model(student, STUDENT_DIR)
    print(f"  Student accuracy: {acc:.2f}%")

    # Quick check: compare confidence margins on test set
    _print_confidence_comparison(teacher_model, student, test_loader, device)
    return student


def _print_confidence_comparison(teacher, student, test_loader, device):
    """Print average confidence margins to sanity-check delta_diff direction."""
    teacher.eval()
    student.eval()
    teacher_margins = []
    student_margins = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            for model, margins_list in [(teacher, teacher_margins), (student, student_margins)]:
                out = model(images)
                # C(N, x, c) = N(x)[c] - max_{k!=c} N(x)[k]
                correct_mask = (out.argmax(dim=1) == labels)
                if correct_mask.sum() == 0:
                    continue
                out_correct = out[correct_mask]
                labels_correct = labels[correct_mask]
                target_scores = out_correct.gather(1, labels_correct.unsqueeze(1)).squeeze()
                out_masked = out_correct.clone()
                out_masked.scatter_(1, labels_correct.unsqueeze(1), float('-inf'))
                max_other = out_masked.max(dim=1).values
                margins_list.append((target_scores - max_other).mean().item())

    avg_teacher = np.mean(teacher_margins)
    avg_student = np.mean(student_margins)
    print(f"  Avg confidence margin — Teacher: {avg_teacher:.4f}, Student: {avg_student:.4f}")
    print(f"  Expected delta_diff direction: {avg_teacher - avg_student:.4f} "
          f"({'positive ✓' if avg_teacher > avg_student else 'WARNING: negative'})")


# ── step 3 & 4: run VHAGaR standard ─────────────────────────────────────

def run_julia(args_list, step_name):
    """Run julia run.jl with given arguments."""
    cmd = ['julia', 'run.jl'] + args_list
    print(f"\n  Running: {' '.join(cmd[:6])}...")
    proc = subprocess.run(cmd, cwd=RUN_JL_DIR)
    if proc.returncode != 0:
        print(f"  WARNING: {step_name} exited with code {proc.returncode}")
    return proc.returncode


def run_vaghar_standard(model_name, model_path, output_dir, perturbation_size='0.05',
                        ctag=1, ct='4,5,6,7,8,9,10', timeout=1000,
                        use_perturbed_intervals=False, perturbation='linf'):
    """Run VHAGaR in standard mode for a single network."""
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
        '--use_hyper_attack', 'true',
        '--activate_vaghgar_deps', 'true',
        '--use_perturbed_intervals', str(use_perturbed_intervals).lower(),
    ]
    return run_julia(args, f'VHAGaR standard {model_name}')


def step3_vaghar_3x10(perturbation_size, ctag, ct, timeout, perturbation):
    print("=" * 60)
    print("STEP 3: Running VHAGaR standard for 3x10")
    print("=" * 60)
    model_path = os.path.join(STUDENT_DIR, 'model.p')
    run_vaghar_standard('3x10', model_path, VAGHAR_3x10_DIR,
                        perturbation_size=perturbation_size, ctag=ctag, ct=ct, timeout=timeout, perturbation=perturbation)
    # Also run with perturbed intervals for transfer mode's vaghar_results
    print("\n  Running with perturbed intervals...")
    run_vaghar_standard('3x10', model_path, VAGHAR_PERT_3x10_DIR,
                        perturbation_size=perturbation_size, ctag=ctag, ct=ct, timeout=timeout,
                        use_perturbed_intervals=True, perturbation=perturbation)


def step4_vaghar_6x10(perturbation_size, ctag, ct, timeout, perturbation):
    print("=" * 60)
    print("STEP 4: Running VHAGaR standard for 6x10")
    print("=" * 60)
    model_path = os.path.join(TEACHER_DIR, 'model.p')
    run_vaghar_standard('6x10', model_path, VAGHAR_6x10_DIR,
                        perturbation_size=perturbation_size, ctag=ctag, ct=ct, timeout=timeout, perturbation=perturbation)
    # Also run with perturbed intervals
    print("\n  Running with perturbed intervals...")
    run_vaghar_standard('6x10', model_path, VAGHAR_PERT_6x10_DIR,
                        perturbation_size=perturbation_size, ctag=ctag, ct=ct, timeout=timeout,
                        use_perturbed_intervals=True, perturbation=perturbation)


# ── step 5: run transfer_distilation ─────────────────────────────────────

def run_transfer_distilation_from_results(vaghar_results_dir, output_dir, timeout, composed_interval, perturbation):
    """
    Iterate over VHAGaR results files for N1 (3x10).
    Each file contains delta_1 values for a specific perturbation_size and c_tag.
    Parse these from the filename and launch a transfer_distilation run.
    Same logic as run_exp.py.
    """
    pattern = re.compile(rf"_{perturbation}_(.*?)_ctag.*cTag(\d+)\.txt$")

    if not os.path.exists(vaghar_results_dir):
        print(f"  Error: Directory {vaghar_results_dir} not found.")
        return

    student_path = os.path.join(STUDENT_DIR, 'model.p')
    teacher_path = os.path.join(TEACHER_DIR, 'model.p')

    for filename in sorted(os.listdir(vaghar_results_dir)):
        match = pattern.search(filename)
        if not match:
            continue

        perturbation_size = match.group(1)
        c_tag_n = match.group(2)
        if c_tag_n != '2' and c_tag_n != '3' and perturbation_size != '0.05':  # Only run for c_tag=1 to limit total runs
            continue
        vaghar_results_path = os.path.join(vaghar_results_dir, filename)

        print(f"  Processing: {filename}  (eps={perturbation_size}, ctag={c_tag_n})")

        command = [
            '--mode', 'transfer_distilation',
            '--dataset', 'mnist',
            '--model_name', '3x10',
            '--model_name2', '6x10',
            '--model_path', student_path,
            '--model_path2', teacher_path,
            '--vaghar_results', vaghar_results_path,
            '--perturbation', perturbation,
            '--perturbation_size', perturbation_size,
            '--ctag', c_tag_n,
            '--ct', '1,2,3,4,5,6,7,8,9,10',
            '--timout', str(timeout),
            '--output_dir', output_dir + '/',
            '--c_tag_mode', 'false',
            '--use_intervals', 'true',
            '--use_perturbed_intervals', 'true',
            '--composed_interval', str(composed_interval).lower(),
            '--use_hyper_attack', 'true',
            '--n1_p_mode', 'false',
            "--n2_fewer_binars_encoding", 'true',
        ]
        run_julia(command, f'transfer_distilation (composed={composed_interval}, ctag={c_tag_n})')


def step5_transfer_distilation(timeout, perturbation):
    print("=" * 60)
    print("STEP 5: Running transfer_distilation (3x10 → 6x10)")
    print("=" * 60)

    # Run WITHOUT composed intervals
    print("\n  --- Without composed intervals ---")
    run_transfer_distilation_from_results(VAGHAR_3x10_DIR, TRANSFER_NO_COMPOSED_DIR, timeout, composed_interval=False, perturbation=perturbation)

    # Run WITH composed intervals
    print("\n  --- With composed intervals ---")
    run_transfer_distilation_from_results(VAGHAR_3x10_DIR, TRANSFER_COMPOSED_DIR, timeout, composed_interval=True, perturbation=perturbation)


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Distillation experiment: 3x10 student ← 6x10 teacher',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--teacher_epochs', type=int, default=10, help='Teacher training epochs')
    parser.add_argument('--student_epochs', type=int, default=10, help='Student distillation epochs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.05, help='SGD learning rate')
    parser.add_argument('--T', type=float, default=4.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='KD loss weight (higher = more teacher signal)')
    parser.add_argument('--perturbation_size', type=str, default='0.05')
    parser.add_argument('--perturbation', type=str, default='linf')
    parser.add_argument('--ct', type=str, default='2,3,4,5,6,7,8,9,10', help='Target classes')
    parser.add_argument('--timeout', type=int, default=4000, help='MIP timeout per class pair')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, use existing models')
    parser.add_argument('--skip_vaghar', action='store_true', help='Skip standard VHAGaR, go to transfer')
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    if not args.skip_training:
        # Step 1: Train teacher
        teacher = train_teacher(epochs=args.teacher_epochs, batch_size=args.batch_size, lr=args.lr)

        # Step 2: Distill student
        train_student(teacher, epochs=args.student_epochs, batch_size=args.batch_size,
                      lr=args.lr, T=args.T, alpha=args.alpha)
    else:
        print("Skipping training (--skip_training)")

    if not args.skip_vaghar:
        for ctag in range(1, 5):
        # Step 3: VHAGaR standard for 3x10
            step3_vaghar_3x10(args.perturbation_size, ctag, args.ct, args.timeout, args.perturbation)

            # Step 4: VHAGaR standard for 6x10
            step4_vaghar_6x10(args.perturbation_size, ctag, args.ct, args.timeout, args.perturbation)
    else:
        print("Skipping standard VHAGaR (--skip_vaghar)")

    # Step 5: Transfer distilation
    step5_transfer_distilation(args.timeout, args.perturbation)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Teacher model:        {TEACHER_DIR}/model.p")
    print(f"  Student model:        {STUDENT_DIR}/model.p")
    print(f"  VHAGaR 3x10:          {VAGHAR_3x10_DIR}/")
    print(f"  VHAGaR 6x10:          {VAGHAR_6x10_DIR}/")
    print(f"  Transfer (no composed): {TRANSFER_NO_COMPOSED_DIR}/")
    print(f"  Transfer (composed):    {TRANSFER_COMPOSED_DIR}/")


if __name__ == '__main__':
    main()
