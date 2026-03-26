#!/usr/bin/env python3
"""
Benchmark: VHAGaR direct vs Transfer verification.

Setup: Take a trained network N1. Create N2 by adding small random
perturbation to the LAST layer weights only. All earlier layers are
identical → diff bounds are zero for layers 1..L-1 → nearly all
neurons get relaxed in transfer → transfer MIP should be much faster.

Compares:
  - VHAGaR standard on N1
  - VHAGaR standard on N2 (direct verification)
  - Transfer from N1 → N2 (with various relaxation thresholds)

Usage:
  # Upload to server, run from /root/Downloads/for_dana/vaghar_org/:
  python3 dana_benchmark_transfer.py --phase all
  python3 dana_benchmark_transfer.py --phase train
  python3 dana_benchmark_transfer.py --phase standard
  python3 dana_benchmark_transfer.py --phase transfer
  python3 dana_benchmark_transfer.py --phase report

All outputs go to dana_exp/ (never touches Neta's directories).
"""
import os
import sys
import subprocess
import time
import argparse
import glob
import copy

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VAGHAR_DIR = os.environ.get("VAGHAR_DIR", SCRIPT_DIR)
JULIA = "/root/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia"
RUN_JL = os.path.join(VAGHAR_DIR, "run.jl")
DANA_EXP = os.path.join(VAGHAR_DIR, "dana_exp")

# ── Experiment configs ───────────────────────────────────────────────────
# Weight perturbation magnitudes for the last layer
WEIGHT_NOISE_SCALES = [0.01, 0.05, 0.1]

EXPERIMENTS = [
    {
        "arch": "3x50",
        "dataset": "mnist",
        "train_epochs": 100,
        "lr": 1e-3,
        "seed": 42,
        "perturbations": [
            ("linf", "0.05", 1800),
            ("brightness", "0.25", 600),
        ],
        "transfer_thresholds": [0.5, 1.0, 2.0, 5.0],
    },
    {
        "arch": "cnn1",
        "dataset": "mnist",
        "train_epochs": 20,
        "lr": 1e-3,
        "seed": 42,
        "perturbations": [
            ("linf", "0.05", 1800),
            ("patch", "1,14,14,3", 1800),
        ],
        "transfer_thresholds": [0.5, 1.0, 2.0, 5.0],
    },
]

MAX_PARALLEL = 3
GUROBI_THREADS = 16


# ── Model definitions (must match Neta's models.py) ─────────────────────

def get_models():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FNN_3_50(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 50)
            self.fc2 = nn.Linear(50, 50)
            self.fc3 = nn.Linear(50, 10)
        def forward(self, x):
            x = x.reshape(-1, 784)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    class CNN1(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1)
            self.flatten1 = nn.Flatten()
            self.fc1 = nn.Linear(16 * 7 * 7, 50)
            self.fc2 = nn.Linear(50, 10)
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.flatten1(x)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    return {"3x50": FNN_3_50, "cnn1": CNN1}


# ── Training + weight perturbation ──────────────────────────────────────

def train_and_perturb(arch, dataset, epochs, lr, seed, noise_scales):
    """Train N1, then create N2 variants by perturbing last-layer weights.
    Returns {noise_scale: (n1_dir, n2_dir)}."""
    import torch
    import torch.nn as nn
    import torchvision.datasets as dsets
    import torchvision.transforms as transforms
    import numpy as np
    import pickle

    torch.manual_seed(seed)
    np.random.seed(seed)

    models = get_models()
    model = models[arch]()
    device = torch.device("cpu")
    model = model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "mnist":
        train_ds = dsets.MNIST(root='./data/', train=True, transform=transform, download=True)
        test_ds = dsets.MNIST(root='./data/', train=False, transform=transform, download=True)
    elif dataset == "fmnist":
        train_ds = dsets.FashionMNIST(root='./data/', train=True, transform=transform, download=True)
        test_ds = dsets.FashionMNIST(root='./data/', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    def evaluate(m):
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = m(images.to(device))
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
        return 100.0 * correct / total

    def save_model(m, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        params = []
        for p in m.parameters():
            arr = p.cpu().detach().numpy()
            params.append(np.transpose(arr))
        with open(os.path.join(save_dir, 'model.p'), 'wb') as f:
            pickle.dump(params, f)
        torch.save(m.state_dict(), os.path.join(save_dir, 'model.pth'))

    # Train
    print(f"Training {arch} on {dataset}, seed={seed}, {epochs} epochs")
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch == epochs:
            acc = evaluate(model)
            print(f"  Epoch {epoch}/{epochs} — acc: {acc:.2f}%")

    # Save N1
    n1_dir = os.path.join(DANA_EXP, f"{arch}_base_seed{seed}")
    save_model(model, n1_dir)
    n1_acc = evaluate(model)
    print(f"  N1 saved: {n1_dir} (acc={n1_acc:.2f}%)")

    # Create N2 variants by perturbing last layer
    results = {}
    # Find last layer parameters (weight and bias)
    param_names = list(model.state_dict().keys())
    # Last layer: last weight and last bias
    last_weight_name = [n for n in param_names if "weight" in n][-1]
    last_bias_name = [n for n in param_names if "bias" in n][-1]
    print(f"  Perturbing last layer: {last_weight_name}, {last_bias_name}")

    for scale in noise_scales:
        n2_model = models[arch]()
        n2_model.load_state_dict(copy.deepcopy(model.state_dict()))

        # Add noise to last layer only
        with torch.no_grad():
            sd = n2_model.state_dict()
            w = sd[last_weight_name]
            b = sd[last_bias_name]
            torch.manual_seed(seed + int(scale * 1000))
            sd[last_weight_name] = w + scale * torch.randn_like(w)
            sd[last_bias_name] = b + scale * torch.randn_like(b)
            n2_model.load_state_dict(sd)

        n2_dir = os.path.join(DANA_EXP, f"{arch}_lastlayer_noise{scale}_seed{seed}")
        save_model(n2_model, n2_dir)
        n2_acc = evaluate(n2_model)
        print(f"  N2 (noise={scale}): acc={n2_acc:.2f}%, saved to {n2_dir}")
        results[scale] = (n1_dir, n2_dir)

    return results


# ── VHAGaR runners ──────────────────────────────────────────────────────

def run_julia(args, label, log_file):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    env["MKL_NUM_THREADS"] = "8"
    print(f"  Launching: {label}")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [JULIA, RUN_JL] + args,
            stdout=f, stderr=subprocess.STDOUT,
            cwd=VAGHAR_DIR, env=env,
        )
    return proc


def run_vaghar_standard(arch, dataset, model_dir, perturbation, perturbation_size,
                        timeout, output_dir, ctag=1):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.p")
    use_pert_int = "false" if perturbation == "brightness" else "true"
    args = [
        "--mode", "standard",
        "--dataset", dataset,
        "--model_name", arch,
        "--model_path", model_path,
        "--perturbation", perturbation,
        "--perturbation_size", perturbation_size,
        "--ctag", str(ctag),
        "--ct", "2",
        "--timout", str(timeout),
        "--output_dir", output_dir + "/",
        "--c_tag_mode", "false",
        "--use_hyper_attack", "true",
        "--activate_vaghgar_deps", "true",
        "--use_perturbed_intervals", use_pert_int,
        "--force_cpu", "true",
    ]
    label = f"standard {arch} {perturbation} {os.path.basename(model_dir)}"
    log_file = os.path.join(output_dir, "log.txt")
    return run_julia(args, label, log_file)


def run_vaghar_transfer(arch, dataset, n1_dir, n2_dir, vaghar_results_file,
                        perturbation, perturbation_size, timeout, output_dir,
                        relaxation_threshold, ctag=1):
    os.makedirs(output_dir, exist_ok=True)
    n1_path = os.path.join(n1_dir, "model.p")
    n2_path = os.path.join(n2_dir, "model.p")
    use_pert_int = "false" if perturbation == "brightness" else "true"
    args = [
        "--mode", "transfer",
        "--dataset", dataset,
        "--model_name", arch,
        "--model_path", n1_path,
        "--model_path2", n2_path,
        "--vaghar_results", vaghar_results_file,
        "--perturbation", perturbation,
        "--perturbation_size", perturbation_size,
        "--ctag", str(ctag),
        "--ct", "2",
        "--timout", str(timeout),
        "--output_dir", output_dir + "/",
        "--c_tag_mode", "false",
        "--use_hyper_attack", "true",
        "--activate_vaghgar_deps", "true",
        "--use_intervals", "true",
        "--use_perturbed_intervals", use_pert_int,
        "--n2_fewer_binars_encoding", "true",
        "--use_relaxations", "true",
        "--delta_diff_positive", "false",
        "--force_cpu", "true",
        "--relaxation_threshold", str(relaxation_threshold),
    ]
    label = f"transfer {arch} {perturbation} noise→thresh={relaxation_threshold}"
    log_file = os.path.join(output_dir, "log.txt")
    return run_julia(args, label, log_file)


# ── Helpers ──────────────────────────────────────────────────────────────

def find_result_file(result_dir):
    """Find VHAGaR result .txt (not log.txt)."""
    if not os.path.exists(result_dir):
        return None
    for f in sorted(glob.glob(os.path.join(result_dir, "*.txt"))):
        if os.path.basename(f) != "log.txt":
            return f
    return None


def parse_results(result_dir):
    results = []
    if not os.path.exists(result_dir):
        return results
    for f in sorted(glob.glob(os.path.join(result_dir, "*.txt"))):
        if os.path.basename(f) == "log.txt":
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = {}
                for pair in line.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        try:
                            d[k] = float(v)
                        except ValueError:
                            d[k] = v
                if d:
                    results.append(d)
    return results


def wait_for_procs(procs, poll_interval=15):
    while any(p.poll() is None for p in procs):
        running = sum(1 for p in procs if p.poll() is None)
        print(f"    ... {running} still running")
        time.sleep(poll_interval)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["train", "standard", "transfer", "report", "all"])
    args = parser.parse_args()

    os.makedirs(DANA_EXP, exist_ok=True)

    for exp in EXPERIMENTS:
        arch = exp["arch"]
        dataset = exp["dataset"]
        seed = exp["seed"]
        n1_tag = f"{arch}_base_seed{seed}"
        n1_dir = os.path.join(DANA_EXP, n1_tag)

        # ── Phase 1: Train + create perturbed variants ───────────────
        if args.phase in ("train", "all"):
            last_n2 = os.path.join(DANA_EXP,
                f"{arch}_lastlayer_noise{WEIGHT_NOISE_SCALES[-1]}_seed{seed}", "model.p")
            if os.path.exists(last_n2):
                print(f"Models exist for {arch}, skipping training")
            else:
                train_and_perturb(
                    arch, dataset, exp["train_epochs"], exp["lr"], seed,
                    WEIGHT_NOISE_SCALES
                )

        # ── Phase 2: VHAGaR standard on N1 and each N2 ──────────────
        if args.phase in ("standard", "all"):
            print(f"\n{'='*60}")
            print(f"VHAGaR standard: {arch}")
            print(f"{'='*60}")
            procs = []

            for pert, psize, timeout in exp["perturbations"]:
                # N1
                out_n1 = os.path.join(DANA_EXP, f"vaghar_{n1_tag}_{pert}_{psize}")
                if not find_result_file(out_n1):
                    p = run_vaghar_standard(arch, dataset, n1_dir, pert, psize, timeout, out_n1)
                    procs.append(p)
                else:
                    print(f"  Skip (exists): {os.path.basename(out_n1)}")

                # Each N2 variant
                for scale in WEIGHT_NOISE_SCALES:
                    n2_tag = f"{arch}_lastlayer_noise{scale}_seed{seed}"
                    n2_dir = os.path.join(DANA_EXP, n2_tag)
                    out_n2 = os.path.join(DANA_EXP, f"vaghar_{n2_tag}_{pert}_{psize}")
                    if not find_result_file(out_n2):
                        p = run_vaghar_standard(arch, dataset, n2_dir, pert, psize, timeout, out_n2)
                        procs.append(p)
                    else:
                        print(f"  Skip (exists): {os.path.basename(out_n2)}")

                    # Rate limit
                    while sum(1 for pp in procs if pp.poll() is None) >= MAX_PARALLEL:
                        time.sleep(10)

            if procs:
                wait_for_procs(procs)
            print("  Standard phase done.")

        # ── Phase 3: Transfer N1 → each N2 ──────────────────────────
        if args.phase in ("transfer", "all"):
            print(f"\n{'='*60}")
            print(f"Transfer: {arch}")
            print(f"{'='*60}")
            procs = []

            for pert, psize, timeout in exp["perturbations"]:
                # Need N1 standard results
                n1_results_dir = os.path.join(DANA_EXP, f"vaghar_{n1_tag}_{pert}_{psize}")
                vaghar_file = find_result_file(n1_results_dir)
                if not vaghar_file:
                    print(f"  No N1 results for {pert} {psize}, skipping")
                    continue

                for scale in WEIGHT_NOISE_SCALES:
                    n2_tag = f"{arch}_lastlayer_noise{scale}_seed{seed}"
                    n2_dir = os.path.join(DANA_EXP, n2_tag)

                    for thresh in exp["transfer_thresholds"]:
                        out_dir = os.path.join(
                            DANA_EXP,
                            f"transfer_{n1_tag}_noise{scale}_{pert}_{psize}_relax{thresh}"
                        )
                        if find_result_file(out_dir):
                            print(f"  Skip (exists): {os.path.basename(out_dir)}")
                            continue
                        p = run_vaghar_transfer(
                            arch, dataset, n1_dir, n2_dir, vaghar_file,
                            pert, psize, timeout, out_dir, thresh,
                        )
                        procs.append(p)
                        while sum(1 for pp in procs if pp.poll() is None) >= MAX_PARALLEL:
                            time.sleep(10)

            if procs:
                wait_for_procs(procs)
            print("  Transfer phase done.")

        # ── Phase 4: Report ──────────────────────────────────────────
        if args.phase in ("report", "all"):
            print(f"\n{'='*60}")
            print(f"RESULTS: {arch} — last-layer weight perturbation")
            print(f"{'='*60}")

            for pert, psize, timeout in exp["perturbations"]:
                print(f"\n  Perturbation: {pert} {psize} (timeout={timeout}s)")
                print(f"  {'-'*55}")

                # N1 standard
                n1_res = parse_results(os.path.join(DANA_EXP, f"vaghar_{n1_tag}_{pert}_{psize}"))
                if n1_res:
                    r = n1_res[0]
                    print(f"    N1 (base):     LB={r.get('lower_bound','?'):>8.2f}  "
                          f"UB={r.get('upper_bound','?'):>8.2f}  "
                          f"time={r.get('optimization_time',0):>7.0f}s")

                for scale in WEIGHT_NOISE_SCALES:
                    n2_tag = f"{arch}_lastlayer_noise{scale}_seed{seed}"
                    print(f"\n    --- noise={scale} ---")

                    # N2 direct
                    n2_res = parse_results(
                        os.path.join(DANA_EXP, f"vaghar_{n2_tag}_{pert}_{psize}"))
                    if n2_res:
                        r = n2_res[0]
                        direct_time = r.get('optimization_time', 0)
                        print(f"    N2 direct:     LB={r.get('lower_bound','?'):>8.2f}  "
                              f"UB={r.get('upper_bound','?'):>8.2f}  "
                              f"time={direct_time:>7.0f}s")
                    else:
                        direct_time = None
                        print(f"    N2 direct:     no results")

                    # Transfer results
                    for thresh in exp["transfer_thresholds"]:
                        t_res = parse_results(os.path.join(
                            DANA_EXP,
                            f"transfer_{n1_tag}_noise{scale}_{pert}_{psize}_relax{thresh}"
                        ))
                        if t_res:
                            r = t_res[0]
                            t_time = r.get('optimization_time', 0)
                            speedup = ""
                            if direct_time and direct_time > 0:
                                speedup = f"  ({direct_time/t_time:.1f}x)"
                            print(f"    Transfer t={thresh:<4}: LB={r.get('lower_bound','?'):>8.2f}  "
                                  f"UB={r.get('upper_bound','?'):>8.2f}  "
                                  f"time={t_time:>7.0f}s{speedup}")
                        else:
                            print(f"    Transfer t={thresh:<4}: no results")

    print("\nDone!")


if __name__ == "__main__":
    main()
