#!/usr/bin/env python3
"""
Benchmark experiment: Train N1/N2 pairs and compare VHAGaR standard vs transfer mode
across different flag combinations for model types 3x10, 4x10, 10x10.
"""

import subprocess
import os
import sys
import time
import glob
import csv
import pickle
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Import model classes from utils/models.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "utils"))
from models import FNN_3_10, FNN_4_10, FNN_10_10

# ─── Configuration ───────────────────────────────────────────────────────────

JULIA_SCRIPT = os.path.join(SCRIPT_DIR, "run.jl")
EXPERIMENT_DIR = os.path.join(SCRIPT_DIR, "experiment_models")
RESULTS_BASE = os.path.join(SCRIPT_DIR, "experiment_results")

TIMEOUT = 1200  # seconds

# Perturbation types and sizes to examine
PERTURBATION_TYPES = ["linf"]#, "brightness", "patch", "contrast"]
PERTURBATION_SIZES = [0.05]#[0.005, 0.01, 0.05]

# For patch perturbation: format is "eps,i,j,width" — we fix i=5, j=5, width=5
PATCH_POSITION = (5, 5, 5)  # (i, j, width)
TRAIN_EPOCHS_N1 = 20
FINETUNE_EPOCHS_N2 = 5
N2_WEIGHT_NOISE_EPS = 0.01
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

MODEL_TYPES = ["3x10", "4x10", "10x10"]

# Standard mode flag combinations
# activate_vaghgar_deps=true, use_hyper_attack=true, c_tag_mode=false (always)
STANDARD_CONFIGS = [
    {
        "name": "S1_deps_hyper",
        "activate_vaghgar_deps": True,
        "use_perturbed_intervals": False,
        "use_hyper_attack": True,
    },
    {
        "name": "S2_deps_hyper_pertIntervals",
        "activate_vaghgar_deps": True,
        "use_perturbed_intervals": True,
        "use_hyper_attack": True,
    },
]

# Transfer mode flag combinations
# c_tag_mode=false, n1_p_mode=false (always)
TRANSFER_CONFIGS = [
    {
        "name": "T1_baseline",
        "activate_vaghgar_deps": False,
        "use_intervals": False,
        "n2_fewer_binars_encoding": False,
    },
    {
        "name": "T2_fewerBinars",
        "activate_vaghgar_deps": False,
        "use_intervals": False,
        "n2_fewer_binars_encoding": True,
    },
    {
        "name": "T3_intervals",
        "activate_vaghgar_deps": False,
        "use_intervals": True,
        "n2_fewer_binars_encoding": False,
    },
    {
        "name": "T4_intervals_fewerBinars",
        "activate_vaghgar_deps": False,
        "use_intervals": True,
        "n2_fewer_binars_encoding": True,
    },
    {
        "name": "T5_deps",
        "activate_vaghgar_deps": True,
        "use_intervals": False,
        "n2_fewer_binars_encoding": False,
    },
    {
        "name": "T6_deps_fewerBinars",
        "activate_vaghgar_deps": True,
        "use_intervals": False,
        "n2_fewer_binars_encoding": True,
    },
    {
        "name": "T7_deps_intervals",
        "activate_vaghgar_deps": True,
        "use_intervals": True,
        "n2_fewer_binars_encoding": False,
    },
    {
        "name": "T8_deps_intervals_fewerBinars",
        "activate_vaghgar_deps": True,
        "use_intervals": True,
        "n2_fewer_binars_encoding": True,
    },
]

MODEL_CLASSES = {
    "3x10": FNN_3_10,
    "4x10": FNN_4_10,
    "10x10": FNN_10_10,
}

# ─── Perturbation Utilities ──────────────────────────────────────────────────


def format_perturbation_size(perturbation_type, eps):
    """Format the --perturbation_size string for a given perturbation type and eps value."""
    if perturbation_type == "patch":
        # Format: eps,i,j,width
        i, j, w = PATCH_POSITION
        return f"{eps},{i},{j},{w}"
    else:
        # linf, brightness, contrast all take a single eps
        return str(eps)


def perturbation_dir_label(perturbation_type, eps):
    """Short directory-safe label for a perturbation configuration."""
    return f"{perturbation_type}_{eps}"


# ─── Training Utilities ─────────────────────────────────────────────────────


def save_model_pickle(model, path):
    """Save model in the .p pickle format expected by Julia's get_nn()."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params = []
    for p in model.parameters():
        params.append(np.transpose(p.cpu().detach().numpy()))
    with open(path, "wb") as f:
        pickle.dump(params, f)
    print(f"  Saved model to {path}")


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


def train_model(model, train_loader, test_loader, device, epochs, lr=LEARNING_RATE):
    """Train a model and return it."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate(model, test_loader, device)
        print(f"    Epoch {epoch+1}/{epochs} - Test accuracy: {acc:.2f}%")

    return model


def create_n2_from_n1(n1_model, eps=N2_WEIGHT_NOISE_EPS):
    """Create N2 by copying N1 weights and adding small random noise."""
    n2_model = copy.deepcopy(n1_model)
    with torch.no_grad():
        for p in n2_model.parameters():
            noise = (torch.rand_like(p) * 2 - 1) * eps
            p.add_(noise)
    return n2_model


# ─── Part 1: Training ───────────────────────────────────────────────────────


def train_all_models():
    """Train N1 and N2 for each model type."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = dsets.MNIST(root="./data/", train=True, transform=transform, download=True)
    mnist_test = dsets.MNIST(root="./data/", train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

    model_paths = {}

    for model_type in MODEL_TYPES:
        print(f"\n{'='*60}")
        print(f"Training {model_type}")
        print(f"{'='*60}")

        n1_dir = os.path.join(EXPERIMENT_DIR, model_type, "N1")
        n2_dir = os.path.join(EXPERIMENT_DIR, model_type, "N2")
        n1_path = os.path.join(n1_dir, "model.p")
        n2_path = os.path.join(n2_dir, "model.p")

        # Skip if already trained
        if os.path.exists(n1_path) and os.path.exists(n2_path):
            print(f"  Models already exist, skipping training.")
            model_paths[model_type] = {"N1": n1_path, "N2": n2_path}
            continue

        # Train N1
        print(f"  Training N1 ({TRAIN_EPOCHS_N1} epochs)...")
        n1 = MODEL_CLASSES[model_type]()
        n1 = train_model(n1, train_loader, test_loader, device, TRAIN_EPOCHS_N1)
        save_model_pickle(n1, n1_path)
        torch.save(n1.state_dict(), n1_path+'th')

        # Create N2 from N1 with weight perturbation + fine-tuning
        print(f"  Creating N2 (noise eps={N2_WEIGHT_NOISE_EPS}, fine-tune {FINETUNE_EPOCHS_N2} epochs)...")
        n2 = create_n2_from_n1(n1, eps=N2_WEIGHT_NOISE_EPS)
        n2 = train_model(n2, train_loader, test_loader, device, FINETUNE_EPOCHS_N2, lr=LEARNING_RATE * 0.1)
        save_model_pickle(n2, n2_path)
        torch.save(n2.state_dict(), n2_path+'th')

        model_paths[model_type] = {"N1": n1_path, "N2": n2_path}

    return model_paths


# ─── Part 2 & 3: Julia Verification Runs ────────────────────────────────────


def run_julia(args_list, label=""):
    """Run julia run.jl with given arguments. Returns (success, stdout, stderr)."""
    cmd = ["julia", JULIA_SCRIPT] + args_list
    print(f"\n  [{label}] Running: {' '.join(cmd[:6])}...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT * 12 + 600,  # generous overall timeout
            cwd=SCRIPT_DIR,
        )
        if result.returncode != 0:
            print(f"  [{label}] FAILED (exit code {result.returncode})")
            print(f"  STDERR: {result.stderr[-2000:]}")
            return False, result.stdout, result.stderr
        print(f"  [{label}] Completed successfully.")
        return True, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"  [{label}] TIMED OUT")
        return False, "", "timeout"


def find_latest_result(results_dir, pattern="*.txt"):
    """Find the most recently created result file matching pattern."""
    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def run_standard_mode(model_type, n1_path, config, perturbation_type, eps):
    """Run standard mode verification on N1."""
    config_name = config["name"]
    pert_label = perturbation_dir_label(perturbation_type, eps)
    results_dir = os.path.join(RESULTS_BASE, model_type, "standard", pert_label, config_name)
    os.makedirs(results_dir, exist_ok=True)

    size_str = format_perturbation_size(perturbation_type, eps)
    name_to_save = f"{config_name}_{pert_label}"

    args = [
        "--mode", "standard",
        "--dataset", "mnist",
        "--model_name", model_type,
        "--model_path", n1_path,
        "--perturbation", perturbation_type,
        "--perturbation_size", size_str,
        "--timout", str(TIMEOUT),
        "--output_dir", results_dir + "/",
        "--name_to_save", name_to_save,
        "--activate_vaghgar_deps", str(config["activate_vaghgar_deps"]).lower(),
        "--use_perturbed_intervals", str(config["use_perturbed_intervals"]).lower(),
        "--use_hyper_attack", str(config["use_hyper_attack"]).lower(),
        "--c_tag_mode", "false",
    ]

    label = f"STD {model_type} {pert_label} {config_name}"
    success, stdout, stderr = run_julia(args, label)
    return results_dir, success


def run_transfer_mode(model_type, n1_path, n2_path, vaghar_results_path, config,
                      perturbation_type, eps):
    """Run transfer mode verification."""
    config_name = config["name"]
    pert_label = perturbation_dir_label(perturbation_type, eps)
    results_dir = os.path.join(RESULTS_BASE, model_type, "transfer", pert_label, config_name)
    os.makedirs(results_dir, exist_ok=True)

    size_str = format_perturbation_size(perturbation_type, eps)
    name_to_save = f"{config_name}_{pert_label}"

    args = [
        "--mode", "transfer",
        "--dataset", "mnist",
        "--model_name", model_type,
        "--model_path", n1_path,
        "--model_path2", n2_path,
        "--vaghar_results", vaghar_results_path,
        "--perturbation", perturbation_type,
        "--perturbation_size", size_str,
        "--timout", str(TIMEOUT),
        "--output_dir", results_dir + "/",
        "--name_to_save", name_to_save,
        "--c_tag_mode", "false",
        "--n1_p_mode", "false",
        "--activate_vaghgar_deps", str(config["activate_vaghgar_deps"]).lower(),
        "--use_intervals", str(config.get("use_intervals", False)).lower(),
        "--n2_fewer_binars_encoding", str(config.get("n2_fewer_binars_encoding", False)).lower(),
        "--use_hyper_attack", "false",
        "--use_perturbed_intervals", "false",
    ]

    label = f"TRANSFER {model_type} {pert_label} {config_name}"
    success, stdout, stderr = run_julia(args, label)
    return results_dir, success


# ─── Part 4: Results Parsing & Comparison ────────────────────────────────────


def parse_result_file(filepath):
    """Parse a VHAGaR result CSV file. Returns list of dicts."""
    rows = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 5:
                    try:
                        rows.append({
                            "source": int(parts[0]),
                            "target": int(parts[1]),
                            "incumbent_obj": float(parts[2]),
                            "best_bound": float(parts[3]),
                            "solve_time": float(parts[4]),
                        })
                    except ValueError:
                        continue
    except FileNotFoundError:
        pass
    return rows


def collect_all_results():
    """Walk experiment_results/ and collect all results into a unified table."""
    all_results = []

    for model_type in MODEL_TYPES:
        for perturbation_type in PERTURBATION_TYPES:
            for eps in PERTURBATION_SIZES:
                pert_label = perturbation_dir_label(perturbation_type, eps)

                # Standard mode results
                for config in STANDARD_CONFIGS:
                    config_name = config["name"]
                    results_dir = os.path.join(
                        RESULTS_BASE, model_type, "standard", pert_label, config_name)
                    result_files = glob.glob(os.path.join(results_dir, "*.txt"))
                    for rf in result_files:
                        rows = parse_result_file(rf)
                        for row in rows:
                            all_results.append({
                                "mode": "standard",
                                "model_type": model_type,
                                "perturbation": perturbation_type,
                                "eps": eps,
                                "config": config_name,
                                "file": os.path.basename(rf),
                                **row,
                            })

                # Transfer mode results
                for config in TRANSFER_CONFIGS:
                    config_name = config["name"]
                    results_dir = os.path.join(
                        RESULTS_BASE, model_type, "transfer", pert_label, config_name)
                    result_files = glob.glob(os.path.join(results_dir, "*.txt"))
                    for rf in result_files:
                        rows = parse_result_file(rf)
                        for row in rows:
                            all_results.append({
                                "mode": "transfer",
                                "model_type": model_type,
                                "perturbation": perturbation_type,
                                "eps": eps,
                                "config": config_name,
                                "file": os.path.basename(rf),
                                **row,
                            })

    return all_results


def print_comparison(all_results):
    """Print comparison tables for standard and transfer mode results."""
    if not all_results:
        print("\nNo results to compare.")
        return

    # Save to CSV
    summary_path = os.path.join(RESULTS_BASE, "summary.csv")
    os.makedirs(RESULTS_BASE, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "model_type", "perturbation", "eps", "config",
            "source", "target", "incumbent_obj", "best_bound", "solve_time", "file",
        ])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nFull results saved to: {summary_path}")

    # Print summary tables
    for mode_label in ["standard", "transfer"]:
        mode_results = [r for r in all_results if r["mode"] == mode_label]
        if not mode_results:
            continue

        print(f"\n{'='*100}")
        print(f"  {mode_label.upper()} MODE COMPARISON")
        print(f"{'='*100}")
        print(f"{'Model':<8} {'Pert':<12} {'Eps':<7} {'Config':<35} {'Src':>3} {'Tgt':>3} "
              f"{'Incumbent':>12} {'BestBound':>12} {'Time(s)':>10}")
        print("-" * 104)

        for model_type in MODEL_TYPES:
            mt_results = [r for r in mode_results if r["model_type"] == model_type]
            if not mt_results:
                continue
            mt_results.sort(key=lambda r: (r["perturbation"], r["eps"], r["config"],
                                           r["source"], r["target"]))
            for r in mt_results:
                print(
                    f"{r['model_type']:<8} {r['perturbation']:<12} {r['eps']:<7} "
                    f"{r['config']:<35} {r['source']:>3} {r['target']:>3} "
                    f"{r['incumbent_obj']:>12.6f} {r['best_bound']:>12.6f} {r['solve_time']:>10.2f}"
                )

    # Aggregate: mean solve_time and mean best_bound per (mode, model_type, perturbation, eps, config)
    print(f"\n{'='*110}")
    print(f"  AGGREGATE SUMMARY (mean across source/target pairs)")
    print(f"{'='*110}")
    print(f"{'Mode':<10} {'Model':<8} {'Pert':<12} {'Eps':<7} {'Config':<35} "
          f"{'MeanBound':>12} {'MeanTime(s)':>12} {'#Pairs':>7}")
    print("-" * 105)

    from collections import defaultdict
    agg = defaultdict(lambda: {"bounds": [], "times": []})
    for r in all_results:
        key = (r["mode"], r["model_type"], r["perturbation"], r["eps"], r["config"])
        agg[key]["bounds"].append(r["best_bound"])
        agg[key]["times"].append(r["solve_time"])

    for (mode, mt, pert, eps, config), vals in sorted(agg.items()):
        mean_bound = np.mean(vals["bounds"])
        mean_time = np.mean(vals["times"])
        n = len(vals["bounds"])
        print(f"{mode:<10} {mt:<8} {pert:<12} {eps:<7} {config:<35} "
              f"{mean_bound:>12.6f} {mean_time:>12.2f} {n:>7}")


# ─── Main Orchestration ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="VHAGaR benchmark experiment")
    parser.add_argument("--skip-training", default=False, action="store_true", help="Skip training, use existing models")
    parser.add_argument("--skip-standard", default=False, action="store_true", help="Skip standard mode runs")
    parser.add_argument("--skip-transfer", default=False, action="store_true", help="Skip transfer mode runs")
    parser.add_argument("--only-compare", default=False, action="store_true", help="Only compare existing results")
    parser.add_argument("--model-types", type=str, default="3x10,4x10,10x10",
                        help="Comma-separated model types to run (default: 3x10,4x10,10x10)")
    args = parser.parse_args()

    global MODEL_TYPES
    if args.model_types:
        MODEL_TYPES = args.model_types.split(",")

    if args.only_compare:
        all_results = collect_all_results()
        print_comparison(all_results)
        return

    # ── Step 1: Train models ──
    if not args.skip_training:
        print("\n" + "=" * 60)
        print("  STEP 1: Training N1 and N2 for each model type")
        print("=" * 60)
        model_paths = train_all_models()
    else:
        model_paths = {}
        for mt in MODEL_TYPES:
            model_paths[mt] = {
                "N1": os.path.join(EXPERIMENT_DIR, mt, "N1", "model.p"),
                "N2": os.path.join(EXPERIMENT_DIR, mt, "N2", "model.p"),
            }

    # ── Step 2: Standard mode runs ──
    # We need standard results first (to get vaghar_results for transfer mode)
    if not args.skip_standard:
        print("\n" + "=" * 60)
        print("  STEP 2: Standard mode verification on N1")
        print("=" * 60)

        for model_type in MODEL_TYPES:
            n1_path = model_paths[model_type]["N1"]
            for perturbation_type in PERTURBATION_TYPES:
                for eps in PERTURBATION_SIZES:
                    for config in STANDARD_CONFIGS:
                        run_standard_mode(model_type, n1_path, config,
                                          perturbation_type, eps)

    # ── Step 3: Transfer mode runs ──
    if not args.skip_transfer:
        print("\n" + "=" * 60)
        print("  STEP 3: Transfer mode verification (delta_diff)")
        print("=" * 60)
        c_tag_list = [1, 2, 3]  # We will run transfer mode for all cTag values, using the same vaghar_results for each cTag since it doesn't affect them
        for model_type in MODEL_TYPES:
            for c_tag in c_tag_list:
                n1_path = model_paths[model_type]["N1"]
                n2_path = model_paths[model_type]["N2"]

                for perturbation_type in PERTURBATION_TYPES:
                    for eps in PERTURBATION_SIZES:
                        pert_label = perturbation_dir_label(perturbation_type, eps)

                        # Find the standard result file for this perturbation config
                        # Use S1 (baseline standard) results for delta_1 values
                        s1_dir = os.path.join(
                            RESULTS_BASE, model_type, "standard", pert_label, "S1_deps_hyper")
                        vaghar_file = find_latest_result(s1_dir, "*_cTag"+str(c_tag)+".txt")
                        if vaghar_file is None:
                            # Try any standard config for this perturbation
                            for config in STANDARD_CONFIGS:
                                d = os.path.join(
                                    RESULTS_BASE, model_type, "standard",
                                    pert_label, config["name"])
                                vaghar_file = find_latest_result(d, "*_cTag"+str(c_tag)+".txt")
                                if vaghar_file:
                                    break

                        if vaghar_file is None:
                            print(f"\n  WARNING: No standard results for {model_type} "
                                f"{pert_label}. Skipping transfer mode for this config.")
                            continue

                        print(f"\n  Using vaghar_results for {model_type} {pert_label}: "
                            f"{vaghar_file}")

                        for config in TRANSFER_CONFIGS:
                            run_transfer_mode(model_type, n1_path, n2_path,
                                            vaghar_file, config,
                                            perturbation_type, eps)

    # ── Step 4: Compare results ──
    print("\n" + "=" * 60)
    print("  STEP 4: Results Comparison")
    print("=" * 60)

    all_results = collect_all_results()
    print_comparison(all_results)


if __name__ == "__main__":
    main()
