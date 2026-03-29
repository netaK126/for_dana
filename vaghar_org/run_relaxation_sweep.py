#!/usr/bin/env python3
"""
Run VHAGaR experiments in two phases:
  Phase 1: Standard experiments — one job per perturbation type (all run in parallel).
  Phase 2: Transfer experiments — one job per (perturbation, T, optimizing_intervals)
            combination (run in parallel after all Phase 1 jobs finish).

Two modes for obtaining N1/N2:
  Default (dual-seed): Uses two independently trained models from different seeds.
  --model_path MODE:   N1 = given model, N2 = N1 + extra SGD epoch(s).

Maintains up to N concurrent jobs (based on available cores).
When a job finishes, its core slot is immediately reused by the next job.

Usage:
    python3 run_relaxation_sweep.py                            # dual-seed mode (default)
    python3 run_relaxation_sweep.py --model_path /path/to/model_dir  # +1 epoch mode
    python3 run_relaxation_sweep.py --perturbations patch occ  # only these perturbations
    python3 run_relaxation_sweep.py --max_cores 100            # limit core usage
    python3 run_relaxation_sweep.py --skip_standard            # skip phase 1
    python3 run_relaxation_sweep.py --skip_transfer            # skip phase 2

Stop all with Ctrl+C.
"""

import subprocess
import signal
import sys
import os
import argparse
import time
import re
import glob

# ── Perturbation configs ─────────────────────────────────────────────────
# Each entry: (name, perturbation_spec)
PERTURBATIONS = [
    # ("contrast(1.5)",      "contrast:1.5"),
    ("patch(1,14,14,3)",  "patch:1,14,14,3"),
    ("occ(5,5,5)",      "occ:5,5,5"),
    ("occ(3,3,5)",      "occ:3,3,5"),
    # ("trans(1,1)",        "translation:1,1"),
    # ("trans(1,3)",        "translation:1,3"),
    # ("trans(3,1)",        "translation:3,1"),
    # ("trans(3,3)",        "translation:3,3"),
    # ("rotation(10)",      "rotation:10"),
    ("occ(1,1,5)",      "occ:1,1,5"),
    # ("linf(0.05)",        "linf:0.05"),
    # ("linf(0.1)",         "linf:0.1"),
    # ("brightness(0.25)",   "brightness:0.25"),
]

# ── Transfer sweep parameters ────────────────────────────────────────────
THRESHOLDS = [0, 0.05] # focused on best T_relax candidate
OPT_INTERVALS = ["true"]#["true", "false"]

# ── CPU pinning ──────────────────────────────────────────────────────────
CORES_PER_JOB = 32
CORE_START = 8  # first core to use (reserve 0-7)
TOTAL_CORES = 255


def standard_results_exist(pert_spec, cwd, arch="cnn1", dataset="mnist"):
    """Check if standard N2 results (.txt files) already exist for this perturbation.

    Returns True if at least one vagharNoPerturbed_*_sgd_itr* directory contains .txt
    result files (these are the standard N2 results that transfer needs).
    """
    pert_type, eps_str = pert_spec.split(":", 1)
    pert_dir_map = {
        "patch": "patch", "occ": "occ", "translation": "translation",
        "rotation": "rotation", "brightness": "brightness", "linf": "linf",
    }
    pert_dir = pert_dir_map.get(pert_type, pert_type)
    eps_dir = os.path.join(cwd, "paper_experiments", dataset, f"{arch}_exp",
                           pert_dir, f"eps_{eps_str}")
    if not os.path.isdir(eps_dir):
        return False
    # Look for vagharNoPerturbed_*_sgd_itr* dirs with .txt results
    n2_dirs = glob.glob(os.path.join(eps_dir, "vagharNoPerturbed_*_sgd_itr*"))
    for d in n2_dirs:
        if glob.glob(os.path.join(d, "*.txt")):
            return True
    return False


def run_pool(ready_jobs, max_slots, cwd, cores_per_job, phase_name="", locked_jobs=None):
    """Run jobs with CPU-pinned slot pooling.

    ready_jobs:  list of (label, cmd) that can start immediately.
    locked_jobs: dict of  standard_label -> [(label, cmd), ...]
                 Transfer jobs that become eligible only after the standard job
                 with the matching label finishes. Pass None (default) for no
                 dependencies (original single-phase behaviour).
    """
    if locked_jobs is None:
        locked_jobs = {}

    # Make a mutable copy so we can pop entries as they are unlocked.
    locked_jobs = {k: list(v) for k, v in locked_jobs.items()}

    total_jobs = len(ready_jobs) + sum(len(v) for v in locked_jobs.values())
    if total_jobs == 0:
        return

    print(f"\n{'=' * 60}")
    print(f"{phase_name}: {total_jobs} jobs total  "
          f"({len(ready_jobs)} ready now, {total_jobs - len(ready_jobs)} waiting on deps)  "
          f"{max_slots} concurrent slots  "
          f"({cores_per_job} cores/job, cores {CORE_START}-{CORE_START + max_slots * cores_per_job - 1})")
    print(f"{'=' * 60}\n")

    slots = [None] * max_slots
    job_queue = list(ready_jobs)
    finished = 0

    log_dir = os.path.join(cwd, "sweep_logs")
    os.makedirs(log_dir, exist_ok=True)

    def launch_in_slot(slot_idx, label, cmd):
        core_lo = CORE_START + slot_idx * cores_per_job
        core_hi = core_lo + cores_per_job - 1
        full_cmd = ["taskset", "-c", f"{core_lo}-{core_hi}"] + cmd
        print(f"  [{label:<50s}] cores {core_lo}-{core_hi}  "
              f"({finished}/{total_jobs} done, {len(job_queue)} queued, "
              f"{sum(len(v) for v in locked_jobs.values())} locked)")
        safe_label = label.replace("/", "_").replace(" ", "_")
        log_file = open(os.path.join(log_dir, f"{phase_name}_{safe_label}.log"), "w")
        proc = subprocess.Popen(
            full_cmd, cwd=cwd,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        slots[slot_idx] = (label, proc, log_file)

    # Fill initial slots from the ready queue.
    for i in range(max_slots):
        if job_queue:
            label, cmd = job_queue.pop(0)
            launch_in_slot(i, label, cmd)

    # Poll for finished jobs, unlock transfer jobs, refill slots.
    while any(s is not None for s in slots) or job_queue:
        for i in range(max_slots):
            if slots[i] is None:
                continue
            label, proc, log_file = slots[i]
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                status = "OK" if ret == 0 else f"EXIT {ret}"
                finished += 1

                # Unlock transfer jobs that were waiting on this standard job.
                if label in locked_jobs:
                    unlocked = locked_jobs.pop(label)
                    job_queue.extend(unlocked)
                    print(f"  [{label:<50s}] finished ({status})  "
                          f"[{finished}/{total_jobs}]  "
                          f"-> unlocked {len(unlocked)} transfer job(s)")
                else:
                    print(f"  [{label:<50s}] finished ({status})  "
                          f"[{finished}/{total_jobs}]")

                if ret != 0:
                    print(f"    -> see log: {log_file.name}")
                slots[i] = None
                if job_queue:
                    next_label, next_cmd = job_queue.pop(0)
                    launch_in_slot(i, next_label, next_cmd)
        time.sleep(1)

    print(f"\n{phase_name}: all {total_jobs} jobs done.")


def train_extra_epochs(model_path, arch, dataset, sgd_epochs=1, lr=1e-3, batch_size=128):
    """Load model from model_path, train sgd_epochs more with SGD, save N2.

    N1 = model_path (unchanged, already exists)
    N2 = {model_path}_sgd_itr{sgd_epochs}/

    Returns (n1_dir, n2_dir) — absolute paths to the model directories.
    """
    import torch
    import torch.nn as nn
    import torchvision.datasets as dsets
    import torchvision.transforms as transforms

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(script_dir, 'utils'))
    from run_experiment import ARCH_REGISTRY, DATASET_CONFIG, save_model, evaluate

    model_cls, _ = ARCH_REGISTRY[arch]
    ds_cls, channels, w, h, julia_ds = DATASET_CONFIG[dataset]

    # Accept both directory and file path (e.g. .../model_seed42_itr20 or .../model_seed42_itr20/model.p)
    model_path = os.path.normpath(model_path)
    if os.path.isfile(model_path):
        model_path = os.path.dirname(model_path)
    n1_dir = model_path
    n2_dir = f"{n1_dir}_sgd_itr{sgd_epochs}"

    # Skip training if N2 already exists
    n2_model_p = os.path.join(n2_dir, 'model.p')
    n2_model_pth = os.path.join(n2_dir, 'model.pth')
    if os.path.exists(n2_model_p) and os.path.exists(n2_model_pth):
        print(f"  N2 already exists at {n2_dir}, skipping training.")
        print(f"  N1: {n1_dir}")
        print(f"  N2: {n2_dir}")
        return n1_dir, n2_dir

    # Load model
    model_pth = os.path.join(n1_dir, 'model.pth')
    if not os.path.exists(model_pth):
        print(f"ERROR: {model_pth} not found")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_cls().to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device))
    print(f"  Loaded model from {model_pth}")

    # Prepare data
    transform = transforms.Compose([transforms.ToTensor()])
    data_root = os.path.join(script_dir, '..', 'MNIST') if dataset == 'mnist' else os.path.join(script_dir, '..', dataset)
    train_dataset = ds_cls(root=data_root, train=True, transform=transform, download=True)
    test_dataset = ds_cls(root=data_root, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    acc_before = evaluate(model, test_loader, device)
    print(f"  N1 accuracy: {acc_before:.2f}%")

    # Train extra epochs with SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(sgd_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        acc = evaluate(model, test_loader, device)
        print(f"  SGD epoch {epoch + 1}/{sgd_epochs}: loss={avg_loss:.4f}, acc={acc:.2f}%")

    # Save N2
    save_model(model, n2_dir)
    print(f"  N1: {n1_dir}")
    print(f"  N2: {n2_dir}")
    return n1_dir, n2_dir


def parse_result_file(filepath):
    """Parse a result .txt file.

    Supports two formats:
      New (key=value): c_source=0,c_target=3,lower_bound=...,upper_bound=...,optimization_time=...,hyper_attack_time=...
      Old (positional CSV): source,target,incumbent_obj,best_bound,solve_time

    Returns dict of (c_source, c_target) -> {
        'optimization_time': float,
        'lower_bound': float,
        'upper_bound': float,
    }
    """
    results = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try new key=value format first
            fields = {}
            for pair in line.split(","):
                if "=" in pair:
                    key, val = pair.split("=", 1)
                    fields[key] = val
            if fields:
                try:
                    cs = int(fields["c_source"])
                    ct = int(fields["c_target"])
                    opt_time = float(fields["optimization_time"])
                    hyper_time = float(fields.get("hyper_attack_time", "0"))
                    results[(cs, ct)] = {
                        "optimization_time": opt_time,
                        "hyper_attack_time": hyper_time,
                        "total_time": opt_time + hyper_time,
                        "lower_bound": float(fields.get("lower_bound", "nan")),
                        "upper_bound": float(fields.get("upper_bound", "nan")),
                        "solve_status": fields.get("solve_status", ""),
                    }
                except (KeyError, ValueError):
                    continue
            else:
                # Old positional CSV format: source,target,incumbent_obj,best_bound,solve_time
                try:
                    parts = line.split(",")
                    cs = int(parts[0])
                    ct = int(parts[1])
                    incumbent = float(parts[2])
                    best_bound = float(parts[3])
                    solve_time = float(parts[4])
                    results[(cs, ct)] = {
                        "optimization_time": solve_time,
                        "hyper_attack_time": 0,
                        "total_time": solve_time,
                        "lower_bound": incumbent,
                        "upper_bound": best_bound,
                        "solve_status": "",
                    }
                except (IndexError, ValueError):
                    continue
    return results


def _extract_transfer_file_metadata(filename):
    """Extract threads, relax_count, optimizing_intervals, and no_n1_bin from a transfer result filename."""
    threads_match = re.search(r"Therads(\d+)", filename)
    relax_count_match = re.search(r"RelaxCount(\d+)", filename)
    opt_intervals = "yes" if "OptimizingIntervals" in filename else "no"
    no_n1_bin = "yes" if "NoN1BinRelaxOnN2only" in filename else "no"
    has_last_layer = "N1LastLayer" in filename
    has_no_bin = "NoBin" in filename
    has_n1xp = "N1xpConf" in filename
    has_zonotope = "Zonotope" in filename
    if has_last_layer and has_no_bin:
        no_n1_enc = "last_layer_no_bin"
    elif has_last_layer and has_n1xp:
        no_n1_enc = "last_layer+n1xp"
    elif has_last_layer:
        no_n1_enc = "last_layer"
    elif "NoN1Encoding" in filename and has_n1xp:
        no_n1_enc = "yes+n1xp"
    elif "NoN1Encoding" in filename:
        no_n1_enc = "yes"
    else:
        no_n1_enc = "no"
    if has_zonotope:
        no_n1_enc += "+zono"
    return {
        "threads": int(threads_match.group(1)) if threads_match else "",
        "relax_count": int(relax_count_match.group(1)) if relax_count_match else "",
        "optimizing_intervals": opt_intervals,
        "no_n1_bin_relax_on_n2": no_n1_bin,
        "no_n1_encoding": no_n1_enc,
    }


def find_transfer_faster_than_standard(perts, exp_base, csv_transfer_faster, csv_standard_faster,
                                       csv_transfer_tighter_at_timeout, csv_standard_tighter_at_timeout,
                                       arch="cnn1", double_check_standard=False,
                                       compare_to_with_perturbed=False):
    """For each perturbation/eps, compare transfer vs standard N2 (NoPerturbed).

    Returns four lists of row dicts (transfer_faster, standard_faster,
    transfer_tighter, standard_tighter). If csv paths are provided,
    also writes them to CSVs.
    """
    import csv

    # Map perturbation name prefix to directory name
    pert_dir_map = {
        "patch": "patch",
        "occ": "occ",
        "trans": "translation",
        "rotation": "rotation",
        "brightness": "brightness",
        "occ": "occ",
    }

    fieldnames = [
        "arch",
        "perturbation",
        "perturbation_size",
        "c_source",
        "c_target",
        "time_standard",
        "time_transfer",
        "delta_standard_lower_bound",
        "delta_standard_upper_bound",
        "delta_diff_transfer_lower_bound",
        "delta_diff_transfer_upper_bound",
        "transfer_threads",
        "T_relax",
        "relax_count",
        "optimizing_intervals",
        "no_n1_bin_relax_on_n2",
        "no_n1_encoding",
        "how_much_faster",
    ]
    if not compare_to_with_perturbed:
        fieldnames += ["gap_standard", "gap_transfer"]
    fieldnames += ["solve_status_standard", "solve_status_transfer"]

    rows_transfer_faster = []
    rows_standard_faster = []
    rows_transfer_tighter = []
    rows_standard_tighter = []

    for pert_name, pert_spec in perts:
        # e.g. pert_spec = "patch:1,14,14,3" -> dir = "patch", eps = "1,14,14,3"
        pert_type, eps_str = pert_spec.split(":", 1)
        pert_dir = pert_dir_map.get(pert_type, pert_type)
        eps_dir = os.path.join(exp_base, pert_dir, f"eps_{eps_str}")

        if not os.path.isdir(eps_dir):
            print(f"  [{pert_name}] No results directory: {eps_dir}")
            continue

        # Find standard N2 directories
        if double_check_standard:
            std_pattern = "double_check_vhagarNoPertubed_*_sgd_itr*"
        elif compare_to_with_perturbed:
            std_pattern = "vagharWithPerturbed_*_sgd_itr*"
        else:
            std_pattern = "vagharNoPerturbed_*_sgd_itr*"
        standard_n2_dirs = sorted(glob.glob(os.path.join(eps_dir, std_pattern)))
        if not standard_n2_dirs:
            print(f"  [{pert_name}] No standard N2 ({std_pattern}) found in {eps_dir}")
            continue

        # Load all standard N2 results: (c_source, c_target) -> result dict
        standard_results = {}
        for sd in standard_n2_dirs:
            txt_files = glob.glob(os.path.join(sd, "*.txt"))
            for tf in txt_files:
                parsed = parse_result_file(tf)
                standard_results.update(parsed)

        if not standard_results:
            print(f"  [{pert_name}] No results parsed from standard N2 directories")
            continue

        # Find transfer directories
        transfer_dirs = sorted(glob.glob(os.path.join(eps_dir, "transfer_*")))
        if not transfer_dirs:
            print(f"  [{pert_name}] No transfer directories found")
            continue

        for td in transfer_dirs:
            td_name = os.path.basename(td)
            relax_match = re.search(r"relax([\d.]+|[Ii]nf)", td_name)
            relax_val = relax_match.group(1) if relax_match else ""
            if relax_val and "GapArea" in td_name:
                relax_val = f"rga{relax_val}"

            txt_files = sorted(glob.glob(os.path.join(td, "*.txt")))
            for tf in txt_files:
                tf_name = os.path.basename(tf)
                meta = _extract_transfer_file_metadata(tf_name)
                transfer_results = parse_result_file(tf)

                for (cs, ct), t_info in sorted(transfer_results.items()):
                    key = (cs, ct)
                    if key not in standard_results:
                        continue
                    # Skip old-style optimizing_intervals runs (but allow NoN1BinRelaxOnN2only and NoN1Encoding)
                    if meta["optimizing_intervals"] == "yes" and meta["no_n1_bin_relax_on_n2"] == "no" and meta["no_n1_encoding"] == "no":
                        continue
                    s_info = standard_results[key]
                    t_time = t_info["total_time"]
                    s_time = s_info["total_time"]

                    row = {
                        "arch": arch,
                        "perturbation": pert_type,
                        "perturbation_size": eps_str,
                        "c_source": cs,
                        "c_target": ct,
                        "time_standard": f"{s_time:.2f}",
                        "time_transfer": f"{t_time:.2f}",
                        "delta_standard_lower_bound": f"{s_info['lower_bound']:.6f}",
                        "delta_standard_upper_bound": f"{s_info['upper_bound']:.6f}",
                        "delta_diff_transfer_lower_bound": f"{t_info['lower_bound']:.6f}",
                        "delta_diff_transfer_upper_bound": f"{t_info['upper_bound']:.6f}",
                        "transfer_threads": meta["threads"],
                        "T_relax": relax_val,
                        "relax_count": meta["relax_count"],
                        "optimizing_intervals": meta["optimizing_intervals"],
                        "no_n1_bin_relax_on_n2": meta["no_n1_bin_relax_on_n2"],
                        "no_n1_encoding": meta["no_n1_encoding"],
                    }

                    if not compare_to_with_perturbed:
                        s_gap = s_info["upper_bound"] - s_info["lower_bound"]
                        t_gap = t_info["upper_bound"] - t_info["lower_bound"]
                        row["gap_standard"] = f"{s_gap:.6f}"
                        row["gap_transfer"] = f"{t_gap:.6f}"
                    row["solve_status_standard"] = s_info.get("solve_status", "")
                    row["solve_status_transfer"] = t_info.get("solve_status", "")

                    if s_info.get("solve_status", "") == "INTERRUPTED":
                        continue

                    if t_time < s_time * 0.99:  # transfer is faster
                        row["how_much_faster"] = f"{s_time / t_time:.2f}x"
                        rows_transfer_faster.append(row)
                    elif s_time < t_time * 0.99:  # standard is faster
                        row["how_much_faster"] = f"{t_time / s_time:.2f}x"
                        rows_standard_faster.append(row)
                    else:  # both hit timeout (~same time)
                        row["how_much_faster"] = ""
                        if not compare_to_with_perturbed:
                            if t_gap < s_gap * 0.99:  # transfer has tighter gap
                                rows_transfer_tighter.append(row)
                            elif s_gap < t_gap * 0.99:  # standard has tighter gap
                                rows_standard_tighter.append(row)

    # Sort helper: (perturbation, perturbation_size, c_source, c_target, numeric_key)
    def _parse_speed(val):
        """Parse '2.50x' -> 2.50, empty -> inf."""
        if not val:
            return float('inf')
        return float(val.rstrip('x'))

    def _sort_key_faster(row):
        return (row["arch"], row["perturbation"], row["perturbation_size"],
                int(row["c_source"]), int(row["c_target"]),
                _parse_speed(row["how_much_faster"]))

    def _sort_key_tighter(row):
        return (row["arch"], row["perturbation"], row["perturbation_size"],
                int(row["c_source"]), int(row["c_target"]),
                float(row["gap_transfer"]))

    rows_transfer_faster.sort(key=_sort_key_faster)
    rows_standard_faster.sort(key=_sort_key_faster)
    rows_transfer_tighter.sort(key=_sort_key_tighter)
    rows_standard_tighter.sort(key=_sort_key_tighter)

    def _group_key(row):
        return (row["arch"], row["perturbation"], row["perturbation_size"],
                row["c_source"], row["c_target"])

    # Write sorted rows to CSVs, inserting a blank row between groups
    empty_row = {fn: "" for fn in fieldnames}
    for filepath, rows in [
        (csv_transfer_faster, rows_transfer_faster),
        (csv_standard_faster, rows_standard_faster),
        (csv_transfer_tighter_at_timeout, rows_transfer_tighter),
        (csv_standard_tighter_at_timeout, rows_standard_tighter),
    ]:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            prev_key = None
            for row in rows:
                key = _group_key(row)
                if prev_key is not None and key != prev_key:
                    writer.writerow(empty_row)
                writer.writerow(row)
                prev_key = key

    print(f"  Wrote {len(rows_transfer_faster)} rows to {csv_transfer_faster}")
    print(f"  Wrote {len(rows_standard_faster)} rows to {csv_standard_faster}")
    print(f"  Wrote {len(rows_transfer_tighter)} rows to {csv_transfer_tighter_at_timeout}")
    print(f"  Wrote {len(rows_standard_tighter)} rows to {csv_standard_tighter_at_timeout}")

    return rows_transfer_faster, rows_standard_faster, rows_transfer_tighter, rows_standard_tighter


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--perturbations", nargs="*", default=None,
                        help="Filter perturbations by name prefix (e.g. 'patch' 'occ' 'trans' 'rotation')")
    parser.add_argument("--max_cores", type=int, default=TOTAL_CORES,
                        help=f"Total cores available (default: {TOTAL_CORES})")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="MIP timeout per class pair in seconds (default: 1800)")
    parser.add_argument("--thresholds", nargs="*", type=float, default=None,
                        help="Override relaxation thresholds for transfer phase (default: all)")
    parser.add_argument("--opt_intervals", nargs="*", default=None,
                        help="Override optimizing_intervals values for transfer phase (e.g. 'true' 'false')")
    parser.add_argument("--relaxation_gap_area", type=str, default="true",
                        help="Use triangle relaxation-gap area scoring instead of interval width (true/false)")
    parser.add_argument("--skip_standard", action="store_true",
                        help="Skip phase 1 (standard experiments)")
    parser.add_argument("--skip_transfer", action="store_true",
                        help="Skip phase 2 (transfer experiments)")
    parser.add_argument("--double_check_standard", action="store_true",
                        help="Also run double-check standard using /root/Downloads/for_dana/code/run.jl")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to existing model directory (containing model.pth). "
                             "N1 = this model, N2 = N1 + extra SGD epoch(s). Replaces dual-seed mode.")
    parser.add_argument("--sgd_epochs", type=int, default=1,
                        help="Number of extra SGD epochs for N2 when using --model_path (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="SGD learning rate for extra training (default: 1e-3)")
    parser.add_argument("--arch", type=str, default="cnn1",
                        help="Network architecture (e.g. cnn1, cnn2, 3x10, 3x50, 4x10, 5x10, 5x50, 10x10, 3x100)")
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Dataset name (default: mnist)")
    parser.add_argument("--arch_models", nargs="*", default=None,
                        help="Run multiple architectures, each with its own model path. "
                             "Format: arch=model_path (e.g. cnn1=/path/to/cnn1_model cnn2=/path/to/cnn2_model). "
                             "Overrides --arch and --model_path when specified.")
    parser.add_argument("--find_transfer_faster_than_standard", action="store_true",
                        help="Scan existing results and report transfer experiments that are "
                             "faster than standard N2 (vagharNoPerturbed with sgd) for each "
                             "perturbation and (c_source, c_target) pair.")
    parser.add_argument("--standard_only", action="store_true",
                        help="Run standard verification only on the given model(s), without "
                             "extra SGD training or creating N2. Implies --skip_transfer.")
    parser.add_argument("--standard_relaxation_thresholds", type=str, default=None,
                        help="Comma-separated relaxation thresholds for standard mode "
                             "(use_perturbed_intervals=true + use_relaxations=true). "
                             "e.g. '0.05,0.1,0.5'. Passed to run_experiment.py. "
                             "If not set, standard relaxation step is skipped.")
    parser.add_argument("--no_n1_binaries_and_relaxtions_only_on_n2", action="store_true",
                        help="LP-relax all N1 binaries and relax N2(x_p) by conditioning on N2(x) "
                             "instead of N1(x). Keeps N2(x) exact as anchor.")
    parser.add_argument("--no_n1_encoding_at_all", action="store_true",
                        help="Skip N1 encoding entirely; replace conf(N1,x,c)>=delta_1 with "
                             "interval-bounded constraints on N2 outputs using weight diff bounds.")
    parser.add_argument("--encode_n1_last_layer", action="store_true",
                        help="When no_n1_encoding_at_all is active, encode N1 last linear layer "
                             "exactly using interval-bounded hidden variables; gives exact delta_diff.")
    parser.add_argument("--n1_last_layer_no_binaries", action="store_true",
                        help="When encode_n1_last_layer is active, use pre-computed scalar lower bound "
                             "on conf_n1 instead of binary max encoding; zero extra binaries.")
    parser.add_argument("--constrain_n1_xp", action="store_true",
                        help="Add interval-based constraint that conf(N1,x',c_target)<=0; "
                             "no extra variables, uses pre-computed pert bounds through N1.")
    parser.add_argument("--use_zonotope", action="store_true",
                        help="Use zonotope (affine arithmetic) for diff bound propagation; "
                             "tighter bounds by tracking correlations between neurons.")
    parser.add_argument("--compare_to_with_perturbed", action="store_true",
                        help="Compare transfer results to vagharWithPerturbed (standard with perturbed "
                             "intervals) instead of vagharNoPerturbed.")
    args = parser.parse_args()

    total_cores = args.max_cores
    thresholds = args.thresholds if args.thresholds else THRESHOLDS
    opt_intervals = args.opt_intervals if args.opt_intervals else OPT_INTERVALS
    dataset = args.dataset

    if args.standard_only:
        args.skip_transfer = True

    # Build list of (arch, model_path|None) to run
    if args.arch_models:
        arch_runs = []
        for pair in args.arch_models:
            if "=" not in pair:
                print(f"ERROR: --arch_models entry must be arch=model_path, got: {pair}")
                sys.exit(1)
            a, mp = pair.split("=", 1)
            arch_runs.append((a, mp))
    else:
        arch_runs = [(args.arch, args.model_path)]

    # Filter perturbations if requested
    perts = PERTURBATIONS
    if args.perturbations:
        prefixes = [p.lower() for p in args.perturbations]
        perts = [p for p in PERTURBATIONS if any(p[0].lower().startswith(pf) for pf in prefixes)]
        if not perts:
            print(f"ERROR: No perturbations matched {args.perturbations}")
            print(f"Available: {[p[0] for p in PERTURBATIONS]}")
            sys.exit(1)

    cwd = os.path.dirname(os.path.abspath(__file__))

    # ── Analysis mode: find transfer faster than standard ─────────
    if args.find_transfer_faster_than_standard:
        # Always scan all perturbation types (not just the ones enabled for running)
        all_perts = [
            ("patch(1,14,14,3)",  "patch:1,14,14,3"),
            ("occ(14,14,9)",      "occ:14,14,9"),
            ("occ(1,1,5)",        "occ:1,1,5"),
            ("brightness(0.25)",  "brightness:0.25"),
            ("trans(1,1)",        "translation:1,1"),
            ("trans(1,3)",        "translation:1,3"),
            ("trans(3,1)",        "translation:3,1"),
            ("trans(3,3)",        "translation:3,3"),
            ("rotation(10)",      "rotation:10"),
        ]
        # Write combined CSVs to the dataset-level directory (not per-arch)
        dblchk = args.double_check_standard
        suffix = "_double_check_standard" if dblchk else ""
        if args.compare_to_with_perturbed:
            suffix += "_vs_withPerturbed"
        combined_base = os.path.join(cwd, "paper_experiments", dataset)
        os.makedirs(combined_base, exist_ok=True)
        csv_transfer_faster = os.path.join(combined_base, f"transfer_faster_than_standard{suffix}.csv")
        csv_standard_faster = os.path.join(combined_base, f"standard_faster_than_transfer{suffix}.csv")
        csv_transfer_tighter = os.path.join(combined_base, f"transfer_tighter_at_timeout{suffix}.csv")
        csv_standard_tighter = os.path.join(combined_base, f"standard_tighter_at_timeout{suffix}.csv")

        # Collect rows across all archs, write CSVs once with no per-arch files
        all_tf, all_sf, all_tt, all_st = [], [], [], []
        for arch, _ in arch_runs:
            exp_base = os.path.join(cwd, "paper_experiments", dataset, f"{arch}_exp")
            print(f"\nScanning results for {arch} in: {exp_base}")
            tf, sf, tt, st = find_transfer_faster_than_standard(
                all_perts, exp_base, csv_transfer_faster, csv_standard_faster,
                csv_transfer_tighter, csv_standard_tighter, arch=arch,
                double_check_standard=dblchk,
                compare_to_with_perturbed=args.compare_to_with_perturbed)
            all_tf.extend(tf)
            all_sf.extend(sf)
            all_tt.extend(tt)
            all_st.extend(st)

        # Re-write combined CSVs with all archs together
        find_transfer_faster_than_standard.__doc__  # just to access fieldnames
        import csv as _csv
        _fieldnames = [
            "arch", "perturbation", "perturbation_size", "c_source", "c_target",
            "time_standard", "time_transfer", "delta_standard_lower_bound",
            "delta_standard_upper_bound", "delta_diff_transfer_lower_bound",
            "delta_diff_transfer_upper_bound", "transfer_threads", "T_relax",
            "relax_count", "optimizing_intervals", "no_n1_bin_relax_on_n2", "no_n1_encoding", "how_much_faster",
        ]
        if not args.compare_to_with_perturbed:
            _fieldnames += ["gap_standard", "gap_transfer"]
        _fieldnames += ["solve_status_standard", "solve_status_transfer"]

        def _parse_speed(val):
            if not val:
                return float('inf')
            return float(val.rstrip('x'))

        def _sort_faster(row):
            return (row["arch"], row["perturbation"], row["perturbation_size"],
                    int(row["c_source"]), int(row["c_target"]),
                    _parse_speed(row["how_much_faster"]))

        def _sort_tighter(row):
            return (row["arch"], row["perturbation"], row["perturbation_size"],
                    int(row["c_source"]), int(row["c_target"]),
                    float(row["gap_transfer"]))

        def _group_key(row):
            return (row["arch"], row["perturbation"], row["perturbation_size"],
                    row["c_source"], row["c_target"])

        all_tf.sort(key=_sort_faster)
        all_sf.sort(key=_sort_faster)
        all_tt.sort(key=_sort_tighter)
        all_st.sort(key=_sort_tighter)

        empty_row = {fn: "" for fn in _fieldnames}
        for filepath, rows in [
            (csv_transfer_faster, all_tf), (csv_standard_faster, all_sf),
            (csv_transfer_tighter, all_tt), (csv_standard_tighter, all_st),
        ]:
            with open(filepath, "w", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=_fieldnames)
                writer.writeheader()
                prev_key = None
                for row in rows:
                    key = _group_key(row)
                    if prev_key is not None and key != prev_key:
                        writer.writerow(empty_row)
                    writer.writerow(row)
                    prev_key = key

        print(f"\nCombined CSVs ({len(arch_runs)} arch(s)):")
        print(f"  {len(all_tf)} rows -> {csv_transfer_faster}")
        print(f"  {len(all_sf)} rows -> {csv_standard_faster}")
        print(f"  {len(all_tt)} rows -> {csv_transfer_tighter}")
        print(f"  {len(all_st)} rows -> {csv_standard_tighter}")
        return

    cores_per_job = CORES_PER_JOB
    max_slots = (total_cores - CORE_START) // cores_per_job

    try:
        # ── Build job lists across all arch runs ──────────────────────
        Threads_num = 32
        cores_per_job = Threads_num
        max_slots = (total_cores - CORE_START) // cores_per_job

        standard_jobs = []   # (pert_name, label, cmd) — pert_name used as dep key
        transfer_by_pert = {}  # pert_name -> [(label, cmd)]
        skipped_standard = []  # pert_names where standard results already exist

        for arch, model_path in arch_runs:
            use_model_path = model_path is not None

            # ── Phase 0: Train +epoch (only in --model_path mode, skipped in --standard_only) ─────
            n1_dir, n2_dir = None, None
            if use_model_path and not args.standard_only:
                print(f"\n{'=' * 60}")
                print(f"Phase 0: Training N2 = N1 + {args.sgd_epochs} SGD epoch(s) [{arch}]")
                print(f"{'=' * 60}\n")
                n1_dir, n2_dir = train_extra_epochs(
                    model_path, arch, dataset,
                    sgd_epochs=args.sgd_epochs, lr=args.lr)
            elif use_model_path and args.standard_only:
                # standard_only: use the given model as both N1 and N2 (no extra training)
                n1_dir = os.path.normpath(model_path)
                if os.path.isfile(n1_dir):
                    n1_dir = os.path.dirname(n1_dir)
                n2_dir = n1_dir

            # Build the extra args for run_experiment.py depending on mode
            if use_model_path:
                _model_args = ["--model_n1_dir", n1_dir, "--model_n2_dir", n2_dir]
            else:
                _model_args = ["--dual_seed"]

            arch_prefix = f"[{arch}] "

            for pert_name, pert_spec in perts:
                job_key = f"{arch}/{pert_name}"
                std_exists = not args.skip_standard and standard_results_exist(pert_spec, cwd, arch, dataset)
                if std_exists and not args.double_check_standard:
                    print(f"  {arch_prefix}{pert_name} Standard results already exist — skipping, "
                          f"transfer jobs will start immediately.")
                    skipped_standard.append(job_key)
                else:
                    std_label = f"{arch_prefix}{pert_name}"
                    std_cmd = [
                        "python3", "utils/run_experiment.py",
                        "--skip_training",
                        "--skip_transfer",
                        "--perturbations", pert_spec,
                        "--timeout", str(args.timeout),
                        "--dataset", dataset,
                        "--arch", arch,
                    ] + _model_args
                    if args.double_check_standard:
                        std_cmd.append("--double_check_standard")
                    if args.standard_relaxation_thresholds is not None:
                        std_cmd += ["--standard_relaxation_thresholds", args.standard_relaxation_thresholds]
                    standard_jobs.append((job_key, std_label, std_cmd))

                t_jobs = []
                for oi in opt_intervals:
                    for t in thresholds :
                        rga_tag = "true" if args.relaxation_gap_area.lower() == "true" else "false"
                        t_label = f"{arch_prefix}{pert_name} T={t} oi={oi} rga={rga_tag}"
                        t_cmd = [
                            "python3", "utils/run_experiment.py",
                            "--skip_training",
                            "--skip_standard",
                            "--perturbations", pert_spec,
                            "--timeout", str(args.timeout),
                            "--dataset", dataset,
                            "--arch", arch,
                            "--relaxation_thresholds", str(t),
                            "--optimizing_intervals", oi,
                            "--Threads_num", str(Threads_num),
                            "--relaxation_gap_area", args.relaxation_gap_area,
                        ] + _model_args
                        if args.no_n1_binaries_and_relaxtions_only_on_n2:
                            t_cmd.append("--no_n1_binaries_and_relaxtions_only_on_n2")
                        if args.no_n1_encoding_at_all:
                            t_cmd.append("--no_n1_encoding_at_all")
                        if args.encode_n1_last_layer:
                            t_cmd.append("--encode_n1_last_layer")
                        if args.n1_last_layer_no_binaries:
                            t_cmd.append("--n1_last_layer_no_binaries")
                        if args.constrain_n1_xp:
                            t_cmd.append("--constrain_n1_xp")
                        if args.use_zonotope:
                            t_cmd.append("--use_zonotope")
                        t_jobs.append((t_label, t_cmd))
                transfer_by_pert[job_key] = t_jobs

        # Transfer jobs for skipped-standard perturbations are immediately ready
        skipped_transfer_ready = [
            (lbl, cmd)
            for pn in skipped_standard
            if pn in transfer_by_pert
            for (lbl, cmd) in transfer_by_pert[pn]
        ]

        # ── Phase 1 only ───────────────────────────────────────────────
        if not args.skip_standard and args.skip_transfer:
            ready = [(lbl, cmd) for (_, lbl, cmd) in standard_jobs]
            run_pool(ready, max_slots, cwd, cores_per_job, "Phase 1 (standard)")

        # ── Phase 2 only (all transfer jobs are immediately ready) ─────
        elif args.skip_standard and not args.skip_transfer:
            ready = [(lbl, cmd) for jobs in transfer_by_pert.values() for (lbl, cmd) in jobs]
            run_pool(ready, max_slots, cwd, cores_per_job, "Phase 2 (transfer)")

        # ── Both phases: transfer jobs unlock as each standard job finishes
        elif not args.skip_standard and not args.skip_transfer:
            ready = [(lbl, cmd) for (_, lbl, cmd) in standard_jobs] + skipped_transfer_ready
            # locked_jobs key = standard job label; value = its transfer jobs
            locked = {lbl: transfer_by_pert[pn] for (pn, lbl, _) in standard_jobs}
            run_pool(ready, max_slots, cwd, cores_per_job,
                     "Sweep", locked_jobs=locked)

    except KeyboardInterrupt:
        print("\nCtrl+C received — terminating all running jobs...")
        sys.exit(1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
