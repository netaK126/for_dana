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
    ("patch(1,14,14,3)",  "patch:1,14,14,3"),
    ("occ(14,14,9)",      "occ:14,14,9"),
    ("trans(1,1)",        "translation:1,1"),
    ("trans(1,3)",        "translation:1,3"),
    ("trans(3,1)",        "translation:3,1"),
    ("trans(3,3)",        "translation:3,3"),
    ("rotation(10)",      "rotation:10"),
]

# ── Transfer sweep parameters ────────────────────────────────────────────
THRESHOLDS = [0.5, 1.2, 3.0, 8.0]
OPT_INTERVALS = ["true", "false"]

# ── CPU pinning ──────────────────────────────────────────────────────────
CORES_PER_JOB = 32
CORE_START = 8  # first core to use (reserve 0-7)
TOTAL_CORES = 200


def run_pool(all_jobs, max_slots, cwd,cores_per_job, phase_name=""):
    """Run a list of (label, cmd) jobs with CPU-pinned slot pooling."""
    if not all_jobs:
        return

    print(f"\n{'=' * 60}")
    print(f"{phase_name}: {len(all_jobs)} jobs, {str(max_slots)} concurrent slots "
          f"({cores_per_job} cores/job, cores {CORE_START}-{CORE_START + max_slots * cores_per_job - 1})")
    print(f"{'=' * 60}\n")

    slots = [None] * max_slots
    job_queue = list(all_jobs)
    finished = 0

    log_dir = os.path.join(cwd, "sweep_logs")
    os.makedirs(log_dir, exist_ok=True)

    def launch_in_slot(slot_idx, label, cmd):
        nonlocal finished
        core_lo = CORE_START + slot_idx * cores_per_job
        core_hi = core_lo + cores_per_job - 1
        full_cmd = ["taskset", "-c", f"{core_lo}-{core_hi}"] + cmd
        print(f"  [{label:<50s}] cores {core_lo}-{core_hi}  "
              f"({finished}/{len(all_jobs)} done, {len(job_queue)} queued)")
        safe_label = label.replace("/", "_").replace(" ", "_")
        log_file = open(os.path.join(log_dir, f"{phase_name}_{safe_label}.log"), "w")
        proc = subprocess.Popen(
            full_cmd, cwd=cwd,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        slots[slot_idx] = (label, proc, log_file)

    # Fill initial slots
    for i in range(max_slots):
        if job_queue:
            label, cmd = job_queue.pop(0)
            launch_in_slot(i, label, cmd)

    # Poll for finished jobs, refill slots
    while any(s is not None for s in slots):
        for i in range(max_slots):
            if slots[i] is None:
                continue
            label, proc, log_file = slots[i]
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                status = "OK" if ret == 0 else f"EXIT {ret}"
                finished += 1
                print(f"  [{label:<50s}] finished ({status})  "
                      f"[{finished}/{len(all_jobs)}]")
                if ret != 0:
                    print(f"    -> see log: {log_file.name}")
                slots[i] = None
                if job_queue:
                    next_label, next_cmd = job_queue.pop(0)
                    launch_in_slot(i, next_label, next_cmd)
        time.sleep(1)

    print(f"\n{phase_name}: all {len(all_jobs)} jobs done.")


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
            fields = {}
            for pair in line.split(","):
                if "=" in pair:
                    key, val = pair.split("=", 1)
                    fields[key] = val
            try:
                cs = int(fields["c_source"])
                ct = int(fields["c_target"])
                results[(cs, ct)] = {
                    "optimization_time": float(fields["optimization_time"]),
                    "lower_bound": float(fields.get("lower_bound", "nan")),
                    "upper_bound": float(fields.get("upper_bound", "nan")),
                }
            except (KeyError, ValueError):
                continue
    return results


def _extract_transfer_file_metadata(filename):
    """Extract threads, relax_count, and optimizing_intervals from a transfer result filename."""
    threads_match = re.search(r"Therads(\d+)", filename)
    relax_count_match = re.search(r"RelaxCount(\d+)", filename)
    opt_intervals = "yes" if "OptimizingIntervals" in filename else "no"
    return {
        "threads": int(threads_match.group(1)) if threads_match else "",
        "relax_count": int(relax_count_match.group(1)) if relax_count_match else "",
        "optimizing_intervals": opt_intervals,
    }


def find_transfer_faster_than_standard(perts, exp_base, output_csv):
    """For each perturbation/eps, find transfer results faster than standard N2 (NoPerturbed).

    Writes matching rows to output_csv.
    """
    import csv

    # Map perturbation name prefix to directory name
    pert_dir_map = {
        "patch": "patch",
        "occ": "occ",
        "trans": "translation",
        "rotation": "rotation",
    }

    fieldnames = [
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
        "how_much_faster",
    ]

    total_rows = 0
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for pert_name, pert_spec in perts:
            # e.g. pert_spec = "patch:1,14,14,3" -> dir = "patch", eps = "1,14,14,3"
            pert_type, eps_str = pert_spec.split(":", 1)
            pert_dir = pert_dir_map.get(pert_type, pert_type)
            eps_dir = os.path.join(exp_base, pert_dir, f"eps_{eps_str}")

            if not os.path.isdir(eps_dir):
                print(f"  [{pert_name}] No results directory: {eps_dir}")
                continue

            # Find standard N2 directories (vagharNoPerturbed_*_sgd_itr*)
            standard_n2_dirs = sorted(glob.glob(os.path.join(eps_dir, "vagharNoPerturbed_*_sgd_itr*")))
            if not standard_n2_dirs:
                print(f"  [{pert_name}] No standard N2 (vagharNoPerturbed_*_sgd_itr*) found in {eps_dir}")
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
                relax_match = re.search(r"relax([\d.]+)", td_name)
                relax_val = relax_match.group(1) if relax_match else ""

                txt_files = sorted(glob.glob(os.path.join(td, "*.txt")))
                for tf in txt_files:
                    tf_name = os.path.basename(tf)
                    meta = _extract_transfer_file_metadata(tf_name)
                    transfer_results = parse_result_file(tf)

                    for (cs, ct), t_info in sorted(transfer_results.items()):
                        key = (cs, ct)
                        if key not in standard_results:
                            continue
                        s_info = standard_results[key]
                        t_time = t_info["optimization_time"]
                        s_time = s_info["optimization_time"]
                        if t_time < s_time * 0.99:  # at least 1% faster
                            speedup = s_time / t_time
                            writer.writerow({
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
                                "how_much_faster": f"{speedup:.2f}x",
                            })
                            total_rows += 1

    print(f"  Wrote {total_rows} rows to {output_csv}")


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
    parser.add_argument("--skip_standard", action="store_true",
                        help="Skip phase 1 (standard experiments)")
    parser.add_argument("--skip_transfer", action="store_true",
                        help="Skip phase 2 (transfer experiments)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to existing model directory (containing model.pth). "
                             "N1 = this model, N2 = N1 + extra SGD epoch(s). Replaces dual-seed mode.")
    parser.add_argument("--sgd_epochs", type=int, default=1,
                        help="Number of extra SGD epochs for N2 when using --model_path (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="SGD learning rate for extra training (default: 1e-3)")
    parser.add_argument("--find_transfer_faster_than_standard", action="store_true",
                        help="Scan existing results and report transfer experiments that are "
                             "faster than standard N2 (vagharNoPerturbed with sgd) for each "
                             "perturbation and (c_source, c_target) pair.")
    args = parser.parse_args()

    total_cores = args.max_cores
    thresholds = args.thresholds if args.thresholds else THRESHOLDS
    opt_intervals = args.opt_intervals if args.opt_intervals else OPT_INTERVALS
    use_model_path = args.model_path is not None

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
        exp_base = os.path.join(cwd, "paper_experiments", "mnist", "cnn1_exp")
        output_csv = os.path.join(exp_base, "transfer_faster_than_standard.csv")
        print(f"\nScanning results in: {exp_base}")
        find_transfer_faster_than_standard(perts, exp_base, output_csv)
        return

    cores_per_job = CORES_PER_JOB
    max_slots = (total_cores - CORE_START) // cores_per_job

    try:
        # ── Phase 0: Train +epoch (only in --model_path mode) ─────────
        n1_dir, n2_dir = None, None
        if use_model_path:
            print(f"\n{'=' * 60}")
            print(f"Phase 0: Training N2 = N1 + {args.sgd_epochs} SGD epoch(s)")
            print(f"{'=' * 60}\n")
            n1_dir, n2_dir = train_extra_epochs(
                args.model_path, "cnn1", "mnist",
                sgd_epochs=args.sgd_epochs, lr=args.lr)

        # Build the extra args for run_experiment.py depending on mode
        def model_args():
            if use_model_path:
                return ["--model_n1_dir", n1_dir, "--model_n2_dir", n2_dir]
            else:
                return ["--dual_seed"]

        # ── Phase 1: Standard experiments ─────────────────────────────
        if not args.skip_standard:
            standard_jobs = []
            for pert_name, pert_spec in perts:
                label = f"{pert_name}"
                cmd = [
                    "python3", "utils/run_experiment.py",
                    "--skip_training",
                    "--skip_transfer",
                    "--perturbations", pert_spec,
                    "--timeout", str(args.timeout),
                    "--dataset", "mnist",
                    "--arch", "cnn1",
                ] + model_args()
                standard_jobs.append((label, cmd))
            run_pool(standard_jobs, max_slots, cwd, cores_per_job, "Phase 1 (standard)")

        # ── Phase 2: Transfer experiments ─────────────────────────────
        if not args.skip_transfer:

            Threads_num_list = [32,48]
            transfer_jobs = []
            for pert_name, pert_spec in perts:
                for oi in opt_intervals:
                    for t in thresholds:
                        for Threads_num in Threads_num_list:
                            cores_per_job = Threads_num 
                            max_slots = (total_cores - cores_per_job) // cores_per_job
                            label = f"{pert_name} T={t} oi={oi}"
                            cmd = [
                                "python3", "utils/run_experiment.py",
                                "--skip_training",
                                "--skip_standard",
                                "--perturbations", pert_spec,
                                "--timeout", str(args.timeout),
                                "--dataset", "mnist",
                                "--arch", "cnn1",
                                "--relaxation_thresholds", str(t),
                                "--optimizing_intervals", oi,
                                "--Threads_num", str(Threads_num),
                            ] + model_args()
                            transfer_jobs.append((label, cmd))
            run_pool(transfer_jobs, max_slots, cwd, cores_per_job, "Phase 2 (transfer)")

    except KeyboardInterrupt:
        print("\nCtrl+C received — terminating all running jobs...")
        sys.exit(1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
