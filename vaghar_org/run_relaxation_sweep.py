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
import itertools

# ── Perturbation configs ─────────────────────────────────────────────────
# Each entry: (name, perturbation_spec)
PERTURBATIONS = [
    ("contrast(1.5)",      "contrast:1.5"),
    ("patch(1,14,14,3)",  "patch:1,14,14,3"),
    ("occ(5,5,5)",        "occ:5,5,5"),
    ("occ(3,3,5)",        "occ:3,3,5"),
    ("trans(1,1)",        "translation:1,1"),
    ("trans(1,3)",        "translation:1,3"),
    ("trans(3,1)",        "translation:3,1"),
    ("trans(3,3)",        "translation:3,3"),
    ("rotation(10)",      "rotation:10"),
    ("occ(1,1,5)",        "occ:1,1,5"),
    ("linf(0.05)",        "linf:0.05"),    
    ("linf(0.1)",         "linf:0.1"),     
    ("brightness(0.25)",  "brightness:0.25"), 
]

# ── Transfer sweep parameters ────────────────────────────────────────────
THRESHOLDS = [0]#[0, 0.05] # focused on best T_relax candidate
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


def _advstd_result_exists(cwd, dataset, arch, pert_type, eps_str, n1_tag,
                          base_name_to_save, seed):
    """Check if an advstd N2 result file already exists for this combo/seed.

    Output filenames written by Julia's run.jl follow the pattern:
      {hash}_n2_{arch}_{pert_type}_{eps_str}_ctag0_{base_name_to_save}_seed{seed}_*.txt

    base_name_to_save captures every technique-flag setting (boundTight,
    zonoBounds, n1ProbeLP, relaxT{x}, varHint, ...), so matching that exact
    substring before `_seed{seed}_` uniquely identifies a combo. Returns
    True if any matching file exists.
    """
    out_dir = os.path.join(
        cwd, "paper_experiments", dataset, f"{arch}_exp",
        pert_type, f"eps_{eps_str}",
        f"advStd_{arch}_N1_{n1_tag}",
    )
    if not os.path.isdir(out_dir):
        return False
    pattern = os.path.join(
        out_dir,
        f"*_n2_{arch}_{pert_type}_{eps_str}_ctag0_"
        f"{base_name_to_save}_seed{seed}_*.txt",
    )
    return bool(glob.glob(pattern))


def standard_with_perturbed_results_exist(pert_spec, cwd, arch="cnn1", dataset="mnist", n2_tag=None):
    """Check if standard N2 vagharWithPerturbed results exist for this perturbation.

    When n2_tag is given (e.g. 'seed42_itr20_sgd_itr1'), checks for that
    specific tag.  Otherwise checks for any vagharWithPerturbed_*_sgd_itr* dir.
    """
    pert_type, eps_str = pert_spec.split(":", 1)
    pert_dir_map = {
        "patch": "patch", "occ": "occ", "translation": "translation",
        "rotation": "rotation", "brightness": "brightness", "linf": "linf",
        "contrast": "contrast",
    }
    pert_dir = pert_dir_map.get(pert_type, pert_type)
    eps_dir = os.path.join(cwd, "paper_experiments", dataset, f"{arch}_exp",
                           pert_dir, f"eps_{eps_str}")
    if not os.path.isdir(eps_dir):
        return False
    if n2_tag:
        pattern = f"vagharWithPerturbed_{arch}_{n2_tag}"
    else:
        pattern = "vagharWithPerturbed_*_sgd_itr*"
    n2_dirs = glob.glob(os.path.join(eps_dir, pattern))
    for d in n2_dirs:
        if glob.glob(os.path.join(d, "*.txt")):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
# N1 state readiness + cross-process lock coordination
#
# Multiple sweep processes can run in parallel and share the same
# n1_state_<arch>_<tag>/ directory. Without coordination they would all
# independently re-solve N1 and race on writing the state files. A lock
# file in the state dir serializes "who solves N1", and other processes
# wait for the winner to finish before proceeding.
# ─────────────────────────────────────────────────────────────────────────

N1_LOCK_FILENAME = ".solving.lock"


def _n1_state_complete(n1_state_dir, need_pseudocosts, need_n1_preact=False):
    """Return True if the N1 state directory already contains everything we need.

    The legacy three-file layout (n1_vars / n1_layers / n1_vbasis) is enough
    when the caller only needs the existing techniques. If any N2 job is
    going to request bp=pseudocost or var_hint=true, the new
    n1_pseudocosts_*.bin file must also be present. If any N2 job is going
    to request --adv_std_n1_probe=lp, the n1_preact_bounds.bin file (added
    by save_n1_diff_bounds when --adv_std_zono_bounds or the probe flag is
    active) must also be present.
    """
    if not os.path.isdir(n1_state_dir):
        return False
    has_vars = bool(glob.glob(os.path.join(n1_state_dir, "n1_vars_*.bin")))
    if not has_vars:
        return False
    if need_pseudocosts:
        has_pseudocosts = bool(glob.glob(os.path.join(n1_state_dir, "n1_pseudocosts_*.bin")))
        if not has_pseudocosts:
            return False
    if need_n1_preact:
        has_preact = os.path.isfile(os.path.join(n1_state_dir, "n1_preact_bounds.bin"))
        if not has_preact:
            return False
    return True


def _acquire_n1_solve_lock(n1_state_dir, stale_after_sec):
    """Try to atomically claim the right to solve N1 for this state directory.

    Uses O_CREAT|O_EXCL via `open(path, 'x')` which is atomic across
    concurrent Python processes on the same POSIX filesystem. If another
    process already holds the lock, check whether the lock is stale (older
    than `stale_after_sec`) and steal it if so.

    Returns (True, lock_path) on success — the caller is responsible for
    calling `_release_n1_solve_lock(lock_path)` after the solve finishes.
    Returns (False, lock_path) if another live process holds the lock.
    """
    os.makedirs(n1_state_dir, exist_ok=True)
    lock_path = os.path.join(n1_state_dir, N1_LOCK_FILENAME)
    while True:
        try:
            with open(lock_path, "x") as f:
                f.write(f"pid={os.getpid()}\nstarted={time.time()}\n")
            return True, lock_path
        except FileExistsError:
            try:
                mtime = os.stat(lock_path).st_mtime
            except FileNotFoundError:
                # Raced with another process releasing it. Retry.
                continue
            age = time.time() - mtime
            if age > stale_after_sec:
                print(f"  WARNING: stealing stale N1 solve lock at {lock_path} (age {age:.0f}s > {stale_after_sec:.0f}s)")
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
                continue
            return False, lock_path


def _release_n1_solve_lock(lock_path):
    """Release a previously-acquired N1 solve lock. No-op if already gone."""
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def _wait_for_n1_state(n1_state_dir, need_pseudocosts, wait_timeout_sec, poll_interval_sec=30, need_n1_preact=False):
    """Block until another process finishes solving N1 and the state is ready.

    Returns True if the state became ready within `wait_timeout_sec`, False
    on timeout or if the other process released its lock but left the state
    incomplete (indicates the other process crashed or errored).
    """
    lock_path = os.path.join(n1_state_dir, N1_LOCK_FILENAME)
    start = time.time()
    while time.time() - start < wait_timeout_sec:
        if os.path.exists(lock_path):
            time.sleep(poll_interval_sec)
            continue
        # Lock gone. Verify the state is actually complete.
        if _n1_state_complete(n1_state_dir, need_pseudocosts, need_n1_preact=need_n1_preact):
            return True
        # Lock released but state incomplete — the other process likely
        # crashed or was killed. Surface this loudly; we do not try to
        # recover silently, because continuing with partial state would
        # produce subtly wrong results.
        return False
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
                    lp_time_str = fields.get("lp_optimization_time", "")
                    lp_time = float(lp_time_str) if lp_time_str else None
                    results[(cs, ct)] = {
                        "optimization_time": opt_time,
                        "hyper_attack_time": hyper_time,
                        "total_time": opt_time + hyper_time,
                        "lp_optimization_time": lp_time,
                        "total_time_with_lp": (opt_time + hyper_time + lp_time) if lp_time is not None else None,
                        "lower_bound": float(fields.get("lower_bound", "nan")),
                        "upper_bound": float(fields.get("upper_bound", "nan")),
                        "solve_status": fields.get("solve_status", ""),
                        "n2_org_relaxed_binaries": fields.get("n2_org_relaxed_binaries", ""),
                        "n2_pert_relaxed_binaries": fields.get("n2_pert_relaxed_binaries", ""),
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
    # New tag is _BoxScalarL; legacy tag is _NoBin. Match either.
    has_no_bin = ("BoxScalarL" in filename) or ("NoBin" in filename)
    has_n1xp = "N1xpConf" in filename
    prune_tol_match = re.search(r"PruneTol([\d.]+)", filename)
    has_zonotope = "Zonotope" in filename
    # Legacy tags (pre-merge): now folded into --use_zonotope but still appear
    # in historical filenames.
    has_refined_relu_legacy = "RefinedReLU" in filename
    has_zonotope_conv_legacy = "ZonoConv" in filename
    has_sparse_zono_legacy = "SparseZono" in filename
    gen_budget_match = re.search(r"GenBudget(\d+)", filename)
    has_no_n2_xp = "NoN2xpEnc" in filename
    # Combine no_n1_encoding and no_n2_xp_encoding into a single field:
    #   "no" = all networks encoded, "no_n1_encoding+..." = N1(x) skipped,
    #   "no_n2_xp_encoding+..." = N2(x') skipped
    has_n1_lp_relax = "NoN1BinRelaxOnN2only" in filename
    if has_no_n2_xp:
        encoding_skip = "no_n2_xp_encoding"
    elif has_n1_lp_relax:
        encoding_skip = "n1_lp_relax"
    elif has_last_layer and has_no_bin:
        encoding_skip = "no_n1_encoding+last_layer_no_bin"
    elif has_last_layer and has_n1xp:
        encoding_skip = "no_n1_encoding+last_layer+n1xp"
    elif has_last_layer:
        encoding_skip = "no_n1_encoding+last_layer"
    elif "NoN1Enc" in filename and has_n1xp:
        encoding_skip = "no_n1_encoding+n1xp"
    elif "NoN1Enc" in filename:
        encoding_skip = "no_n1_encoding"
    else:
        encoding_skip = "no"
    if prune_tol_match:
        encoding_skip += "+pruneTol" + prune_tol_match.group(1)
    adapt_prune_match = re.search(r"AdaptPrune([\d.]+)", filename)
    if adapt_prune_match:
        encoding_skip += "+adaptPrune" + adapt_prune_match.group(1)
    n1_stab_match = re.search(r"N1StabRelax([\d.]+)", filename)
    if n1_stab_match:
        encoding_skip += "+n1StabRelax" + n1_stab_match.group(1)
    if has_zonotope:
        encoding_skip += "+zono"
    zono_ord_match = re.search(r"ZonoOrd(\d+)", filename)
    if zono_ord_match:
        encoding_skip += "+zonoOrd" + zono_ord_match.group(1)
    # Legacy tags — stop cluttering the label; they're implied by +zono now.
    if has_refined_relu_legacy:
        encoding_skip += "+refinedReLU"
    if has_zonotope_conv_legacy:
        encoding_skip += "+zonoConv"
    if has_sparse_zono_legacy:
        encoding_skip += "+sparseZono"
    if gen_budget_match:
        encoding_skip += "+genK" + gen_budget_match.group(1)
    # Legacy --n2_xp_k_value / --bridge_at_split tags (flags removed).
    n2_xp_k_match = re.search(r"N2xpK(\d+)", filename)
    if n2_xp_k_match:
        encoding_skip += "+n2xpK" + n2_xp_k_match.group(1)
    if "SplitBridge" in filename:
        encoding_skip += "+splitBridge"
    if "BoundN2xpOut" in filename:
        encoding_skip += "+boundN2xpOut"
    if "BoundN2xpComp" in filename:
        encoding_skip += "+boundN2xpComp"
    if "N2xpViaN1Zono" in filename:
        encoding_skip += "+n2xpViaN1Zono"
    if "BranchPriN2x" in filename:
        encoding_skip += "+branchPriN2x"
    if "capDD" in filename:
        encoding_skip += "+capDD"
    # N2 bound tightening (current tags + legacy TightenN2 tag)
    has_std_warmstart = "StdWarmstart" in filename
    if has_std_warmstart:
        encoding_skip += "+stdWarmstart"
    has_bound_n2_relu = ("BoundN2ReLU" in filename) or ("TightenN2" in filename)
    has_bound_n2_non_relu = "BoundN2NonReLU" in filename
    return {
        "threads": int(threads_match.group(1)) if threads_match else "",
        "relax_count": int(relax_count_match.group(1)) if relax_count_match else "",
        "optimizing_intervals": opt_intervals,
        "encoding_skip": encoding_skip,
        "bound_n2_relu_using_zonotope": "yes" if has_bound_n2_relu else "no",
        "bound_n2_non_relu_using_zonotope": "yes" if has_bound_n2_non_relu else "no",
    }


def find_transfer_faster_than_standard(perts, exp_base, csv_transfer_faster, csv_standard_faster,
                                       csv_transfer_tighter_at_timeout, csv_standard_tighter_at_timeout,
                                       arch="cnn1", double_check_standard=False,
                                       compare_to_with_perturbed=False,
                                       transfer_opt_time_only=False):
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
        "encoding_skip",
        "bound_n2_relu_using_zonotope",
        "bound_n2_non_relu_using_zonotope",
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
                    # Skip old-style optimizing_intervals runs (but allow NoN1Encoding and NoN2xpEncoding)
                    if meta["optimizing_intervals"] == "yes" and meta["encoding_skip"] == "no":
                        continue
                    s_info = standard_results[key]
                    if s_info.get("solve_status", "") == "INTERRUPTED":
                        continue
                    if t_info.get("solve_status", "") == "INTERRUPTED":
                        continue

                    t_time = t_info["optimization_time"] if transfer_opt_time_only else t_info["total_time"]
                    s_time = s_info["total_time"]
                    
                    
                    # #TODO NETA
                    # if float(t_info['lower_bound']) > 5:
                    #     continue
                    # #TODO NETA

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
                        "encoding_skip": meta["encoding_skip"],
                        "bound_n2_relu_using_zonotope": meta["bound_n2_relu_using_zonotope"],
                        "bound_n2_non_relu_using_zonotope": meta["bound_n2_non_relu_using_zonotope"],
                    }

                    if not compare_to_with_perturbed:
                        s_gap = s_info["upper_bound"] - s_info["lower_bound"]
                        t_gap = t_info["upper_bound"] - t_info["lower_bound"]
                        row["gap_standard"] = f"{s_gap:.6f}"
                        row["gap_transfer"] = f"{t_gap:.6f}"
                    row["solve_status_standard"] = s_info.get("solve_status", "")
                    row["solve_status_transfer"] = t_info.get("solve_status", "")

                    
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


def _extract_advstd_file_metadata(filename):
    """Extract advanced-standard technique flags from a result filename."""
    # Match "_seed<N>" for the Gurobi seed only — exclude "_seed<N>_itr<M>"
    # which is part of the model name. (?=_) anchors so greedy \d+ can't
    # backtrack and partially match "seed42" as "seed4".
    seed_match = re.search(r"_seed(\d+)(?=_)(?!_itr)", filename)
    # Branch priorities: distinguish 3 modes. Check the more specific
    # "branchPriPsd" tag first so it doesn't get masked by "branchPri".
    if "branchPriPsd" in filename:
        bp = "pseudocost"
    elif "branchPri" in filename:
        bp = "bounds"
    else:
        bp = "off"
    relax_match = re.search(r"_relaxT([-0-9.]+)", filename)
    elim_org_match = re.search(r"_elimOrg(\d+)", filename)
    elim_pert_match = re.search(r"_elimPert(\d+)", filename)
    return {
        "mip_start": "yes" if "mipStart" in filename else "no",
        "branch_priorities": bp,
        "lp_basis": "yes" if "lpBasis" in filename else "no",
        "bound_tightening": "yes" if "boundTight" in filename else "no",
        "var_hint": "yes" if "varHint" in filename else "no",
        "var_hint_fix": "yes" if "varHintFix" in filename else "no",
        "zono_bounds": "yes" if "zonoBounds" in filename else "no",
        "n1_probe": "lp" if "n1ProbeLP" in filename else "off",
        "relax_threshold": relax_match.group(1) if relax_match else "off",
        "elim_org": elim_org_match.group(1) if elim_org_match else "elim not activated",
        "elim_pert": elim_pert_match.group(1) if elim_pert_match else "elim not activated",
        "seed": seed_match.group(1) if seed_match else "0",
    }


def find_advstd_faster_than_standard(perts, exp_base, csv_advstd_faster, csv_standard_faster,
                                      csv_advstd_tighter_at_timeout, csv_standard_tighter_at_timeout,
                                      arch="cnn1", compare_to_with_perturbed=False):
    """For each perturbation/eps, compare advanced-standard N2 vs regular standard N2.

    Returns four lists of row dicts (advstd_faster, standard_faster,
    advstd_tighter, standard_tighter). Also writes them to CSVs.
    """
    import csv

    pert_dir_map = {
        "patch": "patch", "occ": "occ", "trans": "translation",
        "rotation": "rotation", "brightness": "brightness",
    }

    fieldnames = [
        "arch", "perturbation", "perturbation_size", "c_source", "c_target",
        "time_standard", "time_advstd",
        "delta_standard_lower_bound", "delta_standard_upper_bound",
        "delta_advstd_lower_bound", "delta_advstd_upper_bound",
        "delta_error",
        "mip_start", "branch_priorities", "lp_basis", "bound_tightening",
        "var_hint", "zono_bounds", "n1_probe", "relax_threshold",
        "elim_org", "elim_pert", "relaxed_org", "relaxed_pert",
        "seed",
        "how_much_faster",
        "lp_optimization_time", "time_advstd_with_lp", "how_much_faster_with_lp",
        "gap_standard", "gap_advstd",
        "solve_status_standard", "solve_status_advstd",
        "standard_file", "advstd_file",
    ]

    rows_advstd_faster = []
    rows_standard_faster = []
    rows_advstd_tighter = []
    rows_standard_tighter = []

    for pert_name, pert_spec in perts:
        pert_type, eps_str = pert_spec.split(":", 1)
        pert_dir = pert_dir_map.get(pert_type, pert_type)
        eps_dir = os.path.join(exp_base, pert_dir, f"eps_{eps_str}")

        if not os.path.isdir(eps_dir):
            print(f"  [{pert_name}] No results directory: {eps_dir}")
            continue

        # Find standard N2 directories (regular standard mode results)
        if compare_to_with_perturbed:
            std_pattern = "vagharWithPerturbed_*_sgd_itr*"
        else:
            std_pattern = "vagharNoPerturbed_*_sgd_itr*"
        standard_n2_dirs = sorted(glob.glob(os.path.join(eps_dir, std_pattern)))
        if not standard_n2_dirs:
            print(f"  [{pert_name}] No standard N2 ({std_pattern}) found in {eps_dir}")
            continue

        # Load all standard N2 results: (c_source, c_target) -> (result_dict, filepath)
        standard_results = {}
        for sd in standard_n2_dirs:
            txt_files = glob.glob(os.path.join(sd, "*.txt"))
            for tf in txt_files:
                parsed = parse_result_file(tf)
                for key, val in parsed.items():
                    standard_results[key] = (val, tf)

        if not standard_results:
            print(f"  [{pert_name}] No results parsed from standard N2 directories")
            continue

        # Load N1 results to detect incomplete tightening (TIME_LIMIT)
        if compare_to_with_perturbed:
            n1_pattern = "vagharWithPerturbed_*"
        else:
            n1_pattern = "vagharNoPerturbed_*"
        n1_dirs = [d for d in sorted(glob.glob(os.path.join(eps_dir, n1_pattern)))
                    if not re.search(r"_sgd_itr\d+", os.path.basename(d))]
        n1_results = {}  # (c_source, c_target) -> result_dict
        for nd in n1_dirs:
            txt_files = glob.glob(os.path.join(nd, "*.txt"))
            for tf in txt_files:
                parsed = parse_result_file(tf)
                for key, val in parsed.items():
                    n1_results[key] = val

        # Find advanced-standard directories
        advstd_dirs = sorted(glob.glob(os.path.join(eps_dir, "advStd_*")))
        if not advstd_dirs:
            print(f"  [{pert_name}] No advStd directories found")
            continue

        for ad in advstd_dirs:
            txt_files = sorted(glob.glob(os.path.join(ad, "*.txt")))
            for tf in txt_files:
                tf_name = os.path.basename(tf)
                # Only process N2 result files
                if "_N2_advStd" not in tf_name:
                    continue
                meta = _extract_advstd_file_metadata(tf_name)
                advstd_results = parse_result_file(tf)

                for (cs, ct), a_info in sorted(advstd_results.items()):
                    key = (cs, ct)
                    if key not in standard_results:
                        continue
                    s_info, std_file = standard_results[key]
                    if s_info.get("solve_status", "") == "INTERRUPTED":
                        continue
                    if a_info.get("solve_status", "") == "INTERRUPTED":
                        continue
                    # Skip if N1 tightening didn't finish — incomplete
                    # tightening makes the advstd comparison unreliable
                    n1_info = n1_results.get(key)
                    if n1_info and n1_info.get("solve_status", "") == "TIME_LIMIT":
                        continue

                    a_time = a_info["total_time"]
                    s_time = s_info["total_time"]
                    a_lp_time = a_info.get("lp_optimization_time")
                    a_time_with_lp = a_info.get("total_time_with_lp")

                    s_gap = s_info["upper_bound"] - s_info["lower_bound"]
                    a_gap = a_info["upper_bound"] - a_info["lower_bound"]

                    # Compute speedup including LP time (empty if LP time unavailable)
                    if a_time_with_lp is not None and a_time_with_lp > 0 and s_time > 0:
                        how_much_faster_with_lp = f"{a_time_with_lp / s_time:.2f}x"
                    else:
                        how_much_faster_with_lp = ""

                    row = {
                        "arch": arch,
                        "perturbation": pert_type,
                        "perturbation_size": eps_str,
                        "c_source": cs,
                        "c_target": ct,
                        "time_standard": f"{s_time:.2f}",
                        "time_advstd": f"{a_time:.2f}",
                        "delta_standard_lower_bound": f"{s_info['lower_bound']:.6f}",
                        "delta_standard_upper_bound": f"{s_info['upper_bound']:.6f}",
                        "delta_advstd_lower_bound": f"{a_info['lower_bound']:.6f}",
                        "delta_advstd_upper_bound": f"{a_info['upper_bound']:.6f}",
                        "delta_error": f"{a_info['upper_bound'] - s_info['upper_bound']:.6f}",
                        "mip_start": meta["mip_start"],
                        "branch_priorities": meta["branch_priorities"],
                        "lp_basis": meta["lp_basis"],
                        "bound_tightening": meta["bound_tightening"],
                        "var_hint": meta["var_hint"],
                        "zono_bounds": meta["zono_bounds"],
                        "n1_probe": meta["n1_probe"],
                        "relax_threshold": meta["relax_threshold"],
                        "elim_org": meta["elim_org"],
                        "elim_pert": meta["elim_pert"],
                        "relaxed_org": (a_info.get("n2_org_relaxed_binaries", "")
                                        if meta["relax_threshold"] != "off"
                                        else "relax not activated"),
                        "relaxed_pert": (a_info.get("n2_pert_relaxed_binaries", "")
                                         if meta["relax_threshold"] != "off"
                                         else "relax not activated"),
                        "seed": meta["seed"],
                        "how_much_faster": "",
                        "lp_optimization_time": f"{a_lp_time:.2f}" if a_lp_time is not None else "",
                        "time_advstd_with_lp": f"{a_time_with_lp:.2f}" if a_time_with_lp is not None else "",
                        "how_much_faster_with_lp": how_much_faster_with_lp,
                        "gap_standard": f"{s_gap:.6f}",
                        "gap_advstd": f"{a_gap:.6f}",
                        "solve_status_standard": s_info.get("solve_status", ""),
                        "solve_status_advstd": a_info.get("solve_status", ""),
                        "standard_file": std_file,
                        "advstd_file": tf,
                    }

                    if a_time < s_time * 0.99:  # advstd is faster
                        row["how_much_faster"] = f"{a_time / s_time:.2f}x"
                        rows_advstd_faster.append(row)
                    elif s_time < a_time * 0.99:  # standard is faster
                        row["how_much_faster"] = f"{a_time / s_time:.2f}x"
                        rows_standard_faster.append(row)
                    else:  # both hit timeout (~same time)
                        s_is_timeout = s_info.get("solve_status", "") == "TIME_LIMIT"
                        a_is_timeout = a_info.get("solve_status", "") == "TIME_LIMIT"
                        if s_is_timeout and a_is_timeout:
                            if a_gap < s_gap * 0.99:
                                rows_advstd_tighter.append(row)
                            elif s_gap < a_gap * 0.99:
                                rows_standard_tighter.append(row)

    # Sort
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
                float(row["gap_advstd"]))

    rows_advstd_faster.sort(key=_sort_faster)
    rows_standard_faster.sort(key=_sort_faster)
    rows_advstd_tighter.sort(key=_sort_tighter)
    rows_standard_tighter.sort(key=_sort_tighter)

    def _group_key(row):
        return (row["arch"], row["perturbation"], row["perturbation_size"],
                row["c_source"], row["c_target"])

    empty_row = {fn: "" for fn in fieldnames}
    for filepath, rows in [
        (csv_advstd_faster, rows_advstd_faster),
        (csv_standard_faster, rows_standard_faster),
        (csv_advstd_tighter_at_timeout, rows_advstd_tighter),
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

    print(f"  Wrote {len(rows_advstd_faster)} rows to {csv_advstd_faster}")
    print(f"  Wrote {len(rows_standard_faster)} rows to {csv_standard_faster}")
    print(f"  Wrote {len(rows_advstd_tighter)} rows to {csv_advstd_tighter_at_timeout}")
    print(f"  Wrote {len(rows_standard_tighter)} rows to {csv_standard_tighter_at_timeout}")

    return rows_advstd_faster, rows_standard_faster, rows_advstd_tighter, rows_standard_tighter


def write_advstd_combo_ranking_csv(rows_advstd_faster, rows_standard_faster, csv_path, seeds=None):
    """Aggregate per-cell sweep rows into one row per flag combination.

    Each combo = the 8-tuple (mip_start, branch_priorities, lp_basis,
    bound_tightening, var_hint, zono_bounds, n1_probe, relax_threshold).
    Within a combo we group the per-cell rows by (arch, perturbation, size,
    c_source, c_target, seed), dedupe duplicate runs of the same cell via
    geometric mean of `time_advstd / time_standard`, then compute win/loss
    counts and geomean speedups (gm_win, gm_lose, gm_all).

    The row is classified per seed (WIN / LOSE / flip / miss) and given an
    overall label (STRICT / GENERAL / MIXED / LOSER). See the docstring of
    the calling branch for the definitions.
    """
    import csv as _csv
    import math
    from collections import defaultdict

    FLAG_FIELDS = (
        "mip_start", "branch_priorities", "lp_basis", "bound_tightening",
        "var_hint", "zono_bounds", "n1_probe", "relax_threshold",
    )
    TC_FIELDS = ("arch", "perturbation", "perturbation_size", "c_source", "c_target")

    def _combo_key(row):
        return tuple(row[f] for f in FLAG_FIELDS)

    def _tc_key(row):
        return tuple(row[f] for f in TC_FIELDS)

    # cells[(combo, tc, seed)] -> list of speedup values
    cells = defaultdict(list)
    cells_with_lp = defaultdict(list)  # same but using total_time_with_lp
    archs_per_combo = defaultdict(set)
    perts_per_combo = defaultdict(set)
    ctargets_per_combo = defaultdict(set)
    elim_org_per_combo = defaultdict(list)
    elim_pert_per_combo = defaultdict(list)
    relaxed_org_per_combo = defaultdict(list)
    relaxed_pert_per_combo = defaultdict(list)
    delta_error_per_combo = defaultdict(list)
    for row in list(rows_advstd_faster) + list(rows_standard_faster):
        try:
            t_std = float(row["time_standard"])
            t_adv = float(row["time_advstd"])
        except (TypeError, ValueError):
            continue
        if t_adv <= 0 or t_std <= 0:
            continue
        combo = _combo_key(row)
        cell_key = (combo, _tc_key(row), row["seed"])
        cells[cell_key].append(t_adv / t_std)
        # Collect with-LP speedup when LP time is available
        t_adv_with_lp_str = row.get("time_advstd_with_lp", "")
        if t_adv_with_lp_str:
            try:
                t_adv_with_lp = float(t_adv_with_lp_str)
                if t_adv_with_lp > 0:
                    cells_with_lp[cell_key].append(t_adv_with_lp / t_std)
            except (TypeError, ValueError):
                pass
        archs_per_combo[combo].add(row["arch"])
        perts_per_combo[combo].add((row["perturbation"], row["perturbation_size"]))
        ctargets_per_combo[combo].add(row["c_target"])
        for field, bucket in (("elim_org", elim_org_per_combo),
                              ("elim_pert", elim_pert_per_combo),
                              ("relaxed_org", relaxed_org_per_combo),
                              ("relaxed_pert", relaxed_pert_per_combo)):
            raw = (row.get(field) or "").strip()
            try:
                bucket[combo].append(int(raw))
            except ValueError:
                pass  # "elim not activated" / "relax not activated" / empty — skip
        raw_err = (row.get("delta_error") or "").strip()
        try:
            delta_error_per_combo[combo].append(float(raw_err))
        except ValueError:
            pass

    def _geomean(xs):
        xs = list(xs)
        if not xs:
            return None
        return math.exp(sum(math.log(x) for x in xs) / len(xs))

    # Dedupe each (combo, tc, seed) cell via geomean, then bucket per combo.
    combo_cells = defaultdict(list)  # combo -> list of (tc, seed, speedup)
    for (combo, tc, seed), sp_list in cells.items():
        combo_cells[combo].append((tc, seed, _geomean(sp_list)))

    # Same dedup for with-LP speedups
    combo_cells_with_lp = defaultdict(list)
    for (combo, tc, seed), sp_list in cells_with_lp.items():
        combo_cells_with_lp[combo].append((tc, seed, _geomean(sp_list)))

    if seeds is None:
        seeds = sorted({k[2] for k in cells.keys()},
                       key=lambda s: int(s) if s.isdigit() else s)

    def _classify_combo(cells_list):
        by_seed = defaultdict(list)
        for _tc, seed, sp in cells_list:
            by_seed[seed].append(sp)
        per_seed = {}
        for seed in seeds:
            vals = by_seed.get(seed, [])
            if not vals:
                per_seed[seed] = ""
                continue
            n_win = sum(1 for v in vals if v < 1)
            n_lose = len(vals) - n_win
            if n_win > 0 and n_lose == 0:
                per_seed[seed] = "WIN"
            elif n_win == 0 and n_lose > 0:
                per_seed[seed] = "LOSE"
            else:
                per_seed[seed] = "flip"
        return per_seed

    agg_rows = []
    for combo, cells_list in combo_cells.items():
        wins = [sp for _, _, sp in cells_list if sp < 1]
        losses = [sp for _, _, sp in cells_list if sp >= 1]
        all_sp = [sp for _, _, sp in cells_list]
        per_seed = _classify_combo(cells_list)
        n_win_seeds = sum(1 for v in per_seed.values() if v == "WIN")
        n_flip_seeds = sum(1 for v in per_seed.values() if v == "flip")
        n_lose_seeds = sum(1 for v in per_seed.values() if v == "LOSE")

        gm_all_raw = _geomean(all_sp)
        min_speedup_raw = max(all_sp) if all_sp else None

        # ── Coverage tier: how varied was the test slice? ──
        n_perts_covered = len(perts_per_combo[combo])
        n_archs_covered = len(archs_per_combo[combo])
        if n_perts_covered >= 4:
            coverage_tier = "broad"
        elif n_perts_covered >= 2:
            coverage_tier = "medium"
        else:
            coverage_tier = "narrow"

        # ── Performance tier: aggregate speedup × worst-case regression ──
        n_lose_cells = len(losses)
        if gm_all_raw is None:
            perf_tier = "unknown"
        elif n_lose_cells == 0 and gm_all_raw < 1.0 / 1.05:
            perf_tier = "dominant"
        elif gm_all_raw < 1.0 / 1.05 and min_speedup_raw is not None and min_speedup_raw <= 1.0 / 0.75:
            perf_tier = "avg-win"
        elif gm_all_raw < 1.0 / 1.05:
            perf_tier = "avg-win-risky"
        elif gm_all_raw <= 1.0 / 0.9:
            perf_tier = "neutral"
        else:
            perf_tier = "loser"

        _PERF_MEANING = {
            "dominant": "no cell slower, gm_all < 0.952x",
            "avg-win": "gm_all < 0.952x, worst cell <= 1.333x",
            "avg-win-risky": "gm_all < 0.952x, worst cell > 1.333x",
            "neutral": "gm_all in [0.952x, 1.111x]",
            "loser": "gm_all > 1.111x",
            "unknown": "no usable speedup data",
        }
        label = f"{coverage_tier}-{perf_tier}"
        label_meaning = (
            f"{n_perts_covered} perturbation(s), {n_archs_covered} arch(s); "
            f"{_PERF_MEANING[perf_tier]}"
        )

        def _fmt_gm(x):
            return f"{x:.3f}x" if x is not None else ""

        agg = {f: combo[i] for i, f in enumerate(FLAG_FIELDS)}
        agg["label"] = label
        agg["label_meaning"] = label_meaning
        agg["coverage_tier"] = coverage_tier
        agg["perf_tier"] = perf_tier
        agg["n_perturbations_covered"] = n_perts_covered
        agg["_coverage_tier"] = coverage_tier
        agg["_perf_tier"] = perf_tier
        agg["n_tested"] = len(all_sp)
        agg["n_win"] = len(wins)
        agg["n_lose"] = len(losses)
        agg["gm_win"] = _fmt_gm(_geomean(wins))
        agg["gm_lose"] = _fmt_gm(_geomean(losses))
        agg["gm_all"] = _fmt_gm(_geomean(all_sp))
        # With-LP speedup: geomean over cells that have LP time data
        lp_cells = combo_cells_with_lp.get(combo, [])
        all_sp_with_lp = [sp for _, _, sp in lp_cells]
        agg["gm_all_with_lp"] = _fmt_gm(_geomean(all_sp_with_lp))
        agg["max_speed_up"] = _fmt_gm(min(all_sp)) if all_sp else ""
        agg["min_speed_up"] = _fmt_gm(max(all_sp)) if all_sp else ""
        agg["_gm_all_raw"] = _geomean(all_sp) or 0.0
        agg["n_win_seeds"] = n_win_seeds
        agg["n_flip_seeds"] = n_flip_seeds
        agg["n_lose_seeds"] = n_lose_seeds
        def _maybe_int(x):
            try:
                return int(x)
            except (TypeError, ValueError):
                return x
        agg["archs_covered"] = ",".join(sorted(str(a) for a in archs_per_combo[combo]))
        agg["tested_perturbations"] = " | ".join(
            sorted(f"{p}(eps_{s})" for p, s in perts_per_combo[combo])
        )
        agg["c_targets_covered"] = ",".join(
            str(t) for t in sorted(ctargets_per_combo[combo], key=_maybe_int)
        )

        def _mean_int(xs, inactive_label):
            if not xs:
                return inactive_label
            avg = sum(xs) / len(xs)
            return str(int(avg)) if avg == int(avg) else f"{avg:.1f}"
        agg["elim_org_avg"] = _mean_int(elim_org_per_combo[combo], "elim not activated")
        agg["elim_pert_avg"] = _mean_int(elim_pert_per_combo[combo], "elim not activated")
        agg["relaxed_org_avg"] = _mean_int(relaxed_org_per_combo[combo], "relax not activated")
        agg["relaxed_pert_avg"] = _mean_int(relaxed_pert_per_combo[combo], "relax not activated")
        err_vals = delta_error_per_combo[combo]
        if err_vals:
            agg["delta_error_avg"] = f"{sum(err_vals)/len(err_vals):.6f}"
            agg["delta_error_abs_avg"] = f"{sum(abs(x) for x in err_vals)/len(err_vals):.6f}"
            agg["delta_error_max"] = f"{max(err_vals):.6f}"
            agg["delta_error_min"] = f"{min(err_vals):.6f}"
        else:
            agg["delta_error_avg"] = ""
            agg["delta_error_abs_avg"] = ""
            agg["delta_error_max"] = ""
            agg["delta_error_min"] = ""
        agg_rows.append(agg)

    coverage_order = {"broad": 0, "medium": 1, "narrow": 2}
    perf_order = {
        "dominant": 0, "avg-win": 1, "avg-win-risky": 2,
        "neutral": 3, "loser": 4, "unknown": 5,
    }
    agg_rows.sort(key=lambda r: (perf_order.get(r["_perf_tier"], 9),
                                 coverage_order.get(r["_coverage_tier"], 9),
                                 r["_gm_all_raw"],
                                 -r["n_tested"]))

    fieldnames = (
        ["label", "label_meaning", "coverage_tier", "perf_tier", "n_perturbations_covered"]
        + list(FLAG_FIELDS)
        + ["n_tested", "n_win", "n_lose",
           "gm_win", "gm_lose", "gm_all", "gm_all_with_lp",
           "max_speed_up", "min_speed_up",
           "delta_error_avg", "delta_error_abs_avg",
           "delta_error_max", "delta_error_min",
           "elim_org_avg", "elim_pert_avg",
           "relaxed_org_avg", "relaxed_pert_avg",
           "n_win_seeds", "n_flip_seeds", "n_lose_seeds"]
        + ["archs_covered", "tested_perturbations", "c_targets_covered"]
    )

    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in agg_rows:
            r.pop("_gm_all_raw", None)
            r.pop("_coverage_tier", None)
            r.pop("_perf_tier", None)
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    return len(agg_rows)


def _generate_combo_ranking_csv(arch_runs, cwd, dataset,
                                compare_to_with_perturbed, combo_ranking_seeds):
    """Scan existing per-cell results and write the combo-ranking CSV.

    Encapsulates the logic behind --find_advstd_faster_than_standard so the
    sweep can regenerate the CSV on the fly before applying
    --advstd_safe_combos_only. Returns the path of the combo-ranking CSV.
    """
    import csv as _csv
    all_perts = PERTURBATIONS
    combined_base = os.path.join(cwd, "paper_experiments", dataset)
    os.makedirs(combined_base, exist_ok=True)
    suffix = "_vs_withPerturbed" if compare_to_with_perturbed else ""
    csv_advstd_faster = os.path.join(combined_base, f"advstd_faster_than_standard{suffix}.csv")
    csv_standard_faster = os.path.join(combined_base, f"standard_faster_than_advstd{suffix}.csv")
    csv_advstd_tighter = os.path.join(combined_base, f"advstd_tighter_at_timeout{suffix}.csv")
    csv_standard_tighter = os.path.join(combined_base, f"standard_tighter_at_timeout_vs_advstd{suffix}.csv")

    all_af, all_sf, all_at, all_st = [], [], [], []
    for arch, _ in arch_runs:
        exp_base = os.path.join(cwd, "paper_experiments", dataset, f"{arch}_exp")
        print(f"\nScanning advanced-standard results for {arch} in: {exp_base}")
        af, sf, at, st = find_advstd_faster_than_standard(
            all_perts, exp_base, csv_advstd_faster, csv_standard_faster,
            csv_advstd_tighter, csv_standard_tighter, arch=arch,
            compare_to_with_perturbed=compare_to_with_perturbed)
        all_af.extend(af); all_sf.extend(sf); all_at.extend(at); all_st.extend(st)

    _fieldnames = [
        "arch", "perturbation", "perturbation_size", "c_source", "c_target",
        "time_standard", "time_advstd",
        "delta_standard_lower_bound", "delta_standard_upper_bound",
        "delta_advstd_lower_bound", "delta_advstd_upper_bound",
        "delta_error",
        "mip_start", "branch_priorities", "lp_basis", "bound_tightening",
        "var_hint", "zono_bounds", "n1_probe", "relax_threshold",
        "elim_org", "elim_pert", "relaxed_org", "relaxed_pert",
        "seed",
        "how_much_faster",
        "lp_optimization_time", "time_advstd_with_lp", "how_much_faster_with_lp",
        "gap_standard", "gap_advstd",
        "solve_status_standard", "solve_status_advstd",
        "standard_file", "advstd_file",
    ]

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
                float(row["gap_advstd"]))

    def _group_key(row):
        return (row["arch"], row["perturbation"], row["perturbation_size"],
                row["c_source"], row["c_target"])

    all_af.sort(key=_sort_faster)
    all_sf.sort(key=_sort_faster)
    all_at.sort(key=_sort_tighter)
    all_st.sort(key=_sort_tighter)

    empty_row = {fn: "" for fn in _fieldnames}
    for filepath, rows in [
        (csv_advstd_faster, all_af), (csv_standard_faster, all_sf),
        (csv_advstd_tighter, all_at), (csv_standard_tighter, all_st),
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
    print(f"  {len(all_af)} rows -> {csv_advstd_faster}")
    print(f"  {len(all_sf)} rows -> {csv_standard_faster}")
    print(f"  {len(all_at)} rows -> {csv_advstd_tighter}")
    print(f"  {len(all_st)} rows -> {csv_standard_tighter}")

    csv_combo_ranking = os.path.join(combined_base, f"advstd_combo_ranking{suffix}.csv")
    if combo_ranking_seeds:
        all_af = [r for r in all_af if r.get("seed") in combo_ranking_seeds]
        all_sf = [r for r in all_sf if r.get("seed") in combo_ranking_seeds]
    n_combos = write_advstd_combo_ranking_csv(
        all_af, all_sf, csv_combo_ranking, seeds=combo_ranking_seeds)
    print(f"  {n_combos} combos -> {csv_combo_ranking}"
          + (f" (seeds={combo_ranking_seeds})" if combo_ranking_seeds else ""))

    _update_advstd_tex_tables(cwd, combined_base, arch_runs,
                              compare_to_with_perturbed, combo_ranking_seeds)
    return csv_combo_ranking


def _update_advstd_tex_tables(cwd, combined_base, arch_runs,
                              compare_to_with_perturbed, combo_ranking_seeds):
    """Rewrite advstd_techniques.tex tables after regenerating CSVs."""
    try:
        sys.path.insert(0, cwd)
        import update_advstd_tex_tables as updater
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[tex-update] skipped (import failed: {exc})")
        return
    tex_path = os.path.join(cwd, "advstd_techniques.tex")
    if not os.path.exists(tex_path):
        print(f"[tex-update] skipped (missing {tex_path})")
        return
    suffix = "_vs_withPerturbed" if compare_to_with_perturbed else ""
    rows = updater.load_rows(combined_base, suffix)
    if not rows:
        print(f"[tex-update] skipped (no rows in {combined_base})")
        return
    archs = [arch for arch, _ in arch_runs]
    seeds = combo_ranking_seeds or sorted({r["seed"] for r in rows
                                           if r.get("seed")})
    seed = seeds[0] if seeds else "4"
    tau = "0.1"
    try:
        body = updater.render_all(archs, rows, seed, tau)
        updater.update_tex(tex_path, body)
    except SystemExit as exc:
        print(f"[tex-update] {exc}")
    except Exception as exc:
        print(f"[tex-update] error: {exc}")


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
    parser.add_argument("--find_advstd_faster_than_standard", action="store_true",
                        help="Scan existing results and report advanced-standard N2 experiments "
                             "that are faster than regular standard N2 for each perturbation "
                             "and (c_source, c_target) pair.")
    parser.add_argument("--skip_vaghar_no_perturbed", action="store_true",
                        help="When running standard, skip vagharNoPerturbed (without perturbed intervals) "
                             "and only run vagharWithPerturbed.")
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
    parser.add_argument("--no_n2_xp_encoding", action="store_true",
                        help="Skip N2(x') encoding entirely; replace conf(N2,x',c) with "
                             "interval-bounded output variables using perturbation bounds through N2. "
                             "Assumes no_n1_encoding_at_all=false.")
    parser.add_argument("--encode_n1_last_layer", action="store_true",
                        help="When no_n1_encoding_at_all is active, encode N1 last linear layer "
                             "exactly using interval-bounded hidden variables; gives exact delta_diff.")
    parser.add_argument("--cap_delta_diff", action="store_true",
                        help="Add delta_diff <= max_k(d_hi[c]-d_lo[k]) as a valid upper bound "
                             "constraint. Tightens LP relaxation for faster solving.")
    parser.add_argument("--n1_last_layer_prune_tol", type=float, default=0.0,
                        help="Drop h_n1 variables with interval width <= this and use "
                             "worst-case constants. 0.0 = only exact singletons. "
                             "Requires --encode_n1_last_layer.")
    parser.add_argument("--sweep_n1_adaptive_prune_budget", nargs="*", type=float, default=None,
                        help="Cross-product: sweep over adaptive pruning budget values. "
                             "E.g. --sweep_n1_adaptive_prune_budget 0 0.1 0.5 1.0")
    parser.add_argument("--sweep_n1_stability_relax_threshold", nargs="*", type=float, default=None,
                        help="Cross-product: sweep over N1 stability relaxation threshold values. "
                             "E.g. --sweep_n1_stability_relax_threshold -1 0 0.05 0.1")
    parser.add_argument("--constrain_n1_xp", action="store_true",
                        help="Add interval-based constraint that conf(N1,x',c_target)<=0; "
                             "no extra variables, uses pre-computed pert bounds through N1.")
    parser.add_argument("--use_zonotope", action="store_true",
                        help="Use zonotope (affine arithmetic) for diff bound propagation; "
                             "tighter bounds by tracking correlations between neurons.")
    parser.add_argument("--sweep_zonotope_max_order", nargs="*", type=int, default=None,
                        help="Cross-product: sweep over zonotope max order values. "
                             "E.g. --sweep_zonotope_max_order 0 3 5 10. Requires --use_zonotope.")
    parser.add_argument("--sweep_bound_n2_xp_output_using_composed", action="store_true",
                        help="Cross-product: run each transfer job twice, once with "
                             "--bound_n2_xp_output_using_composed true and once false.")
    parser.add_argument("--sweep_bound_n2_xp_using_composed", action="store_true",
                        help="Cross-product: run each transfer job twice, once with "
                             "--bound_n2_xp_using_composed true and once false.")
    parser.add_argument("--sweep_branch_priority_n2x_first", action="store_true",
                        help="Cross-product: run each transfer job twice, once with "
                             "--branch_priority_n2x_first true and once false.")
    parser.add_argument("--sweep_constrain_n2_xp_via_n1_zonotope", action="store_true",
                        help="Cross-product: run each transfer job twice, once with "
                             "--constrain_n2_xp_via_n1_zonotope true and once false.")
    parser.add_argument("--sweep_bound_n2_relu_using_zonotope", action="store_true",
                        help="Cross-product: run each transfer job twice, once with "
                             "--bound_n2_relu_using_zonotope true and once false. If omitted, runs once "
                             "with the flag off.")
    parser.add_argument("--sweep_bound_by_zonotope_n2_hidden_neurons_which_are_not_relu", action="store_true",
                        help="Cross-product: run each transfer job twice, once with "
                             "--bound_by_zonotope_n2_hidden_neurons_which_are_not_relu true and once false. "
                             "If omitted, runs once with the flag off.")
    parser.add_argument("--compare_to_with_perturbed", action="store_true",
                        help="Compare transfer results to vagharWithPerturbed (standard with perturbed "
                             "intervals) instead of vagharNoPerturbed.")
    parser.add_argument("--combo_ranking_seeds", nargs="+", default=None,
                        help="Restrict the combo-ranking aggregation to these seeds (e.g. "
                             "--combo_ranking_seeds 4). Rows from other seeds are dropped before "
                             "aggregation. With a single seed, the STRICT/GENERAL/MIXED/LOSER labels "
                             "are assigned by gm_all thresholds instead of per-seed WIN/LOSE/flip.")
    parser.add_argument("--transfer_opt_time_only", action="store_true",
                        help="When comparing times, use only optimization_time for transfer "
                             "(no hyper_attack_time) while standard still uses total_time.")
    parser.add_argument("--skip_hyper_transfer_attack", action="store_true",
                        help="Disable hyper attack (PGD warm-start) in transfer runs.")
    parser.add_argument("--standard_warmstart", action="store_true",
                        help="In transfer mode: first solve standard MIP for N1 per (c_tag,c_target) "
                             "to get delta_1 and binary values, then use those binaries as warm-start "
                             "hints for the transfer MIP. Transfer jobs become self-contained "
                             "(no dependency on standard phase). Implies --skip_standard.")
    parser.add_argument("--standard_warmstart_n1_only", action="store_true",
                        help="Restrict --standard_warmstart so only N1(x) (n1_org) binaries are "
                             "hinted in the transfer MIP — skip n1_pert, n2_org, and n2_pert. "
                             "Only meaningful with the 'full' encoding mode (N1 encoded).")
    # ── Advanced-standard mode ───────────────────────────────────────────
    parser.add_argument("--advanced_standard", action="store_true",
                        help="Run advanced_standard mode: solve standard on N1, then accelerated "
                             "standard on N2 using N1's solver info. Replaces standard+transfer. "
                             "Sweeps over technique flag combinations (excluding all-false).")
    parser.add_argument("--sweep_adv_std_mip_start", nargs="*", type=str, default=None,
                        help="Values for adv_std_mip_start (e.g. 'true false'). Default: ['true'].")
    parser.add_argument("--sweep_adv_std_branch_priorities", nargs="*", type=str, default=None,
                        help="Values for adv_std_branch_priorities: off | bounds | pseudocost "
                             "(legacy true/false accepted). Default: ['bounds'].")
    parser.add_argument("--sweep_adv_std_lp_basis", nargs="*", type=str, default=None,
                        help="Values for adv_std_lp_basis (e.g. 'true false'). Default: ['true'].")
    parser.add_argument("--sweep_adv_std_bound_tightening", nargs="*", type=str, default=None,
                        help="Values for adv_std_bound_tightening (e.g. 'true false'). Default: ['true'].")
    parser.add_argument("--sweep_adv_std_zono_bounds", nargs="*", type=str, default=None,
                        help="Values for adv_std_zono_bounds (e.g. 'true false'). Default: ['false']. "
                             "When true, Technique 4's bound pre-compute uses zonotope propagation "
                             "(Source A) + a second N1-tightened absolute N2 zonotope pass (Source B), "
                             "intersected at each ReLU. Requires adv_std_bound_tightening=true; has no "
                             "effect when combined with bound_tightening=false.")
    parser.add_argument("--sweep_adv_std_n1_probe", nargs="*", type=str, default=None,
                        help="Values for adv_std_n1_probe: off | lp. Default: ['off']. "
                             "When 'lp', runs a post-Phase-1 joint N1+N2 LP probing pass to derive "
                             "tighter per-neuron N2 bounds, eliminating more N2 binaries via "
                             "stable-flip. Requires adv_std_bound_tightening=true; combos with "
                             "bound_tightening=false are auto-pruned.")
    parser.add_argument("--sweep_adv_std_n2_relax_threshold", nargs="*", type=float, default=None,
                        help="Values for adv_std_n2_relax_threshold (floats, e.g. '-1 0.1 0.5 1.0'). "
                             "Default: [-1.0] (disabled). When >= 0, replaces N2/N2p ReLU binaries "
                             "with a triangle LP relaxation (no binary) whenever the triangle-gap-area "
                             "of N1's interval at the same neuron is <= the threshold. Sound over-"
                             "approximation: delta_relaxed >= delta_exact. Requires "
                             "adv_std_bound_tightening=true; combos with bound_tightening=false are "
                             "auto-pruned.")
    parser.add_argument("--sweep_adv_std_var_hint", nargs="*", type=str, default=None,
                        help="Values for adv_std_var_hint (e.g. 'true false'). Default: ['false']. "
                             "Variable hints (VarHintVal/VarHintPri) are orthogonal to branch priorities.")
    parser.add_argument("--sweep_adv_std_var_hint_fix", nargs="*", type=str, default=None,
                        help="Values for adv_std_var_hint_fix (e.g. 'true false'). Default: ['false']. "
                             "When true, filter n1_pseudocosts to N2-surviving binaries before "
                             "rank_to_priority. Combos with vhf=true and vh=false are auto-pruned.")
    parser.add_argument("--sweep_gurobi_seed", nargs="*", type=int, default=None,
                        help="Gurobi seeds to sweep (e.g. '0 1 2 3 4') for variance measurement. Default: [0].")
    parser.add_argument("--refresh_ranking_csv", action="store_true",
                        help="Before applying --advstd_safe_combos_only, regenerate the combo-ranking "
                             "CSV from the latest per-cell results (equivalent to running "
                             "--find_advstd_faster_than_standard first with the same --arch_models, "
                             "--compare_to_with_perturbed, and --combo_ranking_seeds / --sweep_gurobi_seed). "
                             "The regenerated CSV path then replaces the one passed to "
                             "--advstd_safe_combos_only.")
    parser.add_argument("--advstd_safe_combos_only", type=str, default=None, metavar="CSV_PATH",
                        help="Path to advstd_combo_ranking CSV. When given, filters out combos "
                             "that are present in the CSV with a non-safe perf_tier. Safe perf_tiers "
                             "are {dominant, avg-win} by default; unsafe are {avg-win-risky, neutral, "
                             "loser, unknown}. Combos not in the CSV are allowed through (untested, "
                             "worth exploring). Use --advstd_safe_labels or --advstd_safe_perf_tiers "
                             "to override the default safe set.")
    parser.add_argument("--advstd_safe_labels", nargs="*", default=None, metavar="LABEL",
                        help="Override safe set by exact label (e.g. narrow-dominant narrow-avg-win "
                             "broad-avg-win-risky). Only combos whose CSV 'label' column matches one "
                             "of these are treated as safe. Requires --advstd_safe_combos_only.")
    parser.add_argument("--advstd_safe_perf_tiers", nargs="*", default=None, metavar="TIER",
                        help="Override safe set by perf_tier only (e.g. dominant avg-win avg-win-risky). "
                             "Requires --advstd_safe_combos_only. Ignored if --advstd_safe_labels is given.")
    parser.add_argument("--ct", type=str, default=None,
                        help="Comma-separated Julia-indexed c_target values (1-indexed). "
                             "Default: 2,3,4,5,6,7,8,9,10. Use to restrict to specific scenarios.")
    args = parser.parse_args()

    total_cores = args.max_cores
    thresholds = args.thresholds if args.thresholds else THRESHOLDS
    opt_intervals = args.opt_intervals if args.opt_intervals else OPT_INTERVALS
    dataset = args.dataset

    if args.standard_only:
        args.skip_transfer = True
    if args.standard_warmstart:
        args.skip_standard = True  # standard is done inside each transfer Julia process

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
        # all_perts = [
        #     ("patch(1,14,14,3)",  "patch:1,14,14,3"),
        #     ("occ(14,14,9)",      "occ:14,14,9"),
        #     ("occ(1,1,5)",        "occ:1,1,5"),
        #     ("occ(5,5,5)",        "occ:5,5,5"),
        #     ("brightness(0.25)",  "brightness:0.25"),
        #     ("contrast(1.5)",     "contrast:1.5"),
        #     ("trans(1,1)",        "translation:1,1"),
        #     ("trans(1,3)",        "translation:1,3"),
        #     ("trans(3,1)",        "translation:3,1"),
        #     ("trans(3,3)",        "translation:3,3"),
        #     ("rotation(10)",      "rotation:10"),
        # ]
        all_perts = PERTURBATIONS
        # Write combined CSVs to the dataset-level directory (not per-arch)
        dblchk = args.double_check_standard
        suffix = "_double_check_standard" if dblchk else ""
        if args.compare_to_with_perturbed:
            suffix += "_vs_withPerturbed"
        if args.transfer_opt_time_only:
            suffix += "_transferOptOnly"
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
                compare_to_with_perturbed=args.compare_to_with_perturbed,
                transfer_opt_time_only=args.transfer_opt_time_only)
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
            "relax_count", "optimizing_intervals", "encoding_skip",
            "bound_n2_relu_using_zonotope", "bound_n2_non_relu_using_zonotope", "how_much_faster",
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

    # ── Analysis mode: find advanced-standard faster than standard ───
    if args.find_advstd_faster_than_standard:
        _generate_combo_ranking_csv(
            arch_runs, cwd, dataset,
            args.compare_to_with_perturbed, args.combo_ranking_seeds)
        return

    cores_per_job = CORES_PER_JOB
    max_slots = (total_cores - CORE_START) // cores_per_job

    # ── Advanced-standard mode (two-phase: N1 once, then N2 sweep) ──────
    if args.advanced_standard:
        try:
            Threads_num = 32
            cores_per_job = Threads_num
            max_slots = (total_cores - CORE_START) // cores_per_job

            # Resolve technique sweep values
            mip_start_vals = [v.lower() for v in args.sweep_adv_std_mip_start] if args.sweep_adv_std_mip_start else ["true"]
            # Branch priorities is now a 3-mode enum; normalize legacy true/false.
            def _norm_bp(v):
                v = v.lower()
                if v == "true":  return "bounds"
                if v == "false": return "off"
                return v
            branch_pri_vals = [_norm_bp(v) for v in args.sweep_adv_std_branch_priorities] if args.sweep_adv_std_branch_priorities else ["bounds"]
            for v in branch_pri_vals:
                if v not in ("off", "bounds", "pseudocost"):
                    print(f"ERROR: unknown --sweep_adv_std_branch_priorities value '{v}' "
                          "(expected off | bounds | pseudocost)")
                    sys.exit(1)
            lp_basis_vals = [v.lower() for v in args.sweep_adv_std_lp_basis] if args.sweep_adv_std_lp_basis else ["true"]
            bound_tight_vals = [v.lower() for v in args.sweep_adv_std_bound_tightening] if args.sweep_adv_std_bound_tightening else ["true"]
            zono_bounds_vals = [v.lower() for v in args.sweep_adv_std_zono_bounds] if args.sweep_adv_std_zono_bounds else ["false"]
            n1_probe_vals = [v.lower() for v in args.sweep_adv_std_n1_probe] if args.sweep_adv_std_n1_probe else ["off"]
            for v in n1_probe_vals:
                if v not in ("off", "lp"):
                    print(f"ERROR: unknown --sweep_adv_std_n1_probe value '{v}' (expected off | lp)")
                    sys.exit(1)
            relax_t_vals = args.sweep_adv_std_n2_relax_threshold if args.sweep_adv_std_n2_relax_threshold else [-1.0]
            var_hint_vals = [v.lower() for v in args.sweep_adv_std_var_hint] if args.sweep_adv_std_var_hint else ["false"]
            var_hint_fix_vals = [v.lower() for v in args.sweep_adv_std_var_hint_fix] if args.sweep_adv_std_var_hint_fix else ["false"]
            seed_vals = args.sweep_gurobi_seed if args.sweep_gurobi_seed else [0]

            # Generate all combinations, excluding the all-off case and any
            # combo where a flag that requires bound_tightening is on while
            # bound_tightening=false (gated on Technique 4's pre-compute block).
            technique_combos = [
                (ms, bp, lb, bt, zb, np_, rt, vh, vhf)
                for ms, bp, lb, bt, zb, np_, rt, vh, vhf in itertools.product(
                    mip_start_vals, branch_pri_vals, lp_basis_vals, bound_tight_vals,
                    zono_bounds_vals, n1_probe_vals, relax_t_vals, var_hint_vals, var_hint_fix_vals)
                if not (ms == "false" and bp == "off" and lb == "false" and bt == "false"
                        and zb == "false" and np_ == "off" and rt < 0.0 and vh == "false")
                and not (zb == "true" and bt == "false")
                and not (np_ != "off" and bt == "false")
                and not (rt >= 0.0 and bt == "false")
                and not (vhf == "true" and vh == "false")
            ]

            # ── Optional: regenerate the ranking CSV before filtering ──
            if args.refresh_ranking_csv and args.advstd_safe_combos_only:
                regen_seeds = args.combo_ranking_seeds
                if regen_seeds is None and args.sweep_gurobi_seed:
                    regen_seeds = [str(s) for s in args.sweep_gurobi_seed]
                print(f"\n--refresh_ranking_csv: regenerating ranking CSV "
                      f"(compare_to_with_perturbed={args.compare_to_with_perturbed}, "
                      f"seeds={regen_seeds})")
                regen_path = _generate_combo_ranking_csv(
                    arch_runs, cwd, dataset,
                    args.compare_to_with_perturbed, regen_seeds)
                if os.path.realpath(regen_path) != os.path.realpath(args.advstd_safe_combos_only):
                    print(f"  (was: {args.advstd_safe_combos_only})")
                    print(f"  (now: {regen_path})")
                args.advstd_safe_combos_only = regen_path

            # ── Optional: filter to "safe" combos from ranking CSV ──
            # Safe set is one of (checked in order):
            #   1. --advstd_safe_labels  → exact label match
            #   2. --advstd_safe_perf_tiers → perf_tier match
            #   3. default: perf_tier in {dominant, avg-win}
            # Within the safe set, combos are ordered by the priority list
            # above: earlier entries run first, untested combos run last,
            # original flag-product order breaks ties.
            if args.advstd_safe_combos_only:
                import csv as _csv_filter
                if args.advstd_safe_labels:
                    _priority_list = [s.strip().lower() for s in args.advstd_safe_labels]
                    _match_column = "label"
                    _mode = f"labels={_priority_list}"
                elif args.advstd_safe_perf_tiers:
                    _priority_list = [s.strip().lower() for s in args.advstd_safe_perf_tiers]
                    _match_column = "perf_tier"
                    _mode = f"perf_tiers={_priority_list}"
                else:
                    _priority_list = ["dominant", "avg-win"]
                    _match_column = "perf_tier"
                    _mode = f"perf_tiers={_priority_list} (default)"
                _priority_rank = {v: i for i, v in enumerate(_priority_list)}
                _UNTESTED_RANK = len(_priority_list) + 1
                safe_keys = set()
                safe_key_rank = {}
                unsafe_keys = set()
                with open(args.advstd_safe_combos_only) as _f:
                    reader = _csv_filter.DictReader(_f, skipinitialspace=True)
                    for _row in reader:
                        _row = {k: (v.strip() if isinstance(v, str) else v) for k, v in _row.items()}
                        # CSV uses yes/no; sweep uses true/false.
                        _yn = {"yes": "true", "no": "false"}
                        _ms = _yn.get(_row["mip_start"], _row["mip_start"])
                        _bp = _row["branch_priorities"]  # off / bounds — same in both
                        _lb = _yn.get(_row["lp_basis"], _row["lp_basis"])
                        _bt = _yn.get(_row["bound_tightening"], _row["bound_tightening"])
                        _vh = _yn.get(_row["var_hint"], _row["var_hint"])
                        _zb = _yn.get(_row["zono_bounds"], _row["zono_bounds"])
                        _np = _row["n1_probe"]  # off / lp — same in both
                        _rt_str = _row["relax_threshold"]
                        _rt = float(_rt_str) if _rt_str not in ("off",) else -1.0
                        # var_hint_fix is a post-hoc column; CSVs written before
                        # the flag existed don't have it. Default to "false" so
                        # legacy safe-combos CSVs match vhf=false combos.
                        _vhf_raw = _row.get("var_hint_fix", "no")
                        _vhf = _yn.get(_vhf_raw, _vhf_raw)
                        _key = (_ms, _bp, _lb, _bt, _zb, _np, _rt, _vh, _vhf)
                        _match_value = _row.get(_match_column, "").lower()
                        if _match_value in _priority_rank:
                            safe_keys.add(_key)
                            _new_rank = _priority_rank[_match_value]
                            _existing = safe_key_rank.get(_key)
                            if _existing is None or _new_rank < _existing:
                                safe_key_rank[_key] = _new_rank
                        else:
                            unsafe_keys.add(_key)
                print(f"  safe-set mode: {_mode}")
                pre_filter = len(technique_combos)
                blocked = [c for c in technique_combos if c in unsafe_keys]
                technique_combos = [c for c in technique_combos if c not in unsafe_keys]
                n_safe = sum(1 for c in technique_combos if c in safe_keys)
                n_untested = sum(1 for c in technique_combos if c not in safe_keys)
                # Preserve flag-product order as the tiebreaker inside each rank.
                _orig_pos = {c: i for i, c in enumerate(technique_combos)}
                technique_combos.sort(
                    key=lambda c: (safe_key_rank.get(c, _UNTESTED_RANK), _orig_pos[c])
                )
                print(f"\n--advstd_safe_combos_only: filtered {pre_filter} -> {len(technique_combos)} combos "
                      f"({n_safe} safe, {n_untested} untested, {len(blocked)} blocked) "
                      f"from {args.advstd_safe_combos_only}")

            print(f"\nAdvanced-standard: {len(technique_combos)} technique combinations × {len(seed_vals)} seed(s) (all-off + zono/probe/relax-without-boundTight excluded):")
            for ms, bp, lb, bt, zb, np_, rt, vh, vhf in technique_combos:
                print(f"  mipStart={ms}  branchPri={bp}  lpBasis={lb}  boundTight={bt}  zonoBounds={zb}  n1Probe={np_}  relaxT={rt}  varHint={vh}  varHintFix={vhf}")
            print(f"  seeds: {seed_vals}")

            sys.path.insert(0, os.path.join(cwd, 'utils'))
            from run_experiment import ARCH_REGISTRY, DATASET_CONFIG

            # ── Phase 1: Solve N1 once per (arch, perturbation) ──────────
            n1_jobs = []   # (label, cmd)
            # Track state dirs so N2 jobs can reference them
            # Key: (arch, pert_spec) → (n1_state_dir, n1_model_p, n2_model_p, n1_tag, n2_tag)
            n1_info = {}

            # Does any combo in this sweep need pseudo-costs? If so, the
            # legacy three-file N1 state is not enough and we must either
            # re-solve N1 or wait for another process that's re-solving.
            need_pseudocosts = ("pseudocost" in branch_pri_vals) or ("true" in var_hint_vals)
            if need_pseudocosts:
                print("This sweep requires N1 pseudo-costs (bp=pseudocost or var_hint=true).")
            # Does any combo in this sweep need the N1 probe? If so, the
            # state dir must also contain n1_preact_bounds.bin (written by
            # save_n1_diff_bounds when it has n1_preact_up_bounds populated
            # during the Phase-1 save). Legacy state dirs without this
            # file will trigger an N1 re-solve.
            need_n1_preact = any(v != "off" for v in n1_probe_vals)
            if need_n1_preact:
                print("This sweep requires n1_preact_bounds.bin (adv_std_n1_probe != off).")

            # Stale lock heuristic: 2× the Gurobi time limit, i.e. generous
            # enough that a legitimately-long N1 solve is never considered
            # stale, but short enough that a crashed process clears within
            # a reasonable window.
            stale_lock_sec = max(2 * args.timeout, 600)
            wait_timeout_sec = stale_lock_sec

            # Track locks this process acquired so we can release them after
            # Phase 1 finishes (successfully or otherwise).
            acquired_n1_locks = []

            for arch, model_path in arch_runs:
                if model_path is None:
                    print(f"ERROR: --advanced_standard requires --model_path (or --arch_models)")
                    sys.exit(1)

                print(f"\n{'=' * 60}")
                print(f"Phase 0: Training N2 = N1 + {args.sgd_epochs} SGD epoch(s) [{arch}]")
                print(f"{'=' * 60}\n")
                n1_dir, n2_dir = train_extra_epochs(
                    model_path, arch, dataset,
                    sgd_epochs=args.sgd_epochs, lr=args.lr)

                _, model_name = ARCH_REGISTRY[arch]
                _, _, _, _, julia_dataset = DATASET_CONFIG[dataset]
                n1_tag = os.path.basename(os.path.normpath(n1_dir))
                n2_tag = os.path.basename(os.path.normpath(n2_dir))
                n1_model_p = os.path.join(n1_dir, "model.p")
                n2_model_p = os.path.join(n2_dir, "model.p")

                for pert_name, pert_spec in perts:
                    pert_type, eps_str = pert_spec.split(":", 1)
                    arch_prefix = f"[{arch}] "

                    # N1 state directory: one per (arch, perturbation)
                    n1_state_dir = os.path.join(
                        cwd, "paper_experiments", dataset, f"{arch}_exp",
                        pert_type, f"eps_{eps_str}",
                        f"n1_state_{arch}_{n1_tag}")

                    n1_info[(arch, pert_spec)] = (n1_state_dir, n1_model_p, n2_model_p, n1_tag, n2_tag, model_name, julia_dataset)

                    # ── Smart skip + cross-process lock coordination ─────
                    # 1. State already complete → skip
                    # 2. State missing or incomplete (e.g. no pseudocost
                    #    file when we need it) → try to acquire the lock.
                    #    If we get it, queue the N1 job and release the
                    #    lock after the solve finishes.
                    #    If another process holds it, wait for them to
                    #    finish and verify the state.
                    if _n1_state_complete(n1_state_dir, need_pseudocosts, need_n1_preact=need_n1_preact):
                        print(f"  {arch_prefix}{pert_name} N1 state already complete at {n1_state_dir} — skipping N1 solve")
                        continue

                    got_lock, lock_path = _acquire_n1_solve_lock(n1_state_dir, stale_lock_sec)
                    if not got_lock:
                        print(f"  {arch_prefix}{pert_name} another process is solving N1 at {n1_state_dir} — waiting (up to {wait_timeout_sec:.0f}s)")
                        if _wait_for_n1_state(n1_state_dir, need_pseudocosts, wait_timeout_sec, need_n1_preact=need_n1_preact):
                            print(f"  {arch_prefix}{pert_name} N1 state now ready — skipping our own N1 solve")
                            continue
                        # Either a timeout or the other process crashed
                        # without leaving a usable state. Try once to
                        # acquire the (now-released) lock and solve it
                        # ourselves, rather than erroring out.
                        print(f"  {arch_prefix}{pert_name} WARNING: timed out or other process left incomplete state — attempting to solve N1 ourselves")
                        got_lock, lock_path = _acquire_n1_solve_lock(n1_state_dir, stale_lock_sec)
                        if not got_lock:
                            print(f"  {arch_prefix}{pert_name} ERROR: still unable to acquire N1 solve lock at {lock_path}. Aborting.")
                            sys.exit(1)

                    acquired_n1_locks.append(lock_path)
                    n1_label = f"{arch_prefix}{pert_name} N1-solve"
                    n1_cmd = [
                        "julia", "run.jl",
                        "--mode", "advanced_standard_n1",
                        "--dataset", julia_dataset,
                        "--model_name", model_name,
                        "--model_path", n1_model_p,
                        "--model_path2", n2_model_p,
                        "--perturbation", pert_type,
                        "--perturbation_size", eps_str,
                        "--ctag", "1",
                        "--ct", args.ct if args.ct else "2,3,4,5,6,7,8,9,10",
                        "--timout", str(args.timeout),
                        "--output_dir", n1_state_dir + "/",
                        "--n1_state_dir", n1_state_dir,
                        "--use_hyper_attack", "true",
                        "--activate_vaghgar_deps", "true",
                        "--use_perturbed_intervals", "true",
                        "--Threads_num", str(Threads_num),
                    ]
                    n1_jobs.append((n1_label, n1_cmd))

            # Run all N1 jobs (one per perturbation, in parallel).
            # Wrap in try/finally so locks this process acquired are always
            # released — even on KeyboardInterrupt or solver crash — so
            # other parallel sweep processes aren't blocked indefinitely.
            if n1_jobs:
                print(f"\n── Phase 1: {len(n1_jobs)} N1 solve jobs ──")
                try:
                    run_pool(n1_jobs, max_slots, cwd, cores_per_job, "Phase 1 (N1 solves)")
                finally:
                    for lock_path in acquired_n1_locks:
                        _release_n1_solve_lock(lock_path)
                    if acquired_n1_locks:
                        print(f"Phase 1: released {len(acquired_n1_locks)} N1 solve lock(s)")
            else:
                # No jobs — either everything was already complete or another
                # process is handling it. Still release any locks we somehow
                # acquired (shouldn't happen, but belt + suspenders).
                for lock_path in acquired_n1_locks:
                    _release_n1_solve_lock(lock_path)

            # ── Phase 1.5: Run standard N2 (vagharWithPerturbed) if missing ──
            # The ranking comparison needs standard results as baseline.
            # For new perturbation types/sizes these may not exist yet.
            # Runs with: use_hyper_attack=true, activate_vaghgar_deps=true,
            #            use_perturbed_intervals=true, use_relaxations=false.
            std_n2_jobs = []
            for (arch, pert_spec), (n1_state_dir, n1_model_p, n2_model_p, n1_tag, n2_tag, model_name, julia_dataset) in n1_info.items():
                pert_type, eps_str = pert_spec.split(":", 1)
                pert_name = next(pn for pn, ps in perts if ps == pert_spec)
                arch_prefix = f"[{arch}] "

                if standard_with_perturbed_results_exist(pert_spec, cwd, arch, dataset, n2_tag=n2_tag):
                    print(f"  {arch_prefix}{pert_name} standard N2 (vagharWithPerturbed {n2_tag}) already exists — skipping")
                    continue

                std_output_dir = os.path.join(
                    "paper_experiments", dataset, f"{arch}_exp",
                    pert_type, f"eps_{eps_str}",
                    f"vagharWithPerturbed_{arch}_{n2_tag}")

                std_label = f"{arch_prefix}{pert_name} standard-N2 (WithPerturbed)"
                std_cmd = [
                    "julia", "run.jl",
                    "--mode", "standard",
                    "--dataset", julia_dataset,
                    "--model_name", model_name,
                    "--model_path", n2_model_p,
                    "--perturbation", pert_type,
                    "--perturbation_size", eps_str,
                    "--ctag", "1",
                    "--ct", args.ct if args.ct else "2,3,4,5,6,7,8,9,10",
                    "--timout", str(args.timeout),
                    "--output_dir", std_output_dir + "/",
                    "--c_tag_mode", "false",
                    "--use_hyper_attack", "true",
                    "--activate_vaghgar_deps", "true",
                    "--use_perturbed_intervals", "true",
                    "--use_relaxations", "false",
                    "--Threads_num", str(Threads_num),
                ]
                std_n2_jobs.append((std_label, std_cmd))

            if std_n2_jobs:
                print(f"\n── Phase 1.5: {len(std_n2_jobs)} standard N2 jobs (vagharWithPerturbed, missing baselines) ──")
                run_pool(std_n2_jobs, max_slots, cwd, cores_per_job, "Phase 1.5 (standard N2)")
            else:
                print("\n── Phase 1.5: all standard N2 baselines already exist — skipping ──")

            # ── Phase 2: N2 sweep (all technique combos, in parallel) ────
            n2_jobs = []
            n2_skipped = 0
            for (arch, pert_spec), (n1_state_dir, n1_model_p, n2_model_p, n1_tag, n2_tag, model_name, julia_dataset) in n1_info.items():
                pert_type, eps_str = pert_spec.split(":", 1)
                pert_name = next(pn for pn, ps in perts if ps == pert_spec)
                arch_prefix = f"[{arch}] "

                for ms, bp, lb, bt, zb, np_, rt, vh, vhf in technique_combos:
                    tech_tag = ""
                    if ms == "true":          tech_tag += "ms"
                    if bp == "bounds":        tech_tag += "bp"
                    elif bp == "pseudocost":  tech_tag += "bpPsd"
                    if lb == "true":          tech_tag += "lb"
                    if bt == "true":          tech_tag += "bt"
                    if zb == "true":          tech_tag += "zb"
                    if np_ == "lp":           tech_tag += "npLP"
                    if rt >= 0.0:             tech_tag += f"rt{rt}"
                    if vh == "true":          tech_tag += "vh"
                    if vh == "true" and vhf == "true": tech_tag += "fix"

                    adv_output_dir = os.path.join(
                        "paper_experiments", dataset, f"{arch}_exp",
                        pert_type, f"eps_{eps_str}",
                        f"advStd_{arch}_N1_{n1_tag}")

                    base_name_to_save = f"{n2_tag}_N2_advStd"
                    if ms == "true":          base_name_to_save += "_mipStart"
                    if bp == "bounds":        base_name_to_save += "_branchPri"
                    elif bp == "pseudocost":  base_name_to_save += "_branchPriPsd"
                    if lb == "true":          base_name_to_save += "_lpBasis"
                    if bt == "true":          base_name_to_save += "_boundTight"
                    if zb == "true":          base_name_to_save += "_zonoBounds"
                    if np_ == "lp":           base_name_to_save += "_n1ProbeLP"
                    if rt >= 0.0:             base_name_to_save += f"_relaxT{rt}"
                    if vh == "true":          base_name_to_save += "_varHint"
                    if vh == "true" and vhf == "true": base_name_to_save += "_varHintFix"

                    for seed in seed_vals:
                        if _advstd_result_exists(
                                cwd, dataset, arch, pert_type, eps_str,
                                n1_tag, base_name_to_save, seed):
                            n2_skipped += 1
                            continue
                        seed_suffix = f" seed{seed}" if seed != 0 else ""
                        label = f"{arch_prefix}{pert_name} N2({tech_tag}){seed_suffix}"

                        cmd = [
                            "julia", "run.jl",
                            "--mode", "advanced_standard_n2",
                            "--dataset", julia_dataset,
                            "--model_name", model_name,
                            "--model_path", n1_model_p,
                            "--model_path2", n2_model_p,
                            "--perturbation", pert_type,
                            "--perturbation_size", eps_str,
                            "--ctag", "1",
                            "--ct", args.ct if args.ct else "2,3,4,5,6,7,8,9,10",
                            "--timout", str(args.timeout),
                            "--output_dir", adv_output_dir + "/",
                            "--name_to_save", base_name_to_save,
                            "--n1_state_dir", n1_state_dir,
                            "--use_hyper_attack", "true",
                            "--activate_vaghgar_deps", "true",
                            "--use_perturbed_intervals", "true",
                            "--Threads_num", str(Threads_num),
                            "--adv_std_mip_start", ms,
                            "--adv_std_branch_priorities", bp,
                            "--adv_std_lp_basis", lb,
                            "--adv_std_bound_tightening", bt,
                            "--adv_std_zono_bounds", zb,
                            "--adv_std_n1_probe", np_,
                            "--adv_std_n2_relax_threshold", str(rt),
                            "--adv_std_var_hint", vh,
                            "--adv_std_var_hint_fix", vhf,
                            "--gurobi_seed", str(seed),
                        ]
                        n2_jobs.append((label, cmd))

            skip_note = f" (skipped {n2_skipped} already-completed)" if n2_skipped else ""
            print(f"\n── Phase 2: {len(n2_jobs)} N2 jobs ({len(technique_combos)} combos × {len(n1_info)} perturbations){skip_note} ──")
            if n2_jobs:
                run_pool(n2_jobs, max_slots, cwd, cores_per_job, "Phase 2 (N2 sweep)")
            else:
                print("  nothing to run — every requested combo/seed already has a result file")

        except KeyboardInterrupt:
            print("\nCtrl+C received — terminating all running jobs...")
            sys.exit(1)
        return

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
                    if args.skip_vaghar_no_perturbed:
                        std_cmd.append("--skip_vaghar_no_perturbed")
                    if args.standard_relaxation_thresholds is not None:
                        std_cmd += ["--standard_relaxation_thresholds", args.standard_relaxation_thresholds]
                    standard_jobs.append((job_key, std_label, std_cmd))

                # Cross-product values for sweep flags.
                # Default: run with the flag off. --sweep_* enables running multiple values.
                bound_relu_values = [True] if args.sweep_bound_n2_relu_using_zonotope else [False]
                bound_non_relu_values = [False, True] if args.sweep_bound_by_zonotope_n2_hidden_neurons_which_are_not_relu else [False]
                bound_n2xp_out_values = [False, True] if args.sweep_bound_n2_xp_output_using_composed else [False]
                bound_n2xp_comp_values = [False, True] if args.sweep_bound_n2_xp_using_composed else [False]
                link_n2xp_values = [False, True] if args.sweep_constrain_n2_xp_via_n1_zonotope else [False]
                branch_pri_values = [False, True] if args.sweep_branch_priority_n2x_first else [False]
                adapt_prune_values = args.sweep_n1_adaptive_prune_budget if args.sweep_n1_adaptive_prune_budget else [0.0]
                zono_order_values = args.sweep_zonotope_max_order if args.sweep_zonotope_max_order else [0]
                n1_stab_values = args.sweep_n1_stability_relax_threshold if args.sweep_n1_stability_relax_threshold else [-1.0]

                # ── Build encoding mode list ──────────────────────────────
                # When multiple encoding flags are passed, generate separate
                # job groups so they run in parallel (not combined into one cmd).
                encoding_modes = []
                if args.no_n2_xp_encoding:
                    encoding_modes.append("no_n2_xp")
                if args.no_n1_binaries_and_relaxtions_only_on_n2 and not args.no_n1_encoding_at_all:
                    encoding_modes.append("n1_lp_relax")
                if args.no_n1_encoding_at_all:
                    encoding_modes.append("no_n1_enc")
                if not encoding_modes:
                    encoding_modes.append("full")
                # --standard_warmstart: always include "full" so both
                # with-N1 and without-N1 configs run in parallel
                if args.standard_warmstart and "full" not in encoding_modes:
                    encoding_modes.insert(0, "full")

                t_jobs = []
                for enc_mode in encoding_modes:
                  for oi, t, b_relu, b_non_relu, b_n2xp_out, b_n2xp, lnk, bpri, ap_budget, zo, sr in itertools.product(
                        opt_intervals, thresholds, bound_relu_values, bound_non_relu_values,
                        bound_n2xp_out_values, bound_n2xp_comp_values, link_n2xp_values, branch_pri_values,
                        adapt_prune_values, zono_order_values, n1_stab_values):
                                rga_tag = "true" if args.relaxation_gap_area.lower() == "true" else "false"
                                br_tag = "1" if b_relu else "0"
                                bnr_tag = "1" if b_non_relu else "0"
                                ap_tag = f"ap{ap_budget}" if ap_budget > 0 else ""
                                zo_tag = f"zo{zo}" if zo > 0 else ""
                                sr_tag = f"sr{sr}" if sr >= 0 else ""
                                xpout_tag = "bN2xpOut" if b_n2xp_out else ""
                                xp_tag = "bN2xp" if b_n2xp else ""
                                lnk_tag = "n1zono" if lnk else ""
                                bpri_tag = "bpri" if bpri else ""
                                enc_tag = f" enc={enc_mode}" if enc_mode != "full" else ""
                                extra = "".join(f" {x}" for x in [ap_tag, zo_tag, sr_tag, xpout_tag, xp_tag, lnk_tag, bpri_tag] if x)
                                t_label = f"{arch_prefix}{pert_name} T={t} oi={oi} rga={rga_tag} bRelu={br_tag} bNonRelu={bnr_tag}{enc_tag}{extra}"
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
                                # ── Encoding mode flags (mutually exclusive per job) ──
                                if enc_mode == "no_n2_xp":
                                    t_cmd.append("--no_n2_xp_encoding")
                                elif enc_mode == "n1_lp_relax":
                                    t_cmd.append("--no_n1_binaries_and_relaxtions_only_on_n2")
                                elif enc_mode == "no_n1_enc":
                                    t_cmd.append("--no_n1_encoding_at_all")
                                    if args.encode_n1_last_layer:
                                        t_cmd.append("--encode_n1_last_layer")
                                    if args.n1_last_layer_prune_tol > 0:
                                        t_cmd += ["--n1_last_layer_prune_tol", str(args.n1_last_layer_prune_tol)]
                                    if args.constrain_n1_xp:
                                        t_cmd.append("--constrain_n1_xp")
                                # ── Flags applicable to all modes ──
                                if args.cap_delta_diff:
                                    t_cmd.append("--cap_delta_diff")
                                if args.use_zonotope:
                                    t_cmd.append("--use_zonotope")
                                if b_n2xp_out and enc_mode != "no_n2_xp":
                                    t_cmd.append("--bound_n2_xp_output_using_composed")
                                if b_n2xp and enc_mode != "no_n2_xp":
                                    t_cmd.append("--bound_n2_xp_using_composed")
                                if lnk and enc_mode != "no_n2_xp":
                                    t_cmd.append("--constrain_n2_xp_via_n1_zonotope")
                                if bpri:
                                    t_cmd.append("--branch_priority_n2x_first")
                                if b_relu:
                                    t_cmd.append("--bound_n2_relu_using_zonotope")
                                if b_non_relu:
                                    t_cmd.append("--bound_by_zonotope_n2_hidden_neurons_which_are_not_relu")
                                if ap_budget > 0:
                                    t_cmd += ["--n1_adaptive_prune_budget", str(ap_budget)]
                                if zo > 0:
                                    t_cmd += ["--zonotope_max_order", str(zo)]
                                if sr >= 0:
                                    t_cmd += ["--n1_stability_relax_threshold", str(sr)]
                                if args.skip_hyper_transfer_attack:
                                    t_cmd.append("--skip_hyper_transfer_attack")
                                if args.standard_warmstart:
                                    t_cmd.append("--standard_warmstart")
                                if args.standard_warmstart_n1_only:
                                    t_cmd.append("--standard_warmstart_n1_only")
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
