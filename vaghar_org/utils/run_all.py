#!/usr/bin/env python3
"""
Run all experiments sequentially.

Stop cleanly: Ctrl+C (kills current subprocess and stops).
Run in background: nohup python3 run_all.py > run_all.log 2>&1 &
Stop from background: kill <pid>  (reads PID from run_all.pid)
"""
import os
import signal
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(SCRIPT_DIR, 'run_all.pid')

PERTURBATIONS = "linf:0.005,0.01,0.05;brightness:0.25,0.1;contrast:1.5,2"
THRESHOLDS = "1.2"
LR = "0.005"
DATASET = "mnist"

EXPERIMENTS = [
    {"arch": "3x10", "extra": ["--skip_training", "--skip_standard"]},
    {"arch": "3x50", "extra": []},
    {"arch": "3x100", "extra": []},
    {"arch": "5x50", "extra": []},
]

current_proc = None


def cleanup(signum=None, frame=None):
    """Kill the running subprocess and exit."""
    if current_proc and current_proc.poll() is None:
        print(f"\n  Stopping subprocess (pid={current_proc.pid})...")
        os.killpg(os.getpgid(current_proc.pid), signal.SIGTERM)
        current_proc.wait()
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    print("Stopped.")
    sys.exit(1)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def run_experiment(arch, extra_args=None):
    global current_proc
    cmd = [
        sys.executable, "run_experiment.py",
        "--dataset", DATASET,
        "--arch", arch,
        "--lr", LR,
        "--perturbations", PERTURBATIONS,
        "--relaxation_thresholds", THRESHOLDS,
        "--cpu",
    ] + (extra_args or [])

    print(f"\n>>> {arch} {'(' + ' '.join(extra_args) + ')' if extra_args else ''}")
    print(f"  cmd: {' '.join(cmd[:8])}...")

    current_proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR, preexec_fn=os.setsid)
    current_proc.wait()
    rc = current_proc.returncode
    current_proc = None

    if rc != 0:
        print(f"  WARNING: {arch} exited with code {rc}")
    return rc


def main():
    # Write PID file so you can `kill $(cat run_all.pid)` from another terminal
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    print("=" * 50)
    print(f"  Starting full experiment pipeline")
    print(f"  PID: {os.getpid()} (saved to {PID_FILE})")
    print(f"  {datetime.now()}")
    print("=" * 50)

    for exp in EXPERIMENTS:
        run_experiment(exp["arch"], exp.get("extra"))

    print("\n" + "=" * 50)
    print(f"  All experiments complete")
    print(f"  {datetime.now()}")
    print("=" * 50)

    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


if __name__ == "__main__":
    main()
