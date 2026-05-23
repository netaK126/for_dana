"""Strip lines from result .txt files that hit Gurobi's 1800 s time cap.

Lines that ran out at the 1800 s cap are unfinished and should be re-run
by run_relaxation_sweep.py with a longer timeout. Removing the line from
the result file makes the sweep's "have I already done this cell?" check
fail and it re-issues the run. Lines that hit a 3600 s cap (advstd) are
NOT removed: re-running them with the same cap would just hit the same
cap again.

Detection (independent of solve_status, which the old positional CSV
format does not carry):

  A line is a 1800-cap timeout iff
      1800 <= time < 2400
      AND
      |upper_bound - lower_bound| / max(|upper_bound|, eps)  >  MIPGap

  Both conditions are required because:
    - time >= 1800 isolates 1800-cap hits and excludes 3600-cap ones.
    - the gap test confirms Gurobi did not close to MIPGap. A run that
      legitimately closed at MIPGap and happened to take ~1800 s would
      have a tight bound and is left alone.

  MIPGap = 0.01 from utils/mip.jl:36/616.

  EPS = 1e-9 in the denominator handles upper_bound ~= 0 without
  zero-division. A small numerical slack (0.001) is added on top of the
  MIPGap threshold to absorb float noise.

  New (key=value) format reads lower_bound, upper_bound,
  optimization_time. Old positional CSV reads:
      source,target,incumbent_obj,best_bound,solve_time
  where incumbent_obj is the lower (primal) bound and best_bound is the
  dual upper bound.

Usage:
  python3 strip_1800_timeouts.py --dry-run     # report counts only
  python3 strip_1800_timeouts.py               # rewrite files in place
"""
import argparse
import glob
import math
import os
import sys


LOW = 1800.0           # 1800-cap timeouts hit at or just above 1800 s
HIGH = 2400.0          # below this excludes 3600-cap timeouts (~3611 s)
MIPGAP = 0.01          # utils/mip.jl set_optimizer_attribute(m, "MIPGap", 0.01)
GAP_SLACK = 0.001      # absorb float noise around the threshold
EPS = 1e-9             # avoid /0 when upper_bound is near zero


def _is_open(lower, upper):
    """True iff the bound gap exceeds MIPGap (i.e. the run did not close)."""
    if (lower is None or upper is None
            or math.isnan(lower) or math.isnan(upper)):
        return False
    denom = max(abs(upper), EPS)
    gap = abs(upper - lower) / denom
    return gap > (MIPGAP + GAP_SLACK)


def is_1800_timeout(line):
    line = line.strip()
    if not line:
        return False
    # new key=value format
    fields = {}
    for pair in line.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            fields[k] = v
    if fields:
        try:
            t = float(fields.get("optimization_time", "nan"))
            lo = float(fields.get("lower_bound", "nan"))
            hi = float(fields.get("upper_bound", "nan"))
        except ValueError:
            return False
        return LOW <= t < HIGH and _is_open(lo, hi)
    # old positional CSV: source,target,incumbent_obj,best_bound,solve_time
    parts = line.split(",")
    if len(parts) < 5:
        return False
    try:
        lo = float(parts[2])
        hi = float(parts[3])
        t = float(parts[4])
    except ValueError:
        return False
    return LOW <= t < HIGH and _is_open(lo, hi)


def process_file(path, dry_run):
    with open(path) as f:
        lines = f.readlines()
    keep = [ln for ln in lines if not is_1800_timeout(ln)]
    dropped = len(lines) - len(keep)
    if dropped and not dry_run:
        with open(path, "w") as f:
            f.writelines(keep)
    return dropped, len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="paper_experiments",
                    help="directory to scan recursively")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", default=10, type=int,
                    help="how many touched-file lines to print")
    args = ap.parse_args()

    paths = glob.glob(os.path.join(args.root, "**", "*.txt"), recursive=True)
    paths.sort()
    total_files = 0
    files_touched = 0
    total_dropped = 0
    total_lines = 0
    examples = []
    for p in paths:
        dropped, n_lines = process_file(p, args.dry_run)
        total_files += 1
        total_lines += n_lines
        if dropped:
            files_touched += 1
            total_dropped += dropped
            if len(examples) < args.show:
                examples.append((dropped, n_lines, p))

    for dropped, n_lines, p in examples:
        print(f"{'[would drop]' if args.dry_run else '[dropped]'} "
              f"{dropped}/{n_lines} lines :: {p}")
    if files_touched > len(examples):
        print(f"... and {files_touched - len(examples)} more files")

    print(f"\nScanned {total_files} files, {total_lines} total lines.")
    print(f"{'Would drop' if args.dry_run else 'Dropped'} {total_dropped} "
          f"lines across {files_touched} files.")


if __name__ == "__main__":
    sys.exit(main() or 0)
