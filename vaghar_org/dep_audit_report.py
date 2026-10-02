#!/usr/bin/env python3
"""Aggregate the dependency-sign audit CSVs and print the report.

One row per (dataset, architecture, model, perturbation, size). A row is RERUN when the
pre-fix code could have stamped a sign that the audit refutes, `check` when a sign was
neither proved nor refuted, and `clean` otherwise.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dep_audit_driver import discover  # noqa: E402

KEY = ("dataset", "arch", "model", "perturbation", "size")

# run.jl is given the Julia dataset name (fmnist / cifar10) while the results tree is laid
# out under the directory name, so the two must be reconciled to count a cell's files.
DIR_DATASET = {"fmnist": "fashion-mnist", "cifar10": "cifar"}


def dir_key(key):
    d, arch, model, pert, size = key
    return (DIR_DATASET.get(d, d), arch, model, pert, size.replace("-", ","))


def load(audit_dir):
    rows = []
    for fname in sorted(os.listdir(audit_dir)):
        if not fname.endswith(".csv"):
            continue
        with open(os.path.join(audit_dir, fname)) as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def summarise(rows):
    cells = defaultdict(lambda: defaultdict(int))
    witnesses = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in KEY)
        c = cells[key]
        c["reached"] += 1
        guarded = r["guard_ge"] == "true" or r["guard_le"] == "true"
        if not guarded:
            c["no_guard"] += 1
            continue
        c["probed"] += 1
        c["probe_sec"] += float(r["probe_sec"] or 0)
        fired_here = False
        for v, stamped, fired in ((r["verdict_ge"], r["old_stamp_ge"], r["fired_ge"]),
                                  (r["verdict_le"], r["old_stamp_le"], r["fired_le"])):
            if v in ("FLIPPED", "WRONG", "SAFE", "UNDECIDED"):
                c[v] += 1
            if stamped == "true":
                c["stamped"] += 1
                if fired == "true":
                    c["FIRED"] += 1
                    fired_here = True
                elif v == "UNDECIDED":
                    c["unknown"] += 1
        if fired_here:
            witnesses[key].append(r)
    return cells, witnesses


def verdict_of(c):
    """What to do with the cell. Only a sign the pre-fix code actually stamped AND that
    the audit refutes forces a re-run; a false sign it never stamped changes nothing."""
    if c["FIRED"]:
        return "RERUN"
    if c["unknown"]:
        return "check"
    return "clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_dir", default="dep_audit")
    ap.add_argument("--experiments_root", default="paper_experiments_with_zono_fix")
    ap.add_argument("--model_roots", nargs="*",
                    default=["paper_experiments", "paper_experiments_with_zono_fix",
                             "paper_experiments_rerun"])
    args = ap.parse_args()

    rows = load(args.audit_dir)
    cells, witnesses = summarise(rows)
    disc, unmapped = discover(args.experiments_root, args.model_roots)
    files_for = {k: len(v["files"]) for k, v in disc.items()}

    hdr = ["dataset", "arch", "model", "pert", "size", "reached", "probed",
           "stamped", "FIRED", "flip", "wrong", "undec", "safe", "verdict", "files"]
    table = []
    for key in sorted(cells):
        c = cells[key]
        table.append([key[0], key[1], key[2][:34], key[3], key[4],
                      c["reached"], c["probed"], c["stamped"], c["FIRED"],
                      c["FLIPPED"], c["WRONG"], c["UNDECIDED"], c["SAFE"],
                      verdict_of(c), files_for.get(dir_key(key), 0)])

    widths = [max(len(str(r[i])) for r in [hdr] + table) for i in range(len(hdr))]
    line = lambda r: "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r))
    print("\n" + "=" * 100)
    print("DEPENDENCY-SIGN AUDIT  —  " + args.experiments_root)
    print("=" * 100 + "\n")
    print(line(hdr))
    print("-" * sum(w + 2 for w in widths))
    for r in table:
        print(line(r))

    # rolled up per perturbation
    per_pert = defaultdict(lambda: defaultdict(int))
    for key, c in cells.items():
        p = per_pert[key[3]]
        for k in ("reached", "probed", "stamped", "FIRED", "FLIPPED", "WRONG",
                  "UNDECIDED", "SAFE"):
            p[k] += c[k]
        p["cells"] += 1
    print("\nPER PERTURBATION")
    ph = ["pert", "cells", "reached", "probed", "stamped", "FIRED", "flip", "wrong",
          "undec", "safe"]
    pt = [[p, per_pert[p]["cells"], per_pert[p]["reached"], per_pert[p]["probed"],
           per_pert[p]["stamped"], per_pert[p]["FIRED"], per_pert[p]["FLIPPED"],
           per_pert[p]["WRONG"], per_pert[p]["UNDECIDED"], per_pert[p]["SAFE"]]
          for p in sorted(per_pert)]
    pw = [max(len(str(r[i])) for r in [ph] + pt) for i in range(len(ph))]
    print("  ".join(str(v).ljust(pw[i]) for i, v in enumerate(ph)))
    print("-" * sum(w + 2 for w in pw))
    for r in pt:
        print("  ".join(str(v).ljust(pw[i]) for i, v in enumerate(r)))

    tot = defaultdict(int)
    for c in cells.values():
        for k in ("reached", "no_guard", "probed", "stamped", "FIRED", "FLIPPED",
                  "WRONG", "UNDECIDED", "SAFE", "unknown"):
            tot[k] += c[k]
    print(f"\nTOTALS  cells={len(cells)}  neurons reaching the probe branch={tot['reached']}")
    print(f"        of those, no guard held (pre-fix code solved nothing)={tot['no_guard']}"
          f"  probed={tot['probed']}")
    print(f"        signs the pre-fix rule actually stamped={tot['stamped']}  "
          f"of which WRONG (the bug fired)={tot['FIRED']}  undecided={tot['unknown']}")
    print(f"        sign truth over probed: flipped={tot['FLIPPED']} wrong={tot['WRONG']} "
          f"undecided={tot['UNDECIDED']} safe={tot['SAFE']}")

    rerun = [k for k in sorted(cells) if verdict_of(cells[k]) == "RERUN"]
    check = [k for k in sorted(cells) if verdict_of(cells[k]) == "check"]
    print(f"\nRERUN: {len(rerun)} cells "
          f"({sum(files_for.get(dir_key(k), 0) for k in rerun)} result files)")
    for k in rerun:
        print("   " + " | ".join(k))
        for f in disc.get(dir_key(k), {}).get("files", []):
            print("      " + f)
    print(f"\nCHECK (a sign was neither proved nor refuted): {len(check)} cells "
          f"({sum(files_for.get(dir_key(k), 0) for k in check)} result files)")
    for k in check:
        print("   " + " | ".join(k))

    clean = [k for k in sorted(cells) if verdict_of(cells[k]) == "clean"]
    print(f"\nCLEAN: {len(clean)} cells "
          f"({sum(files_for.get(dir_key(k), 0) for k in clean)} result files) — nothing to re-run")

    if witnesses:
        print("\nWITNESSES")
        for key, rs in sorted(witnesses.items()):
            print("  " + " | ".join(key))
            for r in rs[:10]:
                print(f"    layer {r['layer']} neuron {r['neuron']}  "
                      f"ge={r['verdict_ge']}/stamped={r['old_stamp_ge']}  "
                      f"le={r['verdict_le']}/stamped={r['old_stamp_le']}  "
                      f"old_val min={r['old_val_min']} max={r['old_val_max']}  "
                      f"z={r['witness_z']} z^p={r['witness_zp']} "
                      f"forward-checked={r['witness_ok']}  file={r['witness_file']}")

    if unmapped:
        n = sum(len(v) for v in unmapped.values())
        print(f"\nNOT AUDITED: {n} result files")
        for reason, files in sorted(unmapped.items()):
            print(f"   {len(files)}x  {reason}")


if __name__ == "__main__":
    main()
