#!/usr/bin/env python3
"""Enumerate and run the dependency-sign audit over an experiments tree.

One job per (dataset, arch, model, perturbation, size): the sign of a dependency is a
property of the network under the perturbation, so the technique flags and the class pair
do not change it, and a single stripped run answers for every result file in that cell.

Each job runs only the dependency component -- no PGD, no zonotope, no perturbation
intervals, no relaxation -- and stops after perturbation_dependencies, before the solve.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_relaxation_sweep import (run_pool, _max_slots_for, TOTAL_CORES,  # noqa: E402
                                  _julia_dataset_name)

# get_dataset_params (utils/datasets.jl:1-18) knows mnist, fmnist, cifar10 and har only.
BENCHMARK_DATASETS = {"har", "acas"}
UNSUPPORTED_DATASETS = {"acas"}
# get_nn (utils/models.jl) has no branch for these, so run.jl cannot build the network.
UNSUPPORTED_ARCHS = {"convSmallRELU__Point"}

# .../<dataset>/<arch>_exp/<perturbation>/eps_<size>/<subdir>/<file>.txt
PATH_RE = re.compile(r"/(?P<dataset>[a-z-]+)/(?P<arch>[^/]+)_exp/"
                     r"(?P<pert>[^/]+)/eps_(?P<size>[^/]+)/")


def model_index(roots):
    """{(dataset, arch, model_dir_name): path to model.p} over every experiments root."""
    idx = {}
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            if "model.p" not in filenames:
                continue
            parts = dirpath.split(os.sep)
            try:
                i = next(j for j, p in enumerate(parts) if p.endswith("_exp"))
            except StopIteration:
                continue
            dataset, arch = parts[i - 1], parts[i][: -len("_exp")]
            idx.setdefault((dataset, arch, parts[-1]), os.path.join(dirpath, "model.p"))
    return idx


PERTS = ("linf", "brightness", "contrast", "patch_old", "patch", "occ",
         "translation", "rotation", "max")
FNAME_RE = re.compile(
    r"^(?:\d+_)?(?:n2_)?(?P<model_name>.+?)_(?P<pert>" + "|".join(PERTS) + r")_"
    r"(?P<size>[0-9.,]+)_ctag(?P<ctag>\d+)_(?P<nts>.*)$")
TECH_TAG_RE = re.compile(r"_(?:seed\d+_)?HyperAttack|_VagharDeps|_N2_advStd|"
                         r"_N2_vaghar|_N2_seed|_N1_seed|_stdBoost")


def parse_result_name(fname):
    """(model_name, perturbation, size, name_to_save base) from a result filename.

    save_results writes <token>_<model_name>_<pert>_<size>_ctag<N>_<name_to_save>, so the
    filename -- not the directory -- is authoritative. It matters: the har delta_max runs
    sit under a linf/ directory but were run with --perturbation max.
    """
    stem = fname[:-4] if fname.endswith(".txt") else fname
    m = FNAME_RE.match(stem)
    if not m:
        return None
    base = TECH_TAG_RE.split(m.group("nts"))[0]
    return m.group("model_name"), m.group("pert"), m.group("size"), base


def model_dir_candidates(base, arch, parent_dir):
    """Directory names that could hold this cell's model.p. Layouts differ per
    architecture (cnn1 keeps model_<x> directly under <arch>_exp, 3x10 nests <x> under
    model_seed42_itr20), and runs with an empty --name_to_save carry the model identity
    in their parent directory instead, as vagharNoPerturbed_<arch>_<x>."""
    cands = []

    def add(x):
        if x and x not in cands:
            cands.append(x)
            if not x.startswith("model_"):
                add("model_" + x)

    for src in (base, os.path.basename(parent_dir.rstrip("/"))):
        if not src:
            continue
        add(src)
        m = re.match(r"^(.*?)_(N1|N2)(_.*)?$", src)
        if m:
            add(m.group(1))
        m = re.match(r"^vaghar\w*_" + re.escape(arch) + r"_(.*)$", src)
        if m:
            add(m.group(1))
        m = re.match(r"^\w+_" + re.escape(arch) + r"_(.*)$", src)
        if m:
            add(m.group(1))
    return cands


def discover(experiments_root, model_roots):
    idx = model_index(model_roots)
    cells, unmapped = {}, defaultdict(list)
    for dirpath, _dirnames, filenames in os.walk(experiments_root):
        pm = PATH_RE.search(dirpath + "/")
        if not pm:
            continue
        for fname in filenames:
            if not fname.endswith(".txt") or "VagharDeps" not in fname:
                continue
            fpath = os.path.join(dirpath, fname)
            dataset, arch = pm.group("dataset"), pm.group("arch")
            parsed = parse_result_name(fname)
            if not parsed:
                unmapped["unparsable filename"].append(fpath)
                continue
            _model_name, pert, size, base = parsed
            model_p = None
            cands = model_dir_candidates(base, arch, dirpath)
            for cand in cands:
                model_p = idx.get((dataset, arch, cand))
                if model_p:
                    base = base or cand
                    break
            if not model_p:
                unmapped[f"no model.p for {dataset}/{arch} (tried {cands[:4]})"].append(fpath)
                continue
            if dataset in UNSUPPORTED_DATASETS:
                unmapped[f"{dataset}: get_dataset_params has no branch for it"].append(fpath)
                continue
            if arch in UNSUPPORTED_ARCHS:
                unmapped[f"{arch}: get_nn has no branch for it"].append(fpath)
                continue
            key = (dataset, arch, base, pert, size)
            cells.setdefault(key, {"model_p": model_p, "files": []})
            cells[key]["files"].append(fpath)
    return cells, unmapped


def job_cmd(key, info, audit_dir, threads, c_tag, c_target):
    dataset, arch, base, pert, size = key
    cmd = [
        "julia", "run.jl",
        "--mode", "standard",
        "--dataset", _julia_dataset_name(dataset),
        "--model_name", arch,
        "--model_path", info["model_p"],
        "--perturbation", pert,
        "--perturbation_size", size,
        "--ctag", str(c_tag),
        "--ct", str(c_target),
        "--timout", "60",
        "--output_dir", os.path.join(audit_dir, "unused") + "/",
        "--name_to_save", base,
        # only the dependency component
        "--activate_vaghgar_deps", "true",
        "--use_hyper_attack", "false",
        "--use_perturbed_intervals", "false",
        "--nn1_zono_bounds", "false",
        "--nn1_relax_threshold", "0",
        "--nn1_sibling_gate", "false",
        "--dep_probe_audit", audit_dir,
        "--dep_probe_threads", str(threads),
    ]
    if dataset in BENCHMARK_DATASETS:
        cmd += ["--internet_nets_benchmarks", "true"]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_root", default="paper_experiments_with_zono_fix")
    ap.add_argument("--model_roots", nargs="*",
                    default=["paper_experiments", "paper_experiments_with_zono_fix",
                             "paper_experiments_rerun"])
    ap.add_argument("--audit_dir", default="dep_audit")
    ap.add_argument("--threads", type=int, default=2,
                    help="Gurobi Threads per job; also the taskset slot width")
    ap.add_argument("--max_cores", type=int, default=TOTAL_CORES)
    ap.add_argument("--c_tag", type=int, default=1)
    ap.add_argument("--c_target", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cells")
    ap.add_argument("--filter", default="", help="substring filter on the cell key")
    ap.add_argument("--list", action="store_true", help="print the cells and exit")
    args = ap.parse_args()

    cells, unmapped = discover(args.experiments_root, args.model_roots)
    keys = sorted(cells)
    if args.filter:
        keys = [k for k in keys if args.filter in "/".join(k)]
    if args.limit:
        keys = keys[: args.limit]

    n_files = sum(len(cells[k]["files"]) for k in keys)
    print(f"{len(keys)} cells covering {n_files} result files "
          f"(root: {args.experiments_root})")
    if unmapped:
        print(f"UNMAPPED: {sum(len(v) for v in unmapped.values())} files")
        for reason, files in sorted(unmapped.items())[:10]:
            print(f"  {reason}: {len(files)}  e.g. {os.path.basename(files[0])}")
    if args.list:
        for k in keys:
            print("  " + " | ".join(k) + f"   [{len(cells[k]['files'])} files]")
        return

    os.makedirs(args.audit_dir, exist_ok=True)
    jobs = []
    for k in keys:
        label = "__".join(k).replace("/", "_")
        jobs.append((label, job_cmd(k, cells[k], args.audit_dir, args.threads,
                                    args.c_tag, args.c_target)))

    max_slots = max(1, _max_slots_for(args.max_cores, args.threads))
    print(f"{len(jobs)} jobs, {max_slots} slots, {args.threads} threads/job")
    run_pool(jobs, max_slots, os.path.dirname(os.path.abspath(__file__)),
             args.threads, phase_name="dep_audit")


if __name__ == "__main__":
    main()
