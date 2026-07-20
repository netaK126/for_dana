#!/usr/bin/env python3
"""
estimate_sweep_eta.py — schedule / ETA table for the running relaxation sweep.

READ-ONLY OBSERVER. Produces one CSV (eta_out/sweep_schedule.csv) with a row per
  dataset x arch x perturbation_type x perturbation_size
giving status + projected start/finish datetimes, sorted like a timeline; also
prints the table and a 4-line per-arch/dataset summary.

It mirrors the specific command that is currently running (see CONFIG below):

  python3 run_relaxation_sweep.py \
    --dataset_group "mnist|3x50=.../3x50_exp/model_seed42_itr20/19,cnn1=.../cnn1_exp/model_seed42_itr20" \
    --dataset_group "fashion-mnist|cnn1=.../cnn1_exp/model_seed42_itr19,cnn2=.../cnn2_exp/model_seed42_itr19" \
    --dataset_slots 2,5 --timeout 10800 --advanced_standard ... \
    --sweep_ctag 1 3 --ct 2,4,5 --sweep_gurobi_seed 4 --skip_std_n2_baseline \
    --stdboost_combos N1:false:false:0:false,N2:false:false:0:false,N1:true:true:0.5:true,N2:true:true:0.5:true \
    --rerun_timeouts --prioritize_rows --geometric_intervals --n2_tables_only

SAFETY (all enforced below):
  * only reads paper_experiments/**/*.txt + tests n1_state .bin existence;
  * writes exactly one file, atomically, into its own eta_out/ folder;
  * never imports/execs run_relaxation_sweep.py, never launches a job;
  * process inspection is passive (pgrep / ps -o etimes / /proc/<pid>/cmdline) — no signals;
  * pins itself to cores 0-7 and nice 19 so it never contends with the solvers (cores 8-231).

How estimates are formed (honest about uncertainty):
  * per-(c_source,c_target) solve time = optimization_time + hyper_attack_time + lp_optimization_time
    from history, capped at --timeout; any timeout-status (or time >= budget) counts as a full-timeout pair.
  * a per-(arch,pert_type) distribution feeds every job of that cell (N1 / advstd / stdBoost) — the per-pair
    cost is dominated by (network, perturbation, budget), not the technique variant.
  * end_est  = EXPECTED schedule: each remaining class-pair costs p*T + (1-p)*median(solved), p = its
    historical timeout rate.
  * end_worst = GUARANTEED CEILING, computed as an INDEPENDENT schedule where every remaining class-pair is
    charged the full timeout T (dur_worst = encode + #unsolved_pairs*T). A pair can't exceed T, so this
    upper-bounds any real finish; it does NOT depend on end_est. (Wider — that's the price of a hard bound.)
  * ENCODE is a one-time per-job overhead (net encoding + Julia JIT + bound-tightening), per-arch:
    3x50=1.5h, cnn1=1h, cnn2=2h. Added to a QUEUED job's duration; reported in the 'encode_h' column.
  * RUNNING job remaining = Σ(still-missing class-pairs) − time already spent on the in-progress pair
    (= now − mtime of its last completed pair's output, capped at one pair). Not encode, not dur−elapsed.
  * fashion-mnist/cnn2 has almost no history -> low confidence, widest band.
Treat end_est .. end_worst as a RANGE, not a point.
"""

import os
import re
import sys
import glob
import time
import argparse
import subprocess
from datetime import datetime, timedelta
from statistics import median

# ──────────────────────────────────────────────────────────────────────────
# Step 0 — CONFIG (pre-filled from the running command; override via flags)
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_SWEEP_ROOT = "/root/Downloads/for_dana/vaghar_org"

# dataset -> list of (arch, n1_model_dir).  arch order = build order (arch outer).
DATASETS = {
    "mnist": [
        ("3x50", "paper_experiments/mnist/3x50_exp/model_seed42_itr20/19"),
        ("cnn1", "paper_experiments/mnist/cnn1_exp/model_seed42_itr20"),
    ],
    "fashion-mnist": [
        ("cnn1", "paper_experiments/fashion-mnist/cnn1_exp/model_seed42_itr19"),
        ("cnn2", "paper_experiments/fashion-mnist/cnn2_exp/model_seed42_itr19"),
    ],
}
DS_ORDER = ["mnist", "fashion-mnist"]          # --dataset_group order
RESERVE = {"mnist": 2, "fashion-mnist": 5}      # --dataset_slots 2,5  (sum = 7 concurrent slots)

C_TAGS = [1, 3]                                  # --sweep_ctag 1 3
CT = [2, 4, 5]                                   # --ct 2,4,5
SEED = 4                                         # --sweep_gurobi_seed 4
TIMEOUT_DEFAULT = 10800                          # --timeout 10800
# Per-arch one-time encoding overhead (net encoding + Julia JIT + bound-tightening), HOURS.
# Charged once per QUEUED job (bounds are cached/reused across a job's c_targets). A RUNNING job
# has already paid it, so it's excluded from that job's remaining (see attach_durations).
ENCODE_HOURS = {"3x50": 1.5, "cnn1": 1.0, "cnn2": 2.0}
DEFAULT_ENCODE_HOURS = 1.0
GEOMETRIC = True                                 # --geometric_intervals (translation/rotation only)

# 14 perturbations, copied verbatim from run_relaxation_sweep.py PERTURBATIONS (:170-190)
PERTURBATIONS = [
    ("patch(1,14,14,3)", "patch:1,14,14,3"),
    ("trans(1,1)",       "translation:1,1"),
    ("trans(3,1)",       "translation:3,1"),
    ("occ(3,3,5)",       "occ:3,3,5"),
    ("occ(14,14,9)",     "occ:14,14,9"),
    ("occ(1,1,9)",       "occ:1,1,9"),
    ("contrast(1.5)",    "contrast:1.5"),
    ("contrast(1.2)",    "contrast:1.2"),
    ("rotation(10)",     "rotation:10"),
    ("rotation(5)",      "rotation:5"),
    ("linf(0.05)",       "linf:0.05"),
    ("linf(0.1)",        "linf:0.1"),
    ("brightness(0.25)", "brightness:0.25"),
    ("brightness(0.1)",  "brightness:0.1"),
]

# advstd-N2 technique for this run (ms=F,bp=off,lb=F,bt=T,zb=T,np=off,rt=0.5,vh=prev_pgd,sg=T)
ADVSTD_BASE_SUFFIX = "_N2_advStd_BoundTightPertRelax0.5_SibGate_zonoBounds_varHintPrevPGD"
# kept N2 stdBoost combos (N1 combos dropped by --n2_tables_only): (zb, sg, rt, pi) as sweep strings
STDBOOST_COMBOS = [
    ("false", "false", 0.0, "false"),   # label btpr0.0
    ("true",  "true",  0.5, "true"),    # label zbbtpr0.5sgpi
]

HIST_MIN = 3          # >= this many matching historical pairs -> confidence "high"

# Job-type constants
N1, ADVSTD, STDBOOST = "N1", "ADVSTD", "STDBOOST"
PHASE_RANK = {N1: 0, STDBOOST: 1, ADVSTD: 2}

# ──────────────────────────────────────────────────────────────────────────
# Ported read-only skip-check helpers (verbatim logic from run_relaxation_sweep.py)
# ──────────────────────────────────────────────────────────────────────────
_RERUN_TIMEOUTS = True
_RERUN_TIMEOUT_BUDGET = float(TIMEOUT_DEFAULT)
_TIMEOUT_VALUES = (1800.0, 3600.0, float(TIMEOUT_DEFAULT))
_TIMEOUT_MATCH_EPS = 30.0
_TIMEOUT_STATUSES = {
    "TIME_LIMIT", "USER_OBJ_LIMIT", "USER_LIMIT", "ITERATION_LIMIT",
    "NODE_LIMIT", "SOLUTION_LIMIT", "MEMORY_LIMIT", "WORK_LIMIT",
}


def _timeout_row_is_stale_rerun(status, runtime):
    is_timeout = status in _TIMEOUT_STATUSES
    if not is_timeout and not status and runtime is not None:
        is_timeout = any(abs(runtime - cap) <= _TIMEOUT_MATCH_EPS for cap in _TIMEOUT_VALUES)
    if not is_timeout:
        return False
    if runtime is None:
        return True
    return runtime < _RERUN_TIMEOUT_BUDGET - _TIMEOUT_MATCH_EPS


def _parse_kv(line):
    d = {}
    for tok in line.split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def _parse_c_source_target_pairs(filepath):
    """Set of (c_source, c_target) 0-indexed pairs 'covered' (done) in a result file,
    honoring INTERRUPTED exclusion and --rerun_timeouts. Never raises."""
    pairs = set()
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = _parse_kv(line)
                if "c_source" in fields and "c_target" in fields:
                    status = (fields.get("solve_status", "") or "").upper()
                    if status == "INTERRUPTED":
                        continue
                    if _RERUN_TIMEOUTS:
                        opt_t = fields.get("optimization_time", "")
                        try:
                            runtime = float(opt_t) if opt_t else None
                        except ValueError:
                            runtime = None
                        if _timeout_row_is_stale_rerun(status, runtime):
                            continue
                    try:
                        pairs.add((int(fields["c_source"]), int(fields["c_target"])))
                    except ValueError:
                        pass
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        cs = int(parts[0]); ct = int(parts[1])
                    except ValueError:
                        continue
                    if _RERUN_TIMEOUTS and len(parts) >= 5:
                        try:
                            runtime = float(parts[4])
                        except ValueError:
                            runtime = None
                        if runtime is not None and _timeout_row_is_stale_rerun("", runtime):
                            continue
                    pairs.add((cs, ct))
    except OSError:
        pass
    return pairs


def _eps_str_variants(eps_str):
    try:
        floats = [float(tok) for tok in eps_str.split(",")]
    except ValueError:
        return {eps_str}
    julia_form = ",".join((f"{int(v)}.0" if v == int(v) else repr(v)) for v in floats)
    return {eps_str, julia_form}


_DROP_COUNT_RE = re.compile(r"_both(\d+)_orgDrop(\d+)_pertDrop(\d+)")


def _filename_dropped_binaries(fname):
    m = _DROP_COUNT_RE.search(fname)
    if m:
        return any(int(g) > 0 for g in m.groups())
    bm = re.search(r"_(?:BTPR|BoundTightPertRelax)(\d+(?:\.\d+)?)", fname)
    if bm:
        return float(bm.group(1)) > 0.0
    return False


def _is_pre_fix_dropped(fname):
    return _filename_dropped_binaries(fname) and "_depGuardFix" not in fname


def advstd_missing_cts(cwd, dataset, arch, pert_type, eps_str, n1_tag,
                       base_name_to_save, seed, c_tag, c_targets, geom):
    c_targets = [ct for ct in c_targets if ct != c_tag]
    out_dir = os.path.join(cwd, "paper_experiments", dataset, f"{arch}_exp",
                           pert_type, f"eps_{eps_str}", f"advStd_{arch}_N1_{n1_tag}")
    if not os.path.isdir(out_dir):
        return list(c_targets)
    src0 = c_tag - 1
    covered = set()
    for variant in _eps_str_variants(eps_str):
        pattern = os.path.join(
            out_dir, f"*_n2_{arch}_{pert_type}_{variant}_ctag{src0}_{base_name_to_save}_seed{seed}_*.txt")
        for fpath in glob.glob(pattern):
            fname = os.path.basename(fpath)
            if ("_geomInt" in fname) != geom:
                continue
            if _is_pre_fix_dropped(fname):
                continue
            for cs, ct in _parse_c_source_target_pairs(fpath):
                if cs == src0:
                    covered.add(ct)
    return [ct for ct in c_targets if (ct - 1) not in covered]


def _stdboost_filename_regex(base_name_to_save, seed, zb, sg, rt, pi, c_tag, geom):
    parts = [re.escape(base_name_to_save)]
    if seed != 0:
        parts.append(re.escape(f"_seed{seed}"))
    parts.append(re.escape("_HyperAttackHints_VagharDeps") + r"(?:_depGuardFix)?")
    if pi == "true":
        parts.append(re.escape("_PertruebedIntervals"))
        if geom:
            parts.append(re.escape("_geomInt"))
    has_boost = (zb == "true") or (rt >= 0.0) or (sg == "true")
    if has_boost:
        parts.append(re.escape("_stdBoost"))
        if zb == "true":
            parts.append(re.escape("_zono"))
        if rt >= 0.0:
            parts.append(re.escape(f"_BTPR{rt}"))
        if sg == "true":
            parts.append(re.escape("_SibGate"))
    if sg == "true" and rt >= 0.0:
        parts.append(r"_both\d+_orgDrop\d+_pertDrop\d+")
    parts.append(re.escape(f"_cTag{c_tag}"))
    parts.append(r"\.txt$")
    return re.compile("".join(parts))


def stdboost_missing_cts(out_dir, arch, pert_type, eps_str, base_name_to_save,
                         seed, zb, sg, rt, pi, c_tag, c_targets, geom):
    c_targets = [ct for ct in c_targets if ct != c_tag]
    if not os.path.isdir(out_dir):
        return list(c_targets)
    src0 = c_tag - 1
    tail_re = _stdboost_filename_regex(base_name_to_save, seed, zb, sg, rt, pi, c_tag, geom)
    covered = set()
    for variant in _eps_str_variants(eps_str):
        pattern = os.path.join(out_dir, f"*_{arch}_{pert_type}_{variant}_ctag{src0}_{base_name_to_save}_*.txt")
        for fpath in glob.glob(pattern):
            fname = os.path.basename(fpath)
            if tail_re.search(fname) is None:
                continue
            if _is_pre_fix_dropped(fname):
                continue
            for cs, ct in _parse_c_source_target_pairs(fpath):
                if cs == src0:
                    covered.add(ct)
    return [ct for ct in c_targets if (ct - 1) not in covered]


def n1_missing_cts(n1_state_dir, c_tag, c_targets):
    """need_pseudocosts=False, need_n1_preact=False for this run: require vars+layers .bin."""
    c_targets = [ct for ct in c_targets if ct != c_tag]
    if not os.path.isdir(n1_state_dir):
        return list(c_targets)
    missing = []
    for ct in c_targets:
        vars_path = os.path.join(n1_state_dir, f"n1_vars_{c_tag}_{ct}.bin")
        layers_path = os.path.join(n1_state_dir, f"n1_layers_{c_tag}_{ct}.bin")
        if not (os.path.isfile(vars_path) and os.path.isfile(layers_path)):
            missing.append(ct)
    return missing


# ──────────────────────────────────────────────────────────────────────────
# Step 3 — historical solve-time distribution per (arch, pert_type)
# ──────────────────────────────────────────────────────────────────────────
_PATH_RE = re.compile(r"paper_experiments/([^/]+)/([^/]+)_exp/([^/]+)/eps_")


def scan_history(cwd, timeout):
    """Return dict (arch, pert_type) -> list of (per_pair_seconds, is_timeout)."""
    hist = {}
    root = os.path.join(cwd, "paper_experiments")
    for fpath in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True):
        if os.path.basename(fpath) == "_filename_legend.txt":
            continue
        m = _PATH_RE.search(fpath.replace(os.sep, "/"))
        if not m:
            continue
        arch, pert_type = m.group(2), m.group(3)
        key = (arch, pert_type)
        bucket = hist.setdefault(key, [])
        try:
            with open(fpath) as f:
                for line in f:
                    if "optimization_time" not in line:
                        continue
                    d = _parse_kv(line.strip())
                    if "c_source" not in d:
                        continue
                    status = (d.get("solve_status", "") or "").upper()
                    if status == "INTERRUPTED":
                        continue
                    try:
                        opt = float(d.get("optimization_time", "nan"))
                    except ValueError:
                        continue
                    if opt != opt:  # nan
                        continue
                    att = _safe_float(d.get("hyper_attack_time"))
                    lp = _safe_float(d.get("lp_optimization_time"))
                    # Timeout iff: an explicit Gurobi limit status; OR the solve took at least the
                    # current budget (so it would time out now); OR — for legacy status-less rows —
                    # its time sits on a known historical cap (1800/3600/...). Matching the skip-check's
                    # multi-cap rule here keeps old-budget timeouts out of the 'solved' median (which
                    # would otherwise bias the best-case estimate low).
                    is_to = ((status in _TIMEOUT_STATUSES)
                             or (opt >= timeout - _TIMEOUT_MATCH_EPS)
                             or (not status and any(abs(opt - cap) <= _TIMEOUT_MATCH_EPS
                                                    for cap in _TIMEOUT_VALUES)))
                    per_pair = min(opt, float(timeout)) + att + lp
                    bucket.append((per_pair, is_to))
        except OSError:
            continue
    return hist


def _safe_float(s):
    try:
        return float(s) if s not in (None, "") else 0.0
    except ValueError:
        return 0.0


def make_per_ct(hist, timeout):
    """Return per_ct(arch, pert_type) -> (est, conf): the EXPECTED time of ONE class-pair.

        est = p*T + (1-p)*median(solved),   p = empirical timeout rate,  T = budget.
    Using p (not just median of the solved runs) keeps the expected value realistic — a cell that
    times out 99% of the time gets est ~ T, not the median of its 1% lucky fast solves. This feeds
    end_est only; end_worst uses the hard all-timeout ceiling (T per pair), computed separately.
    """
    def dist(H):
        n = len(H)
        solved = [t for (t, to) in H if not to]
        n_to = sum(1 for (_, to) in H if to)
        p = n_to / n
        t_solve = median(solved) if solved else float(timeout)
        return p * float(timeout) + (1.0 - p) * t_solve

    def per_ct(arch, pert_type):
        H = hist.get((arch, pert_type))
        if H:
            return dist(H), ("high" if len(H) >= HIST_MIN else "med")
        # fallback: cnn2 <- cnn1 x scale (capped), else same pert_type across archs, else assume hard.
        if arch == "cnn2" and hist.get(("cnn1", pert_type)):
            return min(dist(hist[("cnn1", pert_type)]) * 1.5, float(timeout)), "low"
        cross = [rec for (a, p), H2 in hist.items() if p == pert_type for rec in H2]
        if cross:
            return dist(cross), "low"
        return 0.5 * float(timeout), "low"

    return per_ct


CONF_RANK = {"high": 0, "med": 1, "low": 2}


def worst_conf(cs):
    cs = [c for c in cs if c]
    return max(cs, key=lambda c: CONF_RANK[c]) if cs else "low"


# ──────────────────────────────────────────────────────────────────────────
# Job model
# ──────────────────────────────────────────────────────────────────────────
class Job:
    __slots__ = ("ds", "arch", "ptype", "psize", "ctag", "jtype", "combo",
                 "pi", "cts", "state", "elapsed", "actual_start", "deps",
                 "dur_lo", "dur_hi", "remaining_lo", "remaining_hi", "conf",
                 "s_lo", "e_lo", "s_hi", "e_hi", "out_glob", "cur_pair")

    def __init__(self, ds, arch, ptype, psize, ctag, jtype, combo=None, pi=None):
        self.ds, self.arch, self.ptype, self.psize = ds, arch, ptype, psize
        self.ctag, self.jtype, self.combo, self.pi = ctag, jtype, combo, pi
        self.cts = [ct for ct in CT if ct != ctag]
        self.state = "QUEUED"
        self.elapsed = 0.0
        self.actual_start = None
        self.deps = []
        self.dur_lo = self.dur_hi = 0.0
        self.remaining_lo = self.remaining_hi = 0.0
        self.conf = "low"
        self.s_lo = self.e_lo = self.s_hi = self.e_hi = 0.0
        self.out_glob = None    # glob for this job's per-pair output files (mtime = last pair completion)
        self.cur_pair = 0.0     # seconds already spent on the in-progress class-pair (running jobs)

    @property
    def cell(self):
        return (self.ds, self.arch, self.ptype, self.psize)


def n1_tags(n1_dir):
    n1_tag = os.path.basename(os.path.normpath(n1_dir))
    # acas/har derive N2 by reducing weight precision rather than extra SGD, so
    # their N2 carries a _int8 tag instead of _sgd_itr1.
    n2_tag = n1_tag + ("_int8" if os.path.exists(n1_dir + "_int8") else "_sgd_itr1")
    return n1_tag, n2_tag


def geom_adv(pert_type):
    return GEOMETRIC and pert_type in ("translation", "rotation")


def geom_std(pert_type, pi):
    return GEOMETRIC and pert_type in ("translation", "rotation") and pi == "true"


def build_jobs(cwd, timeout):
    jobs = []
    for ds in DS_ORDER:
        for arch, n1_dir in DATASETS[ds]:
            n1_tag, n2_tag = n1_tags(n1_dir)
            for pert_name, spec in PERTURBATIONS:
                ptype, eps = spec.split(":", 1)
                n1_state_dir = os.path.join(cwd, "paper_experiments", ds, f"{arch}_exp",
                                            ptype, f"eps_{eps}", f"n1_state_{arch}_{n1_tag}")
                adv_dir_tag = n1_tag
                adv_base = f"{n2_tag}{ADVSTD_BASE_SUFFIX}"
                adv_dir = os.path.join(cwd, "paper_experiments", ds, f"{arch}_exp",
                                       ptype, f"eps_{eps}", f"advStd_{arch}_N1_{n1_tag}")
                std_out = os.path.join(cwd, "paper_experiments", ds, f"{arch}_exp",
                                       ptype, f"eps_{eps}", f"N2stdBoost_{arch}_{n2_tag}")
                std_base = f"{n2_tag}_N2"
                for ctag in C_TAGS:
                    # N1 — per-pair output is n1_vars_{ctag}_{ct}.bin (created when a c_target finishes)
                    j = Job(ds, arch, ptype, eps, ctag, N1)
                    j.cts = n1_missing_cts(n1_state_dir, ctag, CT)
                    j.out_glob = os.path.join(n1_state_dir, f"n1_vars_{ctag}_*.bin")
                    jobs.append(j)
                    # advstd-N2 — per-pair output is the result .txt (appended when a c_target finishes)
                    j = Job(ds, arch, ptype, eps, ctag, ADVSTD, combo="btpr0.5sgzb", pi="true")
                    j.cts = advstd_missing_cts(cwd, ds, arch, ptype, eps, adv_dir_tag,
                                               adv_base, SEED, ctag, CT, geom_adv(ptype))
                    j.out_glob = os.path.join(adv_dir, f"*_ctag{ctag - 1}_*.txt")
                    jobs.append(j)
                    # N2 stdBoost combos
                    for (zb, sg, rt, pi) in STDBOOST_COMBOS:
                        combo = (("zb" if zb == "true" else "") +
                                 f"btpr{rt}" + ("sg" if sg == "true" else "") +
                                 ("pi" if pi == "true" else "")) or "plain"
                        j = Job(ds, arch, ptype, eps, ctag, STDBOOST, combo=combo, pi=pi)
                        j.cts = stdboost_missing_cts(std_out, arch, ptype, eps, std_base, SEED,
                                                     zb, sg, rt, pi, ctag, CT, geom_std(ptype, pi))
                        j.out_glob = os.path.join(std_out, f"*_ctag{ctag - 1}_*.txt")
                        jobs.append(j)
    for j in jobs:
        if not j.cts:
            j.state = "DONE"
    return jobs


# ──────────────────────────────────────────────────────────────────────────
# Step 4 — detect RUNNING jobs from the live process table (passive)
# ──────────────────────────────────────────────────────────────────────────
def _cmd_opt(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def detect_running(cwd, jobs):
    by_cell = {}
    for j in jobs:
        by_cell.setdefault((j.ds, j.arch, j.ptype, j.psize, j.ctag), []).append(j)
    try:
        pids = subprocess.run(["pgrep", "-f", "run.jl"], capture_output=True, text=True).stdout.split()
    except Exception:
        pids = []
    running = []
    ds_arch_re = re.compile(r"paper_experiments/([^/]+)/([^/]+)_exp/")
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        if "run.jl" not in argv:
            continue
        out_dir = _cmd_opt(argv, "--output_dir") or ""
        m = ds_arch_re.search(out_dir)
        if not m:
            continue
        ds, arch = m.group(1), m.group(2)
        ptype = _cmd_opt(argv, "--perturbation")
        eps = _cmd_opt(argv, "--perturbation_size")
        ctag = _cmd_opt(argv, "--ctag")
        mode = _cmd_opt(argv, "--mode")
        # This run has no std-N2 baseline (--skip_std_n2_baseline) and no N1stdBoost (--n2_tables_only),
        # so the only --mode standard jobs are N2 stdBoost. (A concurrent FOREIGN sweep running a
        # vagharWithPerturbed baseline or N1stdBoost in a matching cell would be mis-attributed here.)
        jtype = {"advanced_standard_n1": N1, "advanced_standard_n2": ADVSTD, "standard": STDBOOST}.get(mode)
        if jtype is None or not (ctag and ctag.isdigit()):
            continue
        cands = by_cell.get((ds, arch, ptype, eps, int(ctag)), [])
        cands = [c for c in cands if c.jtype == jtype]
        if not cands:
            continue
        if jtype == STDBOOST and len(cands) > 1:
            pi = _cmd_opt(argv, "--use_perturbed_intervals")
            zb = _cmd_opt(argv, "--nn1_zono_bounds")
            pick = [c for c in cands if (c.pi == pi and (("zb" in c.combo) == (zb == "true")))]
            cands = pick or cands
        j = cands[0]
        if j.state == "RUNNING":
            continue  # already claimed by another PID — don't seed one job into two slots
        et = _etimes(pid)
        if et is None:
            continue
        j.state = "RUNNING"
        j.elapsed = float(et)
        j.actual_start = -j.elapsed  # relative to now
        running.append(j)
    return running


def _etimes(pid):
    try:
        out = subprocess.run(["ps", "-o", "etimes=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────
# Step 3b — attach durations
# ──────────────────────────────────────────────────────────────────────────
def encode_secs(arch):
    """One-time per-job encoding overhead in seconds (per-arch)."""
    return ENCODE_HOURS.get(arch, DEFAULT_ENCODE_HOURS) * 3600.0


def last_output_mtime(out_glob):
    """Most recent mtime among a job's per-pair output files (= when its last class-pair finished),
    or None if it hasn't produced any yet. Read-only."""
    if not out_glob:
        return None
    mt = None
    for f in glob.glob(out_glob):
        try:
            t = os.path.getmtime(f)
        except OSError:
            continue
        if mt is None or t > mt:
            mt = t
    return mt


def attach_durations(jobs, per_ct, timeout):
    T = float(timeout)
    for j in jobs:
        if j.state == "DONE":
            continue
        vals = [per_ct(j.arch, j.ptype) for _ in j.cts] or [(0.0, "low")]
        solve_est = sum(v[0] for v in vals)   # Σ expected over the job's STILL-MISSING class-pairs
        j.conf = worst_conf([v[1] for v in vals])
        m = len(j.cts)
        enc = encode_secs(j.arch)
        j.dur_lo = enc + solve_est            # expected duration for a QUEUED job (encode once + pairs)
        j.dur_hi = enc + m * T                # WORST (guaranteed ceiling): every remaining pair hits T
        if j.state == "RUNNING":
            # Remaining (no encode — already paid) MINUS time already on the ONE in-progress pair
            # (j.cur_pair = now - last-completed-pair mtime), capped at a single pair. Applies to both
            # bounds. Neither `dur - elapsed` (collapses to a past "≈now") nor plain Σ (over-charges a
            # full fresh solve for a pair it's hours into).
            pp_est = solve_est / m if m else 0.0
            j.remaining_lo = max(0.0, solve_est - min(j.cur_pair, pp_est))
            j.remaining_hi = max(0.0, m * T - min(j.cur_pair, T))
        else:
            j.remaining_lo = j.dur_lo
            j.remaining_hi = j.dur_hi


# ──────────────────────────────────────────────────────────────────────────
# Step 5 — dependency edges
# ──────────────────────────────────────────────────────────────────────────
def attach_deps(jobs):
    idx = {}
    for j in jobs:
        idx[(j.ds, j.arch, j.ptype, j.psize, j.ctag, j.jtype)] = j
    for ds in DS_ORDER:
        for arch, _ in DATASETS[ds]:
            for _, spec in PERTURBATIONS:
                ptype, eps = spec.split(":", 1)
                n1 = {ct: idx.get((ds, arch, ptype, eps, ct, N1)) for ct in C_TAGS}
                # N1 chain across c_tags
                prev = None
                for ct in C_TAGS:
                    if n1[ct] and prev and n1[ct].state != "DONE":
                        n1[ct].deps.append(prev)
                    if n1[ct]:
                        prev = n1[ct]
                # advstd locked behind its own N1 iff that N1 is not DONE
                for ct in C_TAGS:
                    adv = idx.get((ds, arch, ptype, eps, ct, ADVSTD))
                    if adv and n1[ct] and n1[ct].state != "DONE":
                        adv.deps.append(n1[ct])


# ──────────────────────────────────────────────────────────────────────────
# Step 6 — project start/end per dataset (live anchor + slot-division)
# ──────────────────────────────────────────────────────────────────────────
def project(jobs, bound):
    """Schedule one bound ('lo' = expected durations, 'hi' = all-timeout ceiling) per dataset via
    live-anchored slot-division. Sets s_lo/e_lo (lo pass) or s_hi/e_hi (hi pass). The two passes are
    INDEPENDENT — end_worst (from the 'hi' pass) never references end_est."""
    arch_index = {ds: {a: i for i, (a, _) in enumerate(DATASETS[ds])} for ds in DS_ORDER}
    pert_index = {f"{spec.split(':', 1)[0]}:{spec.split(':', 1)[1]}": i
                  for i, (_, spec) in enumerate(PERTURBATIONS)}

    def order_key(j):
        return (arch_index[j.ds][j.arch], pert_index[f"{j.ptype}:{j.psize}"], j.ctag, PHASE_RANK[j.jtype])

    lo = (bound == "lo")

    def dur(j):
        base = (j.remaining_lo if lo else j.remaining_hi) if j.state == "RUNNING" else (j.dur_lo if lo else j.dur_hi)
        return max(0.0, base)

    def set_s(j, v): setattr(j, "s_lo" if lo else "s_hi", v)
    def set_e(j, v): setattr(j, "e_lo" if lo else "e_hi", v)
    def get_e(j):    return j.e_lo if lo else j.e_hi

    for D in DS_ORDER:
        run_D = [j for j in jobs if j.ds == D and j.state == "RUNNING"]
        # (1) live anchor: running jobs hold slots; each frees at now+remaining (t=0 is now).
        for j in run_D:
            set_s(j, -j.elapsed)
            set_e(j, dur(j))
        n_slots = max(RESERVE[D], len(run_D))
        slot_free = sorted(dur(j) for j in run_D) + [0.0] * (n_slots - len(run_D))
        # (2) walk queued jobs of D in observed build order; earliest-free slot, after deps.
        queued = sorted([j for j in jobs if j.ds == D and j.state == "QUEUED"], key=order_key)
        for j in queued:
            i = min(range(len(slot_free)), key=lambda k: slot_free[k])
            ready = slot_free[i]
            for d in j.deps:
                if get_e(d) > ready:
                    ready = get_e(d)
            end = ready + dur(j)
            set_s(j, ready)
            set_e(j, end)
            slot_free[i] = end


# ──────────────────────────────────────────────────────────────────────────
# Step 7/8 — aggregate to rows, write CSV, print
# ──────────────────────────────────────────────────────────────────────────
def job_labels(j):
    """(kind, description, sweep_tag) in the sweep's own vocabulary — answers 'what is this job?'."""
    if j.jtype == N1:
        return ("N1", "N1 source-network state solve", "N1-solve")
    if j.jtype == ADVSTD:
        return ("N2 ours (transfer)", "advStd diff-transfer, our method", f"N2({j.combo})")
    # STDBOOST: btpr0.0 = all boosts off + no perturbed-intervals = the VHAGaR baseline column;
    # zbbtpr0.5sgpi = zono+boundTight+sibgate + perturbed-intervals = our standard-mode column.
    if j.combo == "btpr0.0":
        return ("N2 vaghar", "standard N2, boosts off, no perturbed-intervals (VHAGaR baseline)",
                f"N2stdBoost({j.combo})")
    return ("N2 ours", "standard N2, zono+boundTight+sibgate+perturbed-intervals",
            f"N2stdBoost({j.combo})")


def build_rows(jobs, now):
    """One row PER JOB, sorted chronologically by start_est (running/past first, done last)."""
    rows = []
    for j in jobs:
        kind, desc, tag = job_labels(j)
        if j.state == "DONE":
            start_off = None
            start = end = endw = conf = ""
        else:
            start_off = j.actual_start if j.state == "RUNNING" else j.s_lo
            start = _fmt(now + start_off)
            end = _fmt(now + j.e_lo)
            endw = _fmt(now + j.e_hi)
            conf = j.conf
        rows.append({
            "dataset": j.ds, "arch": j.arch, "pert_type": j.ptype, "pert_size": j.psize,
            "c_tag": j.ctag, "job": kind, "description": desc, "sweep_tag": tag,
            "status": j.state.lower(), "start_est": start, "end_est": end,
            "end_worst": endw, "encode_h": f"{ENCODE_HOURS.get(j.arch, DEFAULT_ENCODE_HOURS):.1f}",
            "confidence": conf,
            "_sort": start_off if start_off is not None else float("inf"),
        })
    rows.sort(key=lambda r: r["_sort"])   # by start_est; done (no start) last
    return rows


def _fmt(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


COLS = ["dataset", "arch", "pert_type", "pert_size", "c_tag", "job", "description",
        "sweep_tag", "status", "start_est", "end_est", "end_worst", "encode_h", "confidence"]
PRINT_COLS = ["dataset", "arch", "pert_type", "pert_size", "c_tag", "job",
              "status", "start_est", "end_est", "end_worst", "encode_h", "confidence"]


def write_csv(rows, out_path):
    import csv
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, out_path)


def print_table(rows, cols=PRINT_COLS, limit=None):
    shown = rows if limit is None else rows[:limit]
    if not shown:
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in shown)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in shown:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    if limit is not None and len(rows) > limit:
        print(f"... (+{len(rows) - limit} more rows — see the CSV)")


def print_summary(jobs, now):
    print("\nPer arch x dataset (jobs still to run):")
    agg = {}
    for j in jobs:
        a = agg.setdefault((j.ds, j.arch), {"left": 0, "run": 0, "end_lo": 0.0, "end_hi": 0.0})
        if j.state != "DONE":
            a["left"] += 1
            a["end_lo"] = max(a["end_lo"], j.e_lo)
            a["end_hi"] = max(a["end_hi"], j.e_hi)
            if j.state == "RUNNING":
                a["run"] += 1
    for (ds, arch), a in sorted(agg.items()):
        if a["left"] == 0:
            print(f"  {ds}/{arch}: done")
        else:
            print(f"  {ds}/{arch}: {a['left']} left ({a['run']} running), "
                  f"expected finish {_fmt(now + a['end_lo'])}  |  worst {_fmt(now + a['end_hi'])}")

    enc_all = sum(encode_secs(j.arch) for j in jobs) / 3600.0
    enc_todo = sum(encode_secs(j.arch) for j in jobs if j.state == "QUEUED") / 3600.0
    print(f"\nEncoding overhead (one-time per job): {enc_all:.0f} h across all {len(jobs)} jobs; "
          f"{enc_todo:.0f} h still to be paid on the {sum(1 for j in jobs if j.state=='QUEUED')} queued jobs.")
    per = ", ".join(f"{a}={h:.1f}h" for a, h in sorted(ENCODE_HOURS.items()))
    print(f"  per-job encode by arch: {per} (col 'encode_h').")


def main():
    ap = argparse.ArgumentParser(description="Read-only ETA schedule for the running relaxation sweep.")
    ap.add_argument("--sweep-root", default=DEFAULT_SWEEP_ROOT)
    ap.add_argument("--timeout", type=int, default=TIMEOUT_DEFAULT)
    ap.add_argument("--out-dir", default=None, help="default: <sweep-root>/eta_out")
    ap.add_argument("--tz", default="Asia/Jerusalem",
                    help="timezone for all displayed datetimes (default: Israel). Use e.g. UTC to disable.")
    args = ap.parse_args()

    # Display every datetime in the requested timezone (default Israel). time.tzset() reads the OS
    # tzdata, so DST (IDT +3 summer / IST +2 winter) is handled correctly. Absolute timestamps
    # (time.time()) are unaffected — only the local-time formatting changes.
    os.environ["TZ"] = args.tz
    try:
        time.tzset()
    except AttributeError:
        pass

    # ── safety: never contend with the solvers ──
    try:
        os.sched_setaffinity(0, set(range(0, 8)))
    except (AttributeError, OSError):
        pass
    try:
        os.nice(19)
    except OSError:
        pass

    global _RERUN_TIMEOUT_BUDGET, _TIMEOUT_VALUES
    _RERUN_TIMEOUT_BUDGET = float(args.timeout)
    _TIMEOUT_VALUES = (1800.0, 3600.0, float(args.timeout))

    cwd = os.path.abspath(args.sweep_root)
    out_dir = args.out_dir or os.path.join(cwd, "eta_out")
    os.makedirs(out_dir, exist_ok=True)
    now = time.time()

    print("Scanning history + skip-checks (read-only)...", file=sys.stderr)
    hist = scan_history(cwd, args.timeout)
    per_ct = make_per_ct(hist, args.timeout)

    jobs = build_jobs(cwd, args.timeout)
    detect_running(cwd, jobs)
    for j in jobs:                       # time already spent on each running job's in-progress pair
        if j.state == "RUNNING":
            lm = last_output_mtime(j.out_glob)
            j.cur_pair = max(0.0, now - lm) if lm else 0.0
    attach_durations(jobs, per_ct, args.timeout)
    attach_deps(jobs)
    project(jobs, "lo")
    project(jobs, "hi")

    rows = build_rows(jobs, now)      # one row PER JOB, sorted by start_est
    out_path = os.path.join(out_dir, "sweep_schedule.csv")
    write_csv(rows, out_path)

    n_done = sum(1 for j in jobs if j.state == "DONE")
    n_run = sum(1 for j in jobs if j.state == "RUNNING")
    n_q = sum(1 for j in jobs if j.state == "QUEUED")
    print(f"\nSnapshot: {_fmt(now)}  —  all datetimes below are in {args.tz} ({time.tzname[0]}/{time.tzname[1]})")
    print(f"Jobs: {n_done} done, {n_run} running, {n_q} queued  (total {len(jobs)})  "
          f"| history keys: {len(hist)}\n")
    # Terminal preview: the active jobs (running + queued) in start order; full set incl. done is in the CSV.
    active = [r for r in rows if r["status"] != "done"]
    # print_table(active, limit=50)
    print_summary(jobs, now)
    print(f"\nCSV written: {out_path}")
    # print("Notes:")
    # print("  * end_est = EXPECTED finish: each class-pair costs p*timeout+(1-p)*median(solved), "
    #       "p = its historical timeout rate.")
    # print("    end_worst = every remaining class-pair hits the full timeout (hard upper bound). "
    #       "For timeout-heavy cells est ~ worst.")
    # print("  * Cross-dataset spillover is NOT modeled: each dataset keeps its reserved slots "
    #       "(mnist:2, fashion:5).")
    # print("    So the dataset with fewer slots (mnist) is the most OVER-estimated — in reality it "
    #       "borrows the")
    # print("    other dataset's idle slots once that one finishes, so mnist's dates are an upper bound.")
    # print("  * cnn2 has almost no solve history -> 'low' confidence, widest band.")


if __name__ == "__main__":
    main()
