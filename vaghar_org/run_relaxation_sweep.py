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
import atexit
import time
import re
import glob
import itertools
import shutil

# ── Child-process lifecycle ──────────────────────────────────────────────
# Goal: a Ctrl+C (SIGINT) or `kill <pid>` (SIGTERM) of this supervisor
# tears down every descendant — including grand-children like julia spawned
# from utils/run_experiment.py — rather than leaving them orphaned and
# burning CPU for days.
#
# Two layers:
#   1. Each child is started with start_new_session=True, putting it (and
#      anything it spawns) into a fresh process group. Our signal handler
#      sends SIGTERM / SIGKILL to those groups, which propagates to julia
#      regardless of how deep it sits in the tree.
#   2. preexec_fn sets PR_SET_PDEATHSIG so the kernel itself signals each
#      direct child if the supervisor is SIGKILL'd (no chance to run a
#      handler). The grandchildren survive that path — only the direct
#      children get pdeathsig — but it at least prevents the python
#      run_experiment.py layer from outliving us silently.
try:
    import ctypes
    import ctypes.util as _ctypes_util
    _libc = ctypes.CDLL(_ctypes_util.find_library("c"), use_errno=True)
    _PR_SET_PDEATHSIG = 1
    def _pdeathsig_preexec():
        _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
except Exception:
    def _pdeathsig_preexec():
        pass

# Every live child Popen handle. Populated by launch_in_slot, drained by
# the reaper loop and by the signal handler.
_ACTIVE_CHILDREN = set()

def _kill_descendants(grace_secs=3):
    """Terminate every tracked child's process group, escalate to SIGKILL."""
    pgids = []
    for proc in list(_ACTIVE_CHILDREN):
        if proc.poll() is not None:
            continue
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            pgids.append(pgid)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.time() + grace_secs
    while time.time() < deadline:
        if all(proc.poll() is not None for proc in list(_ACTIVE_CHILDREN)):
            return
        time.sleep(0.2)
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

def _signal_handler(signum, frame):
    try:
        sig_name = signal.Signals(signum).name
    except (AttributeError, ValueError):
        sig_name = str(signum)
    live = sum(1 for p in _ACTIVE_CHILDREN if p.poll() is None)
    print(f"\n[supervisor] Received {sig_name}; terminating {live} live "
          f"child job(s) and their descendants...", flush=True)
    _kill_descendants()
    sys.exit(128 + signum)

atexit.register(_kill_descendants)

# ── Rerun-on-timeout filter ──────────────────────────────────────────────
# Module-level config set from --rerun_timeouts in main(). When enabled,
# `_parse_c_source_target_pairs` excludes (i.e. marks for re-solve) result
# rows that look like a Gurobi termination-limit timeout — but ONLY when the
# new per-cell budget (`_RERUN_TIMEOUT_BUDGET`, = the current --timeout) gives
# meaningfully more time than the cell already consumed. A cell that already
# ran for ≥ new_budget − eps got at least as much time as a fresh solve would
# give it, so re-running it would just reproduce the same timeout — those rows
# are kept as "done".
_RERUN_TIMEOUTS = False
# The new per-cell time budget (seconds) a re-solve would get, = --timeout.
# A timeout row is only worth re-running when its prior runtime is below this
# by more than _TIMEOUT_MATCH_EPS. Populated in main().
_RERUN_TIMEOUT_BUDGET = 0.0
# Timeout values (in seconds) used ONLY to identify legacy/status-less rows as
# timeouts (a row whose optimization_time sits near one of these caps but
# carries no solve_status is treated as a timeout). Populated in main():
# always includes a configurable list of common historical caps (1800, 3600)
# plus the current --timeout. The match tolerance is ±_TIMEOUT_MATCH_EPS s.
_TIMEOUT_VALUES = ()  # type: tuple[float, ...]
_TIMEOUT_MATCH_EPS = 30.0  # seconds


def _timeout_row_is_stale_rerun(status, runtime):
    """Under --rerun_timeouts, decide whether a result row should be EXCLUDED
    from the 'covered' set (i.e. re-solved) because it is a timeout that a
    fresh solve under the current --timeout could plausibly improve.

    `status` is the upper-cased solve_status ("" if absent); `runtime` is the
    row's optimization_time as a float, or None if unknown/unparseable.

    A row is treated as a timeout when its status is in the Gurobi
    termination-limit set, or — for legacy rows lacking a status — when its
    runtime sits within ±_TIMEOUT_MATCH_EPS of a known cap in _TIMEOUT_VALUES.

    A timeout is only worth re-running when the NEW budget
    (`_RERUN_TIMEOUT_BUDGET`, = current --timeout) meaningfully exceeds the
    time the cell already consumed: re-running a cell that already burned the
    same (or a larger) budget would just reproduce the timeout. When the
    runtime is unknown we cannot compare, so we err on the side of re-running
    (the historical behavior).
    """
    timeout_statuses = {
        "TIME_LIMIT", "USER_OBJ_LIMIT", "USER_LIMIT",
        "ITERATION_LIMIT", "NODE_LIMIT",
        "SOLUTION_LIMIT", "MEMORY_LIMIT", "WORK_LIMIT",
    }
    is_timeout = status in timeout_statuses
    if not is_timeout and not status and runtime is not None:
        # Only status-less (legacy/positional) rows fall back to a wall-clock
        # cap match. A row with an explicit non-timeout terminal status
        # (OPTIMAL, INFEASIBLE, ...) is definitively done and is never
        # re-run, even if it happened to finish near a round-number cap.
        is_timeout = any(abs(runtime - cap) <= _TIMEOUT_MATCH_EPS
                         for cap in _TIMEOUT_VALUES)
    if not is_timeout:
        return False  # not a timeout → counts as covered (keep)
    if runtime is None:
        return True   # runtime unknown → re-run (conservative)
    # Re-run only if the new budget gives meaningfully more time than the
    # cell already had; otherwise it's already as solved as a re-run would
    # leave it.
    return runtime < _RERUN_TIMEOUT_BUDGET - _TIMEOUT_MATCH_EPS

# ── Perturbation configs ─────────────────────────────────────────────────
# Each entry: (name, perturbation_spec)
PERTURBATIONS = [
    
    ("patch(1,14,14,3)",  "patch:1,14,14,3"),
    ("trans(1,1)",        "translation:1,1"),
    # ("trans(1,1)",        "translation:1,1"),
    ("occ(14,14,9)",        "occ:14,14,9"),
    # ("contrast(1.5)",      "contrast:1.5"),
    ("rotation(10)",      "rotation:10"),
    # ("linf(0.1)",         "linf:0.1"), 
    # ("brightness(0.25)",  "brightness:0.25"), 
    
    # ("trans(1,3)",        "translation:1,3"),
    # ("trans(3,1)",        "translation:3,1"),
    # ("trans(3,3)",        "translation:3,3"),
    # ("occ(5,5,5)",        "occ:5,5,5"),
    ("occ(3,3,5)",        "occ:3,3,5"),
    # ("occ(1,1,9)",        "occ:1,1,9"),
    # ("contrast(1.2)",      "contrast:1.2"),
    # ("rotation(5)",      "rotation:5"),
    # ("occ(1,1,5)",        "occ:1,1,5"),
    # ("linf(0.05)",        "linf:0.05"),    
    # ("brightness(0.1)",  "brightness:0.1")
]

# Perturbations for the pretrained benchmark nets (acas/har). Only linf is
# usable: every other encoder in perturbation_models.jl hard-codes the [0,1]
# domain and ignores the input box, and the geometric ones (patch/occ/
# translation/rotation) index a 2D pixel grid, which a 5-input vector does not
# have -- patch:1,14,14,3 would index element 79 of 5. eps = 1e-3 matches the
# ACAS row of the TwoSafe paper (arXiv 2606.21282, Table 1).
BENCHMARK_PERTURBATIONS = [
    # linf only. Every other encoder either indexes a 2D pixel grid (patch/occ/
    # translation/rotation -- a 5-vector has no such grid) or hardcodes the
    # [0,1] domain. eps = 1e-3 matches the ACAS row of the TwoSafe paper
    # (arXiv 2606.21282, Table 1).
    ("linf(0.01)", "linf:0.01"),
]

# Per-dataset linf radius for the benchmark nets, overriding the eps baked into
# BENCHMARK_PERTURBATIONS. The two nets live on different input domains -- ACAS
# on the .nnet header normalization (coordinate ranges well under 1) and HAR on
# [-1,1]^561 -- so one shared eps does not mean the same thing for both. Set
# from --benchmark_eps; an absent key falls back to BENCHMARK_PERTURBATIONS.
_BENCHMARK_EPS = {}


# Set from --perturbations in main(); applied inside perturbations_for so the
# filter reaches the per-dataset list rather than only the image constant.
_PERT_NAME_FILTER = None


def _num_classes_for(dataset):
    """Output-class count, so c_target/dummy_ct never exceed the net's outputs."""
    jd = _julia_dataset_name(dataset)
    if jd == "acas":
        return 5
    if jd == "har":
        return 6
    return 10


def _benchmark_perturbations_for(julia_dataset):
    """BENCHMARK_PERTURBATIONS with this dataset's --benchmark_eps applied.

    Returns BENCHMARK_PERTURBATIONS unchanged when no override was given, so
    the ACAS command lines are byte-identical to before unless asked otherwise.
    """
    eps_str = _BENCHMARK_EPS.get(julia_dataset)
    if eps_str is None:
        return BENCHMARK_PERTURBATIONS
    # Kept as the literal CLI string, never a float: eps_str is spliced
    # straight into the "eps_{eps_str}" results-directory name, so any
    # reformatting here (0.001 -> 1e-03) would silently orphan the results.
    return [(f"linf({eps_str})", f"linf:{eps_str}")]


def all_perturbations_for(dataset):
    """Every perturbation valid for `dataset`, IGNORING the --perturbations
    filter. The analysis paths scan all perturbation types on purpose, but they
    still must not scan the image list for a benchmark net: patch/occ/
    translation/rotation index a 2D pixel grid a flat input vector does not
    have, so those cells can never exist and the scan finds nothing at all."""
    _jd = _julia_dataset_name(dataset)
    return (_benchmark_perturbations_for(_jd) if _jd in ("acas", "har")
            else PERTURBATIONS)


def perturbations_for(dataset):
    """Perturbation list valid for `dataset`, after any --perturbations filter."""
    base = all_perturbations_for(dataset)
    if _PERT_NAME_FILTER:
        base = [p for p in base
                if any(p[0].lower().startswith(pf) for pf in _PERT_NAME_FILTER)]
    return base


# ── Transfer sweep parameters ────────────────────────────────────────────
THRESHOLDS = [0]#[0, 0.05] # focused on best T_relax candidate
OPT_INTERVALS = ["true"]#["true", "false"]

# ── CPU pinning ──────────────────────────────────────────────────────────
CORES_PER_JOB = 32
# First core to use (reserve 0-7). Override via SWEEP_CORE_START so two
# concurrent sweeps can claim disjoint core windows and not fight each other.
CORE_START = int(os.environ.get("SWEEP_CORE_START", "8"))
TOTAL_CORES = 255

# Optional explicit per-slot CPU ranges, e.g. SWEEP_CORE_LIST="32-63,96-127".
# Each comma-separated entry is ONE concurrency slot's `taskset -c` spec, letting
# the sweep pin to a NON-CONTIGUOUS CPU set that the CORE_START/--max_cores
# window (a single contiguous block) cannot express. When set, it OVERRIDES that
# slotting: max_slots = number of entries and --max_cores is ignored for the
# slot count. Each entry should be `cores_per_job` (32) wide to give one job a
# full slot; it is passed to `taskset -c` verbatim.
CORE_SLOTS = [p.strip() for p in os.environ.get("SWEEP_CORE_LIST", "").split(",")
              if p.strip()]


def _max_slots_for(total_cores, cores_per_job):
    """Concurrency-slot count: one slot per SWEEP_CORE_LIST entry when that
    override is set, else the contiguous CORE_START..total_cores window divided
    into cores_per_job-wide slots."""
    if CORE_SLOTS:
        return len(CORE_SLOTS)
    return (total_cores - CORE_START) // cores_per_job


def _slot_core_spec(slot_idx, cores_per_job):
    """`taskset -c` CPU spec for a concurrency slot: the explicit SWEEP_CORE_LIST
    range when set, else the contiguous window at CORE_START + slot_idx*width."""
    if CORE_SLOTS:
        return CORE_SLOTS[slot_idx]
    core_lo = CORE_START + slot_idx * cores_per_job
    return f"{core_lo}-{core_lo + cores_per_job - 1}"


def _cores_desc(cores_per_job):
    """Human-readable description of the CPU set the slots cover, for banners."""
    if CORE_SLOTS:
        return "cores " + ",".join(CORE_SLOTS)
    n = _max_slots_for(TOTAL_CORES, cores_per_job)
    return f"cores {CORE_START}-{CORE_START + n * cores_per_job - 1}"


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
    n2_dirs = [d for pat in n2_glob_patterns("vagharNoPerturbed")
               for d in glob.glob(os.path.join(eps_dir, pat))]
    for d in n2_dirs:
        if glob.glob(os.path.join(d, "*.txt")):
            return True
    return False


def _parse_c_source_target_pairs(filepath):
    """Parse a Julia result file and return the set of (c_source, c_target)
    0-indexed pairs present. Tolerates both key=value and legacy positional
    CSV lines. Returns an empty set on any parse/IO error rather than raising
    — a readable-but-empty file is treated as "no pairs covered yet".

    Lines whose `solve_status` is `INTERRUPTED` are always excluded: the
    Julia process was killed mid-solve (sweep supervisor, OOM, etc.) so the
    line carries only partial Gurobi bounds, not a real completion.

    When --rerun_timeouts is enabled (module-level `_RERUN_TIMEOUTS`),
    additionally exclude rows that look like Gurobi timeouts so the caller's
    "missing c_targets" set picks them up for a fresh solve under the new
    --timeout. A row counts as a timeout if either:
      - solve_status ∈ {TIME_LIMIT, USER_OBJ_LIMIT, USER_LIMIT,
        ITERATION_LIMIT, NODE_LIMIT, SOLUTION_LIMIT, MEMORY_LIMIT,
        WORK_LIMIT}; or
      - optimization_time is within ±_TIMEOUT_MATCH_EPS seconds of any
        entry in `_TIMEOUT_VALUES` (catches legacy rows that lack
        solve_status — see the docstring note below about positional CSV).
    A timeout row is only excluded when the new budget (current --timeout)
    meaningfully exceeds the time it already ran, and rows with an explicit
    non-timeout terminal status (OPTIMAL, INFEASIBLE, ...) are never excluded
    — see `_timeout_row_is_stale_rerun`.
    """
    pairs = set()
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = {}
                for tok in line.split(","):
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        fields[k] = v
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
                # Legacy positional CSV: source,target,incumbent_obj,solve_time,...
                # No solve_status field, so we fall back to a wall-clock match
                # against _TIMEOUT_VALUES when --rerun_timeouts is set.
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        cs = int(parts[0]); ct = int(parts[1])
                    except ValueError:
                        continue
                    if _RERUN_TIMEOUTS and len(parts) >= 5:
                        # Common legacy schema: source,target,incumbent,best_bound,solve_time
                        try:
                            runtime = float(parts[4])
                        except ValueError:
                            runtime = None
                        if (runtime is not None
                                and _timeout_row_is_stale_rerun("", runtime)):
                            continue
                    pairs.add((cs, ct))
    except OSError:
        pass
    return pairs


def _eps_str_variants(eps_str):
    """Return the set of filename eps fragments that should match `eps_str`.

    Julia formats numeric perturbation sizes via its default Float64 printer
    when writing result filenames, so an integer comma component becomes a
    float-with-".0" form: `3,3,5` -> `3.0,3.0,5.0`, `1,14,14,3` ->
    `1.0,14.0,14.0,3.0`, `0.1` stays `0.1`. The Python skip-check globs
    must match BOTH the raw `eps_str` (in case any caller writes in that
    form) AND the Julia float form (which is what run.jl actually emits).

    Returns at least {eps_str}; for parseable numeric-comma strings it also
    includes the Julia-style float form. Non-numeric eps strings fall back
    to the single-variant set.
    """
    try:
        floats = [float(tok) for tok in eps_str.split(",")]
    except ValueError:
        return {eps_str}
    julia_form = ",".join(
        (f"{int(v)}.0" if v == int(v) else repr(v)) for v in floats
    )
    return {eps_str, julia_form}


_DROP_COUNT_RE = re.compile(r"_both(\d+)_orgDrop(\d+)_pertDrop(\d+)")


def _filename_dropped_binaries(fname):
    """True if this result file's name shows the run relaxed (dropped) >=1 ReLU
    binary, which is the only condition under which the pre-fix
    perturbation_dependencies bug (missing has_a_o && has_a_p guard) could have
    corrupted the encoding.

    Signal, in priority order:
      * `_both{X}_orgDrop{Y}_pertDrop{Z}` (emitted when sibling_gate is on):
        any non-zero count => binaries were dropped.
      * else a positive relaxation threshold tag `_BTPR{t}` (stdBoost) or
        `_BoundTightPertRelax{t}` (advStd) with t>0: relaxation was enabled but
        per-tier counts weren't recorded, so conservatively treat as dropped.
      * otherwise (no boost / threshold 0): no binary dropped.

    Files that did NOT drop a binary are byte-identical under the fix, so they
    are never forced to re-run regardless of the _depGuardFix tag.
    """
    m = _DROP_COUNT_RE.search(fname)
    if m:
        return any(int(g) > 0 for g in m.groups())
    bm = re.search(r"_(?:BTPR|BoundTightPertRelax)(\d+(?:\.\d+)?)", fname)
    if bm:
        return float(bm.group(1)) > 0.0
    return False


def _is_pre_fix_dropped(fname):
    """A result is stale (must be re-run) iff it dropped binaries AND was not
    produced by the fixed encoder (no _depGuardFix tag)."""
    return _filename_dropped_binaries(fname) and "_depGuardFix" not in fname


def _advstd_missing_c_targets(cwd, dataset, arch, pert_type, eps_str, n1_tag,
                              base_name_to_save, seed, c_tag, c_targets,
                              geometric_intervals=False, perturbed_intervals=True,
                              hyper_attack=True):
    """Return subset of `c_targets` (Julia 1-indexed) for which no advstd
    result line `c_source=<c_tag-1>,c_target=<ct-1>` exists in any matching
    file for (combo=base_name_to_save, seed, c_tag).

    Output filenames written by Julia's run.jl follow the pattern:
      {hash}_n2_{arch}_{pert_type}_{eps_str}_ctag{c_tag-1}_{base_name_to_save}_seed{seed}_*.txt

    Julia formats `{eps_str}` in the filename via its Float64 printer, so we
    match both the raw and float-normalized variants via _eps_str_variants
    (see that helper). Julia accumulates all c_target results for one
    invocation into a single file (results.str is appended across the
    c_target loop), so we union across every matching file to handle
    partial crashes — a previous run that timed out after solving some
    c_targets leaves a file with those lines, and we only need to re-run
    the c_targets it didn't reach.
    """
    c_targets = [ct for ct in c_targets if ct != c_tag]
    out_dir = os.path.join(
        cwd, "paper_experiments", dataset, f"{arch}_exp",
        pert_type, f"eps_{eps_str}",
        f"advStd_{arch}_N1_{n1_tag}",
    )
    if not os.path.isdir(out_dir):
        return list(c_targets)
    src0 = c_tag - 1
    covered = set()  # 0-indexed c_target values already solved for this c_tag
    for variant in _eps_str_variants(eps_str):
        pattern = os.path.join(
            out_dir,
            f"*_n2_{arch}_{pert_type}_{variant}_ctag{src0}_"
            f"{base_name_to_save}_seed{seed}_*.txt",
        )
        for fpath in glob.glob(pattern):
            fname = os.path.basename(fpath)
            # geomInt and non-geomInt runs of the same combo share base_name_to_save,
            # so only count files whose _geomInt presence matches this run.
            if ("_geomInt" in fname) != geometric_intervals:
                continue
            # Perturbed-interval ablation runs share base_name_to_save with
            # their PI siblings too; run.jl stamps _noPI on the pi=false
            # files, so require its presence to match this run's pi setting.
            if ("_noPI" in fname) != (not perturbed_intervals):
                continue
            # Warm-start ablation (use_hyper_attack=false) shares the base
            # name with hyper-on runs; run.jl stamps _HyperAttackHints /
            # _HyperAttackCutoff only when the hyper attack ran, so require
            # the tag's presence to match this run's hyper setting.
            _has_hyper_tag = ("_HyperAttackHints" in fname
                              or "_HyperAttackCutoff" in fname)
            if _has_hyper_tag != hyper_attack:
                continue
            # A pre-fix run that relaxed binaries may have a corrupted encoding;
            # don't let it count as covering its c_targets, so it gets re-run.
            if _is_pre_fix_dropped(fname):
                continue
            for cs, ct in _parse_c_source_target_pairs(fpath):
                if cs == src0:
                    covered.add(ct)
    return [ct for ct in c_targets if (ct - 1) not in covered]


def _stdboost_filename_regex(base_name_to_save, seed, zb, sg, rt, pi, c_tag,
                             geometric_intervals=False):
    """Build a regex matching the exact name_to_save tail Julia's main_standard
    writes for a given (zb, sg, rt, pi) combo. The tail starts right after the
    user-provided --name_to_save base and runs through `_cTag{c_tag}`.

    The same regex applies to both N1 and N2 stdBoost runs — the boost machinery
    in main_standard is network-agnostic (it operates on whatever --model_path
    is passed).

    Order in main_standard (run.jl:459-665):
      base
      + `_seed{seed}` if seed != 0                          (added in main())
      + `_HyperAttackHints` if use_hyper_attack             (line 606)
      + `_VagharDeps` if activate_vaghgar_deps              (line 609)
      + `_PertruebedIntervals` if use_perturbed_intervals   (line 614) [note typo]
      + `_stdBoost` + sub-tags if any boost flag is on      (line 628-638)
          - `_zono`                if nn1_zono_bounds=true
          - `_BTPR{rt}`            if nn1_relax_threshold ≥ 0
          - `_SibGate`             if nn1_sibling_gate=true
      + `_both{X}_orgDrop{Y}_pertDrop{Z}` if sg && rt ≥ 0   (line 656-660)
      + `_cTag{c_tag}`                                      (line 662 save_results)

    The sweep always passes use_hyper_attack=true and activate_vaghgar_deps=true,
    so those two tags are constants here.
    """
    parts = [re.escape(base_name_to_save)]
    if seed != 0:
        parts.append(re.escape(f"_seed{seed}"))
    # run.jl stamps _depGuardFix right after _VagharDeps on every fixed-code dep
    # run. Accept names with OR without it here (old files lack it); whether a
    # missing tag forces a re-run is decided separately by _is_pre_fix_dropped,
    # and only for files whose name shows dropped binaries.
    parts.append(re.escape("_HyperAttackHints_VagharDeps") + r"(?:_depGuardFix)?")
    if pi == "true":
        parts.append(re.escape("_PertruebedIntervals"))
        if geometric_intervals:
            parts.append(re.escape("_geomInt"))   # main_standard adds this right after _PertruebedIntervals
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


def _stdboost_missing_c_targets(out_dir, arch, pert_type, eps_str,
                                base_name_to_save, seed, zb, sg, rt, pi,
                                c_tag, c_targets, geometric_intervals=False):
    """Return subset of `c_targets` (Julia 1-indexed) for which no
    standard-mode boost (Boosting Standard Mode §3 of advstd_techniques.tex)
    result file in `out_dir` matches the exact (zb, sg, rt, pi, seed, c_tag)
    combo for this network.

    Works for both N1 and N2 stdBoost — the caller passes the role-specific
    out_dir and base_name_to_save (e.g. N1stdBoost_{arch}_{n1_tag}/ with
    base=`{n1_tag}_N1`, or N2stdBoost_{arch}_{n2_tag}/ with base=`{n2_tag}_N2`).
    The discriminator regex from _stdboost_filename_regex is network-agnostic.

    We glob broadly by (arch, pert_type, eps_variant, ctag, base_name_to_save)
    and then filter on the filename's tag tail so files written for a
    different (zb/sg/rt/pi) configuration in the same out_dir don't
    false-skip our combo. eps_str is matched via _eps_str_variants because
    Julia writes float-form filenames.
    """
    c_targets = [ct for ct in c_targets if ct != c_tag]
    if not os.path.isdir(out_dir):
        return list(c_targets)
    src0 = c_tag - 1
    tail_re = _stdboost_filename_regex(base_name_to_save, seed, zb, sg, rt, pi, c_tag,
                                       geometric_intervals=geometric_intervals)
    covered = set()
    for variant in _eps_str_variants(eps_str):
        pattern = os.path.join(
            out_dir,
            f"*_{arch}_{pert_type}_{variant}_ctag{src0}_{base_name_to_save}_*.txt",
        )
        for fpath in glob.glob(pattern):
            fname = os.path.basename(fpath)
            if tail_re.search(fname) is None:
                continue  # different combo's file
            # Pre-fix run that relaxed binaries -> potentially corrupted; re-run.
            if _is_pre_fix_dropped(fname):
                continue
            for cs, ct in _parse_c_source_target_pairs(fpath):
                if cs == src0:
                    covered.add(ct)
    return [ct for ct in c_targets if (ct - 1) not in covered]


def _delta_max_missing_c_srcs(out_dir, c_srcs):
    """Return subset of `c_srcs` (Julia 1-indexed) for which no delta_max
    (run.jl --perturbation max) result file exists in `out_dir`.

    delta_max is a per-source-class quantity (no c_target), so we glob
    every `*ctag{c_src-1}*.txt` file in the role-specific delta_max dir
    and count the c_src as covered as soon as any file contains a
    `c_source={c_src-1}` row. Used by the delta_max pre-phase to skip
    re-running cells whose value has already been computed and persisted.
    """
    c_srcs = list(c_srcs)
    if not os.path.isdir(out_dir):
        return c_srcs
    covered = set()
    for fpath in glob.glob(os.path.join(out_dir, "*.txt")):
        if os.path.basename(fpath) == "_filename_legend.txt":
            continue
        for cs, _ct in _parse_c_source_target_pairs(fpath):
            covered.add(cs)
    return [c for c in c_srcs if (c - 1) not in covered]


def _standard_n2_missing_c_targets(pert_spec, cwd, arch, dataset, n2_tag,
                                   c_tag, c_targets):
    """Return subset of `c_targets` (Julia 1-indexed) for which no standard-N2
    (vagharWithPerturbed) result line exists for our c_tag.

    Scans every `vagharWithPerturbed_{arch}_{n2_tag}` directory (or the glob
    `vagharWithPerturbed_*_sgd_itr*` when n2_tag is None) for .txt files
    containing `ctag{c_tag-1}` in their name, then unions the
    (c_source, c_target) pairs found so partial-file runs from a crashed
    previous invocation get their missing c_targets completed rather than
    silently skipped.
    """
    c_targets = [ct for ct in c_targets if ct != c_tag]
    pert_type, eps_str = pert_spec.split(":", 1)
    eps_dir = os.path.join(cwd, "paper_experiments", dataset, f"{arch}_exp",
                           pert_type, f"eps_{eps_str}")
    if not os.path.isdir(eps_dir):
        return list(c_targets)
    if n2_tag:
        n2_dirs = glob.glob(os.path.join(eps_dir, f"vagharWithPerturbed_{arch}_{n2_tag}"))
    else:
        n2_dirs = glob_n2_dirs(eps_dir, "vagharWithPerturbed")
    src0 = c_tag - 1
    covered = set()
    for d in n2_dirs:
        for fpath in glob.glob(os.path.join(d, f"*ctag{src0}*.txt")):
            for cs, ct in _parse_c_source_target_pairs(fpath):
                if cs == src0:
                    covered.add(ct)
    return [ct for ct in c_targets if (ct - 1) not in covered]


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


def _parse_c_targets(ct_str):
    """Parse a Julia --ct string (e.g. "2,3,4,5,6,7,8,9,10") to a sorted
    deduped list of ints, preserving the caller's order otherwise."""
    seen = set()
    out = []
    for tok in ct_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _n1_state_missing_c_targets(n1_state_dir, c_tag, c_targets,
                                need_pseudocosts, need_n1_preact=False):
    """Return the subset of `c_targets` for which the per-pair N1 state is
    incomplete and the N1 solve must be (re-)run.

    Per-pair files Julia writes in save_n1_state:
        n1_vars_{c_tag}_{c_target}.bin, n1_layers_{c_tag}_{c_target}.bin
    If `need_pseudocosts` is True, a non-empty
    n1_pseudocosts_{c_tag}_{c_target}.bin is also required.

    Self-pairs (c_target == c_tag) are filtered out — Julia never produces them.

    The shared n1_preact_bounds.bin is written by save_n1_diff_bounds during
    any N1 solve in this state dir; if `need_n1_preact` and it's missing,
    re-solving any one c_target rewrites it, so all requested c_targets are
    reported as missing in that case to force a rebuild.
    """
    c_targets = [ct for ct in c_targets if ct != c_tag]
    if not os.path.isdir(n1_state_dir):
        return list(c_targets)
    if need_n1_preact and not os.path.isfile(
            os.path.join(n1_state_dir, "n1_preact_bounds.bin")):
        return list(c_targets)
    missing = []
    for ct in c_targets:
        vars_path = os.path.join(n1_state_dir, f"n1_vars_{c_tag}_{ct}.bin")
        layers_path = os.path.join(n1_state_dir, f"n1_layers_{c_tag}_{ct}.bin")
        if not (os.path.isfile(vars_path) and os.path.isfile(layers_path)):
            missing.append(ct)
            continue
        if need_pseudocosts:
            pc_path = os.path.join(
                n1_state_dir, f"n1_pseudocosts_{c_tag}_{ct}.bin")
            # Empty Julia Dict is ~101 bytes; anything <=150 is empty.
            # The uniform-priority fallback has been retired, so we treat
            # empty as incomplete and force a re-solve.
            if not os.path.isfile(pc_path) or os.path.getsize(pc_path) <= 150:
                missing.append(ct)
    return missing


def _n1_state_complete(n1_state_dir, need_pseudocosts, need_n1_preact=False,
                       c_tag=None, c_targets=None):
    """Return True if the N1 state directory already contains everything we need
    for the given `c_tag` (Julia 1-indexed source class).

    When `c_targets` is provided, requires a complete per-pair state file set
    for every requested c_target (delegates to `_n1_state_missing_c_targets`).

    Legacy path (c_targets=None): uses an any-c_target glob — any
    `n1_vars_{c_tag}_*.bin` file suffices. Kept so callers that haven't been
    updated to pass c_targets retain their previous behavior.
    `n1_preact_bounds.bin` is c_tag-agnostic (a single file per state dir).

    If both `c_tag` and `c_targets` are None we fall back further to the
    pre-multi-ctag behaviour (any `n1_vars_*.bin` file is enough).
    """
    if c_targets is not None:
        if c_tag is None:
            raise ValueError("c_targets requires c_tag")
        return not _n1_state_missing_c_targets(
            n1_state_dir, c_tag, c_targets,
            need_pseudocosts, need_n1_preact=need_n1_preact)
    if not os.path.isdir(n1_state_dir):
        return False
    if c_tag is None:
        vars_glob = "n1_vars_*.bin"
        pc_glob = "n1_pseudocosts_*.bin"
    else:
        vars_glob = f"n1_vars_{c_tag}_*.bin"
        pc_glob = f"n1_pseudocosts_{c_tag}_*.bin"
    has_vars = bool(glob.glob(os.path.join(n1_state_dir, vars_glob)))
    if not has_vars:
        return False
    if need_pseudocosts:
        pc_files = glob.glob(os.path.join(n1_state_dir, pc_glob))
        if not pc_files:
            return False
        if not any(os.path.getsize(p) > 150 for p in pc_files):
            return False
    if need_n1_preact:
        has_preact = os.path.isfile(os.path.join(n1_state_dir, "n1_preact_bounds.bin"))
        if not has_preact:
            return False
    return True


def _pid_alive(pid):
    """True if a process with this PID is running on THIS host, False if it is
    gone, None if it can't be determined (unknown/zero PID)."""
    if not pid:
        return None
    try:
        os.kill(pid, 0)
        return True            # signal deliverable -> alive
    except ProcessLookupError:
        return False           # no such process -> dead
    except PermissionError:
        return True            # exists but owned by another user -> alive
    except OSError:
        return None


def _read_lock_owner(lock_path):
    """Parse (pid, host) recorded in a .solving.lock file. Either may be None —
    older locks have no host line, and a crash mid-write can leave it empty."""
    pid = host = None
    try:
        with open(lock_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("pid="):
                    try:
                        pid = int(line[4:])
                    except ValueError:
                        pass
                elif line.startswith("host="):
                    host = line[5:]
    except (FileNotFoundError, OSError):
        pass
    return pid, host


def _acquire_n1_solve_lock(n1_state_dir, stale_after_sec):
    """Try to atomically claim the right to solve N1 for this state directory.

    Uses O_CREAT|O_EXCL via `open(path, 'x')` which is atomic across
    concurrent Python processes on the same POSIX filesystem. If another
    process already holds the lock, reclaim it when the owning process is no
    longer alive (a tombstone from a hard-killed run), or — as a cross-host
    fallback — when it is older than `stale_after_sec`.

    Returns (True, lock_path) on success — the caller is responsible for
    calling `_release_n1_solve_lock(lock_path)` after the solve finishes.
    Returns (False, lock_path) if another LIVE process holds the lock.
    """
    os.makedirs(n1_state_dir, exist_ok=True)
    lock_path = os.path.join(n1_state_dir, N1_LOCK_FILENAME)
    this_host = os.uname().nodename
    while True:
        try:
            with open(lock_path, "x") as f:
                f.write(f"pid={os.getpid()}\nhost={this_host}\nstarted={time.time()}\n")
            return True, lock_path
        except FileExistsError:
            # Liveness first: a lock whose owner process is gone is a leftover
            # from a hard-killed run — reclaim it immediately instead of waiting
            # out `stale_after_sec`. A PID is only meaningful on the host that
            # wrote it, so only trust it when the host matches (or is unknown,
            # for backward compatibility with pre-host lock files).
            owner_pid, owner_host = _read_lock_owner(lock_path)
            same_host = owner_host is None or owner_host == this_host
            if same_host and _pid_alive(owner_pid) is False:
                print(f"  reclaiming dead-owner N1 solve lock at {lock_path} "
                      f"(owner pid {owner_pid} is not running)")
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
                continue
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


def _wait_for_n1_state(n1_state_dir, need_pseudocosts, wait_timeout_sec, poll_interval_sec=30, need_n1_preact=False, c_tag=None, c_targets=None):
    """Block until another process finishes solving N1 and the state is ready
    for the given `c_tag` / `c_targets` (passed through to `_n1_state_complete`).

    Returns True if the state became ready within `wait_timeout_sec`, False
    on timeout or if the other process released its lock but left the state
    incomplete (indicates the other process crashed or errored, or solved a
    different c_target subset than the one we need).
    """
    lock_path = os.path.join(n1_state_dir, N1_LOCK_FILENAME)
    start = time.time()
    while time.time() - start < wait_timeout_sec:
        if os.path.exists(lock_path):
            time.sleep(poll_interval_sec)
            continue
        # Lock gone. Verify the state is actually complete.
        if _n1_state_complete(n1_state_dir, need_pseudocosts, need_n1_preact=need_n1_preact, c_tag=c_tag, c_targets=c_targets):
            return True
        # Lock released but state incomplete — the other process likely
        # crashed or was killed. Surface this loudly; we do not try to
        # recover silently, because continuing with partial state would
        # produce subtly wrong results.
        return False
    return False


def run_pool(ready_jobs, max_slots, cwd, cores_per_job, phase_name="",
             locked_jobs=None, on_job_done=None, priority=None,
             group_slots=None, job_group=None):
    """Run jobs with CPU-pinned slot pooling.

    ready_jobs:  list of (label, cmd) that can start immediately.
    locked_jobs: dict of  unlocker_label -> [(label, cmd), ...]
                 Dependent jobs that become eligible only after the job with
                 the matching label finishes. Pass None (default) for no
                 dependencies (original single-phase behaviour).
    on_job_done: optional callable(label) invoked once for every job as it
                 finishes (used e.g. to release per-job resources like the
                 N1 solve lock the moment its N1 job is done, rather than
                 holding the lock until the entire pool drains).
    priority:    optional callable(job)->sortable-key. When given, the ready
                 queue is sorted by this key before every slot-fill so the
                 lowest-key job is dispatched first (including jobs that have
                 just been unlocked). Pass None (default) for FIFO dispatch.
    group_slots: optional list of length max_slots assigning each slot a
                 "preferred group" key (e.g. a dataset name). When given,
                 a slot is filled with the highest-priority queued job whose
                 job_group(job) matches its preferred group; if that group has
                 nothing queued the slot SPILLS OVER to the highest-priority
                 job of any group (work-conserving — a slot never idles while
                 any job waits). Pass None (default) for ungrouped pooling.
    job_group:   callable(job)->group-key, required when group_slots is given.
    """
    grouped = group_slots is not None and job_group is not None
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
          f"({cores_per_job} cores/job, {_cores_desc(cores_per_job)})")
    if grouped:
        from collections import Counter as _Counter
        split = ", ".join(f"{g}:{n}" for g, n in _Counter(group_slots).items())
        print(f"  reserved slot split (with spillover): {split}")
    print(f"{'=' * 60}\n")

    slots = [None] * max_slots
    job_queue = list(ready_jobs)
    finished = 0

    log_dir = os.path.join(cwd, "sweep_logs")
    os.makedirs(log_dir, exist_ok=True)

    def launch_in_slot(slot_idx, label, cmd):
        core_spec = _slot_core_spec(slot_idx, cores_per_job)
        full_cmd = ["taskset", "-c", core_spec] + cmd
        print(f"  [{label:<50s}] cores {core_spec}  "
              f"({finished}/{total_jobs} done, {len(job_queue)} queued, "
              f"{sum(len(v) for v in locked_jobs.values())} locked)")
        safe_label = label.replace("/", "_").replace(" ", "_")
        log_file = open(os.path.join(log_dir, f"{phase_name}_{safe_label}.log"), "w")
        proc = subprocess.Popen(
            full_cmd, cwd=cwd,
            stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_pdeathsig_preexec,
        )
        _ACTIVE_CHILDREN.add(proc)
        slots[slot_idx] = (label, proc, log_file)

    def pick_for_slot(slot_idx):
        """Pop the best queued job for this slot. With group_slots, prefer the
        slot's owner group (highest-priority match, since job_queue is pre-sorted)
        and spill over to the global highest-priority job if the owner has none
        queued. Returns (label, cmd), or None when the queue is empty."""
        if not job_queue:
            return None
        if grouped:
            want = group_slots[slot_idx]
            for j, job in enumerate(job_queue):
                if job_group(job) == want:
                    return job_queue.pop(j)
            # owner group has nothing queued -> spill over to any group
        return job_queue.pop(0)

    # Fill initial slots from the ready queue.
    if priority:
        job_queue.sort(key=priority)
    for i in range(max_slots):
        job = pick_for_slot(i)
        if job:
            launch_in_slot(i, job[0], job[1])

    # Poll for finished jobs, unlock dependents, refill slots.
    #
    # Two passes per cycle:
    #   1. Reap finished jobs (mark their slot as None, unlock dependents,
    #      run on_job_done).
    #   2. Unified refill — fill *every* empty slot from job_queue, not just
    #      the slot that happened to finish. Without this second pass, any
    #      slot that started the cycle empty (because initial ready_jobs <
    #      max_slots, or because a previous finish had nothing in the queue
    #      yet) would stay empty for the rest of the run — even after a
    #      later unlock dumps many jobs into the queue.
    while any(s is not None for s in slots) or job_queue:
        # Pass 1: reap.
        for i in range(max_slots):
            if slots[i] is None:
                continue
            label, proc, log_file = slots[i]
            ret = proc.poll()
            if ret is None:
                continue
            log_file.close()
            status = "OK" if ret == 0 else f"EXIT {ret}"
            finished += 1

            # Unlock dependent jobs that were waiting on this job.
            if label in locked_jobs:
                unlocked = locked_jobs.pop(label)
                job_queue.extend(unlocked)
                print(f"  [{label:<50s}] finished ({status})  "
                      f"[{finished}/{total_jobs}]  "
                      f"-> unlocked {len(unlocked)} dependent job(s)")
            else:
                print(f"  [{label:<50s}] finished ({status})  "
                      f"[{finished}/{total_jobs}]")

            if on_job_done is not None:
                try:
                    on_job_done(label)
                except Exception as e:
                    print(f"    -> on_job_done({label!r}) raised: {e}")

            if ret != 0:
                print(f"    -> see log: {log_file.name}")
            _ACTIVE_CHILDREN.discard(proc)
            slots[i] = None

        # Pass 2: refill every empty slot from the queue. Keeps all cores
        # busy whenever there is work waiting, regardless of which slots
        # freed up this cycle (or were never filled in the first place).
        if priority:
            job_queue.sort(key=priority)
        for i in range(max_slots):
            if slots[i] is None:
                job = pick_for_slot(i)
                if job:
                    launch_in_slot(i, job[0], job[1])

        time.sleep(1)

    print(f"\n{phase_name}: all {total_jobs} jobs done.")


# ── dataset-name normalization ───────────────────────────────────────────
# Three layers disagree on the Fashion-MNIST name: the exp folder is
# 'fashion-mnist', the Python DATASET_CONFIG is keyed by 'fashion_mnist', and
# the Julia stack (run.jl, datasets.jl, hyper_attack.py) only knows 'fmnist'.
# These helpers translate any Fashion-MNIST spelling to the identifier each
# side expects, so --dataset can match the folder name without adding a
# DATASET_CONFIG entry.
_FASHION_ALIASES = frozenset({"fashion-mnist", "fashion_mnist", "fashion", "fmnist"})

# CIFAR-10 has the same kind of name disagreement: the exp folder is 'cifar'
# while both the Python DATASET_CONFIG and the Julia stack use 'cifar10'. These
# aliases let --dataset cifar match the folder without a new DATASET_CONFIG key.
_CIFAR_ALIASES = frozenset({"cifar", "cifar10", "cifar-10"})


def _dataset_config_key(name):
    """Key into run_experiment.DATASET_CONFIG for a (possibly aliased) dataset name."""
    if name in _FASHION_ALIASES:
        return "fashion_mnist"
    if name in _CIFAR_ALIASES:
        return "cifar10"
    return name


def _julia_dataset_name(name):
    """Dataset identifier that run.jl / hyper_attack.py understand."""
    if name in _FASHION_ALIASES:
        return "fmnist"
    if name in _CIFAR_ALIASES:
        return "cifar10"
    return name


# N2 directory suffixes. The image pipeline derives N2 by extra SGD epochs and
# tags it _sgd_itr{n}; the pretrained benchmark nets (acas/har) have no training
# data, so their N2 comes from reducing weight precision and is tagged for that
# instead. Several tools read the suffix to tell N1 from N2, so both spellings
# have to be recognised anywhere that inference is made.
N2_BENCHMARK_SUFFIX = "_int8"
N2_SUFFIXES = ("_sgd_itr", N2_BENCHMARK_SUFFIX)


def is_n2_model_name(name):
    """True if a model/result directory name denotes an N2 (target) network."""
    return any(s in name for s in N2_SUFFIXES)


def n2_glob_patterns(prefix):
    """Glob patterns matching N2 result dirs for `prefix`, across both suffixes.

    e.g. n2_glob_patterns("vagharNoPerturbed") ->
         ["vagharNoPerturbed_*_sgd_itr*", "vagharNoPerturbed_*_int8*"]
    """
    return [f"{prefix}_*{s}*" for s in N2_SUFFIXES]


def _delta_max_boost_args(julia_dataset):
    """PGD warm start + absolute-zonotope bounds for the delta_max job.

    Gated to the benchmark nets. Both are sound for the image datasets too, but
    they make delta_max solve faster, and a delta_max that previously hit the
    timeout would now reach optimality and change the normaliser behind every
    published image table cell. Widen this only with a before/after check.
    """
    if julia_dataset not in ("acas", "har"):
        return ["--use_hyper_attack", "false"]
    return ["--use_hyper_attack", "true", "--nn1_zono_bounds", "true"]


def _benchmark_args(julia_dataset):
    """--internet_nets_benchmarks for the pretrained tabular nets (acas/har).

    run.jl hard-errors for these datasets without it -- it is the master switch
    that also loads the per-coordinate input box from the <model>_box.txt
    sidecar. Returns [] for the image datasets, so their command lines are
    byte-identical to before.
    """
    return (["--internet_nets_benchmarks", "true"]
            if julia_dataset in ("acas", "har") else [])


# The wall-clock cap the benchmark sweeps were run at, used when
# --arch_timeouts carries no group for the benchmark arch. A cap has to be
# known: it decides which results the cross-cap dedup keeps, whether a bar
# counts as pinned at the timeout (and so belongs in the bounds-difference
# figure rather than the solve-time grid), and the "solver timeout of N hours"
# sentence in each caption. Naming the arch explicitly, as in
# '--arch_timeouts 7200:har:har=.', overrides this.
BENCHMARK_FORCE_TIMEOUT = 10800


def _benchmark_arch_for(dataset):
    """The architecture key of a pretrained benchmark dataset, or None.

    ACAS Xu and HAR ship one pretrained network each, so their arch key IS the
    dataset name (their results live under paper_experiments/<ds>/<ds>_exp) and
    naming the dataset already names the architecture. --arch_timeouts lists
    the image architectures, none of which exist under these datasets, so
    without this the loaders would scan only missing '<arch>_exp' dirs and the
    dataset would render "No data available" even with results on disk.
    """
    jd = _julia_dataset_name(str(dataset).strip().lower())
    return jd if jd in ("acas", "har") else None


def _benchmark_class_grid(cwd, dataset, arch):
    """The (c_targets, c_sources) a benchmark dataset was actually swept over,
    read off its result files, both 0-indexed. Returns (None, None) when the
    dataset has no results.

    This is the '@CT'/'#CS' a caller would otherwise have to write by hand. The
    per-cell tables mark a mean taken over a partial sweep with a red '*', and
    drop such rows from the paper's appendix; with no request to compare
    against they assume every class was intended, so a benchmark net swept over
    three of its target classes would be dropped as incomplete. Naming the arch
    in --arch_timeouts overrides this, since the explicit group is merged first.

    Only the target network's files count, by the same N1/N2 rule the tables
    use (is_n2_model_name, plus the advStd dirs, which are N2 by construction).
    That is what keeps a stray N1 result from widening the grid and flagging
    every row partial.
    """
    exp = os.path.join(cwd, "paper_experiments", dataset, f"{arch}_exp")
    cts, css = set(), set()
    for tf in glob.glob(os.path.join(exp, "*", "eps_*", "*", "*.txt")):
        cell = os.path.basename(os.path.dirname(tf))
        if not (cell.startswith("advStd_") or is_n2_model_name(cell)):
            continue
        try:
            parsed = parse_result_file(tf)
        except Exception:
            continue
        for cs, ct in parsed:
            css.add(int(cs))
            cts.add(int(ct))
    if not cts:
        return None, None
    return cts, css


def glob_n2_dirs(eps_dir, prefix):
    """Sorted N2 result dirs under `eps_dir` for `prefix`, across both suffixes."""
    out = []
    for pat in n2_glob_patterns(prefix):
        out.extend(glob.glob(os.path.join(eps_dir, pat)))
    return sorted(set(out))


def _save_truncated_n2(model, n1_dir, n2_dir, device):
    """Write N2 as N1 with its weights quantized to per-channel int8.

    ReluDiff produces its second network this way ("truncating each network's
    weights from 32-bit floats to 16-bit floats"); VeryDiff, which TwoSafe builds
    on, uses pruning for the same purpose. Either gives a tightly-coupled pair
    without training data, which is what the ACAS nets need since their original
    lookup-table data is not public.

    Post-training int8 quantization rather than a reduced-precision float:
    float formats preserve RELATIVE precision, so shrinking the mantissa barely
    perturbs the network (bfloat16 and fp8 both leave advisory disagreement near
    zero inside the phi1/phi2 region). Integer quantization uses a uniform
    ABSOLUTE step, which on this net's wide weight range (max |w| ~ 20) produces
    a drift comparable to the image models' N1/N2 pairs.

    Scales are PER OUTPUT CHANNEL, not per tensor. A single tensor-wide scale is
    dominated by the largest weight and flattens the small ones, giving ~59%
    disagreement -- a different network, not a perturbed one.

    The quantized values are stored back in the original dtype, so model.p /
    model.pth stay byte-compatible with every consumer.
    """
    import pickle
    import shutil

    import numpy as np
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
    from run_experiment import save_model
    from acas_box import verification_box

    def _quantize_int8_per_channel(t, bits=8):
        qmax = 2 ** (bits - 1) - 1
        if t.dim() >= 2:
            # Linear weight is (out_features, in_features): one scale per output.
            scale = t.abs().amax(dim=1, keepdim=True) / qmax
        else:
            scale = t.abs().max() / qmax
        scale = torch.clamp(scale, min=1e-30)
        return torch.round(t / scale).clamp(-qmax - 1, qmax) * scale

    with torch.no_grad():
        for p in model.parameters():
            p.copy_(_quantize_int8_per_channel(p))
    save_model(model, n2_dir)

    # save_model writes model.p as float32, but N1 came from nnet_to_pickle as
    # float64. Handing the Julia loader a mismatched pair would put a second,
    # unintended rounding on N2 on top of the float16 truncation, so re-dump at
    # N1's precision -- the truncation is meant to change values, not storage.
    params = [np.transpose(p.cpu().detach().numpy()).astype(np.float64)
              for p in model.parameters()]
    with open(os.path.join(n2_dir, 'model.p'), 'wb') as f:
        pickle.dump(params, f, protocol=2)

    # Julia derives the box sidecar from the model path, so N2 needs its own
    # copies or a --model_path2 box lookup would miss.
    for sidecar in ('model_box.txt', 'model_box.json', 'model_domain.txt'):
        src = os.path.join(n1_dir, sidecar)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(n2_dir, sidecar))

    # Report the drift, and how often it flips the advisory inside the region we
    # actually verify -- a pair that never disagrees there makes for a vacuous
    # experiment, so this is worth seeing at build time.
    n1_sd = torch.load(os.path.join(n1_dir, 'model.pth'), map_location='cpu')
    n2_sd = torch.load(os.path.join(n2_dir, 'model.pth'), map_location='cpu')
    # Report the model.p delta: that is the file Julia verifies. N1's .p is
    # float64 straight from the .nnet while its .pth is float32, so the .pth
    # delta is not quite the number that matters.
    with open(os.path.join(n1_dir, 'model.p'), 'rb') as f:
        n1_p = pickle.load(f)
    with open(os.path.join(n2_dir, 'model.p'), 'rb') as f:
        n2_p = pickle.load(f)
    max_d = max(float(np.abs(x - y).max()) for x, y in zip(n1_p, n2_p))
    print(f"  N2 = N1 quantized to per-channel int8: max |w1-w2| = {max_d:.3e} (model.p)")

    lo, hi = verification_box(n1_dir)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(
        rng.uniform(lo, hi, size=(20000, lo.size)).astype(np.float32)
    ).view(-1, 1, int(lo.size), 1).to(device)
    n1_model = type(model)()
    n1_model.load_state_dict(n1_sd)
    n1_model.to(device).eval()
    model.eval()
    with torch.no_grad():
        a = n1_model(x).argmax(dim=1)
        b = model(x).argmax(dim=1)
    disagree = int((a != b).sum())
    print(f"  advisory disagreement inside the verification box: "
          f"{disagree}/{x.shape[0]} ({100.0 * disagree / x.shape[0]:.3f}%)")


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
    ds_cls, channels, w, h, julia_ds = DATASET_CONFIG[_dataset_config_key(dataset)]

    # Accept both directory and file path (e.g. .../model_seed42_itr20 or .../model_seed42_itr20/model.p)
    model_path = os.path.normpath(model_path)
    if os.path.isfile(model_path):
        model_path = os.path.dirname(model_path)
    n1_dir = model_path
    # The benchmark nets get no SGD at all (see below), so their N2 is named for
    # what actually produced it (int8 quantization) rather than inheriting the
    # image pipeline's _sgd_itr tag. is_n2_model_name() knows both spellings.
    n2_dir = (f"{n1_dir}{N2_BENCHMARK_SUFFIX}" if julia_ds in ('acas', 'har')
              else f"{n1_dir}_sgd_itr{sgd_epochs}")

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
    model = model_cls(k=channels, w=w, h=h).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device))
    print(f"  Loaded model from {model_pth}")

    # The benchmark nets (acas/har) ship pretrained with no public training set,
    # and extra SGD cannot produce an N2 for them: any oracle labelled by N1 is
    # fit exactly by N1 at step 0, so the gradient is zero and N2 == N1. Follow
    # the differential-verification literature instead and derive N2 by reducing
    # weight precision -- ReluDiff: "We produce f' by truncating each network's
    # weights from 32-bit floats to 16-bit floats." The truncated values are kept
    # in the original dtype so every downstream consumer is unchanged.
    if julia_ds in ('acas', 'har'):
        _save_truncated_n2(model, n1_dir, n2_dir, device)
        print(f"  N1: {n1_dir}")
        print(f"  N2: {n2_dir}")
        return n1_dir, n2_dir

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
                        "sibgate_both_thin":
                            fields.get("sibgate_both_thin", ""),
                        "sibgate_one_thin_org_dropped":
                            fields.get("sibgate_one_thin_org_dropped", ""),
                        "sibgate_one_thin_pert_dropped":
                            fields.get("sibgate_one_thin_pert_dropped", ""),
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
            std_prefix = "double_check_vhagarNoPertubed"
        elif compare_to_with_perturbed:
            std_prefix = "vagharWithPerturbed"
        else:
            std_prefix = "vagharNoPerturbed"
        std_pattern = " or ".join(n2_glob_patterns(std_prefix))
        standard_n2_dirs = glob_n2_dirs(eps_dir, std_prefix)
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
    # Branch priorities: 'rank' / 'decay' are the active modes. Bare
    # '_branchPri' (legacy magic-number bounds) and '_branchPriPsd' (legacy
    # pseudocost) are retired and tagged 'bp_legacy' so the tex updater can
    # filter them out — old results are not comparable to the new modes.
    # Order matters: most-specific tag first so the bare 'branchPri' check
    # (which is a substring of all the others) doesn't mask them.
    if "branchPriRank" in filename:
        bp = "rank"
    elif "branchPriDecay" in filename:
        bp = "decay"
    elif "branchPriPsd" in filename:
        bp = "bp_legacy"
    elif "branchPri" in filename:
        bp = "bp_legacy"
    else:
        bp = "off"
    # _BoundTightPertRelax{τ} subsumes _boundTight and encodes the relaxation
    # threshold in one tag. The legacy _relaxT{τ} tag is retired — its results
    # are not comparable to the new BoundTightPertRelax (BTPR) mode because
    # the relaxation decision now uses N2's per-copy bounds instead of N1's.
    btpr_match = re.search(r"_BoundTightPertRelax([-0-9.]+)", filename)
    elim_org_match = re.search(r"_elimOrg(\d+)", filename)
    elim_pert_match = re.search(r"_elimPert(\d+)", filename)
    has_bound_tight = ("boundTight" in filename) or (btpr_match is not None)
    return {
        "mip_start": "yes" if "mipStart" in filename else "no",
        "branch_priorities": bp,
        "lp_basis": "yes" if "lpBasis" in filename else "no",
        "bound_tightening": "yes" if has_bound_tight else "no",
        # var_hint: 5-valued column.
        #   'direct_pgd' — Start-consensus variant of direct (tag '_varHintDirectPGD')
        #   'prev_pgd'   — Start-consensus variant of prev   (tag '_varHintPrevPGD')
        #   'direct'     — new rule (tag '_varHintDirect')
        #   'prev'       — previous §4.3 rule (tag '_varHintFixed', the
        #                  merged-behavior tag; still emitted today)
        #   'vh_legacy'  — bare '_varHint' or '_varHint_varHintFix' from
        #                  pre-merger code paths (not directly comparable;
        #                  excluded from auto-tables)
        #   'no'         — no varHint tag present
        # Order matters: most-specific tag first ('varHintDirectPGD' before
        # 'varHintDirect' because the former is a superstring of the latter;
        # 'varHintPrevPGD' checked explicitly to not fall through to the
        # '_varHint' legacy branch; 'varHintFixed' before the legacy fallback).
        "var_hint": ("direct_pgd" if "varHintDirectPGD" in filename
                     else "prev_pgd" if "varHintPrevPGD" in filename
                     else "direct" if "varHintDirect" in filename
                     else "prev" if "varHintFixed" in filename
                     else "vh_legacy" if ("varHintFix" in filename or "varHint" in filename)
                     else "no"),
        "zono_bounds": "yes" if "zonoBounds" in filename else "no",
        "n1_probe": "lp" if "n1ProbeLP" in filename else "off",
        "relax_threshold": btpr_match.group(1) if btpr_match else "off",
        "relax_mode": "btpr" if btpr_match else "off",
        # Technique 4 (SibGate): tag is bare '_SibGate' (no payload). Per-tier
        # neuron counts live as separate fields in the per-cell CSV — keep
        # this dict's value boolean so combo identity stays orthogonal to
        # neuron-population details.
        "sibling_gate": "yes" if "_SibGate" in filename else "no",
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
        "var_hint", "zono_bounds", "n1_probe", "relax_threshold", "relax_mode",
        "sibling_gate",
        "elim_org", "elim_pert", "relaxed_org", "relaxed_pert",
        "sibgate_both_thin", "sibgate_one_thin_org_dropped",
        "sibgate_one_thin_pert_dropped",
        "seed",
        "how_much_faster",
        "lp_optimization_time", "time_advstd_with_lp", "how_much_faster_with_lp",
        "gap_standard", "gap_advstd",
        "solve_status_standard", "solve_status_advstd",
        "standard_file", "advstd_file",
        "geom",
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
        std_prefix = ("vagharWithPerturbed" if compare_to_with_perturbed
                      else "vagharNoPerturbed")
        std_pattern = " or ".join(n2_glob_patterns(std_prefix))
        standard_n2_dirs = glob_n2_dirs(eps_dir, std_prefix)
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
                # Leave-one-out ablation runs are not paper combos — keep them
                # out of these rows and of every CSV built from them (ranking,
                # tex tables). Two markers: _ablation (all component-removed
                # combos from --advstd_ablations) and _noPI (defensive: any
                # pi=false run, e.g. manually launched without the sweep — its
                # (zb, vh, rt, sg) tags are identical to the PI run's and
                # would otherwise silently merge into the same combo row).
                if "_ablation" in tf_name or "_noPI" in tf_name:
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
                        "relax_mode": meta["relax_mode"],
                        "sibling_gate": meta["sibling_gate"],
                        "elim_org": meta["elim_org"],
                        "elim_pert": meta["elim_pert"],
                        "relaxed_org": (a_info.get("n2_org_relaxed_binaries", "")
                                        if meta["relax_threshold"] != "off"
                                        else "relax not activated"),
                        "relaxed_pert": (a_info.get("n2_pert_relaxed_binaries", "")
                                         if meta["relax_threshold"] != "off"
                                         else "relax not activated"),
                        # Per-tier SibGate neuron counts (blank when SibGate
                        # is off — Julia only writes them when the flag is on).
                        "sibgate_both_thin":
                            a_info.get("sibgate_both_thin", ""),
                        "sibgate_one_thin_org_dropped":
                            a_info.get("sibgate_one_thin_org_dropped", ""),
                        "sibgate_one_thin_pert_dropped":
                            a_info.get("sibgate_one_thin_pert_dropped", ""),
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
                        # geometric_intervals twin: same flags/delta as its
                        # baseline, differs only in solve time. Tagged via the
                        # _geomInt filename marker so the table renderer can
                        # pair base,geom and the combo-ranking can skip it.
                        "geom": "yes" if "_geomInt" in tf_name else "no",
                    }

                    s_is_timeout = s_info.get("solve_status", "") == "TIME_LIMIT"
                    a_is_timeout = a_info.get("solve_status", "") == "TIME_LIMIT"
                    if s_is_timeout and a_is_timeout:
                        if a_gap < s_gap * 0.99:
                            rows_advstd_tighter.append(row)
                        elif s_gap < a_gap * 0.99:
                            rows_standard_tighter.append(row)
                    elif a_time <= s_time:
                        row["how_much_faster"] = f"{a_time / s_time:.2f}x"
                        rows_advstd_faster.append(row)
                    else:
                        row["how_much_faster"] = f"{a_time / s_time:.2f}x"
                        rows_standard_faster.append(row)

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

    Each combo = the 9-tuple (mip_start, branch_priorities, lp_basis,
    bound_tightening, var_hint, zono_bounds, n1_probe, relax_threshold,
    sibling_gate).
    Within a combo we group the per-cell rows by (arch, perturbation, size,
    c_source, c_target, seed), dedupe duplicate runs of the same cell by
    averaging `time_advstd` and `time_standard` separately, then compute
    win/loss counts and the aggregate speedups (gm_win, gm_lose, gm_all)
    as sum(time_advstd) / sum(time_standard) over the cells in the combo
    (over the winning subset, losing subset, and all cells respectively).

    The row is classified per seed (WIN / LOSE / flip / miss) and given an
    overall label (STRICT / GENERAL / MIXED / LOSER). See the docstring of
    the calling branch for the definitions.
    """
    import csv as _csv
    from collections import defaultdict

    FLAG_FIELDS = (
        "mip_start", "branch_priorities", "lp_basis", "bound_tightening",
        "var_hint", "zono_bounds", "n1_probe", "relax_threshold", "relax_mode",
        "sibling_gate",
    )
    TC_FIELDS = ("arch", "perturbation", "perturbation_size", "c_source", "c_target")

    def _combo_key(row):
        return tuple(row[f] for f in FLAG_FIELDS)

    def _tc_key(row):
        return tuple(row[f] for f in TC_FIELDS)

    # cell_times[(combo, tc, seed)] -> list of (t_adv, t_std) pairs
    cell_times = defaultdict(list)
    cell_times_with_lp = defaultdict(list)  # (t_adv_with_lp, t_std) pairs
    archs_per_combo = defaultdict(set)
    perts_per_combo = defaultdict(set)
    ctargets_per_combo = defaultdict(set)
    elim_org_per_combo = defaultdict(list)
    elim_pert_per_combo = defaultdict(list)
    relaxed_org_per_combo = defaultdict(list)
    relaxed_pert_per_combo = defaultdict(list)
    delta_error_per_combo = defaultdict(list)
    for row in list(rows_advstd_faster) + list(rows_standard_faster):
        # geometric_intervals twins share every flag with their baseline, so
        # they would group into the same combo and double-count the win/lose
        # and geomean-time tallies. They are a solve-time annotation for the
        # table, not a separate combo -> exclude from the ranking.
        if str(row.get("geom", "no")) == "yes":
            continue
        try:
            t_std = float(row["time_standard"])
            t_adv = float(row["time_advstd"])
        except (TypeError, ValueError):
            continue
        if t_adv <= 0 or t_std <= 0:
            continue
        combo = _combo_key(row)
        cell_key = (combo, _tc_key(row), row["seed"])
        cell_times[cell_key].append((t_adv, t_std))
        # Collect with-LP times when LP time is available
        t_adv_with_lp_str = row.get("time_advstd_with_lp", "")
        if t_adv_with_lp_str:
            try:
                t_adv_with_lp = float(t_adv_with_lp_str)
                if t_adv_with_lp > 0:
                    cell_times_with_lp[cell_key].append((t_adv_with_lp, t_std))
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

    def _mean(xs):
        xs = list(xs)
        if not xs:
            return None
        return sum(xs) / len(xs)

    def _ratio(num, den):
        return (num / den) if (den and den > 0) else None

    # Dedupe each (combo, tc, seed) cell by averaging t_adv and t_std
    # separately, then bucket per combo. Each entry is
    # (tc, seed, t_adv_avg, t_std_avg, sp) where sp = t_adv_avg / t_std_avg.
    combo_cells = defaultdict(list)
    for (combo, tc, seed), pairs in cell_times.items():
        avg_adv = _mean(p[0] for p in pairs)
        avg_std = _mean(p[1] for p in pairs)
        combo_cells[combo].append((tc, seed, avg_adv, avg_std, avg_adv / avg_std))

    # Same dedup for with-LP times
    combo_cells_with_lp = defaultdict(list)
    for (combo, tc, seed), pairs in cell_times_with_lp.items():
        avg_adv_lp = _mean(p[0] for p in pairs)
        avg_std = _mean(p[1] for p in pairs)
        combo_cells_with_lp[combo].append((tc, seed, avg_adv_lp, avg_std))

    if seeds is None:
        seeds = sorted({k[2] for k in cell_times.keys()},
                       key=lambda s: int(s) if s.isdigit() else s)

    def _classify_combo(cells_list):
        by_seed = defaultdict(list)
        for _tc, seed, _adv, _std, sp in cells_list:
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
        win_cells = [(adv, std) for _, _, adv, std, sp in cells_list if sp < 1]
        lose_cells = [(adv, std) for _, _, adv, std, sp in cells_list if sp >= 1]
        all_cells = [(adv, std) for _, _, adv, std, _ in cells_list]
        all_sp = [sp for _, _, _, _, sp in cells_list]
        per_seed = _classify_combo(cells_list)
        n_win_seeds = sum(1 for v in per_seed.values() if v == "WIN")
        n_flip_seeds = sum(1 for v in per_seed.values() if v == "flip")
        n_lose_seeds = sum(1 for v in per_seed.values() if v == "LOSE")

        sum_adv_all = sum(adv for adv, _ in all_cells)
        sum_std_all = sum(std for _, std in all_cells)
        sum_adv_win = sum(adv for adv, _ in win_cells)
        sum_std_win = sum(std for _, std in win_cells)
        sum_adv_lose = sum(adv for adv, _ in lose_cells)
        sum_std_lose = sum(std for _, std in lose_cells)

        gm_all_raw = _ratio(sum_adv_all, sum_std_all)
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
        n_lose_cells = len(lose_cells)
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
        agg["n_win"] = len(win_cells)
        agg["n_lose"] = len(lose_cells)
        agg["gm_win"] = _fmt_gm(_ratio(sum_adv_win, sum_std_win))
        agg["gm_lose"] = _fmt_gm(_ratio(sum_adv_lose, sum_std_lose))
        agg["gm_all"] = _fmt_gm(gm_all_raw)
        # With-LP speedup: sum(t_advstd_with_lp) / sum(t_standard) over cells
        # that have LP time data.
        lp_cells = combo_cells_with_lp.get(combo, [])
        sum_adv_lp = sum(adv for _, _, adv, _ in lp_cells)
        sum_std_lp = sum(std for _, _, _, std in lp_cells)
        agg["gm_all_with_lp"] = _fmt_gm(_ratio(sum_adv_lp, sum_std_lp))
        agg["max_speed_up"] = _fmt_gm(min(all_sp)) if all_sp else ""
        agg["min_speed_up"] = _fmt_gm(max(all_sp)) if all_sp else ""
        agg["_gm_all_raw"] = gm_all_raw or 0.0
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
                                compare_to_with_perturbed, combo_ranking_seeds,
                                combination_table=None,
                                force_timeout=None,
                                rerun_timeout_eps=30.0):
    """Scan existing per-cell results and write the combo-ranking CSV.

    Encapsulates the logic behind --find_advstd_faster_than_standard so the
    sweep can regenerate the CSV on the fly before applying
    --advstd_safe_combos_only. Returns the path of the combo-ranking CSV.
    """
    import csv as _csv
    all_perts = all_perturbations_for(dataset)
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
        "var_hint", "zono_bounds", "n1_probe", "relax_threshold", "relax_mode",
        "sibling_gate",
        "elim_org", "elim_pert", "relaxed_org", "relaxed_pert",
        "sibgate_both_thin", "sibgate_one_thin_org_dropped",
        "sibgate_one_thin_pert_dropped",
        "seed",
        "how_much_faster",
        "lp_optimization_time", "time_advstd_with_lp", "how_much_faster_with_lp",
        "gap_standard", "gap_advstd",
        "solve_status_standard", "solve_status_advstd",
        "standard_file", "advstd_file",
        "geom",
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
                              compare_to_with_perturbed, combo_ranking_seeds,
                              combination_table=combination_table,
                              force_timeout=force_timeout,
                              rerun_timeout_eps=rerun_timeout_eps)
    return csv_combo_ranking


def _update_advstd_tex_tables(cwd, combined_base, arch_runs,
                              compare_to_with_perturbed, combo_ranking_seeds,
                              combination_table=None,
                              force_timeout=None,
                              rerun_timeout_eps=30.0):
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
    archs = [arch for arch, _ in arch_runs]
    # Dataset-scoped AUTO blocks: MNIST (the default) keeps the bare markers
    # and labels so its output is byte-identical; every other dataset
    # (e.g. fashion-mnist) writes into its OWN marker pair (name + ":<ds>")
    # with a "-<ds>" label suffix, so its tables/charts sit ALONGSIDE the
    # MNIST ones instead of overwriting them. The paper tex must contain a
    # matching marker pair for the dataset; if not, the splice is skipped.
    _ds = os.path.basename(combined_base) or "mnist"
    _default_ds = (_ds == "mnist")
    _mslug = "" if _default_ds else f":{_ds}"   # marker-name suffix
    _lslug = "" if _default_ds else f"-{_ds}"   # \label suffix
    # The combo-ranking CSV (`rows`) feeds ONLY the advstd_techniques.tex
    # safe_tables block. The nn1/wide and the AAAI per-cell tables+charts
    # below each scan the per-cell result dirs directly, so they must still
    # run even when the combo ranking is empty (e.g. a dataset that has the
    # per-cell N1/N2 runs but no standard-vs-advstd baseline pairing). Only
    # the safe_tables block is gated on `rows`.
    if not rows:
        print(f"[tex-update] safe_tables skipped (no combo rows in "
              f"{combined_base}); continuing with per-cell tables/charts")
    # Pass seed=None / tau=None so the regenerated tables include every
    # row produced by the sweep (all thresholds, all seeds); tau and seed
    # are already row columns in the tables.
    if rows:
        try:
            combination_filter = updater.parse_combination_spec(
                combination_table)
            body = updater.render_all(archs, rows, None, None,
                                      combination_filter=combination_filter)
            updater.update_tex(tex_path, body,
                               begin_mark=updater.BEGIN_MARK + _mslug,
                               end_mark=updater.END_MARK + _mslug,
                               label_suffix=_lslug)
        except SystemExit as exc:
            print(f"[tex-update] {exc}")
        except Exception as exc:
            print(f"[tex-update] error: {exc}")

    # ── Standard-mode nn1-boost section (sec:safe_nn1) ──
    # Scan _stdBoost_* result files, pair each cell with its
    # with-perturbed-intervals baseline, and rewrite the
    # % BEGIN AUTO: nn1_safe_tables block. Independent of the
    # advstd CSVs above — uses its own filename matcher and
    # iterates the same arch_runs.
    try:
        dataset_guess = os.path.basename(combined_base) or "mnist"
        if hasattr(updater, "regenerate_nn1_section"):
            updater.regenerate_nn1_section(
                tex_path, cwd, dataset_guess, arch_runs,
                parse_result_file,
                seeds_filter=combo_ranking_seeds,
                begin_mark=updater.NN1_BEGIN_MARK + _mslug,
                end_mark=updater.NN1_END_MARK + _mslug,
                ds_label_suffix=_lslug)
        else:
            print("[tex-update] updater missing regenerate_nn1_section "
                  "-- upgrade update_advstd_tex_tables.py to populate "
                  "the nn1 boosting section.")
    except Exception as exc:
        print(f"[tex-update] nn1 block error: {exc}")

    # Per-arch wide comparison section (sec:safe_wide). Scans both
    # vaghar*/N{1,2}stdBoost dirs and emits one table per arch covering
    # the 2 baselines + 6 non-duplicate stdBoost combos.
    try:
        if hasattr(updater, "regenerate_wide_perarch_section"):
            updater.regenerate_wide_perarch_section(
                tex_path, cwd, dataset_guess, arch_runs,
                parse_result_file,
                seeds_filter=combo_ranking_seeds,
                begin_mark=updater.WIDE_BEGIN_MARK + _mslug,
                end_mark=updater.WIDE_END_MARK + _mslug,
                ds_label_suffix=_lslug)
        else:
            print("[tex-update] updater missing "
                  "regenerate_wide_perarch_section -- upgrade "
                  "update_advstd_tex_tables.py to populate the per-arch "
                  "wide comparison section.")
    except Exception as exc:
        print(f"[tex-update] wide_perarch block error: {exc}")

    # AAAI paper companion: same per-arch wide tables, sliced down to
    # the 4 safe-combo columns (standard zono+sg+pi at tau=0/0.5 plus
    # advstd transfer zono+prev_pgd+sg at tau=0/0.5). The tables are split
    # by network role: the target-network (N2) tables live in the
    # Evaluation body (sec_evaluation.tex, the aaai_safe_wide_tables
    # markers) and the source-network (N1) tables live in the appendix
    # (sec_appendix_percell.tex, the aaai_safe_wide_n1_tables markers).
    # Both are sliced from the same per-cell rows as the main
    # advstd_techniques.tex tables.
    if not hasattr(updater, "regenerate_aaai_wide_perarch_section"):
        print("[tex-update] updater missing "
              "regenerate_aaai_wide_perarch_section -- upgrade "
              "update_advstd_tex_tables.py to populate the AAAI "
              "paper's safe-wide tables.")
    else:
        # N2 (target network) -> Evaluation body, as per-perturbation
        # solve-time charts (one ybar figure per arch x source class). The
        # full N2 per-cell tables now live in the appendix (below).
        try:
            body_tex = os.path.join(cwd, "neta-s-paper", "sections",
                                    "sec_evaluation.tex")
            if os.path.exists(body_tex) and hasattr(
                    updater, "regenerate_aaai_n2_charts_section"):
                updater.regenerate_aaai_n2_charts_section(
                    body_tex, cwd, dataset_guess, arch_runs,
                    parse_result_file,
                    seeds_filter=combo_ranking_seeds,
                    force_timeout=force_timeout,
                    rerun_timeout_eps=rerun_timeout_eps,
                    begin_mark=updater.AAAI_N2_CHARTS_BEGIN_MARK + _mslug,
                    end_mark=updater.AAAI_N2_CHARTS_END_MARK + _mslug,
                    ds_label_suffix=_lslug)
        except Exception as exc:
            print(f"[tex-update] aaai_n2_charts (body) block error: {exc}")

        # N2 (target network) per-cell tables -> appendix.
        try:
            n2_tex = os.path.join(cwd, "neta-s-paper", "sections",
                                  "sec_appendix_percell.tex")
            if os.path.exists(n2_tex):
                updater.regenerate_aaai_wide_perarch_section(
                    n2_tex, cwd, dataset_guess, arch_runs,
                    parse_result_file,
                    seeds_filter=combo_ranking_seeds,
                    force_timeout=force_timeout,
                    rerun_timeout_eps=rerun_timeout_eps,
                    roles={"N2"},
                    begin_mark=updater.AAAI_WIDE_N2_APPENDIX_BEGIN_MARK + _mslug,
                    end_mark=updater.AAAI_WIDE_N2_APPENDIX_END_MARK + _mslug,
                    ds_label_suffix=_lslug)
        except Exception as exc:
            print(f"[tex-update] aaai_safe_wide (N2 appendix) block "
                  f"error: {exc}")

        # N1 (source network) appendix tables are intentionally not generated:
        # neta-s-paper no longer carries any per-cell N1 tables.

    # The old per-architecture summary table (former Table 2) was removed
    # from sec_evaluation.tex in favor of the per-cell wide tables above,
    # so there is no aaai_summary_table block left to regenerate.

    # Recompile the AAAI paper so main.pdf reflects the freshly
    # regenerated sec_evaluation.tex tables.
    _recompile_neta_s_paper(cwd)


def _recompile_neta_s_paper(cwd):
    """Rebuild neta-s-paper/main.pdf with pdflatex+bibtex (latexmk absent).

    Builds under a temp jobname and atomically swaps the result into
    main.pdf only on full success. This keeps the VSCode PDF.js preview
    from ever reading a half-written file (which shows up as "Invalid PDF
    structure") and prevents a halted pass from clobbering the last-good
    main.pdf with a truncated partial.
    """
    paper_dir = os.path.join(cwd, "neta-s-paper")
    main_tex = os.path.join(paper_dir, "main.tex")
    if not os.path.exists(main_tex):
        print(f"[paper-build] skipped (missing {main_tex})")
        return
    job = "main_build"
    # Clear stale intermediates first. A run interrupted mid-write can leave a
    # half-written main_build.aux; pgfplots' `legend to name` stores the
    # legend in the .aux and reads it back at \begin{document}, so a corrupt
    # .aux makes the next build die with "File ended while scanning
    # definition of \pgfplots@legend@to@name@...". The first pdflatex pass
    # regenerates these, so deleting them up front is safe.
    for ext in ("aux", "out", "toc"):
        try:
            os.remove(os.path.join(paper_dir, f"{job}.{ext}"))
        except OSError:
            pass
    pdflatex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                "-jobname", job, "main.tex"]
    steps = [pdflatex, ["bibtex", job], pdflatex, pdflatex]
    for step in steps:
        try:
            proc = subprocess.run(step, cwd=paper_dir,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
        except FileNotFoundError:
            print(f"[paper-build] skipped ({step[0]} not installed)")
            return
        if proc.returncode != 0:
            tail = proc.stdout.decode("utf-8", "replace").splitlines()[-20:]
            print(f"[paper-build] {' '.join(step)} failed "
                  f"(exit {proc.returncode}); keeping previous main.pdf. "
                  f"Last lines:")
            for line in tail:
                print(f"  {line}")
            return
    built_pdf = os.path.join(paper_dir, f"{job}.pdf")
    if not os.path.exists(built_pdf):
        print(f"[paper-build] no {job}.pdf produced; keeping previous main.pdf")
        return
    os.replace(built_pdf, os.path.join(paper_dir, "main.pdf"))
    # Seed the plain "main" jobname's .aux/.bbl from this resolved build. An
    # editor (e.g. VSCode LaTeX Workshop) that recompiles main.tex in a single
    # pdflatex pass, with no bibtex, reads these at \begin{document}. Without
    # them main.aux has no \bibcite and main.bbl is absent, so every \cite
    # renders undefined and that stale single-pass main.pdf overwrites this
    # good one. Copying keeps the editor preview fully resolved after each
    # rebuild.
    for _ext in ("aux", "bbl"):
        _src = os.path.join(paper_dir, f"{job}.{_ext}")
        if os.path.exists(_src):
            try:
                shutil.copyfile(_src, os.path.join(paper_dir, f"main.{_ext}"))
            except OSError as _exc:
                print(f"[paper-build] could not seed main.{_ext}: {_exc}")
    print(f"[paper-build] rebuilt {os.path.join(paper_dir, 'main.pdf')}")
    _build_full_results_tex(cwd)


#: Marker pairs mirrored into table_full_results.tex, in output order. Each
#: entry is (source .tex under neta-s-paper/sections, bare marker name); every
#: dataset-scoped variant (":<ds>") of the marker is picked up automatically.
_FULL_RESULTS_BLOCKS = [
    ("sec_full_results_tables.tex", "aaai_safe_wide_n2_appendix_tables"),
    # Per-experiment ablation tables: one per (perturbation, size, c_s), every
    # cell a single run. Only in the full-results document, where a wide table
    # is fine; the Evaluation keeps the aggregated Table 3.
    ("sec_full_results_tables.tex", "ablation_full_tables"),
]

#: Hand-written floats copied verbatim into table_full_results.tex, ahead of
#: the per-cell blocks. Each entry is (source .tex under neta-s-paper/sections,
#: \label of the float). These are NOT auto-generated, so they are lifted by
#: label and travel unchanged; edit them in their own section file.
_FULL_RESULTS_FLOATS = [
    # Table 1: the FULL networks table, read from the frozen master snapshot
    # (see _TAB_NETWORKS_MASTER) rather than from sec_evaluation.tex, whose
    # copy is filtered down to the run's own models by _filter_tab_networks.
    ("tab_networks_full.tex", "tab:networks"),
]

#: Frozen full copy of Table 1. Written ONCE from the hand-written table in
#: sec_evaluation.tex and never rewritten afterwards, so table_full_results.pdf
#: keeps listing every network no matter which subset a given run sweeps. Edit
#: it by hand to change the full table.
_TAB_NETWORKS_MASTER = "tab_networks_full.tex"
_TAB_NETWORKS_BEGIN = "% BEGIN AUTO: tab_networks"
_TAB_NETWORKS_END = "% END AUTO: tab_networks"

#: --dataset key -> the Dataset column value used in Table 1.
_TAB1_DATASET_LABEL = {
    "mnist": "MNIST",
    "fashion-mnist": "Fashion-MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "cifar": "CIFAR-10",
    "cifar-10": "CIFAR-10",
    "acas": "ACAS Xu",
    "har": "HAR",
}


def _blank_tex_comments(text):
    """Replace every LaTeX comment body with spaces, keeping the text's length
    and line structure so a match offset is still valid in the original.

    Table 1 carries an explanatory comment that mentions '\\end{table}' in
    prose; a scan that does not blank comments stops the block there, and the
    resulting fragment holds no \\label -- which silently turned the Table 1
    filter and the table_full_results copy into no-ops."""
    out = []
    for line in text.split("\n"):
        cut = None
        for i, ch in enumerate(line):
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                cut = i
                break
        out.append(line if cut is None
                   else line[:cut] + " " * (len(line) - cut))
    return "\n".join(out)


def _extract_labeled_float(text, label, env="table"):
    """Return the full '\\begin{<env>} ... \\end{<env>}' block whose body
    carries '\\label{<label>}', or None. Matches both the single-column
    (`table`) and double-column (`table*`) form of `env`, so Table 1 can move
    between the two without breaking the snapshot/filter pipeline.
    table*/figure* environments do not nest, so a non-greedy scan over each
    block is unambiguous.

    The scan runs over a comment-blanked copy, so a '\\begin{table}' or
    '\\end{table}' written inside a '%' comment never opens or closes a block;
    the span is then sliced out of the ORIGINAL text, comments included."""
    esc = re.escape(env)
    pat = re.compile(r"\\begin\{" + esc + r"\*?\}.*?"
                     r"\\end\{" + esc + r"\*?\}", re.S)
    scan = _blank_tex_comments(text)
    for m in pat.finditer(scan):
        if "\\label{" + label + "}" in m.group(0):
            return text[m.start():m.end()]
    return None


def _snapshot_tab_networks(cwd):
    """Freeze the FULL Table 1 into sections/tab_networks_full.tex, once.

    The first call copies the hand-written table out of sec_evaluation.tex --
    before any filtering has touched it -- and every later call is a no-op, so
    the frozen copy keeps every network row even after the paper's own copy is
    reduced to a single run's models. Returns the master's table text, or None.
    """
    sec_dir = os.path.join(cwd, "neta-s-paper", "sections")
    master = os.path.join(sec_dir, _TAB_NETWORKS_MASTER)
    if os.path.exists(master):
        with open(master, encoding="utf-8") as fh:
            return _extract_labeled_float(fh.read(), "tab:networks")
    src = os.path.join(sec_dir, "sec_evaluation.tex")
    if not os.path.exists(src):
        return None
    with open(src, encoding="utf-8") as fh:
        block = _extract_labeled_float(fh.read(), "tab:networks")
    if block is None:
        print("[tab1] tab:networks not found in sec_evaluation.tex; "
              "no master snapshot written")
        return None
    with open(master, "w", encoding="utf-8") as fh:
        fh.write("%% FROZEN full copy of Table 1 (tab:networks), snapshotted\n"
                 "%% from sec_evaluation.tex by run_relaxation_sweep.py.\n"
                 "%% run_relaxation_sweep.py NEVER rewrites this file: it feeds\n"
                 "%% table_full_results.tex, which must keep listing every\n"
                 "%% network even when a run sweeps only some of them. The\n"
                 "%% paper's own copy in sec_evaluation.tex IS filtered per run.\n"
                 "%% Edit this file by hand to change the full table.\n"
                 "%% Not \\input by main.tex.\n\n")
        fh.write(block + "\n")
    print(f"[tab1] froze the full Table 1 into sections/{_TAB_NETWORKS_MASTER}")
    return block


def _rendered_ds_arch_pairs(cwd):
    """Return {(dataset, arch_key)} for every per-cell table actually rendered,
    read back from the unfiltered sec_full_results_tables.tex.

    This is the ground truth for Table 1: a (dataset, arch) pair only earns a
    row if the sweep produced a table for it. The cross product of --dataset
    and the --arch_timeouts archs is NOT a substitute -- it would claim, say, a
    CIFAR-10 3x50 network that this sweep has no results for.

    Labels are 'tab:safe-wide-<arch>[-solved|-timeout][-<dataset>]' and no arch
    key contains a hyphen, so the arch is the first '-' segment.
    """
    path = os.path.join(cwd, "neta-s-paper", "sections",
                        "sec_full_results_tables.tex")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    pairs = set()
    for suffix, block in _extract_auto_blocks(
            text, "aaai_safe_wide_n2_appendix_tables"):
        ds = suffix[1:] if suffix else "mnist"
        for lab in re.findall(r"\\label\{tab:safe-wide-([^}]*)\}", block):
            pairs.add((ds, lab.split("-", 1)[0]))
    return pairs


def _tab1_short_caption(block):
    """Cut Table 1's caption down to its FIRST SENTENCE (user request).

    The frozen master carries a long caption that also recounts where the HAR
    network comes from and how its target network is derived. The Setup
    paragraph already says all of that, so the paper's copy keeps only the
    opening sentence; the master is left untouched, so table_full_results.pdf
    still prints the full text.
    """
    i = block.find(r"\caption{")
    if i < 0:
        return block
    j = i + len(r"\caption{")
    depth = 1
    while j < len(block) and depth:
        if block[j] == "{":
            depth += 1
        elif block[j] == "}":
            depth -= 1
        j += 1
    if depth:
        return block
    inner = block[i + len(r"\caption{"):j - 1]
    m = re.search(r"^(.*?[.!?])(?:\s|$)", inner, re.S)
    if not m or len(m.group(1).strip()) < 10:
        return block
    return block[:i] + "\\caption{" + m.group(1).strip() + "}" + block[j:]


#: Column spec and column separation of the PAPER's Table 1. The master keeps
#: its own (wider Architecture column, a text Defense column); the paper's copy
#: is narrower because its Architecture cells are shortened below and its last
#: column holds a centred \checkmark rather than the word "PGD".
_TAB1_PAPER_COLSPEC = (r"{@{}>{\raggedright\arraybackslash}p{1.4cm}"
                       r"l>{\raggedright\arraybackslash}p{3.3cm}rc@{}}")
_TAB1_PAPER_COLSEP = "2pt"
#: A Dataset name longer than this wraps inside the p-column, so its \multirow
#: is given the column width instead of the natural width "*".
_TAB1_WRAPPING_DS = 8
_TAB1_WRAPPING_DS_WIDTH = "1.4cm"


def _tab1_short_arch(cell):
    """The master's Architecture cell in the paper's short form.

    The master spells out the per-layer width and the channel count, which the
    Setup paragraph and the \\#Neurons column already carry:
      '3 FC layers, 50/layer'            -> '3 FC layers'
      '2 CONV (stride 1), 10 ch + 2 FC'  -> '2 conv (stride 1)+2 FC'
    """
    s = cell.strip()
    s = s.replace("CONV", "conv")
    s = re.sub(r",\s*\d+\s*(?:ch|/layer)\b", "", s)
    return re.sub(r"\s*\+\s*", "+", s)


def _tab1_paper_style(block, body_font=None):
    """Restyle a filtered Table 1 into the form the paper's copy uses.

    The frozen master is the SOURCE OF TRUTH for which networks exist and what
    they are, and it is never rewritten (it also feeds table_full_results.tex,
    which wants the long form). The paper's copy differs only in presentation,
    so every difference is produced here rather than kept in a second master:
      * the explanatory comments are dropped (they document the master);
      * the Architecture cells are shortened by _tab1_short_arch;
      * the Defense column becomes a PGD column, a centred \\checkmark for a
        PGD-trained network and an empty cell otherwise;
      * consecutive rows of one dataset share a \\multirow cell, so the dataset
        is named once per group;
      * the column spec and \\tabcolsep narrow to match;
      * the master's \\small gives way to `body_font`, the size the Evaluation's
        other generated tables are pinned to, so all three carry one size. The
        narrower Architecture column above is what pays for the bigger text.
    """
    out, group = [], []

    def _flush():
        """Emit one dataset group, the first row carrying the \\multirow."""
        if not group:
            return
        ds = group[0][0]
        if len(group) == 1:
            first = ds
        else:
            width = (f"{_TAB1_WRAPPING_DS_WIDTH}"
                     if len(ds) > _TAB1_WRAPPING_DS else "*")
            first = r"\multirow{%d}{%s}{%s}" % (len(group), width, ds)
        # The continuation rows align their '&' under the first row's, which
        # is what makes the group readable in the source.
        indent = " " * (4 + len(first) + 1)
        for i, (_ds, net, arch, neurons, pgd) in enumerate(group):
            lead = "    " + first if i == 0 else indent
            out.append(" ".join([lead, "&", net, "&", arch, "&", neurons,
                                 "&", pgd]).rstrip() + r" \\")
        group.clear()

    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("%"):
            continue
        if body_font and stripped == r"{\small":
            _flush()
            out.append(line.replace(r"{\small", "{" + body_font))
            continue
        if r"\begin{tabular}" in line:
            _flush()
            # Lambda replacements: both strings are full of backslashes, which
            # re.sub would read as group references in a plain replacement.
            out.append(re.sub(r"\{@\{\}.*?@\{\}\}",
                              lambda _m: _TAB1_PAPER_COLSPEC, line))
            continue
        if r"\setlength{\tabcolsep}" in line:
            _flush()
            out.append(re.sub(r"\{\d*\.?\d+pt\}",
                              lambda _m: "{%s}" % _TAB1_PAPER_COLSEP, line))
            continue
        cells = line.split("&")
        if len(cells) >= 5 and stripped.endswith(r"\\"):
            if "textbf{Dataset}" in line:
                _flush()
                out.append(line.replace(r"\textbf{Defense}", r"\textbf{PGD}"))
                continue
            ds, net, arch, neurons, defense = [c.strip() for c in cells[:5]]
            group.append((ds, net, _tab1_short_arch(arch),
                          neurons.strip(),
                          r"\checkmark" if defense.rstrip("\\").strip() == "PGD"
                          else ""))
            continue
        _flush()
        out.append(line)
    _flush()
    return "\n".join(out)


def _filter_tab_networks(cwd, pairs, updater):
    """Rewrite Table 1 in sec_evaluation.tex so it lists ONLY the networks this
    run actually produced results for: `pairs` is {(dataset, arch_key)} from
    _rendered_ds_arch_pairs, matched against the table's Dataset column and,
    through the updater's arch display names (cnn5 -> \\emph{conv4}), its
    Network column.

    Rows are taken from the frozen master, so the filter is always applied to
    the FULL table and successive runs with different --arch_timeouts never
    erode it. A \\midrule that ends up separating nothing is dropped with the
    rows it grouped.
    """
    master_block = _snapshot_tab_networks(cwd)
    if master_block is None:
        return
    if not pairs:
        print("[tab1] no rendered per-cell table found; leaving Table 1 "
              "unchanged")
        return
    disp = getattr(updater, "_AAAI_ARCH_DISPLAY", {})
    # (Dataset column value, Network column value) for every rendered pair.
    want = {(_TAB1_DATASET_LABEL.get(d.strip().lower(), d.strip()),
             disp.get(a, a)) for d, a in pairs}

    kept, dropped, out_lines = [], [], []
    for line in master_block.splitlines():
        cells = line.split("&")
        # A data row is '<Dataset> & <Network> & ... \\'; anything else
        # (\toprule, \midrule, \caption, the header row, ...) passes through.
        if len(cells) >= 3 and line.rstrip().endswith(r"\\") \
                and "textbf{Dataset}" not in line:
            ds = cells[0].strip()
            net = cells[1].strip()
            if (ds, net) in want:
                kept.append((ds, net))
                out_lines.append(line)
            else:
                dropped.append((ds, net))
            continue
        out_lines.append(line)

    # Collapse rule runs left behind by dropped groups: a \midrule immediately
    # followed by another rule (or by \bottomrule / \end{tabular}) now groups
    # nothing.
    cleaned, pending = [], []
    for line in out_lines:
        if line.strip() in (r"\midrule",):
            pending.append(line)
            continue
        if pending:
            if not line.strip().startswith((r"\bottomrule", r"\end{tabular}")):
                cleaned.append(pending[0])
            pending = []
        cleaned.append(line)
    cleaned += pending
    new_block = _tab1_paper_style(
        _tab1_short_caption("\n".join(cleaned)),
        body_font=getattr(updater, "_TABLE_BODY_FONT", None))

    if not kept:
        print("[tab1] no Table 1 row matches this run's datasets/archs; "
              "leaving the paper's table unchanged")
        return
    sec_path = os.path.join(cwd, "neta-s-paper", "sections", "sec_evaluation.tex")
    with open(sec_path, encoding="utf-8") as fh:
        text = fh.read()
    cur = _extract_labeled_float(text, "tab:networks")
    if cur is None:
        print("[tab1] tab:networks not found in sec_evaluation.tex; skipped")
        return
    wrapped = f"{_TAB_NETWORKS_BEGIN}\n{new_block}\n{_TAB_NETWORKS_END}"
    text = text.replace(cur, wrapped, 1)
    # Drop the marker pair left by an earlier run so they never nest.
    text = re.sub(re.escape(_TAB_NETWORKS_BEGIN) + r"\n(?="
                  + re.escape(_TAB_NETWORKS_BEGIN) + r")", "", text)
    text = re.sub(re.escape(_TAB_NETWORKS_END) + r"\n"
                  + re.escape(_TAB_NETWORKS_END), _TAB_NETWORKS_END, text)
    with open(sec_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[tab1] sec_evaluation.tex Table 1: kept {len(kept)} row(s) "
          f"({', '.join(f'{d}/{n}' for d, n in kept)}); "
          f"dropped {len(dropped)}")


# The relaxation threshold the paper runs by default. The appendix tables also
# print rows for other thresholds (--paper_taus), but those exist only to show
# tau's impact, so the headline averages count the default rows alone.
PAPER_DEFAULT_TAU = 0.5


def _report_paper_table_averages(cwd):
    """Print the average speedup and average bound-gap ratio of the 'ours' and
    'ours with transfer' columns, computed over the PAPER's appendix tables
    only (sec_appendix_percell.tex), i.e. over exactly the rows the paper
    prints after the full-row filter, restricted to the DEFAULT threshold
    PAPER_DEFAULT_TAU.

    Four numbers in total: {ours, transfer} x {avg speedup, avg gap ratio}.
    The two live in different tables by construction:
      * speedup columns   -- the '-solved' tables (\\aaaisafewideheader), where
        at least one method finished, so wall-clock times carry signal and the
        cell is t_VHAGaR / t_method;
      * gap-ratio columns -- the '-timeout' tables (\\aaaisafewideheaderbd),
        where all three methods hit the cap, so the times are pinned and the
        cell compares the remaining MIP optimality gap instead,
        (delta_u - delta_l)_VHAGaR / (delta_u - delta_l)_method.
    """
    path = os.path.join(cwd, "neta-s-paper", "sections",
                        "sec_appendix_percell.tex")
    if not os.path.exists(path):
        print("[paper-averages] sec_appendix_percell.tex missing; skipped")
        return
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    def _val(cell):
        # Cells render as '1.4$\times$'; '---' means the ratio is undefined for
        # that row and is skipped (it is not a 1.0).
        m = re.match(r"^\s*([0-9.]+)\s*\$\\times\$\s*$", cell)
        return float(m.group(1)) if m else None

    def _tau(cell):
        # The tau column renders as '$0.25$'; '---' (no threshold recorded)
        # returns infinity, so such a row is never mistaken for the default.
        m = re.search(r"([0-9]*\.?[0-9]+)", cell)
        return float(m.group(1)) if m else float("inf")

    # One entry per (table, block, perturbation+size) -- i.e. per (dataset,
    # arch, c_s, pert, size), since each table is one (dataset, arch) and each
    # \midrule-separated block is one c_s. Several tau rows can share that key;
    # only the DEFAULT threshold PAPER_DEFAULT_TAU is counted, so a cell is
    # never represented more than once in the averages and the headline numbers
    # describe the configuration the paper runs by default. The extra tau rows
    # exist only to show tau's impact, and a key whose only row carries another
    # tau contributes nothing.
    picked = {}
    n_rows_off_tau = {"speedup": 0, "gap_ratio": 0}
    n_tables = {"speedup": 0, "gap_ratio": 0}
    n_rows_seen = {"speedup": 0, "gap_ratio": 0}
    for t_i, tab in enumerate(
            re.findall(r"\\begin\{table\*\}.*?\\end\{table\*\}", src, re.S)):
        # 'bd' must be tested first: \aaaisafewideheaderbd starts with
        # \aaaisafewideheader, so the plain test would match both.
        if "\\aaaisafewideheaderbd" in tab:
            metric = "gap_ratio"
        elif "\\aaaisafewideheader" in tab:
            metric = "speedup"
        else:
            continue
        n_tables[metric] += 1
        body = tab[tab.index("\\midrule"):tab.index("\\bottomrule")]
        block = 0
        for line in body.splitlines():
            if line.strip() == r"\midrule":
                block += 1          # next c_s group
                continue
            if not line.rstrip().endswith(r"\\"):
                continue
            if line.lstrip().startswith(r"\multicolumn"):
                # The table's own closing average row, not an experiment.
                # Counting it would average an average back in.
                continue
            cells = line.rstrip().rstrip("\\").split("&")
            if len(cells) < 4:
                continue
            n_rows_seen[metric] += 1
            key = (t_i, block, cells[1].strip())    # pert (size)
            tau = _tau(cells[2])
            if abs(tau - PAPER_DEFAULT_TAU) > 1e-9:
                n_rows_off_tau[metric] += 1
                continue
            picked[key] = (tau, metric, cells[-2], cells[-1])

    # {metric: {"ours": [...], "transfer": [...]}}
    vals = {"speedup": {"ours": [], "transfer": []},
            "gap_ratio": {"ours": [], "transfer": []}}
    for _tau_v, metric, ours_cell, transfer_cell in picked.values():
        for key, cell in (("ours", ours_cell), ("transfer", transfer_cell)):
            v = _val(cell)
            if v is not None:
                vals[metric][key].append(v)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "=" * 72)
    print("PAPER TABLE AVERAGE AND MAX VALUES "
          "(neta-s-paper appendix tables only)")
    print(f"restricted to the default threshold tau = {PAPER_DEFAULT_TAU} "
          f"(skipped {n_rows_off_tau['speedup']} speedup and "
          f"{n_rows_off_tau['gap_ratio']} gap-ratio rows at other thresholds)")
    print("=" * 72)
    for metric, label in (("speedup", "speed_up"),
                          ("gap_ratio", "gap_ratio")):
        for stat, fn in (("avg", _mean), ("max", max)):
            for key, col in (("ours", "ours"),
                             ("transfer", "ours with transfer")):
                xs = vals[metric][key]
                v = fn(xs) if xs else float("nan")
                print(f"  {stat}_{label:<10} [{col:<18}] = {v:7.2f}   "
                      f"(#rows = {len(xs)})")
    n_sp = len(vals["speedup"]["ours"])
    n_gp = len(vals["gap_ratio"]["ours"])
    print(f"\n  avg_speed_up  = sum{{ speed_up values over rows }}  / #rows"
          f"   (#rows = {n_sp})")
    print(f"  max_speed_up  = max{{ speed_up values over rows }}")
    print(f"  avg_gap_ratio = sum{{ gap_ratio values over rows }} / #rows"
          f"   (#rows = {n_gp})")
    print(f"  max_gap_ratio = max{{ gap_ratio values over rows }}")
    print("=" * 72)
    return vals


AAAI_RESULTS_SUMMARY_BEGIN = "% BEGIN AUTO: aaai_results_summary"
AAAI_RESULTS_SUMMARY_END   = "% END AUTO: aaai_results_summary"


def _update_results_sentence(cwd, vals):
    """Fill the Evaluation's headline sentence with the run's own numbers.

    The eight figures come from _report_paper_table_averages, i.e. they are read
    back out of the paper's APPENDIX TABLES, so the sentence can never drift
    from the tables under it and covers exactly the experiments those tables
    print. Leave-one-out ABLATION runs are not among them: `_ablation` files are
    dropped when the paper rows are loaded (both the stdBoost "ours" column and
    the advStd transfer column refuse them), so they reach neither the appendix
    tables nor these averages. The wording states what each metric IS:
      * gap_ratio -- (delta_u - delta_l)_VHAGaR / (delta_u - delta_l)_method,
        over the class pairs where every method reaches the timeout, so it
        reads as "N times smaller bound difference";
      * speedup   -- t_VHAGaR / t_method, over the pairs at least one method
        solves.
    Both are reported for BOTH modes: the "ours" column (single-network) and
    the "ours with transfer" column.
    """
    path = os.path.join(cwd, "neta-s-paper", "sections", "sec_evaluation.tex")
    if not os.path.exists(path) or not vals:
        return

    def _stat(metric, col, fn):
        xs = (vals.get(metric) or {}).get(col) or []
        return fn(xs) if xs else None

    def _mean(xs):
        return sum(xs) / len(xs)

    nums = {}
    for metric in ("gap_ratio", "speedup"):
        for col in ("ours", "transfer"):
            nums[(metric, col, "avg")] = _stat(metric, col, _mean)
            nums[(metric, col, "max")] = _stat(metric, col, max)
    if any(v is None for v in nums.values()):
        print("[results-sentence] skipped: the appendix tables carry no "
              "speedup or gap-ratio rows for both modes yet")
        return False
    g = lambda m, c, s: nums[(m, c, s)]
    # The Evaluation's own wording, with only the eight figures substituted.
    # Line breaks match the hand-written source around it.
    sentence = (
        r"The results show that, on the class pairs where at least one "
        r"verifier solves, \tool computes its" "\n"
        f"bound ${g('speedup','ours','avg'):.1f}\\times$ faster than "
        r"\baseline on average "
        f"(up to ${g('speedup','ours','max'):.1f}\\times$), and "
        f"${g('speedup','transfer','avg'):.1f}\\times$ faster" "\n"
        f"(up to ${g('speedup','transfer','max'):.1f}\\times$) with transfer. "
        r"Figure~\ref{fig:n2-bounddiff} focuses on the most challenging" "\n"
        r"instances, where all three approaches reach the three-hour timeout, "
        r"and thus shows only the tightness." "\n"
        r"There, \tool's bound difference is "
        f"${g('gap_ratio','ours','avg'):.1f}\\times$ smaller than "
        r"\baseline's on average" "\n"
        f"(up to ${g('gap_ratio','ours','max'):.1f}\\times$), and "
        f"${g('gap_ratio','transfer','avg'):.1f}\\times$ smaller "
        f"(up to ${g('gap_ratio','transfer','max'):.1f}\\times$) with transfer. " "\n"
        r"The appendix provides additional results and individual experiment "
        r"results.")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    i = text.find(AAAI_RESULTS_SUMMARY_BEGIN)
    j = text.find(AAAI_RESULTS_SUMMARY_END)
    if i < 0 or j < i:
        print(f"[results-sentence] markers not found in {path}; add "
              f"'{AAAI_RESULTS_SUMMARY_BEGIN}' / "
              f"'{AAAI_RESULTS_SUMMARY_END}' around the sentence")
        return
    new = (text[:i] + AAAI_RESULTS_SUMMARY_BEGIN + "\n" + sentence + "\n"
           + text[j:])
    if new != text:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)
        print(f"[results-sentence] wrote the headline numbers "
              f"(gap ours {g('gap_ratio','ours','avg'):.1f}x/"
              f"{g('gap_ratio','ours','max'):.1f}x, transfer "
              f"{g('gap_ratio','transfer','avg'):.1f}x/"
              f"{g('gap_ratio','transfer','max'):.1f}x; speedup ours "
              f"{g('speedup','ours','avg'):.1f}x/"
              f"{g('speedup','ours','max'):.1f}x, transfer "
              f"{g('speedup','transfer','avg'):.1f}x/"
              f"{g('speedup','transfer','max'):.1f}x)")
        return True
    return False


AAAI_CONCLUSION_SUMMARY_BEGIN = "% BEGIN AUTO: aaai_conclusion_summary"
AAAI_CONCLUSION_SUMMARY_END   = "% END AUTO: aaai_conclusion_summary"


def _update_conclusion_sentence(cwd, vals):
    """Fill the Conclusion's two headline sentences with the run's own numbers.

    Same source as _update_results_sentence -- the four AVERAGES read back out
    of the paper's appendix tables -- so the Conclusion can never quote figures
    the Evaluation and the tables do not. The two sentences cover different row
    sets, exactly as the tables do:
      * the speedup sentence averages t_VHAGaR / t_method over the '-solved'
        tables, i.e. every experiment the appendix prints where at least one
        verifier finishes;
      * the tightness sentence averages
        (delta_u - delta_l)_VHAGaR / (delta_u - delta_l)_method over the
        '-timeout' tables only, i.e. the experiments where \\baseline, \\tool
        and \\tool with transfer ALL reach the timeout.
    Both cover the DEFAULT threshold PAPER_DEFAULT_TAU alone, which the wording
    states, since the appendix also prints rows for other thresholds.
    The maxima are deliberately left out: the Conclusion states the averages.
    """
    path = os.path.join(cwd, "neta-s-paper", "sections", "sec_conclusion.tex")
    if not os.path.exists(path) or not vals:
        return False

    nums = {}
    for metric in ("speedup", "gap_ratio"):
        for col in ("ours", "transfer"):
            xs = (vals.get(metric) or {}).get(col) or []
            nums[(metric, col)] = sum(xs) / len(xs) if xs else None
    if any(v is None for v in nums.values()):
        print("[conclusion-sentence] skipped: the appendix tables carry no "
              "speedup or gap-ratio rows for both modes yet")
        return False
    g = lambda m, c: nums[(m, c)]
    # The Conclusion's own wording, with only the four averages substituted.
    # The threshold is named because the averages cover the default rows only.
    tau_txt = f"{PAPER_DEFAULT_TAU:g}"
    sentence = (
        f"\\tool speeds up "
        f"\\baseline by ${g('speedup','ours'):.1f}\\times$ "
        f"on average, and by ${g('speedup','transfer'):.1f}\\times$ with "
        "transfer.\n"
        "On the hardest tasks, where all verifiers time out, it returns "
        f"bounds ${g('gap_ratio','ours'):.1f}\\times$ tighter, and "
        f"${g('gap_ratio','transfer'):.1f}\\times$ with transfer.")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    i = text.find(AAAI_CONCLUSION_SUMMARY_BEGIN)
    j = text.find(AAAI_CONCLUSION_SUMMARY_END)
    if i < 0 or j < i:
        print(f"[conclusion-sentence] markers not found in {path}; add "
              f"'{AAAI_CONCLUSION_SUMMARY_BEGIN}' / "
              f"'{AAAI_CONCLUSION_SUMMARY_END}' around the sentences")
        return False
    new = (text[:i] + AAAI_CONCLUSION_SUMMARY_BEGIN + "\n" + sentence + "\n"
           + text[j:])
    if new != text:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)
        print(f"[conclusion-sentence] wrote the headline averages "
              f"(speedup ours {g('speedup','ours'):.1f}x, transfer "
              f"{g('speedup','transfer'):.1f}x; bound difference ours "
              f"{g('gap_ratio','ours'):.1f}x, transfer "
              f"{g('gap_ratio','transfer'):.1f}x)")
        return True
    return False


def _ensure_full_results_section(path, updater, marker_suffix=""):
    """Make sure `path` holds a BEGIN/END AUTO marker pair for this dataset,
    creating the file (or appending the dataset's pair) when missing.

    This file is never \\input by main.tex -- it exists only as the unfiltered
    render of the per-cell tables, which _build_full_results_tex wraps into the
    standalone table_full_results.tex.
    """
    begin = updater.AAAI_WIDE_N2_APPENDIX_BEGIN_MARK + marker_suffix
    end = updater.AAAI_WIDE_N2_APPENDIX_END_MARK + marker_suffix
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        text = ("%% AUTO-GENERATED by run_relaxation_sweep.py -- DO NOT EDIT.\n"
                "%% Unfiltered render of the per-cell result tables (rows with\n"
                "%% partial c_target coverage INCLUDED, red \"*\" and all). The\n"
                "%% paper appendix renders the same tables with those rows\n"
                "%% dropped; this file feeds table_full_results.tex only and is\n"
                "%% not \\input by main.tex.\n")
    if begin in text:
        return
    text = text.rstrip("\n") + f"\n\n{begin}\n{end}\n"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError as exc:
        print(f"[full-results] could not prepare {path}: {exc}")


def _ensure_auto_marker_pair(path, begin, end):
    """Append a BEGIN/END AUTO marker pair to an EXISTING `path` if absent, so
    the updater -- which raises SystemExit when its markers are missing -- can
    write this dataset's block instead of bailing. Unlike
    _ensure_full_results_section this adds no file header, so it is safe for
    files that ARE \\input by main.tex (sec_evaluation.tex charts,
    sec_appendix_percell.tex). Without this, a NEW dataset (e.g. HAR) has no
    per-dataset marker in those files, so its charts cells are dropped from the
    combined figure and its rows never reach the main-paper appendix -- while
    sec_full_results_tables.tex (which IS ensured) still shows them.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return
    if begin in text:
        return
    text = text.rstrip("\n") + f"\n\n{begin}\n{end}\n"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError as exc:
        print(f"[paper-tables] could not add AUTO markers to {path}: {exc}")


def _parse_ablation_expected(entries):
    """Parse --ablation_expected into
    {(dataset_or_None, arch): (ct0s, css, tau_or_None, perts_or_None)}.

    ct0s:  frozenset of 0-indexed c_targets (input is Julia-indexed, like --ct).
    css:   frozenset of 0-indexed source classes, or None when the '#CS' part is
           omitted (meaning: whatever sources appear in the data).
    tau:   '~TAU' pins this table's unified relaxation threshold to exactly
           that value; omitted (None) keeps the automatic per-table choice
           (the --paper_taus candidate with the most cells).
    perts: frozenset of (perturbation, size_or_None) from the trailing
           '/PERT(SIZE)' clause, '+'-separated for several. None when omitted,
           meaning every perturbation the ablation variants have data for. A
           selector without a size ('/patch') takes every size of it.
    """
    out = {}
    for entry in entries or ():
        spec = entry.strip()
        if not spec:
            continue
        m = re.match(r"^(?:([A-Za-z0-9_.\-]+):)?([A-Za-z0-9_]+)"
                     r"@([0-9,]+)(?:#([0-9,]+))?(?:~([0-9.]+))?"
                     r"(?:/(.+))?$", spec)
        if not m:
            print(f"ERROR: --ablation_expected entry '{entry}' must be "
                  f"[dataset:]arch@CT[,CT...][#CS[,CS...]][~TAU][/PERT(SIZE)] "
                  f"(CT Julia-indexed like --ct; CS 0-indexed; ~TAU pins "
                  f"the table's unified relaxation threshold; /PERT(SIZE) "
                  f"scopes it to one perturbation, '+'-separated for several).")
            sys.exit(1)
        ds, arch, cts, css, tau, perts = m.groups()
        ct0s = frozenset(int(x) - 1 for x in cts.split(",") if x)
        cs = frozenset(int(x) for x in css.split(",") if x) if css else None
        out[(ds, arch)] = (ct0s, cs, float(tau) if tau else None,
                           _parse_ablation_perts(entry, perts))
    return out


def _parse_ablation_perts(entry, spec):
    """The '/PERT(SIZE)[+PERT(SIZE)...]' clause as {(pert, size_or_None)}."""
    if not spec:
        return None
    out = set()
    for part in spec.split("+"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\(([^)]*)\))?$", part)
        if not m:
            print(f"ERROR: --ablation_expected entry '{entry}': "
                  f"'{part}' must be PERT or PERT(SIZE), e.g. "
                  f"patch(1,14,14,3).")
            sys.exit(1)
        out.add((m.group(1), m.group(2)))
    return frozenset(out) or None


def _extract_auto_blocks(text, marker):
    """Yield (suffix, block_text) for every '% BEGIN/END AUTO: <marker>[:ds]'
    pair in `text`, in file order. The suffix is "" for the default (MNIST)
    block and ":<dataset>" for a dataset-scoped one."""
    pat = re.compile(
        r"^% BEGIN AUTO: " + re.escape(marker) + r"(:[^\n]*)?\n"
        r"(.*?)"
        r"^% END AUTO: " + re.escape(marker) + r"(?::[^\n]*)?$",
        re.S | re.M)
    for m in pat.finditer(text):
        yield (m.group(1) or ""), m.group(2)


def _build_full_results_tex(cwd):
    """Mirror the auto-generated per-cell tables into a STANDALONE
    neta-s-paper/table_full_results.tex and compile it to
    table_full_results.pdf.

    The tables themselves are never re-derived here: the AUTO blocks that
    _regen_paper_tables_from_txt just wrote into the appendix sections are
    copied verbatim, so the two documents can never drift. The preamble is
    main.tex's own preamble (everything before \\begin{document}), which keeps
    the AAAI two-column table* floats, \\baseline/\\tool and the adjustbox /
    multirow / colortbl setup identical to the paper.
    """
    paper_dir = os.path.join(cwd, "neta-s-paper")
    main_tex = os.path.join(paper_dir, "main.tex")
    if not os.path.exists(main_tex):
        return
    with open(main_tex, encoding="utf-8") as fh:
        main_src = fh.read()
    preamble = main_src.split("\\begin{document}", 1)[0]

    chunks, n_blocks = [], 0
    # Hand-written floats first (Table 1 and friends), so the reader meets the
    # networks before the per-cell numbers about them.
    for fname, label in _FULL_RESULTS_FLOATS:
        src = os.path.join(paper_dir, "sections", fname)
        if not os.path.exists(src):
            continue
        with open(src, encoding="utf-8") as fh:
            float_tex = _extract_labeled_float(fh.read(), label)
        if float_tex is None:
            print(f"[full-results] {label} not found in sections/{fname}; "
                  f"skipped")
            continue
        chunks.append(f"%% ---- {label} (copied from sections/{fname}) ----\n"
                      f"{float_tex}")
        n_blocks += 1

    # HAR-only random-input confidence figures: a delta_d stand-in (HAR has no
    # dataset). Computed once here, then placed RIGHT AFTER the HAR tables in
    # the loop below. Appended to this standalone document ONLY, not the paper's
    # own appendix. No-op when HAR has no results or torch/matplotlib is absent.
    har_fig_chunk = ""
    try:
        sys.path.insert(0, cwd)
        import update_advstd_tex_tables as _har_updater
        _har_fig_tex = _har_updater.render_har_confidence_figures(cwd)
        if _har_fig_tex:
            har_fig_chunk = ("%% ---- har_confidence_figures (auto-generated) "
                             "----\n" + _har_fig_tex)
    except Exception as exc:
        print(f"[full-results] HAR confidence figures skipped ({exc})")
    har_fig_emitted = False

    for fname, marker in _FULL_RESULTS_BLOCKS:
        src = os.path.join(paper_dir, "sections", fname)
        if not os.path.exists(src):
            continue
        with open(src, encoding="utf-8") as fh:
            body = fh.read()
        for suffix, block in _extract_auto_blocks(body, marker):
            ds = suffix[1:] if suffix else "mnist"
            chunks.append(f"%% ---- {marker} [{ds}] (copied from "
                          f"sections/{fname}) ----\n"
                          f"\\section*{{{ds}}}\n{block}")
            n_blocks += 1
            # The confidence figures sit immediately after the HAR tables, and
            # ONLY when a HAR table was actually rendered (i.e. `har` was in
            # --dataset). Without this coupling the figure would draw from the
            # on-disk HAR results even on runs that produced no HAR table,
            # leaving a data-less table beside a data-full figure.
            if ds == "har" and har_fig_chunk and not har_fig_emitted:
                chunks.append(har_fig_chunk)
                n_blocks += 1
                har_fig_emitted = True

    if har_fig_chunk and not har_fig_emitted:
        print("[full-results] HAR figures suppressed: no HAR table in this run "
              "(add 'har' to --dataset to generate the HAR tables + figures)")

    if not chunks:
        print("[full-results] no AUTO table blocks found; "
              "table_full_results.tex not written")
        return

    # \label/\ref inside the copied blocks point at labels that live in the
    # paper; they resolve here too because the blocks carry their own \label,
    # and anything they reference outside just prints "??" in this side file.
    out_tex = os.path.join(paper_dir, "table_full_results.tex")
    with open(out_tex, "w", encoding="utf-8") as fh:
        fh.write("%% AUTO-GENERATED by run_relaxation_sweep.py -- DO NOT EDIT.\n"
                 "%% Standalone copy of the per-cell result tables that the\n"
                 "%% paper prints in its appendix. Regenerated on every paper\n"
                 "%% rebuild; hand edits are overwritten.\n")
        fh.write(preamble)
        fh.write("\\begin{document}\n\n")
        fh.write("\n\n".join(chunks))
        fh.write("\n\n\\end{document}\n")
    print(f"[full-results] wrote {out_tex} ({n_blocks} table block(s))")

    job = "table_full_results"
    for ext in ("aux", "out", "toc"):
        try:
            os.remove(os.path.join(paper_dir, f"{job}.{ext}"))
        except OSError:
            pass
    # Two passes so the \ref/\label cross-references inside the tables settle.
    # No -halt-on-error: unresolved references to the paper's own labels are
    # expected in this side document and must not abort the build.
    step = ["pdflatex", "-interaction=nonstopmode", f"{job}.tex"]
    for _ in range(2):
        try:
            proc = subprocess.run(step, cwd=paper_dir,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
        except FileNotFoundError:
            print(f"[full-results] skipped compile ({step[0]} not installed)")
            return
    if not os.path.exists(os.path.join(paper_dir, f"{job}.pdf")):
        tail = proc.stdout.decode("utf-8", "replace").splitlines()[-20:]
        print(f"[full-results] no {job}.pdf produced. Last lines:")
        for line in tail:
            print(f"  {line}")
        return
    print(f"[full-results] built {os.path.join(paper_dir, job + '.pdf')}")


def _check_repeat_arch(arch, prev_path, prev_ct, prev_cs,
                       path, group_ct, group_cs, where):
    """Validate a SECOND --arch_timeouts group for an architecture already
    listed, exiting with an explanation when the repeat is not a per-source-
    class cap split.

    Repeating an arch is only meaningful to give its source classes different
    caps -- MNIST's 3x100 ran c_s 0 and 4 for five hours and c_s 1 for three,
    and one group carries one cap. So each repeat must name a DISJOINT '#CS'
    set, and everything else about the architecture (its model path and its
    '@CT' grid) must agree, since those are per-arch and not per-source-class.
    """
    if not prev_cs or not group_cs:
        print(f"ERROR: --arch_timeouts lists arch '{arch}' more than once"
              f"{where} without a '#CS' list on every group. Repeating an arch "
              f"is only allowed to give different source classes different "
              f"caps, so each group must name the classes its cap applies to.")
        sys.exit(1)
    both = sorted(set(prev_cs) & set(group_cs))
    if both:
        print(f"ERROR: --arch_timeouts gives arch '{arch}'{where} two caps for "
              f"source class(es) {both}; the '#CS' lists of repeated groups "
              f"must be disjoint.")
        sys.exit(1)
    if prev_path != path:
        print(f"ERROR: --arch_timeouts lists arch '{arch}'{where} with two "
              f"model paths ({prev_path!r} and {path!r}); a repeated arch "
              f"splits caps by source class, not by model.")
        sys.exit(1)
    if (prev_ct or set()) != (group_ct or set()):
        print(f"ERROR: --arch_timeouts lists arch '{arch}'{where} with two "
              f"'@CT' target-class grids; a repeated arch splits caps by "
              f"source class and must keep one target grid.")
        sys.exit(1)


def _parse_arch_timeouts(spec):
    """Parse --arch_timeouts into (arch_runs, force_timeout_by_arch,
    c_targets_by_arch, c_sources_by_arch, dataset_scoped).

    Grammar: '|'-separated groups, each 'SECS:arch=path,arch=path[@CT][#CS]':
      * SECS     -- wall-clock cap in seconds, shared by the group's models.
      * arch=path,... -- same syntax as --arch_models. An entry may be
                    DATASET-SCOPED as '<dataset>:<arch>=path', e.g.
                    'cifar:cnn1=.', in which case its cap, '@CT' and '#CS'
                    apply to that dataset ONLY and override any unscoped entry
                    for the same arch. This is what lets one arch carry a
                    different class grid per dataset -- CIFAR's cnn1 was swept
                    on '@3,4,6#0,1' while MNIST's cnn1 used '@2,4,5#0,2', and
                    a single per-arch grid cannot describe both.
      * @CT      -- OPTIONAL per-group target-class list, Julia 1-indexed and
                    comma-separated (same syntax as --ct), e.g. '@2,4,5'.
      * #CS      -- OPTIONAL per-group SOURCE-class (Cs) list, 0-indexed (the
                    exact 'c_s=k' chart labels, NOT shifted -- same syntax as
                    --cs), e.g. '#0,2'. Restricts which source classes the
                    generated tables/charts include, mirroring '@CT' for
                    targets.
      The '@...'/'#...' suffixes are stripped off the END of the group first
      (in any order) so they never collide with the arch=path ':'/'=' seps.

    An architecture may be listed in SEVERAL groups, which is how its source
    classes get different caps: MNIST's 3x100 ran c_s 0 and 4 for five hours and
    c_s 1 for three, and one group carries one cap. Each repeat must then name a
    disjoint '#CS' set and keep the same model path and '@CT' grid, see
    _check_repeat_arch.

    Returns:
      * arch_runs           -- list of (arch, model_path) pairs in spec order.
      * force_timeout_by_arch -- {arch: seconds}, plus a {(arch, c_s): seconds}
                               entry per source class whenever an arch was
                               listed more than once. The bare arch key stays
                               the fallback for classes no group named.
      * c_targets_by_arch   -- {arch: set_of_0indexed_targets | None}, or None
                               overall when NO group carried a '@CT' (so the
                               caller keeps using the global --ct unchanged). A
                               group without '@CT' maps its archs to None.
      * c_sources_by_arch   -- {arch: set_of_0indexed_sources | None}, or None
                               overall when NO group carried a '#CS' (caller
                               keeps using the global --cs). A group without
                               '#CS' maps its archs to None.
      * dataset_scoped      -- {dataset: {arch: {"path", "secs", "ct", "cs"}}}
                               for the '<dataset>:<arch>=path' entries, empty
                               when none were used. Resolve it per dataset with
                               _arch_setup_for_dataset; the four values above
                               describe the UNSCOPED entries only, so a caller
                               that ignores this stays on the old behavior.

    Combines --arch_models, a per-model --force_timeout, a per-model --ct, and a
    per-model --cs so different models can carry different caps, target-class
    sets, and source-class sets."""
    arch_runs = []
    ft_by_arch = {}
    ct_by_arch = {}
    cs_by_arch = {}
    ds_scoped = {}
    any_group_ct = False
    any_group_cs = False
    for group in spec.split("|"):
        group = group.strip()
        if not group:
            continue
        # Optional per-group target-class ('@CT') and source-class ('#CS')
        # lists. Both are stripped off the END of the group, in whatever order
        # they appear, so the arch=path body is left clean for the ':'/'='
        # parsing below.
        group_ct = None
        group_cs = None
        while True:
            m = re.search(r"([@#])([0-9,\s]+)$", group)
            if not m:
                break
            sym, lst = m.group(1), m.group(2).strip()
            kind = "ct" if sym == "@" else "cs"
            try:
                raw = _parse_c_targets(lst)
            except ValueError:
                print(f"ERROR: --arch_timeouts {kind} list after '{sym}' must "
                      f"be comma-separated integers, got: {lst!r}")
                sys.exit(1)
            # '@CT' is Julia 1-indexed (shifted to the 0-indexed c_target the
            # result files store); '#CS' is already the 0-indexed source class
            # shown as "c_s=k" on the chart axis, so it is NOT shifted.
            parsed = {c - 1 for c in raw} if sym == "@" else set(raw)
            if not parsed:
                print(f"ERROR: --arch_timeouts group '{group}{sym}{lst}' has an "
                      f"empty {kind} list after '{sym}'")
                sys.exit(1)
            if sym == "@":
                group_ct = parsed
                any_group_ct = True
            else:
                group_cs = parsed
                any_group_cs = True
            group = group[:m.start()].strip()
        if ":" not in group:
            print(f"ERROR: --arch_timeouts group must be 'SECS:arch=path,...', "
                  f"got: {group}")
            sys.exit(1)
        secs_str, models_part = group.split(":", 1)
        try:
            secs = int(secs_str.strip())
        except ValueError:
            print(f"ERROR: --arch_timeouts timeout must be an integer number of "
                  f"seconds, got: {secs_str.strip()!r} (in '{group}')")
            sys.exit(1)
        for pair in models_part.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                print(f"ERROR: --arch_timeouts entry must be arch=model_path, "
                      f"got: {pair} (in '{group}')")
                sys.exit(1)
            a, mp = pair.split("=", 1)
            a = a.strip()
            # '<dataset>:<arch>' scopes this entry to one dataset. The arch
            # itself never contains ':', so the split is unambiguous.
            ds_key = None
            if ":" in a:
                ds_key, a = (p.strip() for p in a.split(":", 1))
                ds_key = ds_key.lower()
            if ds_key is not None:
                prev = ds_scoped.get(ds_key, {}).get(a)
                if prev is not None:
                    _check_repeat_arch(a, prev["path"], prev["ct"], prev["cs"],
                                       mp.strip(), group_ct, group_cs,
                                       f" for dataset '{ds_key}'")
                    prev["cs"] = set(prev["cs"]) | set(group_cs)
                    prev["secs_by_cs"].update({c: secs for c in group_cs})
                    continue
                ds_scoped.setdefault(ds_key, {})[a] = {
                    "path": mp.strip(), "secs": secs,
                    "ct": group_ct, "cs": group_cs,
                    "secs_by_cs": ({c: secs for c in group_cs}
                                   if group_cs else {})}
                continue
            if a in ft_by_arch:
                _check_repeat_arch(a, dict(arch_runs).get(a), ct_by_arch[a],
                                   cs_by_arch[a], mp.strip(), group_ct,
                                   group_cs, "")
                cs_by_arch[a] = set(cs_by_arch[a]) | set(group_cs)
                for c in group_cs:
                    ft_by_arch[(a, c)] = secs
                continue
            arch_runs.append((a, mp.strip()))
            ft_by_arch[a] = secs
            ct_by_arch[a] = group_ct
            cs_by_arch[a] = group_cs
            for c in (group_cs or ()):
                ft_by_arch[(a, c)] = secs
    if not arch_runs and not ds_scoped:
        print("ERROR: --arch_timeouts listed no arch=model_path pairs")
        sys.exit(1)
    return (arch_runs, ft_by_arch,
            (ct_by_arch if any_group_ct else None),
            (cs_by_arch if any_group_cs else None),
            ds_scoped)


def _arch_setup_for_dataset(dataset, arch_runs, ft_by_arch, ct_by_arch,
                            cs_by_arch, ds_scoped, cwd=None):
    """Resolve the dataset-scoped '<dataset>:<arch>=path' entries for one
    dataset, returning (arch_runs, force_timeout, c_targets, c_sources) with the
    scoped values layered over the unscoped ones.

    A scoped entry REPLACES the unscoped entry for the same arch (keeping its
    position, so the table order does not change) and is appended when the arch
    is otherwise absent. Datasets with no scoped entry get the unscoped setup
    unchanged.

    The pretrained benchmark datasets (acas/har) are then given their own arch
    for free, see _benchmark_arch_for.
    """
    scoped = (ds_scoped or {}).get(str(dataset).strip().lower(), {})
    runs = list(arch_runs)
    ft = dict(ft_by_arch or {})
    ct = dict(ct_by_arch) if ct_by_arch is not None else None
    cs = dict(cs_by_arch) if cs_by_arch is not None else None
    for arch, cfg in scoped.items():
        for i, (a, _p) in enumerate(runs):
            if a == arch:
                runs[i] = (arch, cfg["path"])
                break
        else:
            runs.append((arch, cfg["path"]))
        # A scoped entry replaces the unscoped one wholesale, per-source-class
        # caps included, so the unscoped arch's (arch, c_s) keys go first.
        for k in [k for k in ft
                  if isinstance(k, tuple) and k and k[0] == arch]:
            del ft[k]
        ft[arch] = cfg["secs"]
        for c, c_secs in (cfg.get("secs_by_cs") or {}).items():
            ft[(arch, c)] = c_secs
        if cfg["ct"] is not None:
            ct = dict(ct or {})
            ct[arch] = cfg["ct"]
        if cfg["cs"] is not None:
            cs = dict(cs or {})
            cs[arch] = cfg["cs"]
    if scoped:
        print(f"[arch-scope] {dataset}: applied dataset-scoped entries for "
              + ", ".join(sorted(scoped)))
    bench = _benchmark_arch_for(dataset)
    if bench is not None and not any(a == bench for a, _ in runs):
        runs.append((bench, "."))
        ft.setdefault(bench, BENCHMARK_FORCE_TIMEOUT)
        b_ct, b_cs = ((None, None) if cwd is None
                      else _benchmark_class_grid(cwd, dataset, bench))
        if b_ct:
            ct = dict(ct or {})
            ct[bench] = b_ct
            cs = dict(cs or {})
            cs[bench] = b_cs
        print(f"[arch-scope] {dataset}: added its own architecture '{bench}' "
              f"(the dataset ships a single pretrained network, so naming the "
              f"dataset names the architecture), capped at {ft[bench]}s"
              + (f", swept over c_targets (Julia-indexed) "
                 f"{sorted(c + 1 for c in b_ct)} and c_sources / Cs "
                 f"(0-indexed) {sorted(b_cs)}" if b_ct else ""))
    return runs, ft, ct, cs


def _layer_global_request(per_arch, global_request, arch_runs):
    """Give every arch of `arch_runs` missing from the per-arch class request
    the global one (from --ct / --cs).

    A benchmark dataset turns a global request into a per-arch dict, since its
    own arch is swept over a grid of its own (_benchmark_class_grid). Without
    this the other archs of that dataset would silently lose the global
    restriction, which no per-arch entry replaces.
    """
    if not isinstance(per_arch, dict) or isinstance(global_request, dict):
        return per_arch
    if global_request is None:
        return per_arch
    out = dict(per_arch)
    for a, _ in arch_runs:
        out.setdefault(a, global_request)
    return out


def _regen_paper_tables_from_txt(arch_runs, cwd, dataset, combo_ranking_seeds,
                                 combination_table=None, force_timeout=None,
                                 rerun_timeout_eps=30.0,
                                 requested_c_targets=None,
                                 requested_c_sources=None,
                                 paper_taus=None,
                                 paper_chart_taus=None,
                                 recompile=True,
                                 ablation_tables=False,
                                 ablation_expected=None):
    """Regenerate ONLY the neta-s-paper per-cell tables + N2 charts, sourcing
    the transfer (advstd N2) column DIRECTLY from the advStd .txt files (no
    CSVs, no standard-baseline pairing). Honors --combination_table (combo
    filter) and --force_timeout (cross-cap timeout dedup) exactly like the old
    --find_advstd path, but writes no CSV and does not touch
    advstd_techniques.tex.

    `requested_c_targets` (a set of 0-indexed target classes, or None) comes
    from --ct: when set, every generated table/chart is restricted to exactly
    those target classes, and a requested c_target with no data is flagged with
    the same red "*" the tables already use for partial sweeps.

    `requested_c_sources` (a set/dict of 0-indexed source classes, or None) is
    the Cs / c_tag analogue from --cs / '#CS': when set, every table/chart is
    restricted to exactly those source classes, and a requested Cs with no data
    shows up as a "missing Cs=k" note in the bar charts.
    """
    try:
        sys.path.insert(0, cwd)
        import update_advstd_tex_tables as updater
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[paper-tables] skipped (import failed: {exc})")
        return
    if not hasattr(updater, "_load_advstd_rows_for_wide_from_txt"):
        print("[paper-tables] skipped: update_advstd_tex_tables.py is missing "
              "_load_advstd_rows_for_wide_from_txt (upgrade it).")
        return
    try:
        combination_filter = updater.parse_combination_spec(combination_table)
    except SystemExit as exc:
        print(f"[paper-tables] {exc}")
        return

    # --paper_taus owns the tau dimension for these tables; --combination_table
    # owns the technique (bt / varHint / SibGate). They do not overlap: a
    # tau-free spec carries a wildcard tau that is expanded against the tau
    # list here, so the admitted combos are exactly the cross product and the
    # two flags cannot disagree.
    _taus = updater._AAAI_WIDE_TAUS
    if paper_taus and hasattr(updater, "set_aaai_wide_taus"):
        _taus = updater.set_aaai_wide_taus(
            [t for t in str(paper_taus).split(",") if t.strip()])
    print(f"[paper-tables] per-cell tables render tau rows: "
          f"{', '.join(_taus)}")
    _chart_taus = updater._AAAI_CHART_TAUS
    if paper_chart_taus and hasattr(updater, "set_aaai_chart_taus"):
        _chart_taus = updater.set_aaai_chart_taus(
            [t for t in str(paper_chart_taus).split(",") if t.strip()])
    print(f"[paper-tables] charts draw tau series: "
          f"{', '.join(_chart_taus)} "
          f"({1 + 2 * len(_chart_taus)} bars per cluster)")
    if combination_filter is not None and hasattr(
            updater, "expand_combination_spec_taus"):
        # Expand over the UNION so a tau drawn only in the charts (or shown
        # only as a table row) still passes the combo filter.
        _union = list(_taus) + [t for t in _chart_taus if t not in _taus]
        combination_filter = updater.expand_combination_spec_taus(
            combination_filter, _union)
        print(f"[paper-tables] combos admitted: "
              f"{updater._format_combination_filter(combination_filter)}")

    # Dataset-scoped AUTO markers: MNIST (the default) keeps the bare markers
    # and labels; every other dataset writes into its OWN ":<ds>" marker pair
    # with a "-<ds>" label suffix (same convention as _update_advstd_tex_tables).
    _default_ds = (dataset == "mnist")
    _mslug = "" if _default_ds else f":{dataset}"
    _lslug = "" if _default_ds else f"-{dataset}"

    # Shared kwargs: the txt-direct transfer source + the seed/combo/timeout
    # filters. parse_result_file and _extract_advstd_file_metadata are this
    # module's functions; PERTURBATIONS gives the exact same discovery paths
    # the old --find_advstd command walked.
    common = dict(
        parse_result_file=parse_result_file,
        seeds_filter=combo_ranking_seeds,
        force_timeout=force_timeout,
        rerun_timeout_eps=rerun_timeout_eps,
        advstd_meta_fn=_extract_advstd_file_metadata,
        perts=perturbations_for(dataset),
        combination_filter=combination_filter,
        # Restrict every table/chart to the requested target classes (--ct);
        # None keeps the full-sweep behavior.
        requested_c_targets=requested_c_targets,
        # Same for the source classes (Cs / c_tag) via --cs / '#CS'.
        requested_c_sources=requested_c_sources,
        # Drop pre-fix files that relaxed >=1 binary (unsound under the
        # perturbation-dependency fix) on BOTH vaghar/ours and transfer rows --
        # the same predicate the sweep skip-check uses.
        stale_fn=_is_pre_fix_dropped,
    )

    # N2 (target network) -> Evaluation body, as per-perturbation charts. The
    # solve-time cells AND the bounds-difference clusters are returned so the
    # caller can pool BOTH across datasets into one combined figure each.
    time_cells, bd_groups = [], []
    body_tex = os.path.join(cwd, "neta-s-paper", "sections", "sec_evaluation.tex")
    # Ensure this dataset's chart marker exists, else regenerate_aaai_n2_charts_
    # section raises (SystemExit) on the missing marker and its time/bd cells are
    # lost -- so a new dataset (HAR) never reaches the combined figures.
    if os.path.exists(body_tex) and hasattr(updater, "AAAI_N2_CHARTS_BEGIN_MARK"):
        _ensure_auto_marker_pair(
            body_tex, updater.AAAI_N2_CHARTS_BEGIN_MARK + _mslug,
            updater.AAAI_N2_CHARTS_END_MARK + _mslug)
    if os.path.exists(body_tex) and hasattr(
            updater, "regenerate_aaai_n2_charts_section"):
        try:
            time_cells, bd_groups = updater.regenerate_aaai_n2_charts_section(
                body_tex, cwd, dataset, arch_runs,
                begin_mark=updater.AAAI_N2_CHARTS_BEGIN_MARK + _mslug,
                end_mark=updater.AAAI_N2_CHARTS_END_MARK + _mslug,
                ds_label_suffix=_lslug, **common)
        except Exception as exc:
            print(f"[paper-tables] aaai_n2_charts (body) block error: {exc}")

    # N2 (target network) per-cell tables -> appendix. (N1 source tables are
    # intentionally not regenerated in the --paper_tables_from_txt path.)
    percell_tex = os.path.join(cwd, "neta-s-paper", "sections",
                               "sec_appendix_percell.tex")
    # The SAME tables are rendered twice from the same data, differing only in
    # the partial-coverage filter:
    #   * sec_full_results_tables.tex -- every row, including the ones carrying
    #     the red "*" (feeds the standalone table_full_results.pdf);
    #   * sec_appendix_percell.tex    -- only rows with no red "*" in any cell,
    #     i.e. means taken over every expected c_target.
    full_tex = os.path.join(cwd, "neta-s-paper", "sections",
                            "sec_full_results_tables.tex")
    _ensure_full_results_section(full_tex, updater, _mslug)
    # The main-paper appendix needs the same per-dataset marker (the ensure
    # above only covers sec_full_results_tables.tex); without it this dataset's
    # rows are dropped from main.pdf's appendix.
    _ensure_auto_marker_pair(
        percell_tex, updater.AAAI_WIDE_N2_APPENDIX_BEGIN_MARK + _mslug,
        updater.AAAI_WIDE_N2_APPENDIX_END_MARK + _mslug)
    _drop_supported = hasattr(updater, "set_aaai_wide_drop_partial_rows")
    for _target, _drop in ((full_tex, False), (percell_tex, True)):
        if not os.path.exists(_target):
            continue
        _prev = (updater.set_aaai_wide_drop_partial_rows(_drop)
                 if _drop_supported else None)
        try:
            updater.regenerate_aaai_wide_perarch_section(
                _target, cwd, dataset, arch_runs, roles={"N2"},
                begin_mark=updater.AAAI_WIDE_N2_APPENDIX_BEGIN_MARK + _mslug,
                end_mark=updater.AAAI_WIDE_N2_APPENDIX_END_MARK + _mslug,
                ds_label_suffix=_lslug, **common)
        except Exception as exc:
            print(f"[paper-tables] aaai_safe_wide (N2, {os.path.basename(_target)}) "
                  f"block error: {exc}")
        finally:
            if _drop_supported:
                updater.set_aaai_wide_drop_partial_rows(_prev)

    # --ablation_tables: the Evaluation's ablation table (Table 3) is
    # regenerated from the _ablation result files on EVERY run that passes the
    # flag, exactly like the other AUTO blocks. It is NOT frozen or
    # hand-maintained: edits to its numbers, its headers or its caption belong
    # in _render_ablation_table.
    if ablation_tables and hasattr(updater,
                                   "regenerate_ablation_appendix_section"):
        # Only the datasets named in --ablation_expected render the block. The
        # markers are not dataset-scoped, so a dataset with no _ablation files
        # would overwrite the table with "no result files found" on a later
        # turn of the per-dataset loop and the table would vanish.
        _abl_ds = {d for (d, _a) in (ablation_expected or {}) if d}
        if not _abl_ds or dataset in _abl_ds:
            try:
                updater.regenerate_ablation_appendix_section(
                    body_tex, cwd, dataset, arch_runs,
                    parse_result_file=parse_result_file,
                    seeds_filter=combo_ranking_seeds,
                    stale_fn=_is_pre_fix_dropped,
                    expected_map=ablation_expected,
                    taus=_taus)
            except Exception as exc:
                print(f"[ablation-tables] block error: {exc}")
            if hasattr(updater, "regenerate_ablation_full_section"):
                try:
                    updater.regenerate_ablation_full_section(
                        full_tex, cwd, dataset, arch_runs,
                        parse_result_file=parse_result_file,
                        seeds_filter=combo_ranking_seeds,
                        stale_fn=_is_pre_fix_dropped,
                        expected_map=ablation_expected,
                        taus=_taus)
                except Exception as exc:
                    print(f"[ablation-full] block error: {exc}")

    # Per-network relaxation + precision-cost rows for the single Evaluation
    # table. Pooled across datasets by the caller and emitted once, like the
    # combined figures.
    relax_rows = []
    if hasattr(updater, "collect_aaai_relax_precision_rows"):
        try:
            relax_rows = updater.collect_aaai_relax_precision_rows(
                cwd, dataset, arch_runs, **common)
        except Exception as exc:
            print(f"[paper-tables] aaai_relax_precision collect error: {exc}")

    # Both combined figures (solve-time + bounds-difference) are written once,
    # after ALL datasets are collected, so the caller defers the compile
    # (recompile=False) and returns the pooled data.
    if recompile:
        _recompile_neta_s_paper(cwd)
    return time_cells, bd_groups, relax_rows


def _regen_paper_combined_from_txt(cwd, time_cells, bd_groups,
                                   relax_rows=None,
                                   force_timeout=None):
    """Emit the SINGLE combined solve-time grid (columns = networks) AND the
    SINGLE combined bounds-difference figure, pooling `time_cells`/`bd_groups`
    across every dataset, then recompile. Run once after the per-dataset
    _regen_paper_tables_from_txt calls."""
    try:
        sys.path.insert(0, cwd)
        import update_advstd_tex_tables as updater
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[paper-tables] combined figures skipped (import failed: {exc})")
        return
    body_tex = os.path.join(cwd, "neta-s-paper", "sections", "sec_evaluation.tex")
    if os.path.exists(body_tex) and hasattr(
            updater, "regenerate_aaai_time_combined_section"):
        try:
            updater.regenerate_aaai_time_combined_section(
                body_tex, time_cells, force_timeout=force_timeout)
        except Exception as exc:
            print(f"[paper-tables] aaai_n2_time_combined block error: {exc}")
    # The perturbations the body does not chart go to the appendix, from the
    # same pooled cells.
    appendix_tex = os.path.join(cwd, "neta-s-paper", "sections",
                                "sec_appendix_percell.tex")
    if os.path.exists(appendix_tex) and hasattr(
            updater, "regenerate_aaai_time_appendix_section"):
        try:
            updater.regenerate_aaai_time_appendix_section(
                appendix_tex, time_cells, force_timeout=force_timeout)
        except Exception as exc:
            print(f"[paper-tables] aaai_n2_time_appendix block error: {exc}")
    # Same split for the bounds-difference figures: the body keeps the
    # selected cells, the appendix takes the rest.
    if os.path.exists(appendix_tex) and hasattr(
            updater, "regenerate_aaai_bounddiff_appendix_section"):
        try:
            updater.regenerate_aaai_bounddiff_appendix_section(
                appendix_tex, bd_groups, force_timeout=force_timeout)
        except Exception as exc:
            print(f"[paper-tables] aaai_n2_bounddiff_appendix block error: "
                  f"{exc}")
    if os.path.exists(body_tex) and hasattr(
            updater, "regenerate_aaai_bounddiff_section"):
        try:
            # float_spec="t": the two Evaluation figures must not share a
            # float-only page (see the note in main.tex -- the placement
            # specifier is the sanctioned lever, not \dbltopnumber).
            updater.regenerate_aaai_bounddiff_section(
                body_tex, bd_groups, force_timeout=force_timeout,
                float_spec="t")
        except Exception as exc:
            print(f"[paper-tables] aaai_n2_bounddiff block error: {exc}")
    # The relaxation / precision-cost table: one table for every dataset.
    if os.path.exists(body_tex) and hasattr(
            updater, "regenerate_aaai_relax_precision_section"):
        try:
            updater.regenerate_aaai_relax_precision_section(
                body_tex, relax_rows or [])
        except Exception as exc:
            print(f"[paper-tables] aaai_relax_precision block error: {exc}")
    _recompile_neta_s_paper(cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--perturbations", nargs="*", default=None,
                        help="Filter perturbations by name prefix (e.g. 'patch' 'occ' 'trans' 'rotation')")
    parser.add_argument("--benchmark_eps", nargs="*", default=None,
                        help="Per-dataset linf radius for the pretrained benchmark nets, as "
                             "<dataset>=<eps> (e.g. 'har=0.01'). Only linf is encodable for "
                             "acas/har, and the two live on different input domains (ACAS on the "
                             ".nnet header normalization, HAR on [-1,1]^561), so one shared eps "
                             "does not mean the same thing for both. The value is used verbatim "
                             "in the eps_<value> results directory, so pass it exactly as it "
                             "should appear on disk. Without this the built-in "
                             "BENCHMARK_PERTURBATIONS radius is used.")
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
                        help="Dataset name (default: mnist). With "
                             "--find_advstd_faster_than_standard, a comma-separated "
                             "list (e.g. 'mnist,fashion-mnist') regenerates the tables "
                             "for every dataset in turn, reusing the same --arch_models.")
    parser.add_argument("--arch_models", nargs="*", default=None,
                        help="Run multiple architectures, each with its own model path. "
                             "Format: arch=model_path (e.g. cnn1=/path/to/cnn1_model cnn2=/path/to/cnn2_model). "
                             "Overrides --arch and --model_path when specified.")
    parser.add_argument("--dataset_group", action="append", default=None,
                        help="(advanced_standard only) Run MULTIPLE datasets in a single "
                             "merged job pool, interleaving their row priority. Repeatable, "
                             "once per dataset. Format: 'dataset|arch=path,arch=path'. "
                             "e.g. --dataset_group 'mnist|cnn1=/p/cnn1,3x50=/p/3x50' "
                             "--dataset_group 'fashion-mnist|3x10=/p/3x10,cnn1=/p/cnn1'. "
                             "All other flags (--sweep_ctag, --ct, techniques, etc.) are "
                             "shared across groups. Overrides --dataset/--arch_models.")
    parser.add_argument("--dataset_slots", type=str, default=None,
                        help="(with --dataset_group) Reserve a slot split per dataset, "
                             "comma-separated in --dataset_group order, e.g. '2,5' = 2 slots "
                             "for the 1st dataset, 5 for the 2nd. Reserved slots give each "
                             "dataset guaranteed concurrency; idle slots SPILL OVER to the "
                             "other dataset's remaining jobs (work-conserving, no idle cores). "
                             "Default: split the available slots as evenly as possible.")
    parser.add_argument("--find_transfer_faster_than_standard", action="store_true",
                        help="Scan existing results and report transfer experiments that are "
                             "faster than standard N2 (vagharNoPerturbed with sgd) for each "
                             "perturbation and (c_source, c_target) pair.")
    parser.add_argument("--find_advstd_faster_than_standard", action="store_true",
                        help="Scan existing results and report advanced-standard N2 experiments "
                             "that are faster than regular standard N2 for each perturbation "
                             "and (c_source, c_target) pair.")
    parser.add_argument("--paper_tables_from_txt", action="store_true",
                        help="Regenerate ONLY the neta-s-paper per-cell tables + N2 charts, "
                             "sourcing the transfer (advstd N2) column DIRECTLY from the advStd "
                             ".txt files -- no CSVs and no standard-baseline pairing, so advStd "
                             "results are never dropped for want of a vaghar baseline. Honors "
                             "--combo_ranking_seeds (seed filter), --combination_table (combo "
                             "filter), --force_timeout (cross-cap timeout dedup), --arch_models "
                             "and --dataset. Does not write CSVs or touch advstd_techniques.tex.")
    parser.add_argument("--ablation_expected", nargs="*", default=None,
                        metavar="[DS:]ARCH@CT[,CT..][#CS[,CS..]][~TAU]",
                        help="The planned grid, render filter and tau pin for "
                             "--ablation_tables: only the (dataset, arch) pairs listed here "
                             "render a table, '#CS' and '@CT' give the grid each variant's "
                             "mean is expected to cover, and '~TAU' pins the threshold. "
                             "E.g. 'fashion-mnist:cnn1@2,4#0~0.5'.")
    parser.add_argument("--ablation_tables", action="store_true",
                        help="Regenerate the leave-one-out ablation table (Table 3) in "
                             "sec_evaluation.tex from the _ablation result files, and the "
                             "per-experiment ablation tables in sec_full_results_tables.tex. "
                             "Use --ablation_expected to select which (dataset, arch) pairs "
                             "render and to pin the threshold.")
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
    parser.add_argument("--force_timeout", type=int, default=None,
                        metavar="SECS",
                        help="When set, the AAAI wide-comparison tables in "
                             "neta-s-paper/sections/sec_appendix_percell.tex exclude any "
                             "cell whose Gurobi run hit TIME_LIMIT under a different "
                             "wall-clock cap than this value (in seconds). Cells that "
                             "finished optimally are always included regardless. The "
                             "match tolerance is ±--rerun_timeout_eps seconds (default "
                             "30s). Useful when re-running the sweep after lowering "
                             "--timeout so older 1-hour timed-out cells don't pollute "
                             "the means.")
    parser.add_argument("--arch_timeouts", type=str, default=None,
                        metavar="SECS:arch=path,...[@CT]|SECS2:arch=path,...[@CT]",
                        help="Combine --arch_models, a PER-MODEL --force_timeout, and "
                             "an optional PER-MODEL --ct in one flag: '|'-separated "
                             "groups, each 'SECS:arch=path,arch=path[@CT]' sharing one "
                             "wall-clock cap (seconds); the arch=path list uses the "
                             "same syntax as --arch_models, and the optional trailing "
                             "'@CT' gives that group its own comma-separated, Julia "
                             "1-indexed target-class list (same syntax as --ct). "
                             "Example: '10800:3x50=.,cnn1=.@2,4,5|1800:3x100=.@2,4'. "
                             "A group without '@CT' falls back to the global --ct. "
                             "Overrides --arch_models and --force_timeout (and, for "
                             "@CT groups, --ct). Honored by --paper_tables_from_txt "
                             "(per-model ct) and --find_advstd_faster_than_standard "
                             "(per-model cap): every model renders in one pass, each "
                             "filtered/clamped to its own cap and target-class set. "
                             "An arch may be listed in several groups to give its "
                             "SOURCE classes different caps, e.g. "
                             "'18000:3x100=.@3,4,6#0,4|10800:3x100=.@3,4,6#1'; each "
                             "repeat must then carry a disjoint '#CS' list and keep "
                             "the same model path and '@CT' grid.")
    parser.add_argument("--combination_table", type=str, default=None,
                        metavar="BT:VH[:TAU][,BT:VH[:TAU],...]",
                        help="Restrict the advstd tables in advstd_techniques.tex (overall "
                             "ranking, per-arch c_src-tinted green/yellow/pink blocks, and "
                             "per-arch TIME_LIMIT gap-comparison tables) to one or more "
                             "combinations. Format '<bound_tight>:<varHint>[:<tau>]' per "
                             "combo, comma-separated. PREFER the tau-free form, e.g. "
                             "'zono:prev_pgd+sg': it names only the technique and lets "
                             "--paper_taus name the thresholds, so the two flags stay "
                             "orthogonal and the combos read are their cross product. The "
                             "explicit-tau form ('zono:prev_pgd:0.5+sg') is still accepted. "
                             "Use the '+sg' suffix to select SibGate (Technique 4) rows. "
                             "Only takes "
                             "effect when the tex tables are rewritten (after "
                             "--find_advstd_faster_than_standard). 'off' is accepted as an "
                             "alias for varHint='no'.")
    parser.add_argument("--paper_taus", type=str, default=None,
                        metavar="TAU[,TAU,...]",
                        help="Relaxation thresholds the neta-s-paper per-cell appendix "
                             "tables render, as ROWS (one row per tau per cell), in the "
                             "given order. Default '0.0,0.5'; pass e.g. "
                             "'0.0,0.25,0.5' to also pull in the tau=0.25 runs. This "
                             "widens the tables vertically only -- the tau column names "
                             "which threshold each row is -- and never adds a column. A "
                             "tau with no result files simply yields no rows. This flag "
                             "owns the tau dimension; --combination_table owns the "
                             "technique (bt/varHint/+sg), and the combos actually read "
                             "are their cross product. Only affects "
                             "--paper_tables_from_txt.")
    parser.add_argument("--paper_chart_taus", type=str, default=None,
                        metavar="TAU[,TAU,...]",
                        help="Relaxation thresholds the neta-s-paper CHARTS draw, as an "
                             "(ours, ours with transfer) BAR PAIR per tau, on top of the "
                             "\\baseline bar. Default '0.5' (3 bars per cluster), which "
                             "reproduces the existing figures exactly. Kept separate from "
                             "--paper_taus because a cluster only fits a few bars before "
                             "the per-bar value labels stop fitting, whereas a table just "
                             "grows another row: e.g. '0.25,0.5' gives 5 bars. The legend "
                             "names the tau only when more than one is drawn. Only affects "
                             "--paper_tables_from_txt.")
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
                        help="Values for adv_std_branch_priorities: off | rank | decay "
                             "(legacy true/false accepted; legacy 'bounds'/'pseudocost' map to 'rank' with a warning). "
                             "Default: ['off'].")
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
    parser.add_argument("--sweep_adv_std_n2_sibling_gate", nargs="*", type=str, default=None,
                        help="Values for adv_std_n2_sibling_gate (booleans 'true'/'false'). "
                             "Default: ['false']. When 'true', Technique 4 augments Technique 3's "
                             "tiered relaxation with a sibling-gated conditional triangle (one-thin "
                             "tier) and a pre-activation coupling line (both-thin tier). Sound "
                             "over-approximation: delta_relaxed >= delta_exact. Requires "
                             "adv_std_n2_relax_threshold >= 0; combos that pair sibling_gate=true "
                             "with relax_threshold=off are auto-pruned.")
    parser.add_argument("--sweep_adv_std_var_hint", nargs="*", type=str, default=None,
                        help="Values for adv_std_var_hint. Modes: off | prev | direct | direct_pgd | prev_pgd. "
                             "'prev' = previous §4.3 rule (shift ẑ^N1 by diff bound, clip to [l_n2, u_n2]); "
                             "'direct' = new rule (p derived from [l_n2, u_n2] directly, no ẑ^N1 proxy); "
                             "'direct_pgd' = same p as 'direct', but hint_val is routed via "
                             "set_start_value() with PGD consensus filtering (fill-where-silent / "
                             "leave-where-agree / withdraw-where-disagree) and VarHintVal/VarHintPri "
                             "are not set; "
                             "'prev_pgd' = Start-consensus routing applied to 'prev's p_i. "
                             "Legacy 'true'/'false' still accepted (map to prev/off). Default: ['off']. "
                             "Variable hints are orthogonal to branch priorities.")
    parser.add_argument("--sweep_gurobi_seed", nargs="*", type=int, default=None,
                        help="Gurobi seeds to sweep (e.g. '0 1 2 3 4') for variance measurement. Default: [0].")
    parser.add_argument("--include_nn1_boost", action="store_true",
                        help="In --advanced_standard mode: also queue N1 standard-mode boost jobs "
                             "alongside the advstd-N2 sweep. Each job runs --mode standard on the N1 "
                             "model. The boost grid is controlled by --sweep_stdboost_* flags (zono "
                             "bounds, sibling gate, perturbed intervals) plus the non-negative entries "
                             "of --sweep_adv_std_n2_relax_threshold for the per-copy triangle drop. "
                             "Mirrors Section 3 (Boosting Standard Mode) of advstd_techniques.tex. "
                             "Results land in N1stdBoost_{arch}_{n1_tag}/ dirs. Skip-check uses "
                             "_stdboost_missing_c_targets, which discriminates between combos by the "
                             "exact filename tag sequence.")
    parser.add_argument("--include_nn2_boost", action="store_true",
                        help="In --advanced_standard mode: also queue N2 standard-mode boost jobs "
                             "(same boost machinery as --include_nn1_boost, but applied to the N2 model, "
                             "i.e. N1 + sgd_epochs). Identical combo grid via --sweep_stdboost_* and "
                             "non-negative entries of --sweep_adv_std_n2_relax_threshold. Results land "
                             "in N2stdBoost_{arch}_{n2_tag}/ dirs. Independent of --include_nn1_boost; "
                             "enable either or both.")
    parser.add_argument("--sweep_stdboost_zono_bounds", nargs="*", type=str, default=None,
                        help="Values for nn1_zono_bounds ('true'/'false') used in the stdBoost sweep "
                             "for BOTH --include_nn1_boost and --include_nn2_boost. The Julia flag is "
                             "named --nn1_zono_bounds in run.jl but its mechanics are network-agnostic "
                             "(it operates on whatever --model_path is passed). Default: ['true']. "
                             "See §3.2 of advstd_techniques.tex (Absolute Zonotope Bound Tightening).")
    parser.add_argument("--sweep_stdboost_sibling_gate", nargs="*", type=str, default=None,
                        help="Values for nn1_sibling_gate ('true'/'false') used in the stdBoost sweep "
                             "for BOTH --include_nn1_boost and --include_nn2_boost. Default: ['true']. "
                             "See §3.4 of advstd_techniques.tex (SibGate).")
    parser.add_argument("--sweep_stdboost_perturbed_intervals", nargs="*", type=str, default=None,
                        help="Values for use_perturbed_intervals ('true'/'false') used in the stdBoost "
                             "sweep for BOTH --include_nn1_boost and --include_nn2_boost. Default: "
                             "['true']. Required true for soundness when nn1_relax_threshold > 0 "
                             "(unsound combo guard at run.jl:487-494) — those combos are auto-pruned.")
    parser.add_argument("--stdboost_combos", type=str, default=None,
                        metavar="ROLE:ZB:SG:RT:PI[,ROLE:ZB:SG:RT:PI...]",
                        help="Explicit comma-separated list of stdBoost combos to run, "
                             "bypassing the Cartesian product of --sweep_stdboost_* and "
                             "--include_nn{1,2}_boost. Each entry is "
                             "<role>:<zb>:<sg>:<rt>:<pi> with role in {N1,N2}, zb/sg/pi in "
                             "{true,false}, rt a float (>=0). Soundness filters still apply "
                             "(rt>0 requires pi=true; sg=true requires rt>=0). Example: "
                             "'N1:false:false:0:false,N2:false:false:0:false,"
                             "N1:true:true:0:true,N1:true:true:0.5:true'.")
    parser.add_argument("--advstd_ablations", nargs="*", default=None,
                        metavar="COMPONENT",
                        help="Leave-one-out ablation mode for the advstd-N2 ('transfer') jobs. "
                             "Bypasses the Cartesian product: the single-valued --sweep_adv_std_* "
                             "flags define the BASE combo, and each listed component spawns one "
                             "combo with exactly that component removed. Components: "
                             "none (base combo as-is, the control) | "
                             "var_hint (adv_std_var_hint=off, PGD warm start KEPT — removes the "
                             "N1-derived variable hints only) | "
                             "warm_start (adv_std_var_hint=off AND use_hyper_attack=false — no "
                             "warm start at all; its files carry no _HyperAttackHints tag, which "
                             "is what separates them from var_hint) | "
                             "zono (adv_std_zono_bounds=false) | "
                             "zono_npre (adv_std_zono_npre=false: the zonotope is KEPT but its "
                             "N_pre input is removed -- the absolute N2 zonotope still "
                             "propagates, it is just not intersected with N_pre's pre-activation "
                             "bounds or the N_pre->N difference zonotope; needs the base "
                             "zono_bounds=true; filename tag _noNpreZono) | "
                             "triangle (adv_std_n2_relax_threshold=0, sibling_gate=false — whole "
                             "technique off) | "
                             "zono_triangle (BOTH bound-tightening techniques off: "
                             "adv_std_zono_bounds=false AND rt=0/sibling_gate=false — the one "
                             "two-component entry, since zono and triangle tighten the same ReLU "
                             "bounds and either alone can be masked by the other) | "
                             "pert_intervals (alias 'pi': use_perturbed_intervals=false ONLY, "
                             "rt/SibGate kept; the job gets --allow_relax_without_pi true and its "
                             "results carry a _noPI filename tag). "
                             "Requires every --sweep_adv_std_* flag to have exactly one value.")
    parser.add_argument("--stdboost_ablations", nargs="*", default=None,
                        metavar="COMPONENT",
                        help="Leave-one-out ablation mode for the N2stdBoost ('ours') jobs. "
                             "Appends role-N2 combos derived from the full paper combo "
                             "(zb=true, sg=true, rt=<the single non-negative "
                             "--sweep_adv_std_n2_relax_threshold value>, pi=true), one per listed "
                             "component: none (full combo, the control) | zono (zb=false) | "
                             "triangle (rt=0, sg=false — whole technique off) | "
                             "zono_triangle (BOTH off: zb=false AND rt=0/sg=false) | "
                             "pert_intervals (alias 'pi': pi=false ONLY, rt/SibGate kept; the job "
                             "gets --allow_relax_without_pi true). var_hint is not a standard-mode "
                             "component and is rejected. Deduped against --stdboost_combos entries.")
    parser.add_argument("--skip_std_n2_baseline", action="store_true",
                        help="In --advanc ed_standard mode, skip Phase 1.5 (the auto-launched "
                             "vagharWithPerturbed N2 standard baseline). Use when you only want "
                             "the N1-solve, advstd, and stdBoost jobs and don't need the "
                             "wide-table 'pi' baseline.")
    parser.add_argument("--n2_tables_only", action="store_true",
                        help="In --advanced_standard mode, only run the jobs that fill the "
                             "TARGET-network (N2) tables/rows. Skips jobs that solely populate "
                             "the SOURCE-network (N1) tables: Phase 0.5 delta_max for role N1 and "
                             "Phase 2.5 N1stdBoost combos (role=N1 in --stdboost_combos / "
                             "--include_nn1_boost). N1 work that is ESSENTIAL for N2 is still run: "
                             "the Phase 1 N1-solve (advanced_standard_n1) state that advstd N2 "
                             "depends on is kept (and the existing skip-checks already run it only "
                             "when an N2 job actually needs it). N2 jobs — delta_max N2, the "
                             "standard-N2 baseline (Phase 1.5), advstd-N2 (Phase 2), and N2stdBoost "
                             "(Phase 2.5) — are unaffected.")
    parser.add_argument("--rerun_timeouts", action="store_true",
                        help="Treat existing result rows that hit a Gurobi termination "
                             "limit (solve_status in {TIME_LIMIT, USER_OBJ_LIMIT, "
                             "ITERATION_LIMIT, NODE_LIMIT, SOLUTION_LIMIT, MEMORY_LIMIT, "
                             "WORK_LIMIT}) as NOT done, so they get re-solved under the "
                             "current --timeout. Also catches legacy/positional rows whose "
                             "optimization_time falls within ±--rerun_timeout_eps seconds of "
                             "a known cap (--rerun_timeout_values). A timeout row is only "
                             "re-run when the current --timeout exceeds the time it already "
                             "ran by more than --rerun_timeout_eps; a cell that already "
                             "consumed the full (or a larger) budget is left as done, since "
                             "a same-budget re-solve would just reproduce the timeout. "
                             "Affects all skip-check "
                             "phases that scan .txt result files (Phases 1.5, 2, 2.5, 0.5). "
                             "Phase 1 (N1 state) uses .bin files instead — delete those "
                             "manually if you also need to re-solve the N1 state.")
    parser.add_argument("--rerun_timeout_values", nargs="*", type=float, default=None,
                        metavar="SECS",
                        help="Wall-clock optimization_time caps (in seconds) to treat as "
                             "timeouts for the legacy/positional-row fallback in "
                             "--rerun_timeouts. Default: [1800, 3600] plus the current "
                             "--timeout. Only consulted when --rerun_timeouts is set.")
    parser.add_argument("--rerun_timeout_eps", type=float, default=30.0,
                        metavar="SECS",
                        help="Tolerance (in seconds) used when matching a row's "
                             "optimization_time against --rerun_timeout_values. Default: 30.")
    parser.add_argument("--geometric_intervals", action="store_true",
                        help="Pass --geometric_intervals true to run.jl for TRANSLATION/ROTATION jobs "
                             "(advstd N2 and stdBoost standard with use_perturbed_intervals). Exploits the "
                             "pixel-relocation structure in the perturbed-interval coupling (interval-only, no "
                             "zonotope; delta unchanged). run.jl adds a _geomInt filename tag, and the skip-check "
                             "keeps these separate from the non-geomInt baseline so resume is correct. No-op for "
                             "other perturbations / pi=false combos (run.jl warns and falls back).")
    parser.add_argument("--prioritize_rows", action="store_true",
                        help="In --advanced_standard mode, dispatch jobs in row-priority "
                             "order (row = (arch, perturbation)) so earlier rows finish "
                             "first while still running several rows concurrently, instead "
                             "of even FIFO spread. N1->N2 advstd dependency is unaffected.")
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
                             "Default: 2,3,4,5,6,7,8,9,10. Use to restrict to specific scenarios. "
                             "With --paper_tables_from_txt this restricts the generated "
                             "tables/charts to exactly these target classes; any requested "
                             "c_target with no data is flagged with a red '*'.")
    parser.add_argument("--cs", type=str, default=None,
                        help="Comma-separated 0-indexed c_source (Cs) values, the "
                             "SOURCE-class analogue of --ct. These are the exact 'c_s=k' "
                             "labels shown on the chart axis, so -- unlike Julia 1-indexed "
                             "--ct -- they are NOT shifted (e.g. --cs 0,2 keeps c_s=0 and "
                             "c_s=2). With --paper_tables_from_txt this restricts the "
                             "generated tables/charts to exactly these source classes; a "
                             "requested Cs with no data is flagged \"missing Cs=k\" in the "
                             "bar charts. Per-model overrides come from the '#CS' groups of "
                             "--arch_timeouts (which take precedence, like '@CT' for --ct).")
    parser.add_argument("--sweep_ctag", nargs="*", type=int, default=None,
                        help="Julia-indexed c_tag (source class) values to sweep (e.g. '1 2 3'). "
                             "Each value is passed to run.jl as --ctag in a separate invocation; "
                             "N1 state files, advstd result files, and standard-baseline result files "
                             "are keyed per-ctag (Julia writes ctag{ctag-1} into filenames). Default: [1].")
    args = parser.parse_args()

    # Wire the rerun-on-timeout filter into module globals so
    # _parse_c_source_target_pairs can consult them without threading the
    # flag through every caller.
    global _RERUN_TIMEOUTS, _TIMEOUT_VALUES, _TIMEOUT_MATCH_EPS
    global _RERUN_TIMEOUT_BUDGET
    _RERUN_TIMEOUTS = bool(args.rerun_timeouts)
    if _RERUN_TIMEOUTS:
        _RERUN_TIMEOUT_BUDGET = float(args.timeout)
        default_caps = [1800.0, 3600.0, float(args.timeout)]
        if args.rerun_timeout_values:
            caps = list(args.rerun_timeout_values)
        else:
            caps = default_caps
        # De-dupe while preserving order.
        seen = set(); _TIMEOUT_VALUES = tuple(
            c for c in caps if not (c in seen or seen.add(c))
        )
        _TIMEOUT_MATCH_EPS = float(args.rerun_timeout_eps)
        print(f"[rerun_timeouts] enabled — treating rows with "
              f"solve_status ∈ timeout-set OR optimization_time within "
              f"±{_TIMEOUT_MATCH_EPS}s of {list(_TIMEOUT_VALUES)} as not done, "
              f"but only re-running those whose prior runtime is below the new "
              f"{_RERUN_TIMEOUT_BUDGET:g}s budget by more than {_TIMEOUT_MATCH_EPS}s "
              f"(cells that already ran the full budget are kept as done)")

    total_cores = args.max_cores
    thresholds = args.thresholds if args.thresholds else THRESHOLDS
    opt_intervals = args.opt_intervals if args.opt_intervals else OPT_INTERVALS
    dataset = args.dataset

    if args.standard_only:
        args.skip_transfer = True
    if args.standard_warmstart:
        args.skip_standard = True  # standard is done inside each transfer Julia process

    # Build list of (arch, model_path|None) to run. --arch_timeouts folds
    # --arch_models together with a per-model --force_timeout; when given it
    # yields both arch_runs and a {arch: seconds} cap dict that the table
    # regeneration paths use in place of the scalar --force_timeout.
    arch_force_timeout = None
    arch_c_targets = None
    arch_c_sources = None
    arch_ds_scoped = {}
    if args.arch_timeouts:
        (arch_runs, arch_force_timeout, arch_c_targets,
         arch_c_sources, arch_ds_scoped) = _parse_arch_timeouts(
            args.arch_timeouts)
        if args.arch_models:
            print("WARNING: --arch_timeouts overrides --arch_models "
                  "(models are taken from --arch_timeouts)")
        if args.force_timeout is not None:
            print("WARNING: --arch_timeouts overrides the scalar --force_timeout "
                  "(per-model caps are taken from --arch_timeouts)")
    elif args.arch_models:
        arch_runs = []
        for pair in args.arch_models:
            if "=" not in pair:
                print(f"ERROR: --arch_models entry must be arch=model_path, got: {pair}")
                sys.exit(1)
            a, mp = pair.split("=", 1)
            arch_runs.append((a, mp))
    else:
        arch_runs = [(args.arch, args.model_path)]

    # The effective cap passed to the table/chart regenerators: the per-arch
    # dict from --arch_timeouts if present, else the scalar --force_timeout.
    effective_force_timeout = (arch_force_timeout if arch_force_timeout is not None
                               else args.force_timeout)

    # Per-dataset linf radius for the benchmark nets (acas/har). Applied before
    # perturbations_for so every eps_* directory built below sees the override.
    if args.benchmark_eps:
        global _BENCHMARK_EPS
        for entry in args.benchmark_eps:
            if "=" not in entry:
                print(f"ERROR: --benchmark_eps entry '{entry}' must be "
                      f"<dataset>=<eps>, e.g. har=0.01")
                sys.exit(1)
            ds_key, eps_str = entry.split("=", 1)
            ds_key = _julia_dataset_name(ds_key.strip())
            eps_str = eps_str.strip()
            if ds_key not in ("acas", "har"):
                print(f"ERROR: --benchmark_eps only applies to the pretrained "
                      f"benchmark nets (acas/har); got '{ds_key}'.")
                sys.exit(1)
            try:
                float(eps_str)
            except ValueError:
                print(f"ERROR: --benchmark_eps radius '{eps_str}' is not a number.")
                sys.exit(1)
            _BENCHMARK_EPS[ds_key] = eps_str

    # Filter perturbations if requested
    if args.perturbations:
        global _PERT_NAME_FILTER
        _PERT_NAME_FILTER = [p.lower() for p in args.perturbations]
    perts = perturbations_for(args.dataset)
    if args.perturbations and not perts:
        print(f"ERROR: No perturbations matched {args.perturbations}")
        print(f"Available for {args.dataset}: "
              f"{[p[0] for p in perturbations_for(args.dataset)] or _PERT_NAME_FILTER}")
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
        all_perts = all_perturbations_for(args.dataset)
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

    # ── Paper-tables mode: regenerate neta-s-paper tables from advStd txt ──
    if args.paper_tables_from_txt:
        # --dataset may be comma-separated; regenerate each dataset's
        # neta-s-paper per-cell tables + N2 charts straight from the advStd
        # .txt files (no CSVs), reusing the same --arch_models.
        pt_datasets = [d.strip() for d in dataset.split(",") if d.strip()]
        # --ct restricts the tables to specific target classes. The flag is
        # Julia 1-indexed (2..10); result files store 0-indexed c_targets, so
        # convert here. None => full sweep (all classes), the prior behavior.
        pt_global_c_targets = None
        if args.ct:
            pt_global_c_targets = {ct - 1 for ct in _parse_c_targets(args.ct)}
        # Per-model ct lists from --arch_timeouts (@CT groups) take precedence:
        # build a {arch: set} map where a model whose group carried no @CT falls
        # back to the global --ct (or full sweep). Otherwise use the single
        # global set for every model, exactly as before.
        if arch_c_targets is not None:
            pt_requested_c_targets = {
                a: (cts if cts is not None else pt_global_c_targets)
                for a, cts in arch_c_targets.items()}
            print("[paper-tables] restricting tables/charts to PER-MODEL "
                  "c_targets (Julia-indexed): "
                  + ", ".join(
                      f"{a}={sorted(c + 1 for c in cts)}" if cts else f"{a}=all"
                      for a, cts in pt_requested_c_targets.items())
                  + "; requested targets with no data are flagged with a red '*'")
        else:
            pt_requested_c_targets = pt_global_c_targets
            if pt_global_c_targets is not None:
                print(f"[paper-tables] restricting tables/charts to c_targets "
                      f"(Julia-indexed) {_parse_c_targets(args.ct)}; requested "
                      f"targets with no data are flagged with a red '*'")
        # --cs / '#CS' groups restrict the SOURCE classes (Cs / c_tag) exactly
        # like --ct / '@CT' restrict targets: Julia 1-indexed in, 0-indexed out,
        # per-model #CS groups override the global --cs.
        pt_global_c_sources = None
        if args.cs:
            # --cs is 0-indexed (the c_s labels shown on the chart axis), so it
            # is NOT shifted -- unlike --ct.
            pt_global_c_sources = set(_parse_c_targets(args.cs))
        if arch_c_sources is not None:
            pt_requested_c_sources = {
                a: (css if css is not None else pt_global_c_sources)
                for a, css in arch_c_sources.items()}
            print("[paper-tables] restricting tables/charts to PER-MODEL "
                  "c_sources / Cs (0-indexed, matching the c_s chart labels): "
                  + ", ".join(
                      f"{a}={sorted(css)}" if css else f"{a}=all"
                      for a, css in pt_requested_c_sources.items())
                  + "; a requested Cs with no data is flagged \"missing Cs=k\"")
        else:
            pt_requested_c_sources = pt_global_c_sources
            if pt_global_c_sources is not None:
                print(f"[paper-tables] restricting tables/charts to c_sources / "
                      f"Cs (0-indexed) {sorted(pt_global_c_sources)}; a "
                      f"requested Cs with no data is flagged \"missing Cs=k\"")
        # Pool BOTH the solve-time cells and the bounds-difference clusters
        # across every dataset so each lands in ONE combined figure (written
        # after the per-dataset loop). The per-dataset compile is deferred
        # (recompile=False) and done once at the end.
        all_time_cells = []
        all_bd_groups = []
        all_relax_rows = []
        # Caps of every arch that actually ran, MERGED across datasets. The
        # combined figures pool cells from all of them, and a dataset-scoped
        # '<ds>:<arch>=' entry never reaches the unscoped map -- so passing
        # that map alone leaves a scoped-only arch (e.g. a cnn1 scoped to
        # mnist and fashion-mnist) with no cap, and its rows lose the dashed
        # timeout level and the cross-cap dedup.
        all_force_timeout = {}
        for pt_dataset in pt_datasets:
            if len(pt_datasets) > 1:
                print(f"\n===== Regenerating neta-s-paper tables (from advStd "
                      f"txt) for dataset: {pt_dataset} =====")
            # Layer this dataset's '<dataset>:<arch>=path' entries over the
            # unscoped setup, so one arch can carry a different class grid per
            # dataset (CIFAR's cnn1 vs MNIST's cnn1).
            (ds_arch_runs, ds_ft, ds_ct,
             ds_cs) = _arch_setup_for_dataset(
                pt_dataset, arch_runs, arch_force_timeout, arch_c_targets,
                arch_c_sources, arch_ds_scoped, cwd=cwd)
            ds_force_timeout = (ds_ft if arch_force_timeout is not None
                                else effective_force_timeout)
            if isinstance(ds_force_timeout, dict):
                all_force_timeout.update(ds_force_timeout)
            ds_req_ct = _layer_global_request(
                (ds_ct if (arch_c_targets is not None
                           or ds_ct) else pt_requested_c_targets),
                pt_requested_c_targets, ds_arch_runs)
            ds_req_cs = _layer_global_request(
                (ds_cs if (arch_c_sources is not None
                           or ds_cs) else pt_requested_c_sources),
                pt_requested_c_sources, ds_arch_runs)
            tc, bd, rx = _regen_paper_tables_from_txt(
                ds_arch_runs, cwd, pt_dataset, args.combo_ranking_seeds,
                combination_table=args.combination_table,
                force_timeout=ds_force_timeout,
                rerun_timeout_eps=args.rerun_timeout_eps,
                requested_c_targets=ds_req_ct,
                requested_c_sources=ds_req_cs,
                paper_taus=args.paper_taus,
                paper_chart_taus=args.paper_chart_taus,
                recompile=False,
                ablation_tables=args.ablation_tables,
                ablation_expected=_parse_ablation_expected(
                    args.ablation_expected))
            all_time_cells += tc
            all_bd_groups += bd
            all_relax_rows += rx
        # Table 1 in the paper lists only the networks this run produced
        # results for. It runs AFTER the per-dataset loop because it reads back
        # which per-cell tables were rendered; the frozen master keeps the full
        # table for table_full_results.tex, so no row is ever lost for good.
        try:
            sys.path.insert(0, cwd)
            import update_advstd_tex_tables as _tab1_updater
            _filter_tab_networks(cwd, _rendered_ds_arch_pairs(cwd),
                                 _tab1_updater)
        except Exception as _exc:
            print(f"[tab1] skipped ({_exc})")

        # One combined solve-time grid + one combined bounds-difference figure
        # + one relaxation/precision table across all datasets, then the single
        # final recompile.
        _regen_paper_combined_from_txt(
            cwd, all_time_cells, all_bd_groups,
            relax_rows=all_relax_rows,
            force_timeout=(all_force_timeout or effective_force_timeout))
        # Headline averages of the paper's own tables, printed last so they are
        # the final thing on screen.
        # The numbers are read back OUT of the appendix tables, so they are
        # only known after those tables are written. Recompile when the
        # sentence actually changed, or main.pdf would lag one run behind.
        _paper_avgs = _report_paper_table_averages(cwd)
        _sentences_changed = bool(_update_results_sentence(cwd, _paper_avgs))
        if _update_conclusion_sentence(cwd, _paper_avgs):
            _sentences_changed = True
        if _sentences_changed:
            _recompile_neta_s_paper(cwd)
        return

    # ── Analysis mode: find advanced-standard faster than standard ───
    if args.find_advstd_faster_than_standard:
        # --dataset may be a comma-separated list here; regenerate the
        # combo CSVs + appendix tables for each dataset in turn, reusing the
        # same --arch_models. (Single dataset works unchanged.)
        fa_datasets = [d.strip() for d in dataset.split(",") if d.strip()]
        for fa_dataset in fa_datasets:
            if len(fa_datasets) > 1:
                print(f"\n===== Regenerating combo ranking + tables for "
                      f"dataset: {fa_dataset} =====")
            _generate_combo_ranking_csv(
                arch_runs, cwd, fa_dataset,
                args.compare_to_with_perturbed, args.combo_ranking_seeds,
                combination_table=args.combination_table,
                force_timeout=effective_force_timeout,
                rerun_timeout_eps=args.rerun_timeout_eps)
        return

    cores_per_job = CORES_PER_JOB
    max_slots = _max_slots_for(total_cores, cores_per_job)

    # ── Advanced-standard mode (two-phase: N1 once, then N2 sweep) ──────
    if args.advanced_standard:
        try:
            Threads_num = 32
            cores_per_job = Threads_num
            max_slots = _max_slots_for(total_cores, cores_per_job)

            # ── Multi-dataset grouping ──────────────────────────────────
            # By default the sweep runs a single dataset (args.dataset) over
            # arch_runs. With --dataset_group (repeatable) it runs SEVERAL
            # datasets in one merged job pool, interleaving their row priority
            # round-robin. Each group carries its own dataset + arch_models;
            # everything else (sweep_ctag, ct, techniques, …) is shared.
            # arch_runs is reshaped here into (dataset, arch, model_path)
            # triples; every per-arch loop below iterates these triples so the
            # single c_tag loop wraps both datasets and one run_pool schedules
            # them together. (Safe: this branch returns before any later code
            # that expects the original 2-tuple arch_runs.)
            if args.dataset_group:
                dataset_groups = []
                for spec in args.dataset_group:
                    if "|" not in spec:
                        print(f"ERROR: --dataset_group '{spec}' must be "
                              f"'dataset|arch=path,arch=path'")
                        sys.exit(1)
                    ds_part, am_part = spec.split("|", 1)
                    ds_name = ds_part.strip()
                    if not ds_name:
                        print(f"ERROR: --dataset_group '{spec}' has an empty dataset name")
                        sys.exit(1)
                    aruns = []
                    for pair in am_part.split(","):
                        pair = pair.strip()
                        if not pair:
                            continue
                        if "=" not in pair:
                            print(f"ERROR: --dataset_group arch entry '{pair}' "
                                  f"must be arch=model_path (in '{spec}')")
                            sys.exit(1)
                        a, p = pair.split("=", 1)
                        aruns.append((a.strip(), p.strip()))
                    if not aruns:
                        print(f"ERROR: --dataset_group '{spec}' lists no arch=model_path pairs")
                        sys.exit(1)
                    dataset_groups.append((ds_name, aruns))
            else:
                dataset_groups = [(dataset, arch_runs)]
            _multi_ds = len(dataset_groups) > 1
            # NOTE: arch_runs is reshaped to (dataset, arch, model_path) triples
            # later, just before the Phase-0 arch_meta loop — AFTER the
            # combo-ranking analysis block below, which still consumes the
            # original 2-tuple arch_runs.
            if _multi_ds:
                print(f"\nMulti-dataset run: {len(dataset_groups)} datasets — "
                      + ", ".join(f"{ds}({len(aruns)} arch)" for ds, aruns in dataset_groups)
                      + "  [merged pool, interleaved row priority]")

            def _aprefix(dataset, arch):
                """Label prefix. Includes the dataset only in multi-dataset runs
                so single-dataset labels (and sweep_logs filenames) are unchanged."""
                return f"[{dataset}/{arch}] " if _multi_ds else f"[{arch}] "

            # ── Per-dataset slot reservation (work-conserving) ───────────
            # In a multi-dataset run, reserve a share of the concurrent slots
            # for each dataset so both make progress at once; idle slots spill
            # over to the other dataset's remaining jobs so no core sits unused.
            # job_group maps a job to its dataset via the output_dir path.
            _grp_re = re.compile(r"paper_experiments/([^/]+)/")
            def _job_group(job):
                cmd = job[1]
                try:
                    out = cmd[cmd.index("--output_dir") + 1]
                except (ValueError, IndexError):
                    out = ""
                m = _grp_re.search(out or "")
                return m.group(1) if m else None
            group_slots = None
            if _multi_ds:
                n_groups = len(dataset_groups)
                if args.dataset_slots:
                    try:
                        slot_counts = [int(x) for x in args.dataset_slots.split(",")]
                    except ValueError:
                        print(f"ERROR: --dataset_slots must be comma-separated ints, got '{args.dataset_slots}'")
                        sys.exit(1)
                    if len(slot_counts) != n_groups:
                        print(f"ERROR: --dataset_slots has {len(slot_counts)} entries but "
                              f"there are {n_groups} --dataset_group(s)")
                        sys.exit(1)
                    if any(c < 0 for c in slot_counts) or sum(slot_counts) == 0:
                        print("ERROR: --dataset_slots must be non-negative and not all zero")
                        sys.exit(1)
                    if sum(slot_counts) > max_slots:
                        print(f"ERROR: --dataset_slots sums to {sum(slot_counts)} but only "
                              f"{max_slots} slots are available "
                              f"({_cores_desc(cores_per_job)}, {cores_per_job}/job)")
                        sys.exit(1)
                else:
                    base = max_slots // n_groups
                    slot_counts = [base] * n_groups
                    for i in range(max_slots - base * n_groups):  # remainder to first groups
                        slot_counts[i] += 1
                # Expand to a per-slot owner list. Any leftover slots (when the
                # split sums to < max_slots) are handed out round-robin; they
                # still spill over, so no slot ever idles.
                group_slots = []
                for (ds, _), c in zip(dataset_groups, slot_counts):
                    group_slots.extend([ds] * c)
                _gi = 0
                while len(group_slots) < max_slots:
                    group_slots.append(dataset_groups[_gi % n_groups][0])
                    _gi += 1
                group_slots = group_slots[:max_slots]
                _spill = max_slots - sum(slot_counts)
                print("  slot reservation (spillover on): "
                      + ", ".join(f"{ds}={c}" for (ds, _), c in zip(dataset_groups, slot_counts))
                      + (f" +{_spill} shared" if _spill > 0 else "")
                      + f"  of {max_slots} slots")

            # Resolve technique sweep values
            mip_start_vals = [v.lower() for v in args.sweep_adv_std_mip_start] if args.sweep_adv_std_mip_start else ["true"]
            # Branch priorities: rank (uniform-spacing) / decay (magnitude-aware).
            # Legacy true/false → rank/off. Legacy 'bounds'/'pseudocost' map to 'rank'
            # with a deprecation warning (so old sweep configs still run, but won't
            # silently re-execute the retired magic-number heuristic).
            def _norm_bp(v):
                v = v.lower()
                if v == "true":       return "rank"
                if v == "false":      return "off"
                if v == "bounds":
                    print("WARNING: --sweep_adv_std_branch_priorities=bounds is retired; using 'rank' instead.")
                    return "rank"
                if v == "pseudocost":
                    print("WARNING: --sweep_adv_std_branch_priorities=pseudocost is retired; using 'rank' instead.")
                    return "rank"
                return v
            branch_pri_vals = [_norm_bp(v) for v in args.sweep_adv_std_branch_priorities] if args.sweep_adv_std_branch_priorities else ["off"]
            for v in branch_pri_vals:
                if v not in ("off", "rank", "decay"):
                    print(f"ERROR: unknown --sweep_adv_std_branch_priorities value '{v}' "
                          "(expected off | rank | decay)")
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
            # Technique 4 (SibGate) — boolean flag. Augments Technique 3's
            # tiered relaxation with a sibling-gated conditional triangle
            # (one-thin tier) and a pre-act coupling line (both-thin tier).
            # No effect when adv_std_n2_relax_threshold < 0; combos are
            # auto-pruned below.
            sg_vals = [v.lower() for v in args.sweep_adv_std_n2_sibling_gate] if args.sweep_adv_std_n2_sibling_gate else ["false"]
            for v in sg_vals:
                if v not in ("true", "false"):
                    print(f"ERROR: unknown --sweep_adv_std_n2_sibling_gate value '{v}' (expected true | false)")
                    sys.exit(1)
            # 5-valued mode: off | prev | direct | direct_pgd | prev_pgd. Legacy true/false
            # still accepted and normalized to prev/off so historical commands keep working.
            _vh_alias = {"true": "prev", "false": "off"}
            var_hint_vals = [_vh_alias.get(v.lower(), v.lower())
                             for v in (args.sweep_adv_std_var_hint or ["off"])]
            for v in var_hint_vals:
                if v not in ("off", "prev", "direct", "direct_pgd", "prev_pgd"):
                    print(f"ERROR: unknown --sweep_adv_std_var_hint value '{v}' "
                          f"(expected off | prev | direct | direct_pgd | prev_pgd; legacy true/false also accepted)")
                    sys.exit(1)
            seed_vals = args.sweep_gurobi_seed if args.sweep_gurobi_seed else [0]
            sweep_ctag_vals = args.sweep_ctag if args.sweep_ctag else [1]
            for _ct in sweep_ctag_vals:
                if _ct < 1:
                    print(f"ERROR: --sweep_ctag values must be Julia-indexed (>=1); got {_ct}")
                    sys.exit(1)

            # Generate all combinations, excluding the all-off case and any
            # combo where a flag that requires bound_tightening is on while
            # bound_tightening=false (gated on Technique 4's pre-compute block).
            # Note: var_hint_fix has been merged into var_hint (the "fix" is
            # always-on now), so the vhf dimension is gone.
            # Combos are 10-tuples (ms, bp, lb, bt, zb, np_, rt, vh, sg, pi);
            # the trailing pi (use_perturbed_intervals) is fixed to "true" in
            # the Cartesian path and only varies via --advstd_ablations.
            if args.advstd_ablations:
                # ── Leave-one-out ablation path (bypasses the product) ──
                # The single-valued --sweep_adv_std_* flags define the BASE
                # combo; each token spawns one combo with that component
                # removed. See the --advstd_ablations help text.
                _abl_lists = {
                    "--sweep_adv_std_mip_start": mip_start_vals,
                    "--sweep_adv_std_branch_priorities": branch_pri_vals,
                    "--sweep_adv_std_lp_basis": lp_basis_vals,
                    "--sweep_adv_std_bound_tightening": bound_tight_vals,
                    "--sweep_adv_std_zono_bounds": zono_bounds_vals,
                    "--sweep_adv_std_n1_probe": n1_probe_vals,
                    "--sweep_adv_std_n2_relax_threshold": relax_t_vals,
                    "--sweep_adv_std_var_hint": var_hint_vals,
                    "--sweep_adv_std_n2_sibling_gate": sg_vals,
                }
                for _fname, _vals in _abl_lists.items():
                    if len(_vals) != 1:
                        print(f"ERROR: --advstd_ablations needs a unique base combo, but "
                              f"{_fname} has {len(_vals)} values: {_vals}")
                        sys.exit(1)
                # 11th field `hyp` = use_hyper_attack. It is part of the combo
                # key (not a side set) because var_hint and warm_start differ
                # ONLY by it: both set vh=off, so a 10-wide key would make them
                # identical and the dedup below would silently drop the second.
                _base = (mip_start_vals[0], branch_pri_vals[0], lp_basis_vals[0],
                         bound_tight_vals[0], zono_bounds_vals[0], n1_probe_vals[0],
                         relax_t_vals[0], var_hint_vals[0], sg_vals[0], "true", "true")
                technique_combos = []
                _abl_by_combo = {}  # combo -> token (for the banner)
                for _tok_raw in args.advstd_ablations:
                    _tok = _tok_raw.strip().lower()
                    if _tok == "pi":
                        _tok = "pert_intervals"
                    ms, bp, lb, bt, zb, np_, rt, vh, sg, pi, hyp = _base
                    if _tok == "none":
                        pass
                    elif _tok == "var_hint":
                        # N1-derived variable hints (prev_pgd) off, PGD
                        # hyper-attack warm start KEPT. Such files still carry
                        # the _HyperAttackHints tag, which is what separates
                        # them from warm_start below.
                        if vh == "off":
                            print(f"WARNING: --advstd_ablations var_hint — base var_hint is "
                                  f"already off; combo equals the base.")
                        vh = "off"
                    elif _tok == "warm_start":
                        # No warm start at all: variable hints off AND the
                        # hyper-attack PGD warm start off. Filenames carry no
                        # _HyperAttackHints tag, which is what the skip-check
                        # discriminates on.
                        vh = "off"
                        hyp = "false"
                    elif _tok == "zono":
                        if zb == "false":
                            print(f"WARNING: --advstd_ablations zono — base zono_bounds is "
                                  f"already false; combo equals the base.")
                        zb = "false"
                    elif _tok == "triangle":
                        if rt < 0.0 and sg == "false":
                            print(f"WARNING: --advstd_ablations triangle — base triangle is "
                                  f"already off; combo equals the base.")
                        # Whole technique off. τ = 0 emits no relaxations
                        # (gap area > 0 for every split neuron; see run.jl).
                        rt, sg = 0.0, "false"
                    elif _tok == "zono_triangle":
                        # Both bound-tightening techniques off at once: the
                        # only two-component entry, kept because zono and
                        # triangle overlap (both tighten the same ReLU
                        # bounds), so removing either alone can be masked by
                        # the other still doing the work.
                        if zb == "false" and rt < 0.0 and sg == "false":
                            print(f"WARNING: --advstd_ablations zono_triangle — base zono "
                                  f"and triangle are already off; combo equals the base.")
                        zb, rt, sg = "false", 0.0, "false"
                    elif _tok == "zono_npre":
                        # Keep the zonotope, ablate ONLY its N_pre input: the
                        # absolute N2 zonotope is still propagated but is not
                        # intersected with N_pre's pre-activation bounds or the
                        # N_pre->N difference zonotope. Encoded as a distinct
                        # zb VALUE rather than a 12th tuple slot, so the combo
                        # arity (and every unpack of it) is untouched; the job
                        # builder maps it to the two Julia flags.
                        if zb != "true":
                            print(f"ERROR: --advstd_ablations zono_npre needs the base "
                                  f"zono_bounds=true (there is no zonotope to take N_pre "
                                  f"out of); base has zono_bounds={zb}.")
                            sys.exit(1)
                        zb = "true_nonpre"
                    elif _tok == "pert_intervals":
                        # PI off ONLY — rt/SibGate kept (true leave-one-out).
                        # When rt > 0 the job builder adds
                        # --allow_relax_without_pi true (run.jl guard bypass)
                        # and run.jl stamps a _noPI filename tag.
                        pi = "false"
                    else:
                        print(f"ERROR: unknown --advstd_ablations component '{_tok_raw}' "
                              f"(expected none | var_hint | warm_start | zono | "
                              f"zono_npre | triangle | zono_triangle | pert_intervals)")
                        sys.exit(1)
                    combo = (ms, bp, lb, bt, zb, np_, rt, vh, sg, pi, hyp)
                    # A misconfigured base (not a mere no-op) is an error here —
                    # unlike the Cartesian path we don't silently prune.
                    if (zb.startswith("true") and bt == "false") or \
                       (np_ != "off" and bt == "false") or \
                       (rt >= 0.0 and bt == "false"):
                        print(f"ERROR: --advstd_ablations '{_tok}' yields combo {combo} that "
                              f"requires bound_tightening=true (zono/probe/relax without bt).")
                        sys.exit(1)
                    if sg == "true" and rt < 0.0:
                        print(f"ERROR: --advstd_ablations '{_tok}' yields combo {combo} with "
                              f"sibling_gate=true but relax_threshold off (SibGate would be "
                              f"a no-op). Fix the base flags.")
                        sys.exit(1)
                    if combo in _abl_by_combo:
                        print(f"  [--advstd_ablations] '{_tok}' duplicates "
                              f"'{_abl_by_combo[combo]}' — skipping.")
                        continue
                    _abl_by_combo[combo] = _tok
                    technique_combos.append(combo)
                # Combos with a component actually removed get an _ablation
                # filename tag (see the Phase-2 builder). The 'none' control is
                # deliberately untagged: its rows ARE the paper combo's rows,
                # so the skip-check can reuse already-solved cells.
                advstd_ablation_combos = {c for c, t in _abl_by_combo.items()
                                          if t != "none"}
                print(f"\n--advstd_ablations: {len(technique_combos)} combo(s) "
                      f"[{', '.join(_abl_by_combo.values())}] (Cartesian sweep bypassed; "
                      f"{len(advstd_ablation_combos)} tagged _ablation)")
            else:
                advstd_ablation_combos = set()
                technique_combos = [
                    # pi="true", hyp="true": the Cartesian path never varies
                    # perturbed intervals or the hyper-attack warm start.
                    (ms, bp, lb, bt, zb, np_, rt, vh, sg, "true", "true")
                    for ms, bp, lb, bt, zb, np_, rt, vh, sg in itertools.product(
                        mip_start_vals, branch_pri_vals, lp_basis_vals, bound_tight_vals,
                        zono_bounds_vals, n1_probe_vals, relax_t_vals, var_hint_vals, sg_vals)
                    if not (ms == "false" and bp == "off" and lb == "false" and bt == "false"
                            and zb == "false" and np_ == "off" and rt < 0.0 and vh == "off"
                            and sg == "false")
                    and not (zb == "true" and bt == "false")
                    and not (np_ != "off" and bt == "false")
                    and not (rt >= 0.0 and bt == "false")
                    and not (sg == "true" and rt < 0.0)
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
                    args.compare_to_with_perturbed, regen_seeds,
                    combination_table=args.combination_table)
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
                _has_relax_mode_col = False
                with open(args.advstd_safe_combos_only) as _f:
                    reader = _csv_filter.DictReader(_f, skipinitialspace=True)
                    if reader.fieldnames is not None:
                        _has_relax_mode_col = "relax_mode" in reader.fieldnames
                    for _row in reader:
                        _row = {k: (v.strip() if isinstance(v, str) else v) for k, v in _row.items()}
                        # CSV uses yes/no; sweep uses true/false for binary flags.
                        _yn = {"yes": "true", "no": "false"}
                        _ms = _yn.get(_row["mip_start"], _row["mip_start"])
                        _bp = _row["branch_priorities"]  # off / bounds — same in both
                        _lb = _yn.get(_row["lp_basis"], _row["lp_basis"])
                        _bt = _yn.get(_row["bound_tightening"], _row["bound_tightening"])
                        # var_hint is 5-valued (off/prev/direct/direct_pgd/prev_pgd) in the sweep. CSV
                        # rows may contain legacy "yes"/"no" (from pre-tri-valued extractor runs)
                        # or the current tokens ("prev"/"direct"/"direct_pgd"/"prev_pgd"/"no").
                        # Normalize so both layouts produce a matching combo key.
                        _vh_map = {"yes": "prev", "no": "off", "true": "prev", "false": "off"}
                        _vh = _vh_map.get(_row["var_hint"], _row["var_hint"])
                        _zb = _yn.get(_row["zono_bounds"], _row["zono_bounds"])
                        _np = _row["n1_probe"]  # off / lp — same in both
                        _rt_str = _row["relax_threshold"]
                        _rt = float(_rt_str) if _rt_str not in ("off",) else -1.0
                        # var_hint_fix has been merged into var_hint (the "fix"
                        # is always-on now). Old CSV rows that have a
                        # `var_hint_fix` column are ignored — the var_hint
                        # value alone determines the combo identity, and old
                        # `_varHint`/`_varHint_varHintFix` files are tagged
                        # `vh_legacy` by the extractor and do not match new
                        # combos.
                        # Back-compat: existing ranking CSVs predate Technique 4
                        # and have no sibling_gate column. Treat such rows as
                        # describing the sg="false" combos. New CSV layouts can
                        # add a "sibling_gate" column to disambiguate.
                        _sg = _yn.get(_row.get("sibling_gate", "no"), _row.get("sibling_gate", "false"))
                        _key = (_ms, _bp, _lb, _bt, _zb, _np, _rt, _vh, _sg)
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
                if not _has_relax_mode_col:
                    print("  [warning] ranking CSV lacks a 'relax_mode' column — "
                          "all BoundTightPertRelax combos will be treated as 'untested' "
                          "(permissive). Regenerate the ranking after sweeping to populate it.")
                # Combos carry a trailing pi field the ranking CSV predates —
                # match on the first 9 flag fields only.
                pre_filter = len(technique_combos)
                blocked = [c for c in technique_combos if c[:9] in unsafe_keys]
                technique_combos = [c for c in technique_combos if c[:9] not in unsafe_keys]
                n_safe = sum(1 for c in technique_combos if c[:9] in safe_keys)
                n_untested = sum(1 for c in technique_combos if c[:9] not in safe_keys)
                # Preserve flag-product order as the tiebreaker inside each rank.
                _orig_pos = {c: i for i, c in enumerate(technique_combos)}
                technique_combos.sort(
                    key=lambda c: (safe_key_rank.get(c[:9], _UNTESTED_RANK), _orig_pos[c])
                )
                print(f"\n--advstd_safe_combos_only: filtered {pre_filter} -> {len(technique_combos)} combos "
                      f"({n_safe} safe, {n_untested} untested, {len(blocked)} blocked) "
                      f"from {args.advstd_safe_combos_only}")

            print(f"\nAdvanced-standard: {len(technique_combos)} technique combinations × {len(seed_vals)} seed(s) (all-off + zono/probe/relax-without-boundTight + sibgate-without-relax excluded):")
            for ms, bp, lb, bt, zb, np_, rt, vh, sg, pi, hyp in technique_combos:
                # When bt=true and rt>=0, boundTight is subsumed by BoundTightPertRelax in the filename.
                bt_desc = f"BoundTightPertRelax{rt}" if (bt == "true" and rt >= 0.0) else \
                          ("boundTight" if bt == "true" else "off")
                pi_desc = "" if pi == "true" else "  pertIntervals=false(noPI)"
                hyp_desc = "" if hyp == "true" else "  hyperAttack=false(noWarmStart)"
                print(f"  mipStart={ms}  branchPri={bp}  lpBasis={lb}  boundTight/BTPR={bt_desc}  zonoBounds={zb}  n1Probe={np_}  varHint={vh}  sibGate={sg}{pi_desc}{hyp_desc}")
            print(f"  seeds: {seed_vals}")
            if args.n2_tables_only:
                print("  [n2_tables_only] N2 (target-network) tables only — "
                      "skipping N1-only jobs (delta_max N1, N1stdBoost); the "
                      "essential Phase 1 N1-solve state for advstd N2 is kept.")

            sys.path.insert(0, os.path.join(cwd, 'utils'))
            from run_experiment import ARCH_REGISTRY, DATASET_CONFIG

            # Flatten arch_runs to (dataset, arch, model_path) triples spanning
            # all groups. From here on every per-arch loop iterates triples, so
            # the single c_tag loop covers all datasets and one run_pool
            # schedules them together. (Done here, after the combo-ranking
            # analysis block, which uses the original 2-tuple arch_runs.)
            arch_runs = [(ds, a, p) for ds, aruns in dataset_groups for a, p in aruns]

            # ── Phase 0: Train N2 = N1 + sgd_epochs for each arch (once) ─
            # Hoisted out of the c_tag loop — training only depends on
            # (arch, sgd_epochs, lr) and is idempotent if N2 already exists.
            arch_meta = {}  # (dataset, arch) -> (n1_tag, n2_tag, n1_model_p, n2_model_p, model_name, julia_dataset)
            for dataset, arch, model_path in arch_runs:
                if model_path is None:
                    print(f"ERROR: --advanced_standard requires --model_path (or --arch_models)")
                    sys.exit(1)
                print(f"\n{'=' * 60}")
                print(f"Phase 0: Training N2 = N1 + {args.sgd_epochs} SGD epoch(s) {_aprefix(dataset, arch).strip()}")
                print(f"{'=' * 60}\n")
                n1_dir, n2_dir = train_extra_epochs(
                    model_path, arch, dataset,
                    sgd_epochs=args.sgd_epochs, lr=args.lr)
                _, model_name = ARCH_REGISTRY[arch]
                _, _, _, _, julia_dataset = DATASET_CONFIG[_dataset_config_key(dataset)]
                julia_dataset = _julia_dataset_name(julia_dataset)
                n1_tag = os.path.basename(os.path.normpath(n1_dir))
                n2_tag = os.path.basename(os.path.normpath(n2_dir))
                n1_model_p = os.path.join(n1_dir, "model.p")
                n2_model_p = os.path.join(n2_dir, "model.p")
                arch_meta[(dataset, arch)] = (n1_tag, n2_tag, n1_model_p, n2_model_p, model_name, julia_dataset)

            # ── Phase 0.5: delta_max for N1 and N2 (single-network, no perturbation) ─
            # Runs run.jl with --perturbation max once per (arch, network, c_src).
            # The objective is max over the clean input box of
            # N(x)[c_src] - max_{k!=c_src} N(x)[k], so it depends only on c_src
            # (no c_target). We still pass a non-self --ct because run.jl skips
            # the c_target==c_tag iteration; the c_target value is unused for
            # "max" perturbation (mip_set_delta_property skips set_max_indexes).
            # Results are cached per (arch, network) under a sibling
            # delta_max_{arch}_{network}_{tag}/ dir; cells whose c_src already
            # has a result line are skipped.
            ready_delta_max_jobs = []
            for dataset, arch, model_path in arch_runs:
                n1_tag, n2_tag, n1_model_p, n2_model_p, model_name, julia_dataset = arch_meta[(dataset, arch)]
                arch_prefix = _aprefix(dataset, arch)
                # delta_max is computed for N2 (the target network) only: the
                # bound-gap normalization in the target-network tables uses N2's
                # own delta_max, so the source network N1's delta_max is not
                # needed and is never queued.
                for role, role_tag, role_model_p in (("N2", n2_tag, n2_model_p),):
                    dm_out_dir = os.path.join(
                        cwd, "paper_experiments", dataset, f"{arch}_exp",
                        "delta_max", f"delta_max_{arch}_{role}_{role_tag}")
                    missing_c_srcs = _delta_max_missing_c_srcs(dm_out_dir, sweep_ctag_vals)
                    if not missing_c_srcs:
                        print(f"  {arch_prefix}{role} delta_max already complete for c_srcs={list(sweep_ctag_vals)} at {dm_out_dir} — skipping")
                        continue
                    have = [c for c in sweep_ctag_vals if c not in missing_c_srcs]
                    if have:
                        print(f"  {arch_prefix}{role} delta_max partial — have c_srcs={have}, computing missing={missing_c_srcs}")
                    else:
                        print(f"  {arch_prefix}{role} delta_max missing — computing for c_srcs={missing_c_srcs}")
                    for c_src in missing_c_srcs:
                        _ncls = _num_classes_for(dataset)
                        dummy_ct = c_src % _ncls + 1
                        dm_label = f"{arch_prefix}{role} delta_max c_src={c_src}"
                        dm_cmd = [
                            "julia", "run.jl",
                            "--mode", "standard",
                            "--dataset", julia_dataset,
                            *_benchmark_args(julia_dataset),
                            "--model_name", model_name,
                            "--model_path", role_model_p,
                            "--perturbation", "max",
                            "--perturbation_size", "0",
                            "--ctag", str(c_src),
                            "--ct", str(dummy_ct),
                            "--timout", str(args.timeout),
                            "--output_dir", dm_out_dir + "/",
                            "--c_tag_mode", "false",
                            # delta_max maximises the source-class margin over
                            # the input region. The PGD warm start supplies a
                            # feasible incumbent, and the absolute zonotope
                            # (Source B) tightens the per-neuron bounds, which
                            # main_standard already supports for a single
                            # network (run.jl: "Standard-mode reuse ... still
                            # propagate the absolute zonotope"). With
                            # perturbation_size 0 the zonotope seed is the input
                            # box itself, which is exactly what delta_max needs.
                            *_delta_max_boost_args(julia_dataset),
                            "--activate_vaghgar_deps", "false",
                            "--use_perturbed_intervals", "false",
                            "--use_relaxations", "false",
                            "--Threads_num", str(Threads_num),
                        ]
                        ready_delta_max_jobs.append((dm_label, dm_cmd))

            # delta_max jobs feed ONLY the table generators (which run in a
            # separate later invocation) — nothing in the main verification
            # phase depends on them, so they are dependency-free.
            #
            # We therefore ALWAYS fold them into the main pool (ready_jobs_all
            # below) rather than running a separate blocking Phase 0.5 pass, so
            # they run concurrently with every other job — and we rank them FIRST
            # (priority sentinel in _job_priority / FIFO prepend) so they are
            # dispatched ahead of the row jobs. This holds for both single- and
            # multi-dataset runs. It removes two problems the old blocking pass
            # had: (1) MULTI-dataset — a global barrier stalled EVERY dataset's
            # main jobs behind one dataset's fresh delta_max; (2) SINGLE-dataset
            # — the pool idled behind a possibly-hours-long `max` solve. They
            # still complete before this invocation returns (hence before any
            # table regen), since the pool drains fully.
            if ready_delta_max_jobs:
                print(f"\n  delta_max: {len(ready_delta_max_jobs)} job(s) folded into the "
                      f"main pool, dispatched first (no blocking Phase 0.5 barrier).")

            # Pseudo-cost extraction has been retired. Technique 3 (var_hint)
            # now uses a continuous transfer-probability signal built from
            # N1's primal + N1 bounds + diff bounds + N2 bounds — none of
            # which require per-variable branching stats. The completeness
            # gate therefore never needs the n1_pseudocosts_*.bin files.
            need_pseudocosts = False
            # Does any combo in this sweep need the N1 probe? If so, the
            # state dir must also contain n1_preact_bounds.bin. Derived from
            # the final combo list (index 5 = n1_probe) so the
            # --advstd_ablations path is covered too.
            need_n1_preact = any(c[5] != "off" for c in technique_combos)
            if need_n1_preact:
                print("This sweep requires n1_preact_bounds.bin (adv_std_n1_probe != off).")

            # Stale lock heuristic: 2× the Gurobi time limit.
            stale_lock_sec = max(2 * args.timeout, 600)
            wait_timeout_sec = stale_lock_sec

            # Cross-c_tag / cross-pert tracking. The (dataset, arch, pert_spec)
            # lock is acquired once (at the first c_tag that needs N1 for that
            # pert), held while ALL chained N1 jobs for it run sequentially, and
            # released the moment the last N1 in the chain finishes. The dataset
            # field keeps the two datasets' identical archs (e.g. cnn1) from
            # colliding in a merged --dataset_group run.
            n1_lock_by_pert = {}              # (dataset, arch, pert_spec) -> lock_path
            n1_pending_count_per_pert = {}    # (dataset, arch, pert_spec) -> int (countdown)
            n1_last_label_per_pert = {}       # (dataset, arch, pert_spec) -> label of most-recent N1 job queued (next c_tag chains behind it)
            n1_pert_by_label = {}             # n1_label -> (dataset, arch, pert_spec) reverse lookup for on_job_done
            n1_label_by_pert_ctag = {}        # (dataset, arch, pert_spec, c_tag) -> n1_label (None if no N1 queued for that triple)

            ready_n1_jobs = []
            ready_std_n2_jobs = []
            ready_advstd_jobs = []
            locked_jobs_by_label = {}         # unlocker_label -> [(label, cmd), ...]  (N1 chains + N1→advstd unlocks)
            n2_skipped = 0
            std_skipped = 0

            # ── Build phase: iterate c_tag × arch × pert_spec × combos.
            # Within each c_tag iteration, build N1 jobs (chained per pert),
            # std-N2 jobs (always ready), and advstd jobs (locked behind the
            # corresponding c_tag's N1 if our process is solving it).
            for c_tag in sweep_ctag_vals:
                print(f"\n{'━' * 60}\nc_tag = {c_tag}  (Julia 1-indexed; Julia writes ctag{c_tag - 1} into filenames)\n{'━' * 60}")

                # Phase 1 (this c_tag): N1 jobs.
                for dataset, arch, model_path in arch_runs:
                    n1_tag, n2_tag, n1_model_p, n2_model_p, model_name, julia_dataset = arch_meta[(dataset, arch)]
                    for pert_name, pert_spec in perturbations_for(dataset):
                        pert_type, eps_str = pert_spec.split(":", 1)
                        arch_prefix = _aprefix(dataset, arch)
                        pkey = (dataset, arch, pert_spec)  # per-dataset N1 lock/chain key

                        n1_state_dir = os.path.join(
                            cwd, "paper_experiments", dataset, f"{arch}_exp",
                            pert_type, f"eps_{eps_str}",
                            f"n1_state_{arch}_{n1_tag}")

                        requested_c_targets = _parse_c_targets(
                            args.ct if args.ct else "2,3,4,5,6,7,8,9,10")
                        missing_c_targets = _n1_state_missing_c_targets(
                            n1_state_dir, c_tag, requested_c_targets,
                            need_pseudocosts, need_n1_preact=need_n1_preact)
                        if not missing_c_targets:
                            print(f"  {arch_prefix}{pert_name} c_tag={c_tag} N1 state already complete for c_targets={requested_c_targets} at {n1_state_dir} — skipping N1 solve")
                            continue

                        wanted = [ct for ct in requested_c_targets if ct != c_tag]
                        if len(missing_c_targets) < len(wanted):
                            have = [ct for ct in wanted if ct not in missing_c_targets]
                            print(f"  {arch_prefix}{pert_name} c_tag={c_tag} N1 state partial — have c_targets={have}, completing missing={missing_c_targets}")
                        else:
                            print(f"  {arch_prefix}{pert_name} c_tag={c_tag} N1 state missing — solving for c_targets={missing_c_targets}")

                        # Acquire the (dataset, arch, pert_spec) lock if we
                        # don't already hold it from an earlier c_tag in this
                        # build phase. The lock is held until the LAST chained
                        # N1 for this pert finishes (released via on_job_done).
                        if pkey not in n1_lock_by_pert:
                            got_lock, lock_path = _acquire_n1_solve_lock(n1_state_dir, stale_lock_sec)
                            if not got_lock:
                                print(f"  {arch_prefix}{pert_name} c_tag={c_tag} another process is solving N1 at {n1_state_dir} — waiting (up to {wait_timeout_sec:.0f}s)")
                                if _wait_for_n1_state(n1_state_dir, need_pseudocosts, wait_timeout_sec, need_n1_preact=need_n1_preact, c_tag=c_tag, c_targets=missing_c_targets):
                                    print(f"  {arch_prefix}{pert_name} c_tag={c_tag} N1 state now ready — skipping our own N1 solve")
                                    continue
                                print(f"  {arch_prefix}{pert_name} WARNING: timed out or other process left state incomplete for our c_targets — attempting to solve N1 ourselves")
                                missing_c_targets = _n1_state_missing_c_targets(
                                    n1_state_dir, c_tag, requested_c_targets,
                                    need_pseudocosts, need_n1_preact=need_n1_preact)
                                if not missing_c_targets:
                                    print(f"  {arch_prefix}{pert_name} c_tag={c_tag} N1 state became complete after wait — skipping our own N1 solve")
                                    continue
                                got_lock, lock_path = _acquire_n1_solve_lock(n1_state_dir, stale_lock_sec)
                                if not got_lock:
                                    print(f"  {arch_prefix}{pert_name} ERROR: still unable to acquire N1 solve lock at {lock_path}. Aborting.")
                                    sys.exit(1)
                            n1_lock_by_pert[pkey] = lock_path
                        # else: lock already held from an earlier c_tag — reuse.

                        ct_arg = ",".join(str(ct) for ct in missing_c_targets)
                        n1_label = f"{arch_prefix}{pert_name} N1-solve c_tag={c_tag}"
                        n1_cmd = [
                            "julia", "run.jl",
                            "--mode", "advanced_standard_n1",
                            "--dataset", julia_dataset,
                            *_benchmark_args(julia_dataset),
                            "--model_name", model_name,
                            "--model_path", n1_model_p,
                            "--model_path2", n2_model_p,
                            "--perturbation", pert_type,
                            "--perturbation_size", eps_str,
                            "--ctag", str(c_tag),
                            "--ct", ct_arg,
                            "--timout", str(args.timeout),
                            "--output_dir", n1_state_dir + "/",
                            "--n1_state_dir", n1_state_dir,
                            "--use_hyper_attack", "true",
                            "--activate_vaghgar_deps", "true",
                            "--use_perturbed_intervals", "true",
                            "--Threads_num", str(Threads_num),
                        ]

                        # First N1 for this (dataset, arch, pert_spec) goes into
                        # ready; subsequent c_tags chain behind the previous N1
                        # so they serialize on the shared state-dir
                        # (n1_preact_bounds.bin and the lock file).
                        prev_label = n1_last_label_per_pert.get(pkey)
                        if prev_label is None:
                            ready_n1_jobs.append((n1_label, n1_cmd))
                        else:
                            locked_jobs_by_label.setdefault(prev_label, []).append((n1_label, n1_cmd))

                        n1_last_label_per_pert[pkey] = n1_label
                        n1_pending_count_per_pert[pkey] = n1_pending_count_per_pert.get(pkey, 0) + 1
                        n1_pert_by_label[n1_label] = pkey
                        n1_label_by_pert_ctag[(dataset, arch, pert_spec, c_tag)] = n1_label

                # Phase 1.5 (this c_tag): standard-N2 (vagharWithPerturbed)
                # baselines — independent of N1 state, always ready.
                # Skipped entirely when --skip_std_n2_baseline is passed.
                if args.skip_std_n2_baseline:
                    arch_iter_phase1_5 = []
                else:
                    arch_iter_phase1_5 = arch_runs
                for dataset, arch, model_path in arch_iter_phase1_5:
                    n1_tag, n2_tag, n1_model_p, n2_model_p, model_name, julia_dataset = arch_meta[(dataset, arch)]
                    for pert_name, pert_spec in perturbations_for(dataset):
                        pert_type, eps_str = pert_spec.split(":", 1)
                        arch_prefix = _aprefix(dataset, arch)

                        requested_c_targets = _parse_c_targets(
                            args.ct if args.ct else "2,3,4,5,6,7,8,9,10")
                        missing_std_cts = _standard_n2_missing_c_targets(
                            pert_spec, cwd, arch, dataset, n2_tag,
                            c_tag, requested_c_targets)
                        if not missing_std_cts:
                            std_skipped += 1
                            continue

                        wanted_std = [ct for ct in requested_c_targets if ct != c_tag]
                        if len(missing_std_cts) < len(wanted_std):
                            have_std = [ct for ct in wanted_std if ct not in missing_std_cts]
                            print(f"  {arch_prefix}{pert_name} c_tag={c_tag} standard N2 partial — have c_targets={have_std}, completing missing={missing_std_cts}")
                        else:
                            print(f"  {arch_prefix}{pert_name} c_tag={c_tag} standard N2 missing — solving for c_targets={missing_std_cts}")

                        std_output_dir = os.path.join(
                            "paper_experiments", dataset, f"{arch}_exp",
                            pert_type, f"eps_{eps_str}",
                            f"vagharWithPerturbed_{arch}_{n2_tag}")

                        std_ct_arg = ",".join(str(ct) for ct in missing_std_cts)
                        std_label = f"{arch_prefix}{pert_name} c_tag={c_tag} standard-N2 (WithPerturbed)"
                        std_cmd = [
                            "julia", "run.jl",
                            "--mode", "standard",
                            "--dataset", julia_dataset,
                            *_benchmark_args(julia_dataset),
                            "--model_name", model_name,
                            "--model_path", n2_model_p,
                            "--perturbation", pert_type,
                            "--perturbation_size", eps_str,
                            "--ctag", str(c_tag),
                            "--ct", std_ct_arg,
                            "--timout", str(args.timeout),
                            "--output_dir", std_output_dir + "/",
                            "--c_tag_mode", "false",
                            "--use_hyper_attack", "true",
                            "--activate_vaghgar_deps", "true",
                            "--use_perturbed_intervals", "true",
                            "--use_relaxations", "false",
                            "--Threads_num", str(Threads_num),
                        ]
                        ready_std_n2_jobs.append((std_label, std_cmd))

                # Phase 2 (this c_tag): advstd jobs.
                for dataset, arch, model_path in arch_runs:
                    n1_tag, n2_tag, n1_model_p, n2_model_p, model_name, julia_dataset = arch_meta[(dataset, arch)]
                    for pert_name, pert_spec in perturbations_for(dataset):
                        pert_type, eps_str = pert_spec.split(":", 1)
                        arch_prefix = _aprefix(dataset, arch)

                        n1_state_dir = os.path.join(
                            cwd, "paper_experiments", dataset, f"{arch}_exp",
                            pert_type, f"eps_{eps_str}",
                            f"n1_state_{arch}_{n1_tag}")

                        for ms, bp, lb, bt, zb, np_, rt, vh, sg, pi, hyp in technique_combos:
                            tech_tag = ""
                            if ms == "true":          tech_tag += "ms"
                            if bp == "rank":          tech_tag += "bpRank"
                            elif bp == "decay":       tech_tag += "bpDecay"
                            if lb == "true":          tech_tag += "lb"
                            # _BoundTightPertRelax subsumes _boundTight (see run.jl).
                            if bt == "true":
                                if rt >= 0.0:         tech_tag += f"btpr{rt}"
                                else:                 tech_tag += "bt"
                            if sg == "true":          tech_tag += "sg"
                            if zb == "true":          tech_tag += "zb"
                            elif zb == "true_nonpre": tech_tag += "zbNoNpre"
                            if np_ == "lp":           tech_tag += "npLP"
                            if vh == "true":          tech_tag += "vhFixed"
                            if pi == "false":         tech_tag += "noPI"
                            no_hyper = (hyp == "false")
                            if no_hyper:              tech_tag += "noHyp"

                            adv_output_dir = os.path.join(
                                "paper_experiments", dataset, f"{arch}_exp",
                                pert_type, f"eps_{eps_str}",
                                f"advStd_{arch}_N1_{n1_tag}")

                            base_name_to_save = f"{n2_tag}_N2_advStd"
                            if ms == "true":          base_name_to_save += "_mipStart"
                            if bp == "rank":          base_name_to_save += "_branchPriRank"
                            elif bp == "decay":       base_name_to_save += "_branchPriDecay"
                            if lb == "true":          base_name_to_save += "_lpBasis"
                            if bt == "true":
                                if rt >= 0.0:
                                    base_name_to_save += f"_BoundTightPertRelax{rt}"
                                else:
                                    base_name_to_save += "_boundTight"
                            if sg == "true":          base_name_to_save += "_SibGate"
                            if zb.startswith("true"): base_name_to_save += "_zonoBounds"
                            if zb == "true_nonpre":   base_name_to_save += "_noNpreZono"
                            if np_ == "lp":           base_name_to_save += "_n1ProbeLP"
                            if vh == "prev":          base_name_to_save += "_varHintFixed"
                            elif vh == "direct":      base_name_to_save += "_varHintDirect"
                            elif vh == "direct_pgd":  base_name_to_save += "_varHintDirectPGD"
                            elif vh == "prev_pgd":    base_name_to_save += "_varHintPrevPGD"
                            # Component-removed combos from --advstd_ablations
                            # carry an explicit _ablation marker. The tag rides
                            # inside --name_to_save, so the saved filename, the
                            # skip-check glob, and run.jl's n2_check all stay
                            # consistent automatically. ('none' controls stay
                            # untagged — they are the paper combo's own rows.)
                            if (ms, bp, lb, bt, zb, np_, rt, vh, sg, pi, hyp) in advstd_ablation_combos:
                                base_name_to_save += "_ablation"

                            requested_c_targets = _parse_c_targets(
                                args.ct if args.ct else "2,3,4,5,6,7,8,9,10")
                            # --geometric_intervals applies to translation/rotation advstd N2 jobs WITH
                            # perturbed intervals (pi=true); run.jl warns+falls back otherwise. run.jl adds
                            # _geomInt; the skip-check keeps these separate from the non-geomInt baseline
                            # so resume is correct.
                            geom_applies_adv = (args.geometric_intervals
                                                and pert_type in ("translation", "rotation")
                                                and pi == "true")
                            for seed in seed_vals:
                                missing_adv_cts = _advstd_missing_c_targets(
                                    cwd, dataset, arch, pert_type, eps_str,
                                    n1_tag, base_name_to_save, seed,
                                    c_tag, requested_c_targets,
                                    geometric_intervals=geom_applies_adv,
                                    perturbed_intervals=(pi == "true"),
                                    hyper_attack=not no_hyper)
                                if not missing_adv_cts:
                                    n2_skipped += 1
                                    continue
                                seed_suffix = f" seed{seed}" if seed != 0 else ""
                                wanted_adv = [ct for ct in requested_c_targets if ct != c_tag]
                                if len(missing_adv_cts) < len(wanted_adv):
                                    tag_suffix = f" (partial: missing {missing_adv_cts})"
                                else:
                                    tag_suffix = ""
                                # Ablation jobs carry an explicit marker in the
                                # label — and therefore in the sweep_logs/
                                # filename, which is built from the label.
                                n2_kind = ("N2-ablation"
                                           if (ms, bp, lb, bt, zb, np_, rt, vh, sg, pi, hyp)
                                           in advstd_ablation_combos else "N2")
                                label = f"{arch_prefix}{pert_name} c_tag={c_tag} {n2_kind}({tech_tag}){seed_suffix}{tag_suffix}"

                                adv_ct_arg = ",".join(str(ct) for ct in missing_adv_cts)
                                cmd = [
                                    "julia", "run.jl",
                                    "--mode", "advanced_standard_n2",
                                    "--dataset", julia_dataset,
                                    *_benchmark_args(julia_dataset),
                                    "--model_name", model_name,
                                    "--model_path", n1_model_p,
                                    "--model_path2", n2_model_p,
                                    "--perturbation", pert_type,
                                    "--perturbation_size", eps_str,
                                    "--ctag", str(c_tag),
                                    "--ct", adv_ct_arg,
                                    "--timout", str(args.timeout),
                                    "--output_dir", adv_output_dir + "/",
                                    "--name_to_save", base_name_to_save,
                                    "--n1_state_dir", n1_state_dir,
                                    # var_hint ablation removes the PGD warm
                                    # start too (no warm start at all).
                                    "--use_hyper_attack",
                                    ("false" if no_hyper else "true"),
                                    "--activate_vaghgar_deps", "true",
                                    "--use_perturbed_intervals", pi,
                                    "--Threads_num", str(Threads_num),
                                    "--adv_std_mip_start", ms,
                                    "--adv_std_branch_priorities", bp,
                                    "--adv_std_lp_basis", lb,
                                    "--adv_std_bound_tightening", bt,
                                    # "true_nonpre" is the zono_npre ablation:
                                    # zonotope ON, its N_pre input OFF.
                                    "--adv_std_zono_bounds",
                                    ("true" if zb == "true_nonpre" else zb),
                                    "--adv_std_zono_npre",
                                    ("false" if zb == "true_nonpre" else "true"),
                                    "--adv_std_n1_probe", np_,
                                    "--adv_std_n2_relax_threshold", str(rt),
                                    "--adv_std_n2_sibling_gate", sg,
                                    "--adv_std_var_hint", vh,
                                    "--gurobi_seed", str(seed),
                                ]
                                if pi == "false" and rt > 0.0:
                                    # PI ablation with the triangle kept: bypass
                                    # run.jl's rt>0-requires-PI guard (sound —
                                    # see --allow_relax_without_pi help).
                                    cmd += ["--allow_relax_without_pi", "true"]
                                if geom_applies_adv:
                                    cmd += ["--geometric_intervals", "true"]
                                n1_label_for_advstd = n1_label_by_pert_ctag.get((dataset, arch, pert_spec, c_tag))
                                if n1_label_for_advstd is None:
                                    ready_advstd_jobs.append((label, cmd))
                                else:
                                    locked_jobs_by_label.setdefault(n1_label_for_advstd, []).append((label, cmd))

                # Phase 2.5 (this c_tag): N1 standard-mode boost jobs
                # (Section 3 of advstd_techniques.tex). Sweep the full grid
                # of (zb × sg × rt × pi) where:
                #   zb = nn1_zono_bounds        (from --sweep_stdboost_zono_bounds, default ['true'])
                #   sg = nn1_sibling_gate       (from --sweep_stdboost_sibling_gate, default ['true'])
                #   rt = nn1_relax_threshold    (non-negative entries of --sweep_adv_std_n2_relax_threshold)
                #   pi = use_perturbed_intervals (from --sweep_stdboost_perturbed_intervals, default ['true'])
                #
                # Unsound-combo filter (mirrors run.jl:487-494):
                #   rt > 0 requires pi=true.   Combos that violate are dropped.
                # No-op SibGate filter (mirrors run.jl:496):
                #   sg=true with rt<0 emits SibGate inactive; we drop it to
                #   avoid burning solve time on a no-op SibGate setup.
                #
                # These jobs have NO inter-job dependency — main_standard reads
                # only its --model_path, so they go straight into
                # ready_advstd_jobs and share the slot pool with everything else.
                # Same boost grid applies to whichever network is targeted
                # (--include_nn1_boost runs on N1; --include_nn2_boost runs on
                # the same boost grid applied to N2). Both can be enabled
                # simultaneously.
                # Build the (role, zb, sg, rt, pi) combo list either from the
                # explicit --stdboost_combos override or from the legacy
                # Cartesian sweep of --sweep_stdboost_* × --include_nn{1,2}_boost.
                explicit_stdboost_combos = []  # list of (role, zb, sg, rt, pi)
                if args.stdboost_combos:
                    for spec in args.stdboost_combos.split(","):
                        spec = spec.strip()
                        if not spec:
                            continue
                        parts = spec.split(":")
                        if len(parts) != 5:
                            print(f"ERROR: --stdboost_combos entry '{spec}' must have 5 colon-separated "
                                  f"fields: <role>:<zb>:<sg>:<rt>:<pi>")
                            sys.exit(1)
                        c_role, c_zb, c_sg, c_rt_str, c_pi = parts
                        c_role = c_role.upper()
                        if c_role not in ("N1", "N2"):
                            print(f"ERROR: --stdboost_combos role '{c_role}' must be N1 or N2 "
                                  f"(in entry '{spec}')")
                            sys.exit(1)
                        c_zb = c_zb.lower(); c_sg = c_sg.lower(); c_pi = c_pi.lower()
                        for fld, name in ((c_zb, "zb"), (c_sg, "sg"), (c_pi, "pi")):
                            if fld not in ("true", "false"):
                                print(f"ERROR: --stdboost_combos field {name}='{fld}' must be "
                                      f"true|false (in entry '{spec}')")
                                sys.exit(1)
                        try:
                            c_rt = float(c_rt_str)
                        except ValueError:
                            print(f"ERROR: --stdboost_combos rt='{c_rt_str}' must be a float "
                                  f"(in entry '{spec}')")
                            sys.exit(1)
                        if c_rt > 0.0 and c_pi == "false":
                            print(f"ERROR: --stdboost_combos entry '{spec}' is unsound: "
                                  f"rt>0 requires pi=true (run.jl:487-494).")
                            sys.exit(1)
                        if c_sg == "true" and c_rt < 0.0:
                            print(f"ERROR: --stdboost_combos entry '{spec}' is no-op: "
                                  f"sg=true with rt<0 emits inactive SibGate.")
                            sys.exit(1)
                        explicit_stdboost_combos.append((c_role, c_zb, c_sg, c_rt, c_pi))
                else:
                    stdboost_targets = []
                    if args.include_nn1_boost:
                        stdboost_targets.append("N1")
                    if args.include_nn2_boost:
                        stdboost_targets.append("N2")
                    if stdboost_targets:
                        nn_rt_vals = [r for r in relax_t_vals if r >= 0.0]
                        nn_zb_vals = (args.sweep_stdboost_zono_bounds
                                      if args.sweep_stdboost_zono_bounds else ["true"])
                        nn_sg_vals = (args.sweep_stdboost_sibling_gate
                                      if args.sweep_stdboost_sibling_gate else ["true"])
                        nn_pi_vals = (args.sweep_stdboost_perturbed_intervals
                                      if args.sweep_stdboost_perturbed_intervals else ["true"])
                        nn_zb_vals = [v.lower() for v in nn_zb_vals]
                        nn_sg_vals = [v.lower() for v in nn_sg_vals]
                        nn_pi_vals = [v.lower() for v in nn_pi_vals]
                        for role in stdboost_targets:
                            for zb in nn_zb_vals:
                                for sg in nn_sg_vals:
                                    for rt in nn_rt_vals:
                                        for pi in nn_pi_vals:
                                            if rt > 0.0 and pi == "false":
                                                continue
                                            if sg == "true" and rt < 0.0:
                                                continue
                                            explicit_stdboost_combos.append((role, zb, sg, rt, pi))

                # ── --stdboost_ablations: leave-one-out combos for the "ours"
                # column. Appends role-N2 combos derived from the full paper
                # combo (zb=true, sg=true, rt=<single non-negative τ from
                # --sweep_adv_std_n2_relax_threshold>, pi=true), one per
                # component token. pert_intervals keeps rt/SibGate (true
                # leave-one-out) — the job builder below adds
                # --allow_relax_without_pi true for rt>0 ∧ pi=false combos,
                # which only this path can produce (manual --stdboost_combos
                # entries stay strictly validated above).
                stdboost_ablation_combos = set()  # combos to tag _ablation in filenames
                if args.stdboost_ablations:
                    _nn_rts = [r for r in relax_t_vals if r >= 0.0]
                    if len(_nn_rts) != 1:
                        print(f"ERROR: --stdboost_ablations needs exactly one non-negative "
                              f"--sweep_adv_std_n2_relax_threshold value as the base τ; "
                              f"got {_nn_rts}")
                        sys.exit(1)
                    _tau = _nn_rts[0]
                    _stdboost_abl_tokens = []
                    for _tok_raw in args.stdboost_ablations:
                        _tok = _tok_raw.strip().lower()
                        if _tok == "pi":
                            _tok = "pert_intervals"
                        zb, sg, rt, pi = "true", "true", _tau, "true"
                        if _tok == "none":
                            pass
                        elif _tok == "zono":
                            zb = "false"
                        elif _tok == "triangle":
                            # Whole technique off; τ = 0 emits no relaxations.
                            rt, sg = 0.0, "false"
                        elif _tok == "zono_triangle":
                            # Both bound-tightening techniques off at once
                            # (see the --advstd_ablations note): zono and
                            # triangle overlap, so a single-component removal
                            # can be masked by the other still tightening.
                            zb, rt, sg = "false", 0.0, "false"
                        elif _tok == "pert_intervals":
                            pi = "false"
                        elif _tok == "var_hint":
                            print("ERROR: --stdboost_ablations var_hint — variable hints are "
                                  "an advstd (transfer) technique; standard mode has no N1. "
                                  "Use --advstd_ablations var_hint (or warm_start) instead.")
                            sys.exit(1)
                        else:
                            print(f"ERROR: unknown --stdboost_ablations component '{_tok_raw}' "
                                  f"(expected none | zono | triangle | zono_triangle | "
                                  f"pert_intervals)")
                            sys.exit(1)
                        combo = ("N2", zb, sg, rt, pi)
                        if combo in explicit_stdboost_combos:
                            if c_tag == sweep_ctag_vals[0]:
                                print(f"  [--stdboost_ablations] '{_tok}' duplicates an "
                                      f"existing combo {combo} — skipping.")
                            continue
                        explicit_stdboost_combos.append(combo)
                        _stdboost_abl_tokens.append(_tok)
                        # Component-removed combos get an _ablation filename
                        # marker; the 'none' control stays untagged so its rows
                        # double as the paper combo's rows.
                        if _tok != "none":
                            stdboost_ablation_combos.add(combo)
                    if _stdboost_abl_tokens and c_tag == sweep_ctag_vals[0]:
                        print(f"  [--stdboost_ablations] added {len(_stdboost_abl_tokens)} "
                              f"N2stdBoost combo(s): [{', '.join(_stdboost_abl_tokens)}] "
                              f"(base τ = {_tau})")

                # --n2_tables_only: N1stdBoost combos only populate the
                # source-network (N1) wide tables, so drop them here while
                # keeping every N2stdBoost combo. (advstd-N2 still gets its N1
                # state from the Phase 1 N1-solve, which is unaffected.)
                if args.n2_tables_only and explicit_stdboost_combos:
                    n1_combos = [c for c in explicit_stdboost_combos if c[0] == "N1"]
                    explicit_stdboost_combos = [c for c in explicit_stdboost_combos
                                                if c[0] != "N1"]
                    if n1_combos and c_tag == sweep_ctag_vals[0]:
                        print(f"  [n2_tables_only] skipping {len(n1_combos)} N1stdBoost "
                              f"combo(s) (N1-only tables); keeping "
                              f"{len(explicit_stdboost_combos)} N2stdBoost combo(s)")

                if explicit_stdboost_combos:
                    for dataset, arch, model_path in arch_runs:
                        n1_tag, n2_tag, n1_model_p, n2_model_p, model_name, julia_dataset = arch_meta[(dataset, arch)]
                        for pert_name, pert_spec in perturbations_for(dataset):
                            pert_type, eps_str = pert_spec.split(":", 1)
                            arch_prefix = _aprefix(dataset, arch)

                            requested_c_targets = _parse_c_targets(
                                args.ct if args.ct else "2,3,4,5,6,7,8,9,10")

                            for role, zb, sg, rt, pi in explicit_stdboost_combos:
                                if role == "N1":
                                    target_model_p = n1_model_p
                                    target_tag     = n1_tag
                                else:
                                    target_model_p = n2_model_p
                                    target_tag     = n2_tag
                                nn_output_dir = os.path.join(
                                    "paper_experiments", dataset, f"{arch}_exp",
                                    pert_type, f"eps_{eps_str}",
                                    f"{role}stdBoost_{arch}_{target_tag}")
                                base_name_to_save_nn = f"{target_tag}_{role}"
                                # --stdboost_ablations component-removed combos
                                # carry an explicit _ablation marker; it rides
                                # inside --name_to_save so the saved filename
                                # and _stdboost_missing_c_targets' regex (built
                                # from this same base) stay consistent.
                                if (role, zb, sg, rt, pi) in stdboost_ablation_combos:
                                    base_name_to_save_nn += "_ablation"
                                # --geometric_intervals applies to translation/rotation stdBoost jobs WITH
                                # perturbed intervals (pi=true); run.jl warns+falls back otherwise (no _geomInt tag).
                                geom_applies_std = (args.geometric_intervals
                                                    and pert_type in ("translation", "rotation")
                                                    and pi == "true")

                                for seed in seed_vals:
                                    missing_nn_cts = _stdboost_missing_c_targets(
                                        os.path.join(cwd, nn_output_dir),
                                        arch, pert_type, eps_str,
                                        base_name_to_save_nn, seed,
                                        zb, sg, rt, pi,
                                        c_tag, requested_c_targets,
                                        geometric_intervals=geom_applies_std)
                                    if not missing_nn_cts:
                                        n2_skipped += 1
                                        continue
                                    seed_suffix = f" seed{seed}" if seed != 0 else ""
                                    wanted_nn = [ct for ct in requested_c_targets if ct != c_tag]
                                    if len(missing_nn_cts) < len(wanted_nn):
                                        tag_suffix = f" (partial: missing {missing_nn_cts})"
                                    else:
                                        tag_suffix = ""
                                    combo_tag = ""
                                    if zb == "true": combo_tag += "zb"
                                    if rt >= 0.0:    combo_tag += f"btpr{rt}"
                                    if sg == "true": combo_tag += "sg"
                                    if pi == "true": combo_tag += "pi"
                                    if not combo_tag: combo_tag = "plain"
                                    # Ablation jobs carry an explicit marker in
                                    # the label — and therefore in the
                                    # sweep_logs/ filename (built from the label).
                                    sb_kind = (f"{role}stdBoost-ablation"
                                               if (role, zb, sg, rt, pi)
                                               in stdboost_ablation_combos
                                               else f"{role}stdBoost")
                                    nn_label = f"{arch_prefix}{pert_name} c_tag={c_tag} {sb_kind}({combo_tag}){seed_suffix}{tag_suffix}"
                                    nn_ct_arg = ",".join(str(ct) for ct in missing_nn_cts)
                                    nn_cmd = [
                                        "julia", "run.jl",
                                        "--mode", "standard",
                                        "--dataset", julia_dataset,
                                        *_benchmark_args(julia_dataset),
                                        "--model_name", model_name,
                                        "--model_path", target_model_p,
                                        "--perturbation", pert_type,
                                        "--perturbation_size", eps_str,
                                        "--ctag", str(c_tag),
                                        "--ct", nn_ct_arg,
                                        "--timout", str(args.timeout),
                                        "--output_dir", nn_output_dir + "/",
                                        "--name_to_save", base_name_to_save_nn,
                                        "--c_tag_mode", "false",
                                        "--use_hyper_attack", "true",
                                        "--activate_vaghgar_deps", "true",
                                        "--use_perturbed_intervals", pi,
                                        "--use_relaxations", "false",
                                        "--Threads_num", str(Threads_num),
                                        "--nn1_zono_bounds", zb,
                                        "--nn1_relax_threshold", str(rt),
                                        "--nn1_sibling_gate", sg,
                                        "--gurobi_seed", str(seed),
                                    ]
                                    if pi == "false" and rt > 0.0:
                                        # PI ablation with the triangle kept
                                        # (--stdboost_ablations pert_intervals):
                                        # bypass run.jl's rt>0-requires-PI guard.
                                        nn_cmd += ["--allow_relax_without_pi", "true"]
                                    if geom_applies_std:
                                        nn_cmd += ["--geometric_intervals", "true"]
                                    ready_advstd_jobs.append((nn_label, nn_cmd))

            # Move stdBoost jobs with --use_perturbed_intervals=false to the
            # tail of the ready queue so they execute last. Stable sort on a
            # boolean key (False<True) preserves the relative order of every
            # other job (advstd-N2, std-N2, and pi=true stdBoost stay where
            # the build loop placed them).
            def _is_pi_false(job):
                cmd = job[1]
                try:
                    idx = cmd.index("--use_perturbed_intervals")
                    return cmd[idx + 1] == "false"
                except (ValueError, IndexError):
                    return False

            # Row-priority dispatch is opt-in (--prioritize_rows). In the
            # default (FIFO) path we keep the global pi=false tail-sort. In the
            # row-priority path the pi=false ordering is folded into the
            # priority key, so the global sort is skipped and run_pool sorts the
            # queue by (within_row, ds_rank, phase_rank, pi_false) before every
            # slot-fill.
            row_priority = None
            if args.prioritize_rows:
                # Row = (arch, pert_spec) WITHIN a dataset, ordered as built
                # (arch outer, pert inner). The primary key is the within-dataset
                # row index and the secondary key is the dataset's CLI order, so
                # dispatch interleaves datasets round-robin: dataset-A row 0,
                # dataset-B row 0, dataset-A row 1, dataset-B row 1, …  (when one
                # dataset runs out of rows the other keeps filling). Within a
                # single (dataset, row) cell: N1 (which unlocks advstd) before
                # std/stdBoost before advstd-N2, and pi=false last. For a
                # single-dataset run ds_rank is constant, so this degenerates to
                # the original (row_index, phase_rank, pi_false) ordering.
                _ds_rank = {ds: i for i, (ds, _) in enumerate(dataset_groups)}
                _within_row = {}
                for ds, aruns in dataset_groups:
                    ri = 0
                    for a, _ in aruns:
                        for _, pert_spec in perturbations_for(ds):
                            _within_row[(ds, a, pert_spec)] = ri
                            ri += 1
                _n_rows = len(_within_row)
                # output_dir looks like .../paper_experiments/<dataset>/<arch>_exp/...
                _ds_arch_exp_re = re.compile(r"paper_experiments/([^/]+)/([^/]+)_exp/")

                def _cmd_opt(cmd, name):
                    try:
                        return cmd[cmd.index(name) + 1]
                    except (ValueError, IndexError):
                        return None

                def _job_priority(job):
                    cmd = job[1]
                    # delta_max jobs (--perturbation max) are dependency-free and
                    # must be dispatched FIRST: give them a sentinel key that
                    # sorts ahead of every real (within>=0, ...) row job.
                    if _cmd_opt(cmd, "--perturbation") == "max":
                        return (-1, -1, -1, False)
                    pert = _cmd_opt(cmd, "--perturbation")
                    psize = _cmd_opt(cmd, "--perturbation_size")
                    pert_spec = f"{pert}:{psize}" if pert and psize else None
                    out_dir = _cmd_opt(cmd, "--output_dir") or ""
                    m = _ds_arch_exp_re.search(out_dir)
                    ds = m.group(1) if m else None
                    arch = m.group(2) if m else None
                    within = _within_row.get((ds, arch, pert_spec), _n_rows)
                    drank = _ds_rank.get(ds, len(_ds_rank))
                    mode = _cmd_opt(cmd, "--mode")
                    if mode == "advanced_standard_n1":
                        phase_rank = 0
                    elif mode == "advanced_standard_n2":
                        phase_rank = 2
                    else:
                        phase_rank = 1
                    return (within, drank, phase_rank, _is_pi_false(job))

                row_priority = _job_priority
            else:
                ready_advstd_jobs.sort(key=_is_pi_false)

            # ── Run phase: single merged pool across all c_tags ─────────
            # Count N1 vs advstd within locked_jobs_by_label so the banner is
            # meaningful even when N1→N1 chain entries are present.
            n1_locked_count = sum(
                1 for v in locked_jobs_by_label.values()
                for j in v if j[0] in n1_pert_by_label)
            advstd_locked_count = sum(
                len(v) for v in locked_jobs_by_label.values()) - n1_locked_count
            total_n1 = len(ready_n1_jobs) + n1_locked_count
            total_advstd = len(ready_advstd_jobs) + advstd_locked_count

            skip_note = ""
            if n2_skipped:
                skip_note += f" advstd-skipped={n2_skipped}"
            if std_skipped:
                skip_note += f" std-skipped={std_skipped}"
            skip_note = (" " + skip_note.strip()) if skip_note else ""

            order_note = " [row-priority]" if row_priority else ""
            # delta_max jobs are folded into this pool for both single- and
            # multi-dataset runs now (no blocking Phase 0.5); when present they
            # are dispatched first (priority sentinel / FIFO prepend).
            _phase_label = "0.5+1+1.5+2" if ready_delta_max_jobs else "1+1.5+2"
            _dm_seg = f"{len(ready_delta_max_jobs)} delta_max + " if ready_delta_max_jobs else ""
            print(f"\n── Phase {_phase_label} merged across c_tags={list(sweep_ctag_vals)}{order_note}: "
                  f"{_dm_seg}"
                  f"{total_n1} N1 ({len(ready_n1_jobs)} ready, {n1_locked_count} chained) + "
                  f"{len(ready_std_n2_jobs)} std-N2 + "
                  f"{total_advstd} advstd ({len(ready_advstd_jobs)} ready, {advstd_locked_count} waiting on N1)"
                  f"{skip_note} ──")

            def _on_pool_job_done(label):
                """Decrement the per-pert N1 pending count when an N1 job
                finishes; release that pert's lock when the chain is done.
                Non-N1 labels are a no-op."""
                pert_key = n1_pert_by_label.get(label)
                if pert_key is None:
                    return
                remaining = n1_pending_count_per_pert.get(pert_key, 0) - 1
                if remaining <= 0:
                    n1_pending_count_per_pert.pop(pert_key, None)
                    lock_path = n1_lock_by_pert.pop(pert_key, None)
                    if lock_path is not None:
                        _release_n1_solve_lock(lock_path)
                else:
                    n1_pending_count_per_pert[pert_key] = remaining

            # delta_max jobs are dependency-free (nothing in this pool waits on
            # them); folding them in here — instead of a blocking Phase 0.5 —
            # lets them overlap the row jobs rather than barriering the whole
            # run behind a possibly-hours-long `max` solve. They are placed at
            # the FRONT so the FIFO path dispatches them first; the row-priority
            # path pins them first via the _job_priority sentinel above.
            ready_jobs_all = ready_delta_max_jobs + ready_n1_jobs + ready_std_n2_jobs + ready_advstd_jobs
            if ready_jobs_all or locked_jobs_by_label:
                try:
                    run_pool(
                        ready_jobs_all, max_slots, cwd, cores_per_job,
                        "Phase 1+1.5+2 (all c_tags)",
                        locked_jobs=locked_jobs_by_label,
                        on_job_done=_on_pool_job_done,
                        priority=row_priority,
                        group_slots=group_slots, job_group=_job_group,
                    )
                finally:
                    # Defensive: release any (arch, pert_spec) locks still
                    # held (e.g. an N1 job crashed before its on_job_done
                    # fired, or the pool was interrupted).
                    leftover_pert_locks = list(n1_lock_by_pert.items())
                    for pert_key, lock_path in leftover_pert_locks:
                        _release_n1_solve_lock(lock_path)
                        n1_lock_by_pert.pop(pert_key, None)
                    if leftover_pert_locks:
                        print(f"Phase 1+1.5+2: released {len(leftover_pert_locks)} leftover N1 solve lock(s) at shutdown")
            else:
                # Nothing to run at all. Still release any locks we somehow
                # acquired (shouldn't happen but belt+suspenders).
                for pert_key, lock_path in list(n1_lock_by_pert.items()):
                    _release_n1_solve_lock(lock_path)
                    n1_lock_by_pert.pop(pert_key, None)
                print("\n── Phase 1+1.5+2: nothing to run — N1 state, standard-N2, and advstd all complete ──")

        except KeyboardInterrupt:
            print("\nCtrl+C received — terminating all running jobs...")
            sys.exit(1)
        return

    try:
        # ── Build job lists across all arch runs ──────────────────────
        Threads_num = 32
        cores_per_job = Threads_num
        max_slots = _max_slots_for(total_cores, cores_per_job)

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

            for pert_name, pert_spec in perturbations_for(dataset):
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
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    main()
