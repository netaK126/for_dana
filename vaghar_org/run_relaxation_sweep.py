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
    # ("trans(1,3)",        "translation:1,3"),
    ("trans(3,1)",        "translation:3,1"),
    ("trans(3,3)",        "translation:3,3"),
    ("occ(5,5,5)",        "occ:5,5,5"),
    ("occ(3,3,5)",        "occ:3,3,5"),
    ("occ(14,14,9)",        "occ:14,14,9"),
    ("occ(1,1,9)",        "occ:1,1,9"),
    ("contrast(1.5)",      "contrast:1.5"),
    ("contrast(1.2)",      "contrast:1.2"),
    ("rotation(10)",      "rotation:10"),
    ("rotation(5)",      "rotation:5"),
    # ("occ(1,1,5)",        "occ:1,1,5"),
    ("linf(0.05)",        "linf:0.05"),    
    ("linf(0.1)",         "linf:0.1"),     
    ("brightness(0.25)",  "brightness:0.25"), 
    ("brightness(0.1)",  "brightness:0.1")
]

# ── Transfer sweep parameters ────────────────────────────────────────────
THRESHOLDS = [0]#[0, 0.05] # focused on best T_relax candidate
OPT_INTERVALS = ["true"]#["true", "false"]

# ── CPU pinning ──────────────────────────────────────────────────────────
CORES_PER_JOB = 32
# First core to use (reserve 0-7). Override via SWEEP_CORE_START so two
# concurrent sweeps can claim disjoint core windows and not fight each other.
CORE_START = int(os.environ.get("SWEEP_CORE_START", "8"))
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
                              geometric_intervals=False):
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
        dir_pattern = f"vagharWithPerturbed_{arch}_{n2_tag}"
    else:
        dir_pattern = "vagharWithPerturbed_*_sgd_itr*"
    n2_dirs = glob.glob(os.path.join(eps_dir, dir_pattern))
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
          f"({cores_per_job} cores/job, cores {CORE_START}-{CORE_START + max_slots * cores_per_job - 1})")
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


def _dataset_config_key(name):
    """Key into run_experiment.DATASET_CONFIG for a (possibly aliased) dataset name."""
    return "fashion_mnist" if name in _FASHION_ALIASES else name


def _julia_dataset_name(name):
    """Dataset identifier that run.jl / hyper_attack.py understand."""
    return "fmnist" if name in _FASHION_ALIASES else name


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
    model = model_cls(k=channels, w=w, h=h).to(device)
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

        # N1 (source network) -> appendix.
        try:
            n1_tex = os.path.join(cwd, "neta-s-paper", "sections",
                                  "sec_appendix_percell.tex")
            if os.path.exists(n1_tex):
                updater.regenerate_aaai_wide_perarch_section(
                    n1_tex, cwd, dataset_guess, arch_runs,
                    parse_result_file,
                    seeds_filter=combo_ranking_seeds,
                    force_timeout=force_timeout,
                    rerun_timeout_eps=rerun_timeout_eps,
                    roles={"N1"}, label_suffix="-n1",
                    begin_mark=updater.AAAI_WIDE_N1_BEGIN_MARK + _mslug,
                    end_mark=updater.AAAI_WIDE_N1_END_MARK + _mslug,
                    ds_label_suffix=_lslug)
        except Exception as exc:
            print(f"[tex-update] aaai_safe_wide (N1 appendix) block "
                  f"error: {exc}")

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
    print(f"[paper-build] rebuilt {os.path.join(paper_dir, 'main.pdf')}")


def _regen_paper_tables_from_txt(arch_runs, cwd, dataset, combo_ranking_seeds,
                                 combination_table=None, force_timeout=None,
                                 rerun_timeout_eps=30.0):
    """Regenerate ONLY the neta-s-paper per-cell tables + N2 charts, sourcing
    the transfer (advstd N2) column DIRECTLY from the advStd .txt files (no
    CSVs, no standard-baseline pairing). Honors --combination_table (combo
    filter) and --force_timeout (cross-cap timeout dedup) exactly like the old
    --find_advstd path, but writes no CSV and does not touch
    advstd_techniques.tex.
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
        perts=PERTURBATIONS,
        combination_filter=combination_filter,
        # Drop pre-fix files that relaxed >=1 binary (unsound under the
        # perturbation-dependency fix) on BOTH vaghar/ours and transfer rows --
        # the same predicate the sweep skip-check uses.
        stale_fn=_is_pre_fix_dropped,
    )

    # N2 (target network) -> Evaluation body, as per-perturbation charts.
    body_tex = os.path.join(cwd, "neta-s-paper", "sections", "sec_evaluation.tex")
    if os.path.exists(body_tex) and hasattr(
            updater, "regenerate_aaai_n2_charts_section"):
        try:
            updater.regenerate_aaai_n2_charts_section(
                body_tex, cwd, dataset, arch_runs,
                begin_mark=updater.AAAI_N2_CHARTS_BEGIN_MARK + _mslug,
                end_mark=updater.AAAI_N2_CHARTS_END_MARK + _mslug,
                ds_label_suffix=_lslug, **common)
        except Exception as exc:
            print(f"[paper-tables] aaai_n2_charts (body) block error: {exc}")

    # N2 + N1 (source) per-cell tables -> appendix.
    percell_tex = os.path.join(cwd, "neta-s-paper", "sections",
                               "sec_appendix_percell.tex")
    if os.path.exists(percell_tex):
        try:
            updater.regenerate_aaai_wide_perarch_section(
                percell_tex, cwd, dataset, arch_runs, roles={"N2"},
                begin_mark=updater.AAAI_WIDE_N2_APPENDIX_BEGIN_MARK + _mslug,
                end_mark=updater.AAAI_WIDE_N2_APPENDIX_END_MARK + _mslug,
                ds_label_suffix=_lslug, **common)
        except Exception as exc:
            print(f"[paper-tables] aaai_safe_wide (N2 appendix) block "
                  f"error: {exc}")
        try:
            # advstd is N2-only, so the N1 table's transfer column stays '---'.
            updater.regenerate_aaai_wide_perarch_section(
                percell_tex, cwd, dataset, arch_runs, roles={"N1"},
                label_suffix="-n1",
                begin_mark=updater.AAAI_WIDE_N1_BEGIN_MARK + _mslug,
                end_mark=updater.AAAI_WIDE_N1_END_MARK + _mslug,
                ds_label_suffix=_lslug, **common)
        except Exception as exc:
            print(f"[paper-tables] aaai_safe_wide (N1 appendix) block "
                  f"error: {exc}")

    _recompile_neta_s_paper(cwd)


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
    parser.add_argument("--combination_table", type=str, default=None,
                        metavar="BT:VH:TAU[,BT:VH:TAU,...]",
                        help="Restrict the advstd tables in advstd_techniques.tex (overall "
                             "ranking, per-arch c_src-tinted green/yellow/pink blocks, and "
                             "per-arch TIME_LIMIT gap-comparison tables) to one or more "
                             "combinations. Format '<bound_tight>:<varHint>:<tau>' per combo, "
                             "comma-separated. Examples: 'zono:prev_pgd:0.5' or "
                             "'interval:prev_pgd:0.5+sg,zono:prev_pgd:0.5+sg'. Use the '+sg' "
                             "suffix on tau to select SibGate (Technique 4) rows. Only takes "
                             "effect when the tex tables are rewritten (after "
                             "--find_advstd_faster_than_standard). 'off' is accepted as an "
                             "alias for varHint='no'.")
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
                             "Default: ['rank'].")
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
                             "Default: 2,3,4,5,6,7,8,9,10. Use to restrict to specific scenarios.")
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

    # ── Paper-tables mode: regenerate neta-s-paper tables from advStd txt ──
    if args.paper_tables_from_txt:
        # --dataset may be comma-separated; regenerate each dataset's
        # neta-s-paper per-cell tables + N2 charts straight from the advStd
        # .txt files (no CSVs), reusing the same --arch_models.
        pt_datasets = [d.strip() for d in dataset.split(",") if d.strip()]
        for pt_dataset in pt_datasets:
            if len(pt_datasets) > 1:
                print(f"\n===== Regenerating neta-s-paper tables (from advStd "
                      f"txt) for dataset: {pt_dataset} =====")
            _regen_paper_tables_from_txt(
                arch_runs, cwd, pt_dataset, args.combo_ranking_seeds,
                combination_table=args.combination_table,
                force_timeout=args.force_timeout,
                rerun_timeout_eps=args.rerun_timeout_eps)
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
                force_timeout=args.force_timeout,
                rerun_timeout_eps=args.rerun_timeout_eps)
        return

    cores_per_job = CORES_PER_JOB
    max_slots = (total_cores - CORE_START) // cores_per_job

    # ── Advanced-standard mode (two-phase: N1 once, then N2 sweep) ──────
    if args.advanced_standard:
        try:
            Threads_num = 32
            cores_per_job = Threads_num
            max_slots = (total_cores - CORE_START) // cores_per_job

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
                              f"{max_slots} slots are available (cores {CORE_START}-"
                              f"{CORE_START + max_slots * cores_per_job - 1}, {cores_per_job}/job)")
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
            branch_pri_vals = [_norm_bp(v) for v in args.sweep_adv_std_branch_priorities] if args.sweep_adv_std_branch_priorities else ["rank"]
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
            technique_combos = [
                (ms, bp, lb, bt, zb, np_, rt, vh, sg)
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

            print(f"\nAdvanced-standard: {len(technique_combos)} technique combinations × {len(seed_vals)} seed(s) (all-off + zono/probe/relax-without-boundTight + sibgate-without-relax excluded):")
            for ms, bp, lb, bt, zb, np_, rt, vh, sg in technique_combos:
                # When bt=true and rt>=0, boundTight is subsumed by BoundTightPertRelax in the filename.
                bt_desc = f"BoundTightPertRelax{rt}" if (bt == "true" and rt >= 0.0) else \
                          ("boundTight" if bt == "true" else "off")
                print(f"  mipStart={ms}  branchPri={bp}  lpBasis={lb}  boundTight/BTPR={bt_desc}  zonoBounds={zb}  n1Probe={np_}  varHint={vh}  sibGate={sg}")
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
                for role, role_tag, role_model_p in (("N1", n1_tag, n1_model_p),
                                                     ("N2", n2_tag, n2_model_p)):
                    # --n2_tables_only: N1's delta_max only feeds the source-network
                    # (N1) tables; the N2 rows use N2's own delta_max. Skip it.
                    if args.n2_tables_only and role == "N1":
                        continue
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
                        dummy_ct = c_src + 1 if c_src < 10 else 1
                        dm_label = f"{arch_prefix}{role} delta_max c_src={c_src}"
                        dm_cmd = [
                            "julia", "run.jl",
                            "--mode", "standard",
                            "--dataset", julia_dataset,
                            "--model_name", model_name,
                            "--model_path", role_model_p,
                            "--perturbation", "max",
                            "--perturbation_size", "0",
                            "--ctag", str(c_src),
                            "--ct", str(dummy_ct),
                            "--timout", str(args.timeout),
                            "--output_dir", dm_out_dir + "/",
                            "--c_tag_mode", "false",
                            "--use_hyper_attack", "false",
                            "--activate_vaghgar_deps", "false",
                            "--use_perturbed_intervals", "false",
                            "--use_relaxations", "false",
                            "--Threads_num", str(Threads_num),
                        ]
                        ready_delta_max_jobs.append((dm_label, dm_cmd))

            if ready_delta_max_jobs:
                run_pool(
                    ready_delta_max_jobs, max_slots, cwd, cores_per_job,
                    "Phase 0.5 (delta_max)",
                    group_slots=group_slots, job_group=_job_group,
                )

            # Pseudo-cost extraction has been retired. Technique 3 (var_hint)
            # now uses a continuous transfer-probability signal built from
            # N1's primal + N1 bounds + diff bounds + N2 bounds — none of
            # which require per-variable branching stats. The completeness
            # gate therefore never needs the n1_pseudocosts_*.bin files.
            need_pseudocosts = False
            # Does any combo in this sweep need the N1 probe? If so, the
            # state dir must also contain n1_preact_bounds.bin.
            need_n1_preact = any(v != "off" for v in n1_probe_vals)
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
                    for pert_name, pert_spec in perts:
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
                    for pert_name, pert_spec in perts:
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
                    for pert_name, pert_spec in perts:
                        pert_type, eps_str = pert_spec.split(":", 1)
                        arch_prefix = _aprefix(dataset, arch)

                        n1_state_dir = os.path.join(
                            cwd, "paper_experiments", dataset, f"{arch}_exp",
                            pert_type, f"eps_{eps_str}",
                            f"n1_state_{arch}_{n1_tag}")

                        for ms, bp, lb, bt, zb, np_, rt, vh, sg in technique_combos:
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
                            if np_ == "lp":           tech_tag += "npLP"
                            if vh == "true":          tech_tag += "vhFixed"

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
                            if zb == "true":          base_name_to_save += "_zonoBounds"
                            if np_ == "lp":           base_name_to_save += "_n1ProbeLP"
                            if vh == "prev":          base_name_to_save += "_varHintFixed"
                            elif vh == "direct":      base_name_to_save += "_varHintDirect"
                            elif vh == "direct_pgd":  base_name_to_save += "_varHintDirectPGD"
                            elif vh == "prev_pgd":    base_name_to_save += "_varHintPrevPGD"

                            requested_c_targets = _parse_c_targets(
                                args.ct if args.ct else "2,3,4,5,6,7,8,9,10")
                            # --geometric_intervals applies to translation/rotation advstd N2 jobs (which always
                            # have use_perturbed_intervals=true). run.jl adds _geomInt; the skip-check keeps these
                            # separate from the non-geomInt baseline so resume is correct.
                            geom_applies_adv = (args.geometric_intervals
                                                and pert_type in ("translation", "rotation"))
                            for seed in seed_vals:
                                missing_adv_cts = _advstd_missing_c_targets(
                                    cwd, dataset, arch, pert_type, eps_str,
                                    n1_tag, base_name_to_save, seed,
                                    c_tag, requested_c_targets,
                                    geometric_intervals=geom_applies_adv)
                                if not missing_adv_cts:
                                    n2_skipped += 1
                                    continue
                                seed_suffix = f" seed{seed}" if seed != 0 else ""
                                wanted_adv = [ct for ct in requested_c_targets if ct != c_tag]
                                if len(missing_adv_cts) < len(wanted_adv):
                                    tag_suffix = f" (partial: missing {missing_adv_cts})"
                                else:
                                    tag_suffix = ""
                                label = f"{arch_prefix}{pert_name} c_tag={c_tag} N2({tech_tag}){seed_suffix}{tag_suffix}"

                                adv_ct_arg = ",".join(str(ct) for ct in missing_adv_cts)
                                cmd = [
                                    "julia", "run.jl",
                                    "--mode", "advanced_standard_n2",
                                    "--dataset", julia_dataset,
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
                                    "--adv_std_n2_sibling_gate", sg,
                                    "--adv_std_var_hint", vh,
                                    "--gurobi_seed", str(seed),
                                ]
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
                        for pert_name, pert_spec in perts:
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
                                    nn_label = f"{arch_prefix}{pert_name} c_tag={c_tag} {role}stdBoost({combo_tag}){seed_suffix}{tag_suffix}"
                                    nn_ct_arg = ",".join(str(ct) for ct in missing_nn_cts)
                                    nn_cmd = [
                                        "julia", "run.jl",
                                        "--mode", "standard",
                                        "--dataset", julia_dataset,
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
                        for _, pert_spec in perts:
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
            print(f"\n── Phase 1+1.5+2 merged across c_tags={list(sweep_ctag_vals)}{order_note}: "
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

            ready_jobs_all = ready_n1_jobs + ready_std_n2_jobs + ready_advstd_jobs
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
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    main()
