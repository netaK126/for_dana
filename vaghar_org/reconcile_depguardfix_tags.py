#!/usr/bin/env python3
"""Reconcile missing _depGuardFix filename tags.

Background: the fixed dependency encoder (perturbation_dependencies.jl, the
`has_a_o && has_a_p` guard) has been continuously active since 2026-06-28
12:19. run.jl is supposed to stamp `_depGuardFix` right after `_VagharDeps`
so the paper-tables pipeline (run_relaxation_sweep._is_pre_fix_dropped) keeps
binary-dropping runs. That stamping regressed (~Jul 1 -> Jul 5), so sound runs
were written WITHOUT the tag and silently dropped from the tables.

This script re-adds the tag to exactly those regression victims. It is safe to
run repeatedly (idempotent) and safe to run WHILE old-code Julia jobs are still
producing untagged files -- it only touches files that are provably post-fix,
already fully written, and not colliding with an existing tagged twin.

Guards (a file is renamed ONLY if ALL hold):
  * name has `_VagharDeps` but not `_depGuardFix`
  * `_filename_dropped_binaries` is True  (else it's byte-identical under the
    fix and needs no tag -- and pre-fix all-off files must NOT be resurrected)
  * mtime >= CUTOFF  (first-ever tagged file; before this we cannot prove the
    fix was active, so those files stay dropped as genuinely pre-fix)
  * mtime is older than SETTLE_SECS  (avoid the tiny open("w")..close() window
    of a pair a live job is finalizing right now)
  * the tagged target name does not already exist
"""
import os, sys, glob, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_relaxation_sweep import _filename_dropped_binaries, _is_pre_fix_dropped

# First-ever _depGuardFix mtime (2026-06-28 17:22:22). Fixed constant so the
# cutoff can never drift as new tagged files land.
CUTOFF = 1782667342
SETTLE_SECS = 60


def reconcile(root="paper_experiments", verbose=True):
    now = time.time()
    renamed, skipped_collision, skipped_unsettled = 0, 0, 0
    for tf in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True):
        fn = os.path.basename(tf)
        if "_VagharDeps" not in fn or "_depGuardFix" in fn:
            continue
        if not _filename_dropped_binaries(fn):
            continue  # sound without the tag anyway; leave it
        mt = os.path.getmtime(tf)
        if mt < CUTOFF:
            continue  # genuinely pre-fix: must stay dropped
        if now - mt < SETTLE_SECS:
            skipped_unsettled += 1
            continue  # possibly being written right now
        new_fn = fn.replace("_VagharDeps", "_VagharDeps_depGuardFix", 1)
        assert "_depGuardFix" in new_fn and not _is_pre_fix_dropped(new_fn)
        new_tf = os.path.join(os.path.dirname(tf), new_fn)
        if os.path.exists(new_tf):
            skipped_collision += 1
            continue
        os.rename(tf, new_tf)
        renamed += 1
    if verbose:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] reconcile: renamed={renamed} "
              f"collision={skipped_collision} unsettled={skipped_unsettled}")
    return renamed


if __name__ == "__main__":
    reconcile()
