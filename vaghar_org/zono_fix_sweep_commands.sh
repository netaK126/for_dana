#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Zono-fix re-run sweep — SINGLE INVOCATION version (2026-08-05).
# Review before running; nothing has been executed.
#
# Target tree: paper_experiments_with_zono_fix (cleaned copy — only the vaghar
# column [vagharNoPerturbed_* dirs + all-off/no-PI stdBoost cells] and the
# delta_max/'max' family survive; vaghar brightness/contrast deleted).
# The original paper_experiments is untouched.
#
# One run_relaxation_sweep.py call schedules ALL FOUR configurations:
#   (1) vaghar on N2 .......... --stdboost_combos N2:false:false:0:false
#                               (all-off/no-PI standard run; pooled into the
#                               \baseline/vaghar column by the tables)
#   (2) BLEND w/o transfer     --stdboost_combos N2:true:true:0.5:true
#       on N2 ................. (zono Source B + SibGate + tau=0.5 + PI;
#                               Source A cannot exist in standard mode)
#   (3) advanced_standard_n1   Phase 1, auto-scheduled before (4): plain N1
#       on Npre ............... solve + state save (deps+PI+PGD warm start;
#                               technique flags don't apply to Phase 1)
#   (4) advanced_standard_n2   Phase 2 with ALL techniques ON incl. Source A
#       on N .................. (adv_std_zono_npre stays at its default true):
#                               bound_tightening + zono + tau=0.5 + SibGate +
#                               var-hints(prev_pgd) + PI + deps + PGD
#
# Model scope: ONLY the paper's Table 1 (tab:networks) networks —
#   MNIST 3x50 + 3x100 + conv1(=cnn1), Fashion-MNIST 3x50 + conv1(=cnn1),
#   CIFAR-10 conv4(=cnn5), HAR 1x500(=har)
# => 44 cells / 744 vaghar pairs.
#
# Pair restriction: --only_vaghar_pairs (new flag) prunes every phase's job
# list to the Julia-indexed (c_tag, c_target) pairs that exist in the kept
# vaghar-column files of the SAME (dataset, arch, perturbation, eps) cell,
# so the union --sweep_ctag/--ct below can never over-schedule.
# delta_max normalizer runs are pruned to c_srcs with >= 1 vaghar pair.
#
# optimized_interval: always on (flag removed from run.jl).
# Gurobi seed 4 matches the kept vaghar cells' _seed4 tag.
# --skip_std_n2_baseline: the vagharWithPerturbed 'PI' baseline is not one of
#   the four configs. --n2_tables_only: no N1 delta_max / N1stdBoost jobs.
# --geometric_intervals: applies itself only to translation/rotation jobs.
#
# NOT covered (flagged, not guessed):
#   * acas linf eps_0.001/eps_0.01 (12 pairs): the sweep's benchmark wiring
#     (--internet_nets_benchmarks + benchmark perturbation list) is har-only.
#   * brightness/contrast: their vaghar results were deleted (task 3), so the
#     pairs-with-vaghar-results rule schedules nothing for them.
#
# Parallelism: the invocation maintains its own pool — up to --max_cores,
# 32 pinned cores per Julia job, all datasets interleaved via --dataset_group
# (idle slots spill over between datasets; --prioritize_rows finishes rows
# early). TIMEOUT is overridable: TIMEOUT=3600 ./zono_fix_sweep_commands.sh
#
# CPU budget: this run is confined to CPUs 0-160 ONLY.
#   * outer `taskset -c 0-160` pins the supervisor and every unpinned child
#     (PGD hyper-attack subprocesses, table regeneration) to that set;
#   * SWEEP_CORE_START=0 + --max_cores 160 lay the job slots inside it:
#     (160-0)/32 = 5 concurrent Julia jobs on cores 0-31, 32-63, 64-95,
#     96-127, 128-159; CPU 160 is headroom for the supervisor/PGD.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"
: "${TIMEOUT:=10800}"   # seconds per MIP
: "${CORES:=160}"       # slot window width -> 5 jobs x 32 cores (0-159)
export SWEEP_CORE_START="${SWEEP_CORE_START:-0}"

taskset -c 0-160 python3 run_relaxation_sweep.py --advanced_standard \
  --experiments_root paper_experiments_with_zono_fix \
  --only_vaghar_pairs \
  --dataset_group 'mnist|3x50=paper_experiments_with_zono_fix/mnist/3x50_exp/model_seed42_itr20/19,3x100=paper_experiments_with_zono_fix/mnist/3x100_exp/model_seed42_itr19,cnn1=paper_experiments_with_zono_fix/mnist/cnn1_exp/model_seed42_itr20' \
  --dataset_group 'fashion-mnist|3x50=paper_experiments_with_zono_fix/fashion-mnist/3x50_exp/model_seed42_itr19,cnn1=paper_experiments_with_zono_fix/fashion-mnist/cnn1_exp/model_seed42_itr19' \
  --dataset_group 'cifar|cnn5=paper_experiments_with_zono_fix/cifar/cnn5_exp/model_seed42_itr19' \
  --dataset_group 'har|har=paper_experiments_with_zono_fix/har/har_exp/model_har' \
  --perturbations 'linf(0.1)' 'linf(0.05)' 'linf(0.01)' 'occ(14,14,9)' 'occ(3,3,5)' 'occ(5,5,5)' 'occ(1,1,9)' 'occ(1,1,5)' 'patch(1,14,14,3)' 'trans(1,1)' 'trans(1,3)' 'trans(3,1)' 'trans(3,3)' 'rotation(10)' 'rotation(5)' \
  --sweep_ctag 1 2 3 5 \
  --ct 1,2,3,4,5,6,7,8,9,10 \
  --timeout "$TIMEOUT" --max_cores "$CORES" \
  --sweep_adv_std_bound_tightening true \
  --sweep_adv_std_zono_bounds true \
  --sweep_adv_std_n2_relax_threshold 0.5 \
  --sweep_adv_std_n2_sibling_gate true \
  --sweep_adv_std_var_hint prev_pgd \
  --sweep_gurobi_seed 4 \
  --stdboost_combos "N2:false:false:0:false,N2:true:true:0.5:true" \
  --skip_std_n2_baseline \
  --n2_tables_only \
  --geometric_intervals \
  --prioritize_rows
