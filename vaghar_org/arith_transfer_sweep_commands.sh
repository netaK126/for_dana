#!/bin/bash
# Arithmetic-transfer-bounds campaign (--arithmetic_transfer_bounds):
# N1 solves + advstd (transfer) N2 jobs ONLY — no stdBoost, no std-N2 baseline.
# Pinned to CPUs 198-230: one 32-core job slot (198-229; 230 spare).
# Result files carry the _arithTransfer tag; the zonotope paper tables and
# CSVs exclude them, so this campaign never touches the existing columns.
# State dirs are shared with the zonotope campaign: already-solved N1 pairs
# are skipped (mode-independent), only diff_bounds_arithTransfer.bin is added.
set -euo pipefail
cd "$(dirname "$0")"
export SWEEP_CORE_START="${SWEEP_CORE_START:-198}"

taskset -c 198-230 python3 run_relaxation_sweep.py --advanced_standard \
  --experiments_root paper_experiments_with_zono_fix \
  --only_vaghar_pairs \
  --dataset_group 'mnist|3x50=paper_experiments_with_zono_fix/mnist/3x50_exp/model_seed42_itr20/19' \
  --dataset_group 'fashion-mnist|3x50=paper_experiments_with_zono_fix/fashion-mnist/3x50_exp/model_seed42_itr19,cnn1=paper_experiments_with_zono_fix/fashion-mnist/cnn1_exp/model_seed42_itr19' \
  --dataset_group 'cifar|cnn5=paper_experiments_with_zono_fix/cifar/cnn5_exp/model_seed42_itr19' \
  --dataset_group 'har|har=paper_experiments_with_zono_fix/har/har_exp/model_har' \
  --perturbations 'linf(0.1)' 'linf(0.05)' 'linf(0.01)' 'occ(14,14,9)' 'occ(3,3,5)' 'occ(5,5,5)' 'occ(1,1,9)' 'occ(1,1,5)' 'patch(1,14,14,3)' 'trans(1,1)' 'trans(1,3)' 'trans(3,1)' 'trans(3,3)' 'rotation(10)' 'rotation(5)' \
  --sweep_ctag 1 2 3 5 \
  --ct 1,2,3,4,5,6,7,8,9,10 \
  --timeout "10800" --max_cores "230" \
  --sweep_adv_std_bound_tightening true \
  --sweep_adv_std_zono_bounds true \
  --sweep_adv_std_n2_relax_threshold 0.5 \
  --sweep_adv_std_n2_sibling_gate true \
  --sweep_adv_std_var_hint prev_pgd \
  --sweep_gurobi_seed 4 \
  --arithmetic_transfer_bounds \
  --skip_std_n2_baseline \
  --n2_tables_only \
  --geometric_intervals \
  --prioritize_rows
