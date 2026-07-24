#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Re-run "ours" (stdBoost N2) + "transfer" (advStd) for the cells affected by
# the geometric conv/rotation fix (2026-07-23), at exactly the paper's pairs:
#   mnist/fmnist cnn1 + cifar cnn5 : cs {0,2} -> --sweep_ctag 1 3, --ct 2,4,5
#   cifar cnn1                     : cs {0,1} -> --sweep_ctag 1 2, --ct 3,4,6
# (pair sets taken from the tables' --arch_timeouts spec "@ct#cs")
#
#   Step 0: quarantine stale result files into _stale_pre_geomfix/ per cell
#   Step 1: 7 slots in parallel (224/255 cores, no core sharing):
#             2x mnist cnn1 (translation+rotation, tau=0.5)
#             1x fmnist cnn1 (translation+rotation, tau=0.5)
#             1x cifar cnn5 (translation, tau=0.25)
#             3x cifar cnn1 (translation+rotation, tau=0.5)
#   Final : the canonical paper-tables regen command
#
# Run inside tmux or:  nohup ./run_geomfix_reruns.sh > geomfix_master.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd /root/Downloads/for_dana/vaghar_org
P=$PWD/paper_experiments
L=$PWD/geomfix_logs; mkdir -p "$L"
trap 'echo "INTERRUPTED — killing child sweeps"; kill 0' INT TERM

echo "════ preflight ════"
if pgrep -f "run_relaxation_sweep.py" > /dev/null; then
    echo "ABORT: a run_relaxation_sweep.py is already running (PID(s): $(pgrep -f run_relaxation_sweep.py | tr '\n' ' '))."
    echo "Stop it first — the 7-slot budget assumes the machine is free."
    exit 1
fi
for d in "$P/mnist/cnn1_exp/model_seed42_itr20" \
         "$P/fashion-mnist/cnn1_exp/model_seed42_itr19" \
         "$P/cifar/cnn1_exp/model_seed42_itr19" \
         "$P/cifar/cnn5_exp/model_seed42_itr19"; do
    [ -e "$d" ] || { echo "ABORT: model path missing: $d"; exit 1; }
done
CORES=$(nproc)
[ "$CORES" -ge 224 ] || echo "WARNING: only $CORES cores — 7 slots x 32 threads will oversubscribe"
echo "preflight OK ($CORES cores)"

echo "════ step 0: quarantine ════"
quarantine_geom() {  # $1 = cell dir: move fake-geomInt results aside
    local q="$1/_stale_pre_geomfix"
    if [ -d "$q" ] && [ -n "$(ls -A "$q" 2>/dev/null)" ]; then
        echo "  SKIP (already quarantined): $1"; return
    fi
    mkdir -p "$q"
    local n
    n=$(find "$1" -maxdepth 2 ! -path "*_stale*" -name "*geomInt*" -name "*.txt" | wc -l)
    find "$1" -maxdepth 2 ! -path "*_stale*" -name "*geomInt*" -name "*.txt" -exec mv {} "$q/" \;
    echo "  quarantined $n geomInt files: $1"
}
quarantine_geom "$P/mnist/cnn1_exp/translation/eps_1,1"
quarantine_geom "$P/mnist/cnn1_exp/rotation/eps_10"
quarantine_geom "$P/fashion-mnist/cnn1_exp/translation/eps_1,1"
quarantine_geom "$P/fashion-mnist/cnn1_exp/translation/eps_3,1"
quarantine_geom "$P/fashion-mnist/cnn1_exp/rotation/eps_10"
quarantine_geom "$P/cifar/cnn5_exp/translation/eps_1,1"

# cifar cnn1 rotation: encoder semantics changed -> ours + advStd + N1 state all stale
C="$P/cifar/cnn1_exp/rotation/eps_10"; Q="$C/_stale_pre_geomfix"
if [ -d "$Q" ] && [ -n "$(ls -A "$Q" 2>/dev/null)" ]; then
    echo "  SKIP (already quarantined): $C"
else
    mkdir -p "$Q"
    n1=$(find "$C" -maxdepth 2 ! -path "*_stale*" -name "*BTPR0.5*" -name "*.txt" ! -name "_filename*" | wc -l)
    find "$C" -maxdepth 2 ! -path "*_stale*" -name "*BTPR0.5*" -name "*.txt" ! -name "_filename*" -exec mv {} "$Q/" \;
    n2=$(find "$C" -maxdepth 2 ! -path "*_stale*" -path "*advStd*" -name "*.txt" ! -name "_filename*" | wc -l)
    find "$C" -maxdepth 2 ! -path "*_stale*" -path "*advStd*" -name "*.txt" ! -name "_filename*" -exec mv {} "$Q/" \;
    mv "$C"/n1_state_cnn1_* "$Q/" 2>/dev/null || true
    echo "  quarantined $n1 ours + $n2 advStd files + n1_state: $C"
fi
# cifar cnn1 translation: nothing quarantined on purpose — geom runs are additive there.

echo "════ step 1: launch (7 slots) ════"
BASE="--timeout 10800 --advanced_standard --sweep_adv_std_mip_start false --sweep_adv_std_lp_basis false \
 --sweep_adv_std_bound_tightening true --sweep_adv_std_zono_bounds true --sweep_adv_std_n1_probe off \
 --sweep_adv_std_n2_sibling_gate true --sweep_adv_std_branch_priorities off --sweep_adv_std_var_hint prev_pgd \
 --sweep_gurobi_seed 4 --skip_std_n2_baseline --geometric_intervals --prioritize_rows --n2_tables_only"
T05="--sweep_adv_std_n2_relax_threshold 0.5 --stdboost_combos N2:true:true:0.5:true"
T025="--sweep_adv_std_n2_relax_threshold 0.25 --stdboost_combos N2:true:true:0.25:true"
MN="--dataset_group mnist|cnn1=$P/mnist/cnn1_exp/model_seed42_itr20"
FM="--dataset_group fashion-mnist|cnn1=$P/fashion-mnist/cnn1_exp/model_seed42_itr19"
C1="--dataset_group cifar|cnn1=$P/cifar/cnn1_exp/model_seed42_itr19"
C5="--dataset_group cifar|cnn5=$P/cifar/cnn5_exp/model_seed42_itr19"

# ── work-stealing pool: 7 one-slot windows, a single queue of units ─────────
# Each unit = one sweep invocation (dataset x perturbation x ctag), --max_cores 32
# (1 slot). Whenever a window's unit exits, the next queued unit takes that
# window immediately; when the queue is empty, freed windows simply stay free.
# Units are ordered longest-first (LPT) so the heavy cifar work starts at t=0.
# SWEEP_CORE_START pins each window to disjoint cores (all drivers otherwise
# default to core 8 and would stack their jobs onto the SAME cores).
WINDOWS=(8 40 72 104 136 168 200)

UNITS=(
  'c1_trans_ct1|$C1 $T05 --perturbations "trans(1,1)" --sweep_ctag 1 --ct 3,4,6'
  'c1_trans_ct2|$C1 $T05 --perturbations "trans(1,1)" --sweep_ctag 2 --ct 3,4,6'
  'c1_rot_ct1|$C1 $T05 --perturbations "rotation(10)" --sweep_ctag 1 --ct 3,4,6'
  'c1_rot_ct2|$C1 $T05 --perturbations "rotation(10)" --sweep_ctag 2 --ct 3,4,6'
  'mn_rot_ct1|$MN $T05 --perturbations "rotation(10)" --sweep_ctag 1 --ct 2,4,5'
  'mn_rot_ct3|$MN $T05 --perturbations "rotation(10)" --sweep_ctag 3 --ct 2,4,5'
  'c5_trans_ct1|$C5 $T025 --perturbations "trans(1,1)" --sweep_ctag 1 --ct 2,4,5'
  'c5_trans_ct3|$C5 $T025 --perturbations "trans(1,1)" --sweep_ctag 3 --ct 2,4,5'
  'mn_trans_ct1|$MN $T05 --perturbations "trans(1,1)" --sweep_ctag 1 --ct 2,4,5'
  'mn_trans_ct3|$MN $T05 --perturbations "trans(1,1)" --sweep_ctag 3 --ct 2,4,5'
  'fm_trans_ct1|$FM $T05 --perturbations "trans(1,1)" "trans(3,1)" --sweep_ctag 1 --ct 2,4,5'
  'fm_trans_ct3|$FM $T05 --perturbations "trans(1,1)" "trans(3,1)" --sweep_ctag 3 --ct 2,4,5'
  'fm_rot_ct1|$FM $T05 --perturbations "rotation(10)" --sweep_ctag 1 --ct 2,4,5'
  'fm_rot_ct3|$FM $T05 --perturbations "rotation(10)" --sweep_ctag 3 --ct 2,4,5'
)

declare -A PIDS   # pid -> "label:window"
qi=0
launch_next() {   # $1 = free core window; returns 1 when queue empty
    local win=$1
    [ "$qi" -ge "${#UNITS[@]}" ] && return 1
    local spec="${UNITS[$qi]}"; qi=$((qi+1))
    local label="${spec%%|*}" args="${spec#*|}"
    # --max_cores is an ABSOLUTE core-index ceiling, not a count: the driver
    # computes slots = (max_cores - SWEEP_CORE_START) // 32, so a one-slot
    # window starting at W needs max_cores = W + 32.
    eval "SWEEP_CORE_START=$win python3 run_relaxation_sweep.py $args $BASE --max_cores $((win+32)) > \"$L/$label.log\" 2>&1 &"
    local pid=$!
    PIDS[$pid]="$label:$win"
    echo "  [core $win-$((win+31))] $label  pid $pid  ($(date '+%H:%M'))"
    return 0
}

for win in "${WINDOWS[@]}"; do launch_next "$win" || break; done
echo "launched $(date '+%F %T') — queue: $((${#UNITS[@]} - qi)) units waiting; watch: tail -f $L/*.log"

while [ "${#PIDS[@]}" -gt 0 ]; do
    wait -n 2>/dev/null || true
    for pid in "${!PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            IFS=: read -r label win <<< "${PIDS[$pid]}"
            wait "$pid" 2>/dev/null; rc=$?
            echo "  done: $label (rc=$rc) at $(date '+%F %H:%M') — window $win free, $((${#UNITS[@]} - qi)) queued"
            unset "PIDS[$pid]"
            launch_next "$win" || echo "  queue empty — window $win stays free"
        fi
    done
done
echo "════ all sweeps done $(date '+%F %T') — regenerating tables (canonical command) ════"
python3 run_relaxation_sweep.py --paper_tables_from_txt \
    --combo_ranking_seeds 4 \
    --arch_timeouts "10800:cnn1=.,3x50=.,cnn5=.@2,4,5#0,2|10800:cifar:cnn1=.@3,4,6#0,1|18000:3x100=.@3,4,6#0,1,4" \
    --combination_table "zono:prev_pgd+sg" \
    --paper_taus "0.25,0.5" \
    --dataset mnist,fashion-mnist,cifar > "$L/tables.log" 2>&1
echo "════ ALL DONE $(date '+%F %T') ════"
