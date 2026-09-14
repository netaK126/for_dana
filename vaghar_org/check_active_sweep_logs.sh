#!/usr/bin/env bash
#
# Print the sweep_logs files that a running job is writing right now.
#
# "Active" means some live julia / run_relaxation_sweep.py process holds an open
# write fd on the file (/proc/<pid>/fd) -- not that its mtime looks recent. A job
# can sit for hours inside one solve without printing a line, and a finished job
# leaves a fresh mtime behind, so mtime alone is wrong in both directions.
#
# READ-ONLY: reads /proc and stat()s the logs. Never signals or stops anything.
#
# Usage: ./check_active_sweep_logs.sh [LOGDIR]     (default: ./sweep_logs)

set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
logdir=${1:-$here/sweep_logs}
logdir=$(cd "$logdir" 2>/dev/null && pwd) || {
    echo "no such log directory: ${1:-$here/sweep_logs}" >&2; exit 1; }

declare -A active=()
for pid in $(pgrep -f 'run_relaxation_sweep\.py|run\.jl' 2>/dev/null); do
    # Only the sweep's own processes; skips this script's shell, greps, etc.
    read -r comm < "/proc/$pid/comm" 2>/dev/null || continue
    case $comm in julia|python*) ;; *) continue ;; esac
    for fd in "/proc/$pid/fd"/*; do
        target=$(readlink "$fd" 2>/dev/null) || continue
        [[ $target == "$logdir"/* && -f $target ]] && active["$target"]=1
    done
done

if (( ${#active[@]} == 0 )); then
    echo "No active log files in $logdir"
    exit 0
fi

# Newest write first, so the slot that just reported is at the top.
i=0
while IFS=$'\t' read -r _ name; do
    printf '%2d. %s\n' "$(( ++i ))" "$name"
done < <(
    for log in "${!active[@]}"; do
        printf '%s\t%s\n' "$(stat -c %Y "$log")" "$(basename "$log")"
    done | sort -rn
)
