#!/bin/bash
# Append-only poll of ghost-fd julia logs.
# For each (pid, log) pair: every 10s, append any new bytes from
# /proc/<pid>/fd/1 to the on-disk file. Exits when the pid exits.
LOGDIR=/root/Downloads/for_dana/vaghar_org/sweep_logs

poll() {
  local pid=$1
  local name=$2
  local out="$LOGDIR/$name"
  if [ ! -f "$out" ] && [ -r "/proc/$pid/fd/1" ]; then
    cat "/proc/$pid/fd/1" > "$out" 2>/dev/null
  fi
  while kill -0 "$pid" 2>/dev/null; do
    if [ ! -f "$out" ] && [ -r "/proc/$pid/fd/1" ]; then
      cat "/proc/$pid/fd/1" > "$out" 2>/dev/null
    elif [ -r "/proc/$pid/fd/1" ]; then
      local prev
      prev=$(stat -c %s "$out" 2>/dev/null || echo 0)
      tail -c +$((prev + 1)) "/proc/$pid/fd/1" >> "$out" 2>/dev/null
    fi
    sleep 10
  done
}

poll 1740357 "Phase 2 (N2 sweep)_[3x50]_patch(1,14,14,3)_c_tag=2_N2(btpr0.5sgzb)_seed4.log" &
poll 1996306 "Phase 2 (N2 sweep)_[3x50]_trans(1,1)_c_tag=2_N2(btpr0.5sg)_seed4.log"          &
poll 2000282 "Phase 2 (N2 sweep)_[3x50]_trans(1,1)_c_tag=2_N2(btpr0.0sgzb)_seed4.log"        &
poll 2015350 "Phase 2 (N2 sweep)_[3x50]_trans(1,1)_c_tag=2_N2(btpr0.25sgzb)_seed4.log"       &
poll 2039077 "Phase 2 (N2 sweep)_[3x50]_trans(1,1)_c_tag=2_N2(btpr0.5sgzb)_seed4.log"        &
poll 2039140 "Phase 2 (N2 sweep)_[3x50]_trans(3,1)_c_tag=2_N2(btpr0.0sg)_seed4.log"          &
poll 2076975 "Phase 2 (N2 sweep)_[3x50]_trans(3,1)_c_tag=2_N2(btpr0.25sg)_seed4.log"         &
wait
