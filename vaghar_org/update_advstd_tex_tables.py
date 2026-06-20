#!/usr/bin/env python3
"""Regenerate the per-architecture safe-combo tables in advstd_techniques.tex.

Reads the per-cell CSVs produced by --find_advstd_faster_than_standard:
  - advstd_faster_than_standard{_vs_withPerturbed}.csv
  - standard_faster_than_advstd{_vs_withPerturbed}.csv
  - advstd_tighter_at_timeout{_vs_withPerturbed}.csv
  - standard_tighter_at_timeout_vs_advstd{_vs_withPerturbed}.csv
Decisive (>1% wall-clock difference) cells come from the first two; the
last two carry both-timeout cells whose times are included in the sp
averaging and whose upper-lower gaps populate a separate per-arch
timeout-gap table. Rewrites the tables between the AUTO markers in
advstd_techniques.tex.

Invoked automatically from run_relaxation_sweep.py's
_generate_combo_ranking_csv, and also runnable standalone:

    python3 update_advstd_tex_tables.py \\
        --tex advstd_techniques.tex \\
        --csv_dir paper_experiments/mnist \\
        --seed 4 --tau 0.1 --archs cnn1 3x10 3x50
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEX = os.path.join(HERE, "advstd_techniques.tex")
DEFAULT_CSV_DIR = os.path.join(HERE, "paper_experiments", "mnist")

BEGIN_MARK = "% BEGIN AUTO: safe_tables"
END_MARK = "% END AUTO: safe_tables"

PERT_ORDER = ["patch", "occ", "translation", "linf", "brightness",
              "contrast", "rotation"]


def load_rows(csv_dir, suffix):
    decisive_paths = [
        os.path.join(csv_dir, f"advstd_faster_than_standard{suffix}.csv"),
        os.path.join(csv_dir, f"standard_faster_than_advstd{suffix}.csv"),
    ]
    timeout_paths = [
        os.path.join(csv_dir, f"advstd_tighter_at_timeout{suffix}.csv"),
        os.path.join(csv_dir,
                     f"standard_tighter_at_timeout_vs_advstd{suffix}.csv"),
    ]
    rows = []
    for p in decisive_paths + timeout_paths:
        if not os.path.exists(p):
            print(f"[update_advstd_tex_tables] missing: {p}", file=sys.stderr)
            continue
        is_timeout_csv = p in timeout_paths
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get("arch"):
                    r["_is_timeout_pair"] = is_timeout_csv
                    rows.append(r)
    return rows


def classify(mn):
    if mn <= 1.0:
        return "dominant"
    if mn <= 1.0 / 0.75:
        return "avg-win"
    if mn <= 1.0 / 0.5:
        return "avg-win-risky"
    return "loser"


def _parse_int(s):
    s = (s or "").strip()
    try:
        return int(s)
    except ValueError:
        return None


def _parse_float(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# When both runs report TIME_LIMIT but their wall-clocks differ by more
# than this many seconds, treat the cell as having mismatched `--timout`
# caps. Gurobi typically logs `--timout` plus a few seconds when it
# actually hits the limit, so 60s is enough to separate distinct caps
# (e.g. 1800 vs 3600) without flagging the noise around a single cap.
TIMEOUT_MISMATCH_TOL_SEC = 60.0


def _has_timeout_mismatch(status_std, status_adv, t_std, t_adv):
    """True iff both runs hit TIME_LIMIT but were launched under
    different `--timout` caps. The per-cell `sp` is meaningless in that
    case until the slower-budget run is repeated under the same cap, so
    callers replace the row's `sp` column with a warning and exclude the
    cell from per-c_src / per-combo means."""
    if status_std != "TIME_LIMIT" or status_adv != "TIME_LIMIT":
        return False
    if t_std is None or t_adv is None:
        return False
    return abs(float(t_std) - float(t_adv)) > TIMEOUT_MISMATCH_TOL_SEC


def collect_arch(rows, arch, seed, tau):
    # Only include new-mode (BoundTightPertRelax) rows. Legacy _relaxT rows
    # are identifiable by the absence of a "relax_mode" column or a value
    # other than "btpr"; those are retired because the relaxation decision
    # now uses N2's per-copy bounds instead of N1's.
    # Branch priorities is no longer a paper-reported technique; keep only
    # bp=off rows so the tables are consistent with the text.
    # Also drop legacy var-hint rows ('vh_legacy' = pre-merger _varHint or
    # _varHint_varHintFix tags). The var_hint column is now 5-valued:
    #   'prev'       — previous §4.3 rule (filename tag _varHintFixed)
    #   'direct'     — p derived from [l_n2, u_n2] (tag _varHintDirect)
    #   'direct_pgd' — same p as direct, routed to Start with PGD consensus
    #                  (tag _varHintDirectPGD); no VarHintVal/VarHintPri emitted
    #   'prev_pgd'   — same p as prev, routed to Start with PGD consensus
    #                  (tag _varHintPrevPGD); no VarHintVal/VarHintPri emitted
    #   'no'         — varHint disabled
    # All of 'prev', 'direct', 'direct_pgd', and 'prev_pgd' rows render into the tables below.
    # seed=None or tau=None disables that filter (include all seeds / all
    # thresholds). tau is already part of the combo grouping key, so
    # different thresholds simply surface as distinct combo rows.
    filt = [r for r in rows
            if r["arch"] == arch
            and (seed is None or r["seed"] == seed)
            and (tau is None or r["relax_threshold"] == tau)
            and r.get("relax_threshold") in ("0.0", "0.5")
            and r["bound_tightening"] == "yes"
            and r.get("relax_mode") == "btpr"
            and r.get("branch_priorities") == "off"
            and r.get("var_hint") != "vh_legacy"
            and r.get("var_hint") not in ("prev", "direct")
            and r.get("n1_probe", "off") == "off"]
    seen = {}
    for r in filt:
        # sibling_gate (Tech 4) is part of combo identity. Legacy CSVs that
        # predate the column read as "no" via .get() default, which keeps
        # historical combos comparable to the new sg=no runs.
        k = (r["mip_start"], r["branch_priorities"], r["lp_basis"],
             r["bound_tightening"], r["var_hint"], r["zono_bounds"],
             r["n1_probe"], r["relax_threshold"],
             r.get("sibling_gate", "no"))
        grp = (k, r["perturbation"], r["perturbation_size"])
        cell = (r["c_source"], r["c_target"])
        t_adv_lp_raw = r.get("time_advstd_with_lp", "")
        try:
            t_adv_lp = float(t_adv_lp_raw) if t_adv_lp_raw else None
        except ValueError:
            t_adv_lp = None
        r_org = _parse_int(r.get("relaxed_org", ""))
        r_pert = _parse_int(r.get("relaxed_pert", ""))
        r_total = (r_org + r_pert) if (r_org is not None
                                        and r_pert is not None) else None
        is_timeout = bool(r.get("_is_timeout_pair"))
        gap_std = _parse_float(r.get("gap_standard", ""))
        gap_adv = _parse_float(r.get("gap_advstd", ""))
        std_l = _parse_float(r.get("delta_standard_lower_bound", ""))
        std_u = _parse_float(r.get("delta_standard_upper_bound", ""))
        adv_l = _parse_float(r.get("delta_advstd_lower_bound", ""))
        adv_u = _parse_float(r.get("delta_advstd_upper_bound", ""))
        if None not in (std_l, std_u, adv_l, adv_u):
            delta_diff = abs((std_u - std_l) / 2.0 - (adv_u - adv_l) / 2.0)
        else:
            delta_diff = None
        t_std_v = float(r["time_standard"])
        t_adv_v = float(r["time_advstd"])
        status_std = (r.get("solve_status_standard") or "").strip()
        status_adv = (r.get("solve_status_advstd") or "").strip()
        timeout_mismatch = _has_timeout_mismatch(
            status_std, status_adv, t_std_v, t_adv_v)
        seen[(grp, cell)] = (t_std_v,
                             t_adv_v,
                             t_adv_lp,
                             r_total,
                             is_timeout,
                             gap_std,
                             gap_adv,
                             delta_diff,
                             timeout_mismatch)
    by_grp = defaultdict(list)
    for (grp, cell), tup in seen.items():
        by_grp[grp].append((cell, tup))

    all_combos = []
    timeout_all = []
    for grp, pairs in by_grp.items():
        tuples = [tup for _, tup in pairs]
        # Mismatched-timeout cells (different `--timout` caps on the
        # two runs) get excluded from the combo-level wall-clock and
        # speedup means — `sp` for those cells is meaningless until the
        # slower-budget run is repeated under the matching cap.
        valid = [tup for tup in tuples if not tup[8]]
        if valid:
            t_std = sum(tup[0] for tup in valid) / len(valid)
            t_adv = sum(tup[1] for tup in valid) / len(valid)
        else:
            t_std = sum(tup[0] for tup in tuples) / len(tuples)
            t_adv = sum(tup[1] for tup in tuples) / len(tuples)
        lp_vals = [tup[2] for tup in tuples if tup[2] is not None]
        t_adv_lp = (sum(lp_vals) / len(lp_vals)) if lp_vals else None
        r_vals = [tup[3] for tup in tuples if tup[3] is not None]
        n_relax = (sum(r_vals) / len(r_vals)) if r_vals else None
        hmfs = [tup[1] / tup[0] for tup in tuples
                if tup[0] > 0 and not tup[8]]
        if not hmfs:
            continue
        gm = math.exp(sum(math.log(h) for h in hmfs) / len(hmfs))
        mn = max(hmfs)
        tier = classify(mn)
        # Mismatched-timeout cells are also dropped from the orange
        # timeout-gap aggregation: even though both runs report
        # TIME_LIMIT, the gaps are measured under different `--timout`
        # caps, so a "tighter advstd" verdict isn't comparing
        # like-for-like and would mislead the reader.
        timeout_tuples = [(tup[5], tup[6]) for tup in tuples
                          if tup[4] and tup[5] is not None
                          and tup[6] is not None
                          and not tup[8]]
        n_timeout = len(timeout_tuples)
        if n_timeout:
            n_advstd_tighter = sum(1 for gs, ga in timeout_tuples
                                   if ga < gs)
            avg_gap_std = sum(gs for gs, _ in timeout_tuples) / n_timeout
            avg_gap_adv = sum(ga for _, ga in timeout_tuples) / n_timeout
            # Per-c_src breakdown: group timeout cells by c_src so
            # each emitted timeout-table row reports one (combo,
            # c_src) with bounds averaged over that c_src's tested
            # c_tgts.
            by_csrc_to = defaultdict(list)
            for cell_key, tup in pairs:
                if (tup[4] and tup[5] is not None and tup[6] is not None
                        and not tup[8]):
                    by_csrc_to[cell_key[0]].append((tup[5], tup[6]))
            for cs, tt in by_csrc_to.items():
                n_t = len(tt)
                n_at = sum(1 for gs, ga in tt if ga < gs)
                ags = sum(gs for gs, _ in tt) / n_t
                aga = sum(ga for _, ga in tt) / n_t
                timeout_all.append(
                    (grp, cs, n_t, n_at, ags, aga))
        else:
            n_advstd_tighter = 0
            avg_gap_std = None
            avg_gap_adv = None
        su_lp = (t_adv_lp / t_std) if (t_adv_lp is not None
                                        and t_std > 0) else None
        # Per-cell list, each entry is
        # (c_src, c_tgt, t_std, t_adv, t_adv_lp, n_relax, is_timeout,
        #  gap_std, gap_adv, delta_diff, timeout_mismatch).
        # Sorted by (c_src, c_tgt).
        cells = sorted(
            [(c[0], c[1]) + tup for c, tup in pairs],
            key=lambda r: (r[0], r[1]))
        all_combos.append((grp, t_std, t_adv, t_adv / t_std,
                           t_adv_lp, su_lp, gm, mn, tier, n_relax,
                           n_timeout, n_advstd_tighter,
                           avg_gap_std, avg_gap_adv,
                           cells))
    return all_combos, timeout_all


MAX_ROWS_SINGLE_TABLE = 30  # if combined rows > this, split per-block
# Each per-(pert, size, c_src, tau_bucket) sub-table can still pack
# ~20 combos (120+ cell rows) at scriptsize, which overflows one page.
# Cap combos per emitted sub-table so each chunk fits on a page;
# combos are already sorted by descending mean sp, so the chunks form a
# natural top/bottom split of the ranking.
MAX_COMBOS_PER_SUBTABLE = 7
# Max rows per chunk of the overall combo ranking table (the blue/purple
# table at the top). Each row wraps onto several printed lines because
# the loser/winner/missing columns hold many \texttt{<...>} items, so a
# conservative row budget keeps each chunk inside one page.
MAX_OVERALL_ROWS_PER_CHUNK = 10

# Group Technique-4 threshold values into two buckets for green-table
# splitting. Bucket "aggressive" holds the largest and disabled thresholds
# (loosest and tightest encodings); "middle" holds the intermediate values.
_TAU_BUCKET_AGG = {"0.0", "0.5"}
_TAU_BUCKET_ORDER = ("agg", "mid")

# Per-c_src tint: one distinct color per source class so the reader can
# tell at a glance which (pert, size, c_src) block a row belongs to.
_CSRC_COLORS = {
    "0": "green!15",
    "1": "yellow!25",
    "2": "magenta!20",
    "3": "teal!20",
    "4": "violet!20",
    "5": "olive!25",
    "6": "lime!25",
    "7": "pink!25",
    "8": "brown!20",
    "9": "gray!20",
}
_DEFAULT_CSRC_COLOR = "green!15"


def _csrc_color(c_src):
    return _CSRC_COLORS.get(str(c_src), _DEFAULT_CSRC_COLOR)


def parse_combination_spec(spec):
    """Parse comma-separated 'bt:varHint:tau' specs for --combination_table.

    Format: '<bt>:<varHint>:<tau>[+sg][,<bt>:<varHint>:<tau>[+sg],...]'.
    Examples:
      - 'zono:prev_pgd:0.5'
      - 'interval:prev_pgd:0.5+sg,zono:prev_pgd:0.5+sg'
    Per-combo fields:
      - bt   ∈ {none, interval, zono, interval+lp, zono+lp} — matches the
               bound_tight column rendered in the per-arch tables.
      - vh   ∈ {no, off, prev, direct, direct_pgd, prev_pgd} — matches the
               CSV var_hint column verbatim ('off' is accepted as an alias
               for 'no').
      - tau  ∈ {off, 0.0, 0.01, 0.05, 0.1, 0.5, 1.0}, optionally with
               '+sg' suffix for SibGate (Technique 4).
    Returns None if spec is empty / None; otherwise a list of
    (bt, vh, tau) tuples suitable for membership testing against
    _combo_label.
    """
    if not spec:
        return None
    combos = []
    for combo_str in str(spec).split(","):
        combo_str = combo_str.strip()
        if not combo_str:
            continue
        parts = [p.strip() for p in combo_str.split(":")]
        if len(parts) != 3:
            raise SystemExit(
                "--combination_table: expected "
                "'bt:varHint:tau[,bt:varHint:tau,...]', "
                f"got {spec!r}")
        bt, vh, tau = parts
        if vh.lower() == "off":
            vh = "no"
        combo = (bt, vh, tau)
        if combo not in combos:
            combos.append(combo)
    if not combos:
        return None
    return combos


def _format_combination_filter(combination_filter):
    """Return a 'bt:vh:tau[,bt:vh:tau,...]' string for diagnostics."""
    if not combination_filter:
        return ""
    return ",".join(":".join(c) for c in combination_filter)


def _format_combination_filter_tex(combination_filter):
    """Return a LaTeX-safe list of \\texttt{bt:vh:tau} items joined by ', '."""
    if not combination_filter:
        return ""
    items = []
    for bt, vh, tau in combination_filter:
        vh_safe = vh.replace("_", r"\_")
        items.append(f"\\texttt{{{bt}:{vh_safe}:{tau}}}")
    return ", ".join(items)


def _combo_label(grp_key):
    """(bt_label, vh, rt[+sg]) for a combo's grp[0] tuple, matching the format
    rendered into the per-arch tables. Used to filter combos against a
    --combination_table spec. SibGate is appended to rt as '+sg' when on so
    sg=on / sg=off rows show up as different combos in --combination_table.
    Legacy 8-tuple keys are accepted for back-compat."""
    if len(grp_key) == 9:
        _ms, _bp, _lb, bt, vh, zb, np_, rt, sg = grp_key
    else:
        _ms, _bp, _lb, bt, vh, zb, np_, rt = grp_key
        sg = "no"
    if bt != "yes":
        bt_label = "none"
    else:
        base = "zono" if zb == "yes" else "interval"
        bt_label = base + ("+lp" if np_ == "lp" else "")
    rt_label = str(rt).strip() + ("+sg" if sg == "yes" else "")
    return (bt_label, vh, rt_label)


def _tau_bucket(rt):
    return "agg" if rt in _TAU_BUCKET_AGG else "mid"


def _tau_bucket_label(bucket):
    if bucket == "agg":
        return r"$\tau\in\{0.0,\,0.5\}$"
    return r"$\tau\in\{0.01,\,0.05,\,0.1\}$"


def _emit_block_rows(out, ps, block, head_first=True, row_color="green!15"):
    block_head = head_first
    first_combo = True
    for combo in block:
        if not first_combo:
            out.append(r"\hline")
        first_combo = False
        grp = combo[0]
        cells = combo[14]
        # 9-tuple grp[0] (sibling_gate appended); accept 8-tuple legacy too.
        if len(grp[0]) == 9:
            _ms, bp, _lb, bt, vh, zb, np_, rt, sg = grp[0]
        else:
            _ms, bp, _lb, bt, vh, zb, np_, rt = grp[0]
            sg = "no"
        if bt != "yes":
            bt_label = "none"
        else:
            base = "zono" if zb == "yes" else "interval"
            bt_label = base + ("+lp" if np_ == "lp" else "")
        prefix = (f"\\rowcolor{{{row_color}}} " if np_ != "lp" else "")
        pert, psize = ps
        combo_head = True
        # Per-c_src clipped averages of wall-clock:
        #   $\bar t^{c}_{c\_src} = \frac{1}{|C|-1}\sum_{j\neq c\_src}
        #     \min(t_{c\_src, j, c},\, T_{\text{block}})$
        # for $c \in \{\text{std}, \text{adv}\}$, where $|C|$ is the count
        # of distinct classes that appear (as c_src or c_tgt) anywhere in
        # this (combo, pert, p_size) block and $T_{\text{block}}$ is the
        # max wall-clock observed in any timeout-flagged cell of the block
        # (no clip if no cell timed out). The new sp_mean for the c_src
        # group is $\bar t^{\text{std}}/\bar t^{\text{adv}}$; the two
        # clipped averages are also shown so the reader can see what went
        # into the ratio. Mismatched-timeout cells (c[10]) are excluded
        # because their `--timout` caps differ across modes.
        mean_t_std_by_csrc = {}
        mean_t_adv_by_csrc = {}
        mean_sp_by_csrc = {}
        mean_dd_by_csrc = {}
        seen_csrc_in_combo = set()
        by_csrc = defaultdict(list)
        for cell in cells:
            by_csrc[cell[0]].append(cell)
        # T_block: max wall-clock from any non-mismatched timeout cell in
        # the block. +inf when nothing timed out (i.e., min(t, T) is a
        # no-op). c[6] is the per-cell timeout flag.
        timeout_walls = [max(g[2], g[3]) for g in cells
                         if g[6] and not g[10]
                         and g[2] is not None and g[3] is not None]
        T_block = max(timeout_walls) if timeout_walls else float("inf")
        # |C|: distinct classes appearing as c_src or c_tgt anywhere in
        # the block (after dropping mismatched-timeout cells, which we
        # exclude from the totals anyway).
        block_classes = set()
        for g in cells:
            if g[10]:
                continue
            block_classes.add(g[0])
            block_classes.add(g[1])
        denom = max(len(block_classes) - 1, 1)
        for c_src_key, group in by_csrc.items():
            # Sum of min(t, T_block) over this c_src's non-mismatched
            # cells (i.e., the j \neq c_src tested pairs). Missing pairs
            # contribute 0; the denom is the full |C|-1 so coverage gaps
            # bias the row average down rather than being silently dropped.
            std_terms = [min(g[2], T_block) for g in group
                         if g[3] > 0 and not g[10]]
            adv_terms = [min(g[3], T_block) for g in group
                         if g[3] > 0 and not g[10]]
            if std_terms and adv_terms:
                mean_t_std_by_csrc[c_src_key] = sum(std_terms) / denom
                mean_t_adv_by_csrc[c_src_key] = sum(adv_terms) / denom
                tot_std = sum(std_terms)
                tot_adv = sum(adv_terms)
                mean_sp_by_csrc[c_src_key] = (
                    tot_std / tot_adv if tot_adv > 0 else float("nan"))
            else:
                mean_t_std_by_csrc[c_src_key] = float("nan")
                mean_t_adv_by_csrc[c_src_key] = float("nan")
                mean_sp_by_csrc[c_src_key] = float("nan")
            dds = [g[9] for g in group
                   if g[9] is not None and not g[10]]
            mean_dd_by_csrc[c_src_key] = (
                (sum(dds) / len(dds)) if dds else None)
        for cell in cells:
            (c_src, c_tgt, ts, ta, _ta_lp, n_relax, _ito,
             _gs, _ga, _delta_diff, _tom) = cell
            pert_c = pert if (block_head and combo_head) else ""
            size_c = (f"\\texttt{{{psize}}}"
                      if (block_head and combo_head) else "")
            bt_c = bt_label if combo_head else ""
            vh_c = vh.replace("_", r"\_") if combo_head else ""
            rt_label = str(rt).strip() + ("+sg" if sg == "yes" else "")
            rt_c = rt_label if combo_head else ""
            nr_c = (f"{n_relax:5d}"
                    if isinstance(n_relax, int) else "  ---")
            if _tom:
                # Mismatched `--timout` caps on this cell — replace the
                # last 5 columns (sp, $\bar t^{std}_{c\_src}$,
                # $\bar t^{adv}_{c\_src}$, $\overline{sp}_{c\_src}$,
                # $\overline{|\delta_{std}-\delta_{adv}|}_{c\_src}$) with
                # a single \multicolumn warning that records both
                # wall-clocks so the reader knows the comparison is
                # unsound and which leg needs a 2nd attempt under the
                # matching cap. We do NOT add c_src to
                # `seen_csrc_in_combo`, so the per-c_src avg display
                # defers to the next non-mismatched cell.
                warn_inner = (
                    r"\itshape\bfseries timeouts differ; 2nd attempt "
                    rf"required ($T_{{\mathrm{{std}}}}={ts:.0f}\,$s, "
                    rf"$T_{{\mathrm{{adv}}}}={ta:.0f}\,$s)")
                warning_cell = (
                    r"\multicolumn{5}{l}{" + warn_inner + r"}")
                out.append(
                    f"{prefix}{pert_c:11s} & {size_c:20s} & {bt_c:11s} & "
                    f"{vh_c:3s} & {rt_c:4s} & "
                    f"{c_src:>5s} & {c_tgt:>5s} & {nr_c} & "
                    f"{ts:7.2f} & {ta:7.2f} & {warning_cell} \\\\")
                combo_head = False
                continue
            su = ts / ta if ta > 0 else float("nan")
            su_c = (f"\\textbf{{{su:.3f}}}"
                    if (su == su and su > 1.0) else f"{su:.3f}")
            if c_src not in seen_csrc_in_combo:
                mts = mean_t_std_by_csrc[c_src]
                mta = mean_t_adv_by_csrc[c_src]
                msp = mean_sp_by_csrc[c_src]
                mt_std_c = (f"\\textit{{{mts:7.2f}}}"
                            if mts == mts else "  ---")
                mt_adv_c = (f"\\textit{{{mta:7.2f}}}"
                            if mta == mta else "  ---")
                if msp == msp and msp > 1.0:
                    avg_sp_c = f"\\textit{{\\textbf{{{msp:.3f}}}}}"
                elif msp == msp:
                    avg_sp_c = f"\\textit{{{msp:.3f}}}"
                else:
                    avg_sp_c = "  ---"
                mdd = mean_dd_by_csrc[c_src]
                dd_c = (f"\\textit{{{mdd:.2f}}}"
                        if mdd is not None else "  ----")
                seen_csrc_in_combo.add(c_src)
            else:
                mt_std_c = "       "
                mt_adv_c = "       "
                avg_sp_c = "       "
                dd_c = "       "
            out.append(
                f"{prefix}{pert_c:11s} & {size_c:20s} & {bt_c:11s} & "
                f"{vh_c:3s} & {rt_c:4s} & "
                f"{c_src:>5s} & {c_tgt:>5s} & {nr_c} & "
                f"{ts:7.2f} & {ta:7.2f} & {su_c} & "
                f"{mt_std_c} & {mt_adv_c} & {avg_sp_c} & "
                f"{dd_c} \\\\")
            combo_head = False
        block_head = False


def _emit_table_header(out, clear_page=True):
    # `clear_page=True` keeps the historical behaviour of starting each
    # emitted table on a fresh page (used by the unfiltered split-mode
    # tables, which are dense and would otherwise overflow).
    # `clear_page=False` lets LaTeX float `[!htbp]` decide placement on
    # the current page; merged-combo tables (under --combination_table)
    # use this so short tables can pack together instead of one-per-page.
    if clear_page:
        out.append(r"\clearpage")
    out.append(r"\begin{table}[!htbp]")
    out.append(r"\centering")
    out.append(r"\scriptsize")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\begin{adjustbox}{max width=\textwidth,center}%")
    out.append(r"\begin{tabular}{@{}l l l l l | r r r r r r r r r r@{}}")
    out.append(r"\hline")
    out.append(r"p\_type & p\_size & \tech{bound\_tight} & "
               r"\tech{varHint} & $\tau$ & "
               r"c\_src & c\_tgt & n\_relax & "
               r"t\_std & t\_adv & $\text{sp}$ & "
               r"$\overline{t}^{\mathrm{std}}_{c\_\mathrm{src}}$ & "
               r"$\overline{t}^{\mathrm{adv}}_{c\_\mathrm{src}}$ & "
               r"$\overline{\text{sp}}_{c\_\mathrm{src}}$ & "
               r"$\overline{|\delta_\mathrm{std}-\delta_\mathrm{adv}|}_{c\_\mathrm{src}}$ \\")
    out.append(r"\hline")


def _full_caption_body(arch, n_total_cells, has_timeout_table):
    """Short caption used on the first/single sub-table.

    Column meanings are defined once in the intro paragraph above the
    tables, so we do not repeat them here.
    """
    cross_ref = (
        f" Timeout-cell gaps: Table~\\ref{{tab:timeout_gap_"
        f"{arch.replace('=', '_')}}}."
        if has_timeout_table else "")
    return (
        f"Rows are tinted by $c_\\mathrm{{src}}$ (one distinct color per "
        f"source class); \\textsf{{sp}}$>\\!1$ (advstd faster) in "
        f"\\textbf{{bold}}. "
        f"Combos sorted by descending mean \\textsf{{sp}}; cells within "
        f"a combo by $(c_\\mathrm{{src}}, c_\\mathrm{{tgt}})$. "
        f"Total {n_total_cells} cell rows across this architecture."
        f"{cross_ref}")


def _seed_tau_phrase(seed, tau):
    seed_c = f"\\textsf{{seed}}$={seed}$" if seed else r"all seeds"
    tau_c = f"$\\tau={tau}$" if tau else r"all $\tau$ values"
    return f"{seed_c}, {tau_c}"


# ---------------------------------------------------------------------------
# Per-c_src success histograms (one mini TikZ bar chart per (arch, combo))
# ---------------------------------------------------------------------------

# Cap bar heights at this y-axis value. Anything taller is rendered at
# this height with a bold numeric label above so the reader sees the
# real value but the chart's y-scale stays useful for the rest.
HIST_YMAX = 3.0
HIST_BAR_W = 0.4         # bar width in TikZ units
HIST_BAR_DX = 0.55       # horizontal spacing between bar centres
HIST_X_LEFT = 0.4        # x of the first bar centre
HIST_X_RIGHT = HIST_X_LEFT + 9 * HIST_BAR_DX + 0.6  # axis right edge


def _csrc_histogram_data(all_combos, combination_filter=None):
    """For a single arch's combo list, aggregate per-c_src mean sp.

    Returns a list of dicts (one per combo, sorted by descending overall
    mean sp) shaped:
        {"key": grp_key,
         "label": (bt, vh, rt[+sg]),         # from _combo_label
         "bars": {c_src: (mean_sp, n_cells)},
         "overall_mean_sp": float,
         "overall_win_rate": float in [0,1],
         "n_cells": int}

    Cells with `cell[10]` (mismatched-timeout) or `cell[3] == 0` are
    excluded — same rule as `_emit_block_rows` and
    `render_overall_combo_summary`. Combos with zero surviving cells
    are dropped entirely.

    `combination_filter`: list of (bt, vh, rt[+sg]) tuples from
    parse_combination_spec. When set, only combos whose _combo_label
    matches one of them survives.
    """
    # Collapse combos that share the same grp[0] (same flag tuple) but
    # different (pert, psize) into one combined record per combo. This
    # is the key difference from render_table, which keeps the (pert,
    # psize) dimension. Histogram aggregation folds all perturbations
    # into the bar height.
    by_key = defaultdict(list)
    for e in all_combos:
        by_key[e[0][0]].append(e)

    out = []
    for key, combos in by_key.items():
        label = _combo_label(key)
        if (combination_filter is not None
                and label not in combination_filter):
            continue
        per_csrc = defaultdict(list)  # c_src (int) -> [sp values]
        for combo in combos:
            for cell in combo[14]:
                if cell[10]:
                    continue
                if cell[3] <= 0:
                    continue
                # c_source comes from the CSV as a string ("0".."9");
                # coerce to int so the renderer's range(10) lookup
                # matches. Fall back to leaving the value as-is if a
                # non-numeric label ever appears (the renderer will
                # then render that c_src as "--", which is harmless).
                try:
                    cs_key = int(cell[0])
                except (TypeError, ValueError):
                    cs_key = cell[0]
                per_csrc[cs_key].append(cell[2] / cell[3])
        if not per_csrc:
            continue
        bars = {cs: (sum(v) / len(v), len(v))
                for cs, v in per_csrc.items()}
        all_sps = [r for v in per_csrc.values() for r in v]
        overall_mean = sum(all_sps) / len(all_sps)
        n_wins = sum(1 for r in all_sps if r > 1.0)
        out.append({
            "key": key,
            "label": label,
            "bars": bars,
            "overall_mean_sp": overall_mean,
            "overall_win_rate": n_wins / len(all_sps),
            "n_cells": len(all_sps),
        })
    out.sort(key=lambda d: -d["overall_mean_sp"])
    return out


def _render_csrc_histogram(record):
    """Emit one TikZ bar chart for a single (arch, combo) record.

    Renders a self-contained `tikzpicture` wrapped in a minipage so two
    fit per row. Bar height = min(mean_sp, HIST_YMAX); clipped bars
    show the true numeric value in bold above the bar so the reader
    knows the bar is truncated.
    """
    bt, vh, rt = record["label"]
    vh_safe = vh.replace("_", r"\_")
    label_tex = f"\\texttt{{{bt}:{vh_safe}:{rt}}}"

    lines = []
    lines.append(r"\begin{minipage}[t]{0.48\linewidth}")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(label_tex + r"\\[-0.5ex]")
    lines.append(
        r"$\overline{\text{sp}}="
        f"{record['overall_mean_sp']:.3f}$, "
        r"\textsf{wr}="
        f"{record['overall_win_rate']*100:.1f}\\%, "
        f"$n={record['n_cells']}$")
    lines.append(r"\begin{tikzpicture}[scale=0.62, "
                 r"every node/.style={font=\tiny}]")
    # Axes.
    lines.append(
        f"  \\draw[->] (0,0) -- ({HIST_X_RIGHT:.2f},0);")
    lines.append(
        f"  \\draw[->] (0,0) -- (0,{HIST_YMAX + 0.4:.2f}) "
        r"node[left]{$\overline{\text{sp}}$};")
    # Breakeven line.
    lines.append(
        f"  \\draw[dashed, gray] (0,1) -- ({HIST_X_RIGHT:.2f},1) "
        r"node[right]{$=1$};")
    # y-axis ticks at 0, 1, 2, 3
    for y in (1, 2, 3):
        if y <= HIST_YMAX:
            lines.append(
                f"  \\node[left, font=\\tiny] at (0,{y}) {{{y}}};")
    # Per-c_src bars.
    for cs in range(10):
        x_center = HIST_X_LEFT + cs * HIST_BAR_DX
        x_lo = x_center - HIST_BAR_W / 2
        x_hi = x_center + HIST_BAR_W / 2
        if cs not in record["bars"]:
            # Absent c_src: just the axis label, no bar.
            lines.append(
                f"  \\node[below, font=\\tiny] at "
                f"({x_center:.2f},0) {{{cs}}};")
            lines.append(
                f"  \\node[below=0.32cm, font=\\tiny, gray] at "
                f"({x_center:.2f},0) {{--}};")
            continue
        mean_sp, n = record["bars"][cs]
        clipped = mean_sp > HIST_YMAX
        h_drawn = min(mean_sp, HIST_YMAX)
        color = "green!60!black" if mean_sp >= 1.0 else "red!60!black"
        lines.append(
            f"  \\fill[{color}] ({x_lo:.2f},0) rectangle "
            f"({x_hi:.2f},{h_drawn:.3f});")
        lines.append(
            f"  \\draw[gray, line width=0.2pt] ({x_lo:.2f},0) "
            f"rectangle ({x_hi:.2f},{h_drawn:.3f});")
        # Value label above the bar (bold if clipped).
        val_label = (f"\\textbf{{{mean_sp:.2f}}}" if clipped
                     else f"{mean_sp:.2f}")
        lines.append(
            f"  \\node[above, font=\\tiny] at "
            f"({x_center:.2f},{h_drawn:.3f}) {{{val_label}}};")
        # c_src tick label and cell count.
        lines.append(
            f"  \\node[below, font=\\tiny] at "
            f"({x_center:.2f},0) {{{cs}}};")
        lines.append(
            f"  \\node[below=0.32cm, font=\\tiny] at "
            f"({x_center:.2f},0) {{$n{{=}}{n}$}};")
    # x-axis label.
    lines.append(
        f"  \\node[below=0.7cm, font=\\tiny] at "
        f"({(HIST_X_LEFT + 4.5 * HIST_BAR_DX):.2f},0) "
        r"{$c_\mathrm{src}$};")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{minipage}")
    return "\n".join(lines)


def render_csrc_histograms_section(per_arch_combos, seed, tau,
                                   combination_filter=None):
    """Emit a `\\section*{Per-c_src success histograms}` block.

    One mini chart per (arch, combo) with at least one surviving cell.
    Charts within an arch are laid out two per row using minipages.
    Honours `combination_filter` (a list of (bt, vh, rt) tuples) so
    --combination_table narrows the section just like it narrows the
    blue ranking and the per-arch tables.
    """
    arch_blocks = []
    total_charts = 0
    for arch, all_combos in per_arch_combos:
        records = _csrc_histogram_data(all_combos,
                                       combination_filter=combination_filter)
        if not records:
            continue
        arch_label = arch.replace("=", "_")
        arch_out = []
        arch_out.append(r"\clearpage")
        arch_out.append(
            r"\subsection*{Architecture \textbf{" + arch + r"} --- "
            r"per-$c_\mathrm{src}$ success histograms}")
        arch_out.append(
            f"\\label{{tab:hist_{arch_label}}}")
        arch_out.append(
            f"This page shows {len(records)} bar charts, one per "
            r"advstd combination measured on \textbf{" + arch + r"}, "
            r"sorted by descending overall mean speedup. Each bar "
            r"covers one source class $c_\mathrm{src}\in\{0,\ldots,9\}$ "
            r"and reports the mean of "
            r"$\text{sp}=\textsf{t\_std}/\textsf{t\_adv}$ over every "
            r"$c_\mathrm{tgt}\times$ perturbation $\times$ "
            r"\textsf{p\_size}$\times$ seed cell of that "
            r"$(c_\mathrm{src})$ group. Cell counts "
            r"$n{=}\ldots$ appear under each bar; absent "
            r"$c_\mathrm{src}$ values are shown as `--'.")
        # Pairs of charts side-by-side via minipages.
        for i in range(0, len(records), 2):
            arch_out.append(r"\par\medskip\noindent")
            arch_out.append(_render_csrc_histogram(records[i]))
            arch_out.append(r"\hfill")
            if i + 1 < len(records):
                arch_out.append(
                    _render_csrc_histogram(records[i + 1]))
            else:
                # Fill the second slot with an empty minipage so the
                # first chart stays left-aligned.
                arch_out.append(
                    r"\begin{minipage}[t]{0.48\linewidth}\strut\end{minipage}")
            arch_out.append("")
        total_charts += len(records)
        arch_blocks.append("\n".join(arch_out))

    if not arch_blocks:
        filter_repr = _format_combination_filter(combination_filter)
        return ("% no per-c_src histograms"
                + (f" (combo_filter={filter_repr})"
                   if filter_repr else "")
                + "\n")

    header = []
    header.append(r"\clearpage")
    header.append(
        r"\section*{Per-$c_\mathrm{src}$ success histograms}")
    intro = (
        r"For each architecture the figures below give a fast visual "
        r"readout of how each advstd combination performs across "
        r"source classes. Each chart has up to ten bars "
        r"($c_\mathrm{src}\in\{0,\ldots,9\}$); the bar height is the "
        r"arithmetic mean of $\text{sp}=\textsf{t\_std}/\textsf{t\_adv}$ "
        r"over every $c_\mathrm{tgt}\times$ perturbation $\times$ "
        r"\textsf{p\_size}$\times$ seed cell of that "
        r"$(\text{arch},\text{combo},c_\mathrm{src})$ group, with "
        r"cells whose two modes ran under different "
        r"\texttt{--timout} caps excluded (same rule as the green "
        r"per-arch tables and the blue ranking). Bars are coloured "
        r"\textcolor{green!60!black}{green} when "
        r"$\overline{\text{sp}}\geq 1$ (advstd was faster on average for "
        r"that $c_\mathrm{src}$) and \textcolor{red!60!black}{red} "
        r"otherwise. A dashed line marks the $\overline{\text{sp}}=1$ "
        r"breakeven anchor. "
        f"Bar values above ${HIST_YMAX:.1f}$ are clipped at that "
        r"height and shown with the true numeric value in "
        r"\textbf{bold} above the bar. Within each architecture, "
        r"combinations are sorted by descending overall mean "
        r"speedup (the same metric as the blue overall ranking's "
        r"$\overline{\text{sp}}$ column).")
    if combination_filter:
        n_combos_filter = len(combination_filter)
        combo_list_tex = _format_combination_filter_tex(combination_filter)
        intro += (
            "\n\n"
            r"\noindent\textbf{Note:} these histograms are restricted "
            f"to the {n_combos_filter} "
            f"{'combination' if n_combos_filter == 1 else 'combinations'} "
            + combo_list_tex
            + r" (via \texttt{--combination\_table}).")
    intro += (
        "\n\n"
        f"% auto-generated: {total_charts} per-c_src histograms across "
        f"{len(arch_blocks)} archs, seed={seed}, tau={tau}"
        + (f", combo_filter={_format_combination_filter(combination_filter)}"
           if combination_filter else "")
        + "\n")
    header.append(intro)
    return "\n".join(header) + "\n\n" + "\n\n".join(arch_blocks) + "\n"


# ---------------------------------------------------------------------------
# Per-perturbation combo-comparison histograms
# ("for each arch × perturbation, one bar per combo so the reader can pick
#  the best combo at a glance")
# ---------------------------------------------------------------------------

# Chart geometry. One bar per combo, placed at a stable x position
# shared across every chart of the same arch (see `position_map` in
# `_render_perturbation_histogram`), so a given combo lives at the same
# column everywhere.
PERT_HIST_YMAX = 3.0
PERT_HIST_BAR_W = 0.45    # bar width in TikZ units
PERT_HIST_BAR_DX = 0.70   # x-step between bar centres
PERT_HIST_X_LEFT = 0.55   # x of the first bar centre
# Extra vertical headroom above the bars so the diagonal value labels
# placed on top of each bar don't fall outside the chart frame.
PERT_HIST_LABEL_HEADROOM = 1.2
# Rotation (degrees) for the per-bar mean-sp value labels. Higher
# rotation -> less horizontal text extent -> less risk of adjacent
# labels overlapping each other when neighbouring bars have similar
# heights.
PERT_HIST_LABEL_ROT = 60

# Sixteen perceptually-distinct named TikZ colours. With more combos than
# slots the palette wraps, but the doc currently caps at 16 combos and
# the user can `--combination_table`-filter when running with more.
COMBO_COLOR_PALETTE = [
    "blue!75!black",
    "red!75!black",
    "green!55!black",
    "orange!80!black",
    "purple!70!black",
    "teal!75!black",
    "magenta!65!black",
    "brown!80!black",
    "cyan!55!black",
    "olive!70!black",
    "violet!75!black",
    "yellow!60!black",
    "pink!70!black",
    "lime!55!black",
    "gray!55!black",
    "black",
]


def _build_combo_color_map(per_arch_combos, combination_filter=None):
    """Stable abbrev -> TikZ colour mapping shared across the whole section.

    Built once from the union of combos surviving `combination_filter`,
    sorted alphabetically. The same abbreviation gets the same colour in
    every chart of the run, which is what lets the user eyeball "which
    colour won on N slices" across arches and perturbations.
    """
    seen = set()
    for _, all_combos in per_arch_combos:
        for e in all_combos:
            label = _combo_label(e[0][0])
            if (combination_filter is not None
                    and label not in combination_filter):
                continue
            seen.add(_abbrev_combo_label(label))
    ordered = sorted(seen)
    return {
        abbrev: COMBO_COLOR_PALETTE[i % len(COMBO_COLOR_PALETTE)]
        for i, abbrev in enumerate(ordered)
    }


def _full_combo_label(label_tuple):
    """Readable per-bar label spelling out the underlying flag identities.

    Used as the rotated label attached to each bar in the
    per-perturbation histogram so the reader can identify a combo from
    the chart alone, without consulting the abbreviation key. Words
    are spelled out (``interval`` / ``zono`` / ``prev-PGD`` …) rather
    than reduced to one-or-two-letter shortcuts.

    The string is intentionally compact (separator is a bullet, no
    ``tau=`` prefix) so that even the longest combo
    (``interval+LP * direct-PGD * 0.5+sg``) fits inside a bar's
    height when rendered at ``\\tiny`` rotated 90 degrees.
    """
    bt, vh, rt = label_tuple
    bt_name = {
        "none": "no-BT",
        "interval": "interval",
        "zono": "zono",
        "interval+lp": "interval+LP",
        "zono+lp": "zono+LP",
    }.get(bt, bt)
    vh_name = {
        "no": "no-hint",
        "prev": "prev",
        "direct": "direct",
        "prev_pgd": "prev-PGD",
        "direct_pgd": "direct-PGD",
    }.get(vh, vh)
    # rt already carries "+sg" when SibGate is on; escape "_" defensively
    # though no current rt values contain underscores.
    rt_safe = str(rt).replace("_", r"\_")
    return rf"{bt_name} $\bullet$ {vh_name} $\bullet$ {rt_safe}"


def _abbrev_combo_label(label_tuple):
    """Compact two-letter codes for the bt/varHint fields plus the raw tau.

    The full combo strings ("interval:prev_pgd:0.5+sg") are too wide to
    render under a bar at this scale; this routine produces a compact
    code like "I:pP:0.5+sg" that still uniquely identifies the combo
    in the same alphabet as _combo_label.
    """
    bt, vh, rt = label_tuple
    bt_short = {
        "none": "N",
        "interval": "I",
        "zono": "Z",
        "interval+lp": "I+lp",
        "zono+lp": "Z+lp",
    }.get(bt, bt)
    vh_short = {
        "no": "no",
        "prev": "pv",
        "direct": "dr",
        "prev_pgd": "pP",
        "direct_pgd": "dP",
    }.get(vh, vh)
    return f"{bt_short}:{vh_short}:{rt}"


def _perturbation_combo_data(all_combos, combination_filter=None):
    """For one arch, group combos by **perturbation type** (merging
    all perturbation-sizes of that type into one chart entry).

    Returns dict keyed by ``pert`` (the perturbation type only), value
    a list of dicts:
        {"key": grp_key,                # 9-tuple combo key
         "label": (bt, vh, rt[+sg]),    # _combo_label output
         "abbrev": "I:pP:0.5+sg",
         "mean_sp": float,              # mean sp over *all* cells of
                                        # every p_size for this combo
         "n_cells": int,                # total surviving cells
         "n_psizes_tested": int,        # distinct p_sizes with data
         "n_psizes_won":    int,        # of those, the count with
                                        # per-p_size mean sp > 1
         "psize_details": [             # one entry per p_size
            {"psize": ..., "mean_sp": ..., "n_cells": ..., "n_wins": ...},
            ...
         ]}
    The list is sorted alphabetically by ``abbrev`` (so the renderer
    can drop bars into stable column positions via the section-level
    `position_map`). Cells with mismatched timeouts or t_adv<=0 are
    excluded (same rule as the green tables and the per-c_src
    histograms). Honours `combination_filter` (list of (bt, vh, rt)
    tuples) the same way the per-c_src renderer does.

    Merging by perturbation type collapses e.g.\ ``translation(1,1)``,
    ``translation(3,1)``, ``translation(3,3)`` into one bar per combo
    instead of three; the per-p_size breakdown is preserved on the
    record so the renderer can show the win ratio across p_sizes via
    solid + dashed bar segments.
    """
    by_pert = defaultdict(dict)
    for e in all_combos:
        grp = e[0]
        key = grp[0]              # 9-tuple combo key
        label = _combo_label(key)
        if (combination_filter is not None
                and label not in combination_filter):
            continue
        pert = grp[1]
        psize = grp[2]
        # Each per-cell record: (sp, on_gap_table, gap_std, gap_adv)
        # on_gap_table = True when both runs timed out *and* the cell has
        # finite (gap_std, gap_adv) — these are the cells the orange
        # timeout-gap table aggregates.
        cell_recs = []
        for cell in e[14]:
            if cell[10]:               # timeout_mismatch
                continue
            if cell[3] <= 0:           # t_adv <= 0
                continue
            sp = cell[2] / cell[3]
            on_gap = bool(cell[6]
                          and cell[7] is not None
                          and cell[8] is not None)
            cell_recs.append((sp, on_gap, cell[7], cell[8]))
        if not cell_recs:
            continue
        bucket = by_pert[pert].setdefault(key, {
            "key": key,
            "label": label,
            "abbrev": _abbrev_combo_label(label),
            "psize_cells": {},  # psize -> list of cell_rec tuples
        })
        bucket["psize_cells"].setdefault(psize, []).extend(cell_recs)

    result = defaultdict(list)
    for pert, combos in by_pert.items():
        for bucket in combos.values():
            all_sps = []
            n_psizes_won = 0
            psize_details = []
            for psize, recs in bucket["psize_cells"].items():
                sps = [r[0] for r in recs]
                n = len(sps)
                msp = sum(sps) / n
                gap_recs = [r for r in recs if r[1]]
                sp_recs = [r for r in recs if not r[1]]
                if not gap_recs:
                    # Pure non-timeout: keep historical rule.
                    won = msp > 1.0
                elif not sp_recs:
                    # Pure timeout-on-gap-table: compare mean gaps.
                    mean_g_std = sum(r[2] for r in gap_recs) / len(gap_recs)
                    mean_g_adv = sum(r[3] for r in gap_recs) / len(gap_recs)
                    won = mean_g_adv < mean_g_std
                else:
                    # Mix: per-cell verdict, p_size wins iff majority of
                    # cells win on their own criterion.
                    n_win = sum(1 for r in sp_recs if r[0] > 1.0)
                    n_win += sum(1 for r in gap_recs if r[3] < r[2])
                    won = n_win > n / 2
                n_wins = sum(1 for s in sps if s > 1.0)
                psize_details.append({
                    "psize": psize,
                    "mean_sp": msp,
                    "n_cells": n,
                    "n_wins": n_wins,
                    "n_gap_cells": len(gap_recs),
                })
                all_sps.extend(sps)
                if won:
                    n_psizes_won += 1
            n_total = len(all_sps)
            if n_total == 0:
                continue
            psize_details.sort(key=lambda d: str(d["psize"]))
            result[pert].append({
                "key": bucket["key"],
                "label": bucket["label"],
                "abbrev": bucket["abbrev"],
                "mean_sp": sum(all_sps) / n_total,
                "n_cells": n_total,
                "n_psizes_tested": len(bucket["psize_cells"]),
                "n_psizes_won": n_psizes_won,
                "psize_details": psize_details,
            })
        result[pert].sort(key=lambda r: r["abbrev"])
    return result


def _perturbation_winrate_data(all_combos, combination_filter=None):
    """Same grouping as `_perturbation_combo_data`, but each combo's
    cells are classified into win/loss buckets for the win-rate
    histogram. Per cell:
      - finished win: cell is **not** on the timeout-gap table and
        $\\text{sp}>1$ (advstd finished faster on a finished cell);
      - timeout win:  cell **is** on the timeout-gap table and
        $\\overline{\\text{gap}}_\\mathrm{adv}<\\overline{\\text{gap}}_\\mathrm{std}$
        (both modes timed out but advstd's $u-l$ gap is tighter);
      - loss otherwise.
    Returns dict keyed by `pert`, value list of dicts:
        {"key", "label", "abbrev",
         "n_cells", "n_finished_cells", "n_timeout_cells",
         "n_finished_wins", "n_timeout_wins", "n_wins",
         "finished_win_frac", "timeout_win_frac", "win_rate",
         "psizes": [str,...]}
    Sorted alphabetically by abbrev so the section's `position_map`
    keeps stable columns. Same exclusion rules as
    `_perturbation_combo_data` (timeout_mismatch / `t_adv<=0`).
    """
    by_pert = defaultdict(dict)
    for e in all_combos:
        grp = e[0]
        key = grp[0]
        label = _combo_label(key)
        if (combination_filter is not None
                and label not in combination_filter):
            continue
        pert = grp[1]
        psize = grp[2]
        for cell in e[14]:
            if cell[10]:
                continue
            if cell[3] <= 0:
                continue
            sp = cell[2] / cell[3]
            on_gap = bool(cell[6]
                          and cell[7] is not None
                          and cell[8] is not None)
            bucket = by_pert[pert].setdefault(key, {
                "key": key,
                "label": label,
                "abbrev": _abbrev_combo_label(label),
                "n_cells": 0,
                "n_finished_cells": 0,
                "n_finished_wins": 0,
                "n_timeout_cells": 0,
                "n_timeout_wins": 0,
                "psizes": set(),
            })
            bucket["n_cells"] += 1
            bucket["psizes"].add(psize)
            if on_gap:
                bucket["n_timeout_cells"] += 1
                if cell[8] < cell[7]:
                    bucket["n_timeout_wins"] += 1
            else:
                bucket["n_finished_cells"] += 1
                if sp > 1.0:
                    bucket["n_finished_wins"] += 1

    result = defaultdict(list)
    for pert, combos in by_pert.items():
        for bucket in combos.values():
            n = bucket["n_cells"]
            if n == 0:
                continue
            n_wins = bucket["n_finished_wins"] + bucket["n_timeout_wins"]
            result[pert].append({
                "key": bucket["key"],
                "label": bucket["label"],
                "abbrev": bucket["abbrev"],
                "n_cells": n,
                "n_finished_cells": bucket["n_finished_cells"],
                "n_timeout_cells": bucket["n_timeout_cells"],
                "n_finished_wins": bucket["n_finished_wins"],
                "n_timeout_wins": bucket["n_timeout_wins"],
                "n_wins": n_wins,
                "finished_win_frac": bucket["n_finished_wins"] / n,
                "timeout_win_frac": bucket["n_timeout_wins"] / n,
                "win_rate": n_wins / n,
                "psizes": sorted(str(p) for p in bucket["psizes"]),
            })
        result[pert].sort(key=lambda r: r["abbrev"])
    return result


def _render_combo_color_legend(combo_abbrevs, color_map):
    """Emit a compact ``colour swatch + combo abbreviation'' legend.

    Used once at the top of each arch's per-perturbation subsection so
    every chart below it can be read against a single colour key.
    `combo_abbrevs` is the set of abbreviated combo labels that appear
    anywhere in this arch's charts; sorted alphabetically for stable
    output.
    """
    if not combo_abbrevs:
        return ""
    items = []
    for abbrev in sorted(combo_abbrevs):
        color = color_map.get(abbrev, "blue!75!black")
        abbrev_safe = abbrev.replace("_", r"\_")
        items.append(
            f"\\textcolor{{{color}}}{{\\rule{{0.7em}}{{0.7em}}}}"
            f"\\,\\texttt{{{abbrev_safe}}}")
    return (r"\noindent\textbf{Colour key:} \scriptsize "
            + r"\quad ".join(items)
            + r"\normalsize")


def _render_perturbation_histogram(arch, pert, combo_records,
                                   color_map, position_map):
    """Emit one TikZ chart for a single (arch, pert) — **merging all
    p_sizes of that perturbation type into one bar per combo**.

    Each bar's height is the combo's mean sp aggregated across every
    p_size of this perturbation type. The bar is split into two
    vertical segments:
      * a **solid** bottom segment, filled with the combo's colour,
        whose height is ``(n_psizes_won / n_psizes_tested) × h_drawn``;
      * a **faded same-colour** top segment (opacity 0.25) overlaid
        with **diagonal hatching** in the combo's full colour,
        covering the remaining height — same hue as the bottom so
        the combo's identity stays readable, but visibly striped so
        the lost portion is unmistakable.
    So a combo that won on every p_size shows a fully solid bar; one
    that won on none shows only the faded-and-hatched top; and a
    2/3 combo shows 2/3 of its bar solid and the top 1/3 hatched.
    The reader
    gets two facts at a glance: how fast the combo is on average
    (height) and how consistently it wins across p_sizes (solid
    fraction).

    `position_map` reserves a stable column for every combo (shared
    across every chart of the arch) so the same combo always lives
    at the same x position — pan a column down the page and you trace
    one combo across perturbation types.

    A diagonal value label sits above every bar (rotated
    ``PERT_HIST_LABEL_ROT``°) showing the bar's mean sp; the
    highest-sp combo on the chart is starred above its bar.
    """
    n = len(combo_records)
    n_total_positions = (max(position_map.values()) + 1
                         if position_map else 1)
    x_right = (PERT_HIST_X_LEFT
               + (n_total_positions - 1) * PERT_HIST_BAR_DX + 0.6)
    n_total = sum(r["n_cells"] for r in combo_records)

    # The p_sizes that contribute to this merged chart (union across
    # combos). Printed in the header so the reader knows what was
    # rolled up.
    psize_union = set()
    for rec in combo_records:
        for d in rec["psize_details"]:
            psize_union.add(str(d["psize"]))
    psize_items = []
    for ps in sorted(psize_union):
        ps_safe = ps.replace("_", r"\_")
        psize_items.append(f"\\texttt{{{ps_safe}}}")
    psize_list_safe = ", ".join(psize_items)
    max_psizes = max(r["n_psizes_tested"] for r in combo_records)

    # Ranking by descending mean sp (independent of bar positions).
    # The reader sees the ranking inline as a single compact line, and
    # then reads the bars at their stable positions.
    ranked = sorted(combo_records, key=lambda r: -r["mean_sp"])
    best_rec = ranked[0]

    lines = []
    lines.append(r"\begin{center}")
    lines.append(r"\scriptsize")
    # Per-chart identifier only — the bar/solid/hatched/star/clipping
    # rules are described once in the section intro and are not
    # repeated here.
    lines.append(
        r"\colorbox{yellow!40}{\textbf{sp}}\;"
        f"\\textbf{{{arch}}} \\textbar{{}} pert "
        f"\\textbf{{{pert}}} "
        rf"(p\_sizes merged: {psize_list_safe}) "
        rf"\textbar{{}} {n} combos, total $n={n_total}$ cells."
        r"\\[0.4ex]")

    # Best-to-worst ranking summary so the chart still tells the user
    # which combo to pick even though bars are no longer sorted by sp.
    # The wins-per-p_size ratio is appended next to each combo so the
    # reader doesn't have to read it off the bar's fill fraction.
    rank_items = []
    for i, rec in enumerate(ranked):
        color = color_map.get(rec["abbrev"], "blue!75!black")
        marker = (r"\textcolor{orange!85!black}{$\bigstar$}"
                  if i == 0 else "")
        wins = f"{rec['n_psizes_won']}/{rec['n_psizes_tested']}"
        rank_items.append(
            f"{marker}\\textcolor{{{color}}}{{\\rule{{0.55em}}{{0.55em}}}}"
            f"\\,{rec['mean_sp']:.2f}\\,({wins})")
    lines.append(r"\textbf{Ranking (best $\to$ worst):}\ "
                 + " ~ ".join(rank_items) + r"\\[0.4ex]")

    y_chart_top = PERT_HIST_YMAX + PERT_HIST_LABEL_HEADROOM
    lines.append(r"\begin{tikzpicture}[scale=0.62, "
                 r"every node/.style={font=\tiny}]")
    # Axes — extra vertical headroom above the bars to fit the rotated
    # value labels comfortably.
    lines.append(f"  \\draw[->] (0,0) -- ({x_right:.2f},0);")
    lines.append(
        f"  \\draw[->] (0,0) -- (0,{y_chart_top:.2f}) "
        r"node[left]{$\overline{\text{sp}}$};")
    # Breakeven line + tick labels.
    lines.append(
        f"  \\draw[dashed, gray] (0,1) -- ({x_right:.2f},1) "
        r"node[right]{$=1$};")
    for y in (1, 2, 3):
        if y <= PERT_HIST_YMAX:
            lines.append(
                f"  \\node[left, font=\\tiny] at (0,{y}) {{{y}}};")

    # Bars, drawn at their stable positions. Each bar is split into a
    # full-opacity solid-fill bottom (height = win-fraction × total)
    # and a faded same-color top (opacity 0.25) overlaid with diagonal
    # hatching in the combo's full colour (remainder). The hatched
    # top keeps the combo's colour identity readable while making the
    # lost-vs-won split unmistakable. Above the bar sits a diagonal
    # mean-sp label.
    drawn = sorted(combo_records, key=lambda r: position_map[r["abbrev"]])
    for rec in drawn:
        i = position_map[rec["abbrev"]]
        x_center = PERT_HIST_X_LEFT + i * PERT_HIST_BAR_DX
        x_lo = x_center - PERT_HIST_BAR_W / 2
        x_hi = x_center + PERT_HIST_BAR_W / 2
        mean_sp = rec["mean_sp"]
        clipped = mean_sp > PERT_HIST_YMAX
        h_drawn = min(mean_sp, PERT_HIST_YMAX)
        color = color_map.get(rec["abbrev"], "blue!75!black")
        n_tested = rec["n_psizes_tested"]
        n_won = rec["n_psizes_won"]
        win_frac = n_won / n_tested if n_tested > 0 else 0.0
        h_solid = h_drawn * win_frac
        # Solid bottom — represents the fraction of p_sizes the combo
        # won at sp > 1.
        if h_solid > 0:
            lines.append(
                f"  \\fill[{color}] ({x_lo:.2f},0) rectangle "
                f"({x_hi:.2f},{h_solid:.3f});")
            lines.append(
                f"  \\draw[gray!50!black, line width=0.2pt] "
                f"({x_lo:.2f},0) rectangle "
                f"({x_hi:.2f},{h_solid:.3f});")
        # Faded same-colour top with diagonal hatching — represents
        # the fraction of p_sizes the combo did *not* win. We keep
        # the bar's colour (so the reader can still pick out the
        # combo by colour) but render it at low opacity AND overlay
        # diagonal stripes in the combo's full colour so the lost
        # portion is unmistakably distinct from the solid bottom.
        if h_solid < h_drawn:
            lines.append(
                f"  \\fill[{color}, opacity=0.25] "
                f"({x_lo:.2f},{h_solid:.3f}) rectangle "
                f"({x_hi:.2f},{h_drawn:.3f});")
            lines.append(
                f"  \\fill[pattern=north east lines, pattern color={color}] "
                f"({x_lo:.2f},{h_solid:.3f}) rectangle "
                f"({x_hi:.2f},{h_drawn:.3f});")
            lines.append(
                f"  \\draw[gray!50!black, line width=0.2pt] "
                f"({x_lo:.2f},{h_solid:.3f}) rectangle "
                f"({x_hi:.2f},{h_drawn:.3f});")
        # Diagonal value label above the bar; bold-faced when the bar
        # is clipped (so the reader knows the true value exceeds the
        # drawn height).
        val_label = (f"\\textbf{{{mean_sp:.2f}}}" if clipped
                     else f"{mean_sp:.2f}")
        lines.append(
            f"  \\node[anchor=south west, rotate={PERT_HIST_LABEL_ROT}, "
            f"font=\\tiny, inner sep=1pt] at "
            f"({x_center - 0.05:.2f},{h_drawn + 0.05:.3f}) "
            f"{{{val_label}}};")
        # Readable combo name below the x-axis, rotated 90 degrees so
        # each label is confined to its own bar's x-extent (width
        # PERT_HIST_BAR_W) and cannot overlap a neighbour regardless
        # of label length.
        full_name = _full_combo_label(rec["label"])
        lines.append(
            f"  \\node[anchor=east, rotate=90, font=\\tiny, "
            f"text={color}, inner sep=1pt] at "
            f"({x_center:.2f},-0.05) {{{full_name}}};")

    # Star above the best combo's bar. Sits above the diagonal label
    # so the two don't overlap.
    best_i = position_map[best_rec["abbrev"]]
    best_x = PERT_HIST_X_LEFT + best_i * PERT_HIST_BAR_DX
    best_h = min(best_rec["mean_sp"], PERT_HIST_YMAX)
    # 1.05 vertical room covers a ~4-char rotated label at 60°.
    best_y = min(best_h + 1.05, y_chart_top - 0.05)
    lines.append(
        f"  \\node[font=\\small, "
        r"text=orange!85!black] at "
        f"({best_x:.2f},{best_y:.2f}) {{$\\bigstar$}};")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{center}")
    return "\n".join(lines)


def render_perturbation_histograms_section(per_arch_combos, seed, tau,
                                           combination_filter=None):
    """Emit a `\\section*{Per-perturbation combo-comparison histograms}`.

    For each (arch, perturbation type), render a single bar chart
    that **merges every p_size of that perturbation type** into one
    bar per combo. Bar height is the combo's mean sp across all those
    p_sizes, and the bar's solid-vs-dashed split encodes how many of
    those p_sizes the combo won (sp > 1) — see
    `_render_perturbation_histogram` for the exact visual rule.

    Honours `combination_filter` (a list of (bt, vh, rt) tuples) the
    same way the per-c_src renderer does. Combos with zero surviving
    cells on a given (arch, pert) are not drawn for that chart. A
    stable abbrev -> TikZ colour map is built once from the union of
    surviving combos so the same combo appears in the same colour
    across every chart in this section.
    """
    color_map = _build_combo_color_map(
        per_arch_combos, combination_filter=combination_filter)
    arch_blocks = []
    total_charts = 0
    for arch, all_combos in per_arch_combos:
        by_pert = _perturbation_combo_data(
            all_combos, combination_filter=combination_filter)
        if not by_pert:
            continue
        arch_label = arch.replace("=", "_")
        ordered_perts = sorted(
            by_pert.keys(),
            key=lambda p: (PERT_ORDER.index(p) if p in PERT_ORDER
                           else 99, p))
        arch_out = []
        arch_out.append(r"\clearpage")
        arch_out.append(
            r"\subsection*{Architecture \textbf{" + arch + r"} --- "
            r"per-perturbation combo comparison}")
        arch_out.append(
            f"\\label{{tab:perthist_{arch_label}}}")
        arch_out.append(
            f"This page shows {len(ordered_perts)} bar charts, one per "
            r"\textbf{perturbation type} measured on \textbf{" + arch
            + r"} (all p\_sizes of that perturbation type are merged "
            r"into a single bar per combo). Chart construction follows "
            r"the section intro above; the colour key below covers "
            r"every combo on this page.")
        # One colour key per arch covering every combo that appears
        # anywhere in this arch's charts. The position_map reserves a
        # stable column index for each of these abbreviations so a
        # combo lives at the same x position on every chart of the
        # arch — the reader can scan a single column down the page
        # and follow that combo's sp across perturbations.
        arch_abbrevs = set()
        for pert in ordered_perts:
            for rec in by_pert[pert]:
                arch_abbrevs.add(rec["abbrev"])
        position_map = {
            abbrev: i for i, abbrev in enumerate(sorted(arch_abbrevs))
        }
        legend_block = _render_combo_color_legend(arch_abbrevs, color_map)
        if legend_block:
            arch_out.append(r"\par\medskip")
            arch_out.append(legend_block)
            arch_out.append(r"\par\medskip")
        # Pairs of charts side-by-side via minipages, matching the
        # per-c_src histogram layout. Each chart's own `\begin{center}`
        # wrapper centers its contents inside the minipage.
        for i in range(0, len(ordered_perts), 2):
            arch_out.append(r"\par\medskip\noindent")
            arch_out.append(r"\begin{minipage}[t]{0.48\linewidth}")
            arch_out.append(_render_perturbation_histogram(
                arch, ordered_perts[i], by_pert[ordered_perts[i]],
                color_map, position_map))
            arch_out.append(r"\end{minipage}")
            arch_out.append(r"\hfill")
            if i + 1 < len(ordered_perts):
                arch_out.append(r"\begin{minipage}[t]{0.48\linewidth}")
                arch_out.append(_render_perturbation_histogram(
                    arch, ordered_perts[i + 1],
                    by_pert[ordered_perts[i + 1]],
                    color_map, position_map))
                arch_out.append(r"\end{minipage}")
            else:
                # Fill the second slot so the first chart stays
                # left-aligned in the row.
                arch_out.append(
                    r"\begin{minipage}[t]{0.48\linewidth}"
                    r"\strut\end{minipage}")
            arch_out.append("")
        total_charts += len(ordered_perts)
        arch_blocks.append("\n".join(arch_out))

    if not arch_blocks:
        filter_repr = _format_combination_filter(combination_filter)
        return ("% no per-perturbation combo histograms"
                + (f" (combo_filter={filter_repr})"
                   if filter_repr else "")
                + "\n")

    header = []
    header.append(r"\clearpage")
    header.append(
        r"\section*{Per-perturbation combo-comparison histograms}")
    intro = (
        r"The figures below directly answer ``which advstd combination "
        r"should I pick on this perturbation $\times$ architecture?''. "
        r"Each chart is a single $\langle\text{arch}, \text{pert}\rangle$ "
        r"slice: \textbf{all p\_sizes of that perturbation type are "
        r"merged into a single bar per combo}, with the bar height "
        r"equal to the arithmetic mean of $\overline{\text{sp}}$ "
        r"across every $c_\mathrm{src}\times c_\mathrm{tgt}\times$ "
        r"seed cell across all p\_sizes. \textbf{Each bar is "
        r"vertically split} into a \emph{solid-fill bottom} and a "
        r"\emph{faded + diagonally hatched top} in the \emph{same "
        r"combo colour} (rendered at 25\% opacity with diagonal "
        r"stripes in the combo's full colour, so the bar's colour "
        r"identity stays readable while the lost portion is "
        r"unmistakable): "
        r"the solid fraction equals "
        r"$\textsf{wins}/\textsf{p\_sizes}$, where a p\_size's "
        r"\textsf{win} is defined by a \emph{hybrid} rule that mirrors "
        r"how the orange gap-comparison table treats the two cell "
        r"populations. Specifically: a p\_size in which every cell "
        r"finished (no \texttt{TIME\_LIMIT}) wins iff $\overline{\text"
        r"{sp}}>1$; a p\_size in which every cell hit \texttt{TIME\_"
        r"LIMIT} (its sp is pinned to $\approx1$ because both runs "
        r"max out the wall-clock cap) wins iff $\overline{\text{gap}}"
        r"_\mathrm{adv}<\overline{\text{gap}}_\mathrm{std}$ across "
        r"its timeout cells (the same criterion used to bold rows in "
        r"Table~\ref{tab:timeout_gap_cnn1} and siblings); a p\_size "
        r"with cells of both kinds is judged \emph{per cell} and wins "
        r"iff a majority of cells win on their own type---each "
        r"non-timeout cell votes \emph{yes} when $\text{sp}>1$, each "
        r"timeout cell votes \emph{yes} when its $u-l$ gap on advstd "
        r"is tighter than on standard. So a combo that won on every "
        r"tested p\_size is fully solid; one that won on none "
        r"appears entirely faded-and-hatched; \emph{e.g.}\ 2/3 "
        r"wins shows 2/3 of its bar solid and the top 1/3 "
        r"faded-and-hatched, both in the same combo colour. "
        r"\textbf{Combo columns "
        r"occupy a stable x position across every chart of an arch} "
        r"(alphabetical order on the combo abbreviation), so a "
        r"given combo lives in the same column on every chart---"
        r"the reader can follow that column down the page and "
        r"watch the combo's mean speed-up change as the "
        r"perturbation type changes. The "
        r"\textcolor{orange!85!black}{$\bigstar$} marks the combo "
        r"with the highest mean $\overline{\text{sp}}$ on that "
        r"slice (its column may shift between charts because "
        r"best-of-slice shifts; the column itself stays where it "
        r"is). \textbf{Each combo has its own colour}, and the "
        r"same colour is used for that combo in every chart of "
        r"this section, so a colour that wins on one slice can be "
        r"spotted at a glance on another. A compact colour key "
        r"sits above each arch subsection. Each bar is annotated "
        r"with its mean $\overline{\text{sp}}$ value, rotated "
        f"${PERT_HIST_LABEL_ROT}^{{\\circ}}$ "
        r"so adjacent labels don't overlap when neighbouring bars "
        r"have similar heights. A dashed line marks the "
        r"$\overline{\text{sp}}=1$ breakeven anchor. "
        f"Bars above ${PERT_HIST_YMAX:.1f}$ are clipped at that "
        r"height and their true numeric value is shown in \textbf{"
        r"bold} above the bar. A best-to-worst ranking summary "
        r"line sits above each chart, showing the mean sp and the "
        r"$\textsf{wins}/\textsf{p\_sizes}$ ratio for each combo. "
        r"Cells with mismatched \texttt{--timout} caps between the "
        r"two modes are excluded (same rule as the per-arch "
        r"perturbation tables and the blue overall ranking).")
    if combination_filter:
        n_combos_filter = len(combination_filter)
        combo_list_tex = _format_combination_filter_tex(combination_filter)
        intro += (
            "\n\n"
            r"\noindent\textbf{Note:} these histograms are restricted "
            f"to the {n_combos_filter} "
            f"{'combination' if n_combos_filter == 1 else 'combinations'} "
            + combo_list_tex
            + r" (via \texttt{--combination\_table}).")
    intro += (
        "\n\n"
        f"% auto-generated: {total_charts} per-perturbation histograms "
        f"across {len(arch_blocks)} archs, seed={seed}, tau={tau}"
        + (f", combo_filter={_format_combination_filter(combination_filter)}"
           if combination_filter else "")
        + "\n")
    header.append(intro)
    return "\n".join(header) + "\n\n" + "\n\n".join(arch_blocks) + "\n"


# Y-axis upper bound and headroom for the per-perturbation win-rate
# histograms. Bars are bounded in [0, 1] so no clipping is ever needed;
# the headroom leaves space for the rotated value label sitting above.
WINRATE_HIST_YMAX = 1.0
WINRATE_HIST_LABEL_HEADROOM = 0.35
# Vertical stretch applied to every y-coordinate in the win-rate chart
# so its physical height matches the paired sp chart (sp y_chart_top is
# 4.2; with Y_SCALE=3.0 the win chart's y_chart_top becomes
# 1.35*3 = 4.05, close enough that the star sits well clear of the
# value labels).
WINRATE_Y_SCALE = 3.0


def _render_perturbation_winrate_histogram(arch, pert, combo_records,
                                           color_map, position_map):
    """One TikZ chart per (arch, pert) for the win-rate section.

    Each column visually encodes three quantities as nested
    rectangles on an absolute-count y-axis (height = $n / n_{\\max}$
    where $n_{\\max}$ is the largest \\texttt{n\\_cells} across all
    combos drawn on this chart):

      * a faint \\textbf{envelope} rectangle of height $n_{\\max}$ ---
        the maximum coverage anywhere on this chart;
      * a mid-grey \\textbf{tests} rectangle of height
        \\texttt{n\\_cells} --- this combo's actual coverage;
      * the coloured \\textbf{wins} bar inside, split into a solid
        bottom (\\texttt{n\\_finished\\_wins}) and a hatched top
        (\\texttt{n\\_timeout\\_wins}).

    A combo's win-rate is therefore the fraction of the mid-grey
    rectangle that is coloured. An under-tested combo's column is
    visibly shorter than its envelope. The percentage label above
    each column still reports win rate, and the
    \\textcolor{orange!85!black}{$\\bigstar$} marks the highest-win-
    rate combo on this chart.
    """
    n = len(combo_records)
    n_total_positions = (max(position_map.values()) + 1
                         if position_map else 1)
    x_right = (PERT_HIST_X_LEFT
               + (n_total_positions - 1) * PERT_HIST_BAR_DX + 0.6)
    n_total = sum(r["n_cells"] for r in combo_records)
    max_n_cells = max((r["n_cells"] for r in combo_records),
                      default=1)

    # p_sizes folded into this chart (union across combos).
    psize_union = set()
    for rec in combo_records:
        for ps in rec["psizes"]:
            psize_union.add(ps)
    psize_items = []
    for ps in sorted(psize_union):
        ps_safe = ps.replace("_", r"\_")
        psize_items.append(f"\\texttt{{{ps_safe}}}")
    psize_list_safe = ", ".join(psize_items)

    ranked = sorted(combo_records, key=lambda r: -r["win_rate"])
    best_rec = ranked[0]

    lines = []
    lines.append(r"\begin{center}")
    lines.append(r"\scriptsize")
    lines.append(
        r"\colorbox{cyan!30}{\textbf{win}}\;"
        f"\\textbf{{{arch}}} \\textbar{{}} pert "
        f"\\textbf{{{pert}}} "
        rf"(p\_sizes merged: {psize_list_safe}) "
        rf"\textbar{{}} {n} combos, total $n={n_total}$ cells, "
        rf"$n_{{\max}}={max_n_cells}$."
        r"\\[0.4ex]")

    # Best-to-worst ranking summary: marker + colour swatch + win-rate
    # + (finished/timeout) breakdown over total cells.
    rank_items = []
    for i, rec in enumerate(ranked):
        color = color_map.get(rec["abbrev"], "blue!75!black")
        marker = (r"\textcolor{orange!85!black}{$\bigstar$}"
                  if i == 0 else "")
        rank_items.append(
            f"{marker}\\textcolor{{{color}}}{{\\rule{{0.55em}}{{0.55em}}}}"
            f"\\,{rec['win_rate']*100:.0f}\\%"
            f"\\,({rec['n_finished_wins']}f+"
            f"{rec['n_timeout_wins']}t/{rec['n_cells']})")
    lines.append(r"\textbf{Ranking (best $\to$ worst):}\ "
                 + " ~ ".join(rank_items) + r"\\[0.4ex]")

    y_chart_top = (WINRATE_HIST_YMAX + WINRATE_HIST_LABEL_HEADROOM) \
        * WINRATE_Y_SCALE
    # y-axis maps absolute counts: y = WINRATE_Y_SCALE corresponds
    # to count = max_n_cells (the envelope top).
    count_scale = WINRATE_Y_SCALE / max_n_cells if max_n_cells > 0 else 0
    lines.append(r"\begin{tikzpicture}[scale=0.62, "
                 r"every node/.style={font=\tiny}]")
    lines.append(f"  \\draw[->] (0,0) -- ({x_right:.2f},0);")
    lines.append(
        f"  \\draw[->] (0,0) -- (0,{y_chart_top:.2f}) "
        r"node[left]{$n$};")
    # Envelope reference line at n_max.
    envelope_y = WINRATE_Y_SCALE
    lines.append(
        f"  \\draw[dashed, gray] (0,{envelope_y:.2f}) -- "
        f"({x_right:.2f},{envelope_y:.2f}) "
        f"node[right]{{$n_{{\\max}}={max_n_cells}$}};")
    # y-axis ticks at fractions of n_max (absolute counts).
    for frac in (0.25, 0.5, 0.75, 1.0):
        count = int(round(frac * max_n_cells))
        lines.append(
            f"  \\node[left, font=\\tiny] at "
            f"(0,{frac*WINRATE_Y_SCALE:.2f}) {{{count}}};")

    drawn = sorted(combo_records, key=lambda r: position_map[r["abbrev"]])
    for rec in drawn:
        i = position_map[rec["abbrev"]]
        x_center = PERT_HIST_X_LEFT + i * PERT_HIST_BAR_DX
        x_lo = x_center - PERT_HIST_BAR_W / 2
        x_hi = x_center + PERT_HIST_BAR_W / 2
        color = color_map.get(rec["abbrev"], "blue!75!black")
        h_env = WINRATE_Y_SCALE
        h_tests = rec["n_cells"] * count_scale
        h_solid = rec["n_finished_wins"] * count_scale
        h_top = rec["n_timeout_wins"] * count_scale
        h_total = h_solid + h_top
        # 1. Envelope: pale-grey fill to n_max, dashed outline.
        lines.append(
            f"  \\fill[gray!8] ({x_lo:.2f},0) rectangle "
            f"({x_hi:.2f},{h_env:.3f});")
        lines.append(
            f"  \\draw[gray!50, line width=0.2pt, dashed] "
            f"({x_lo:.2f},0) rectangle ({x_hi:.2f},{h_env:.3f});")
        # 2. This column's tests: mid-grey fill to n_cells.
        if h_tests > 0:
            lines.append(
                f"  \\fill[gray!30] ({x_lo:.2f},0) rectangle "
                f"({x_hi:.2f},{h_tests:.3f});")
            lines.append(
                f"  \\draw[gray!50!black, line width=0.2pt] "
                f"({x_lo:.2f},0) rectangle ({x_hi:.2f},{h_tests:.3f});")
        # 3. Solid bottom: finished-cell wins on absolute scale.
        if h_solid > 0:
            lines.append(
                f"  \\fill[{color}] ({x_lo:.2f},0) rectangle "
                f"({x_hi:.2f},{h_solid:.3f});")
            lines.append(
                f"  \\draw[gray!50!black, line width=0.2pt] "
                f"({x_lo:.2f},0) rectangle ({x_hi:.2f},{h_solid:.3f});")
        # 4. Hatched top: timeout-cell wins on absolute scale.
        # White-mask first so the mid-grey tests fill below doesn't
        # bleed through the 25%-opacity colour overlay.
        if h_top > 0:
            lines.append(
                f"  \\fill[white] "
                f"({x_lo:.2f},{h_solid:.3f}) rectangle "
                f"({x_hi:.2f},{h_total:.3f});")
            lines.append(
                f"  \\fill[{color}, opacity=0.25] "
                f"({x_lo:.2f},{h_solid:.3f}) rectangle "
                f"({x_hi:.2f},{h_total:.3f});")
            lines.append(
                f"  \\fill[pattern=north east lines, pattern color={color}] "
                f"({x_lo:.2f},{h_solid:.3f}) rectangle "
                f"({x_hi:.2f},{h_total:.3f});")
            lines.append(
                f"  \\draw[gray!50!black, line width=0.2pt] "
                f"({x_lo:.2f},{h_solid:.3f}) rectangle "
                f"({x_hi:.2f},{h_total:.3f});")
        # Win-rate % label sits just above this column's tests rect
        # (i.e. above the actual coverage, never above the envelope).
        val_label = f"{rec['win_rate']*100:.0f}\\%"
        label_y = h_tests + 0.02 if h_tests > 0 else 0.02
        lines.append(
            f"  \\node[anchor=south west, rotate={PERT_HIST_LABEL_ROT}, "
            f"font=\\tiny, inner sep=1pt] at "
            f"({x_center - 0.05:.2f},{label_y:.3f}) "
            f"{{{val_label}}};")
        full_name = _full_combo_label(rec["label"])
        lines.append(
            f"  \\node[anchor=east, rotate=90, font=\\tiny, "
            f"text={color}, inner sep=1pt] at "
            f"({x_center:.2f},-0.02) {{{full_name}}};")

    # Star above the best combo's tests rectangle.
    best_i = position_map[best_rec["abbrev"]]
    best_x = PERT_HIST_X_LEFT + best_i * PERT_HIST_BAR_DX
    best_h = best_rec["n_cells"] * count_scale
    best_y = min(best_h + 1.05, y_chart_top - 0.05)
    lines.append(
        f"  \\node[font=\\small, "
        r"text=orange!85!black] at "
        f"({best_x:.2f},{best_y:.2f}) {{$\\bigstar$}};")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{center}")
    return "\n".join(lines)


def render_perturbation_winrate_section(per_arch_combos, seed, tau,
                                        combination_filter=None):
    """Emit a `\\section*{Per-perturbation combo win-rate histograms}`.

    Mirrors `render_perturbation_histograms_section` (same arch
    subsections, same colour key, same stable per-arch column order,
    same side-by-side minipage layout), but each bar's height is the
    combo's overall win rate on that (arch, pert) slice instead of
    its mean sp.
    """
    color_map = _build_combo_color_map(
        per_arch_combos, combination_filter=combination_filter)
    arch_blocks = []
    total_charts = 0
    for arch, all_combos in per_arch_combos:
        by_pert = _perturbation_winrate_data(
            all_combos, combination_filter=combination_filter)
        if not by_pert:
            continue
        arch_label = arch.replace("=", "_")
        ordered_perts = sorted(
            by_pert.keys(),
            key=lambda p: (PERT_ORDER.index(p) if p in PERT_ORDER
                           else 99, p))
        arch_out = []
        arch_out.append(r"\clearpage")
        arch_out.append(
            r"\subsection*{Architecture \textbf{" + arch + r"} --- "
            r"per-perturbation win-rate}")
        arch_out.append(
            f"\\label{{tab:wrhist_{arch_label}}}")
        arch_out.append(
            f"This page shows {len(ordered_perts)} bar charts, one per "
            r"\textbf{perturbation type} measured on \textbf{" + arch
            + r"} (all p\_sizes of that perturbation type are merged "
            r"into a single bar per combo). Chart construction follows "
            r"the section intro above; the colour key below covers "
            r"every combo on this page.")
        arch_abbrevs = set()
        for pert in ordered_perts:
            for rec in by_pert[pert]:
                arch_abbrevs.add(rec["abbrev"])
        position_map = {
            abbrev: i for i, abbrev in enumerate(sorted(arch_abbrevs))
        }
        legend_block = _render_combo_color_legend(arch_abbrevs, color_map)
        if legend_block:
            arch_out.append(r"\par\medskip")
            arch_out.append(legend_block)
            arch_out.append(r"\par\medskip")
        for i in range(0, len(ordered_perts), 2):
            arch_out.append(r"\par\medskip\noindent")
            arch_out.append(r"\begin{minipage}[t]{0.48\linewidth}")
            arch_out.append(_render_perturbation_winrate_histogram(
                arch, ordered_perts[i], by_pert[ordered_perts[i]],
                color_map, position_map))
            arch_out.append(r"\end{minipage}")
            arch_out.append(r"\hfill")
            if i + 1 < len(ordered_perts):
                arch_out.append(r"\begin{minipage}[t]{0.48\linewidth}")
                arch_out.append(_render_perturbation_winrate_histogram(
                    arch, ordered_perts[i + 1],
                    by_pert[ordered_perts[i + 1]],
                    color_map, position_map))
                arch_out.append(r"\end{minipage}")
            else:
                arch_out.append(
                    r"\begin{minipage}[t]{0.48\linewidth}"
                    r"\strut\end{minipage}")
            arch_out.append("")
        total_charts += len(ordered_perts)
        arch_blocks.append("\n".join(arch_out))

    if not arch_blocks:
        filter_repr = _format_combination_filter(combination_filter)
        return ("% no per-perturbation win-rate histograms"
                + (f" (combo_filter={filter_repr})"
                   if filter_repr else "")
                + "\n")

    header = []
    header.append(r"\clearpage")
    header.append(
        r"\section*{Per-perturbation combo win-rate histograms}")
    intro = (
        r"The figures below mirror the per-perturbation $\overline{\text{sp}}$ "
        r"histograms above but show each combo's \textbf{win rate} on the "
        r"$\langle\text{arch}, \text{pert}\rangle$ slice instead of its "
        r"mean speed-up. \textbf{Each column encodes three quantities by "
        r"nested rectangles on an absolute-count $y$-axis} (the axis "
        r"scale runs $0\to n_{\max}$, where $n_{\max}$ is the largest "
        r"\textsf{n\_cells} of any combo on the chart, marked by a "
        r"dashed reference line): a faint \emph{envelope} rectangle of "
        r"height $n_{\max}$ (the maximum coverage anywhere on the "
        r"chart); a mid-grey \emph{tests} rectangle of height "
        r"\textsf{n\_cells} (this combo's actual coverage on the slice); "
        r"and the coloured \emph{wins} bar inside, split into a "
        r"\emph{solid-fill bottom} = $\textsf{finished wins}$ "
        r"(wins where the run finished and advstd was faster, "
        r"$\text{sp}>1$) and a \emph{faded + diagonally hatched top} "
        r"= $\textsf{timeout wins}$ (wins where both modes hit "
        r"\texttt{TIME\_LIMIT} and advstd's "
        r"$\overline{\text{gap}}_\mathrm{adv}<\overline{\text{gap}}_\mathrm{std}$ "
        r"on those cells), rendered in the \emph{same combo colour} so "
        r"the bar's identity is readable while the win-type split is "
        r"unmistakable. A combo's \textbf{win rate} is therefore the "
        r"fraction of the mid-grey rectangle that is coloured "
        r"(solid + hatched); an under-tested combo's column is "
        r"visibly shorter than its envelope. A combo with no timeout "
        r"cells shows a fully solid coloured bar, and a combo whose "
        r"wins all came from timeout-tighter-gap cells shows only a "
        r"hatched top. \textbf{Combo columns occupy a stable x "
        r"position across every chart of an arch} (alphabetical "
        r"order on the combo abbreviation), so a given combo lives "
        r"in the same column on every chart --- the reader can "
        r"follow that column down the page and watch both the win "
        r"rate and the sample size change as the perturbation type "
        r"changes. The \textcolor{orange!85!black}{$\bigstar$} marks "
        r"the combo with the highest win rate on each chart. A "
        r"best-to-worst ranking summary line sits above each chart, "
        r"showing the win rate and the "
        r"$(\textsf{finished}+\textsf{timeout})/\textsf{n}$ "
        r"breakdown for each combo. Cells with mismatched "
        r"\texttt{--timout} caps between the two modes are excluded "
        r"(same rule as the per-arch perturbation tables and the blue "
        r"overall ranking).")
    if combination_filter:
        n_combos_filter = len(combination_filter)
        combo_list_tex = _format_combination_filter_tex(combination_filter)
        intro += (
            "\n\n"
            r"\noindent\textbf{Note:} these histograms are restricted "
            f"to the {n_combos_filter} "
            f"{'combination' if n_combos_filter == 1 else 'combinations'} "
            + combo_list_tex
            + r" (via \texttt{--combination\_table}).")
    intro += (
        "\n\n"
        f"% auto-generated: {total_charts} per-perturbation win-rate "
        f"histograms across {len(arch_blocks)} archs, seed={seed}, tau={tau}"
        + (f", combo_filter={_format_combination_filter(combination_filter)}"
           if combination_filter else "")
        + "\n")
    header.append(intro)
    return "\n".join(header) + "\n\n" + "\n\n".join(arch_blocks) + "\n"


def render_perturbation_combined_section(per_arch_combos, seed, tau,
                                         combination_filter=None):
    """Emit a single section that pairs each $(arch, pert)$ slice's
    $\\overline{\\text{sp}}$ histogram and win-rate histogram in one
    row, so both metrics for the same slice can be read at a glance.

    Layout per arch:
      * `\\subsection*` header + one colour key covering every combo
        that appears in either chart on the page.
      * One row per perturbation type, with two minipages side-by-side:
        left = sp chart, right = win-rate chart. Combo x-positions are
        shared between the two charts (via a per-arch `position_map`
        built from the union of abbrevs present on either side), so a
        combo lives in the same column in both halves of every row
        and across every perturbation type on the page.

    The two charts and their on-page captions are the same renderings
    used by the standalone sp / win-rate sections; this section just
    pairs them up.
    """
    color_map = _build_combo_color_map(
        per_arch_combos, combination_filter=combination_filter)
    arch_blocks = []
    total_pairs = 0
    for arch, all_combos in per_arch_combos:
        by_pert_sp = _perturbation_combo_data(
            all_combos, combination_filter=combination_filter)
        by_pert_wr = _perturbation_winrate_data(
            all_combos, combination_filter=combination_filter)
        if not by_pert_sp and not by_pert_wr:
            continue
        arch_label = arch.replace("=", "_")
        all_perts = set(by_pert_sp.keys()) | set(by_pert_wr.keys())
        ordered_perts = sorted(
            all_perts,
            key=lambda p: (PERT_ORDER.index(p) if p in PERT_ORDER
                           else 99, p))
        arch_out = []
        arch_out.append(r"\clearpage")
        arch_out.append(
            r"\subsection*{Architecture \textbf{" + arch + r"} --- "
            r"$\overline{\text{sp}}$ paired with win-rate}")
        arch_out.append(
            f"\\label{{tab:perthist_{arch_label}}}")
        arch_out.append(
            f"This page shows {len(ordered_perts)} paired rows, one "
            r"per \textbf{perturbation type} measured on \textbf{"
            + arch + r"}. Each row repeats the same $\langle"
            r"\text{arch}, \text{pert}\rangle$ slice twice: the "
            r"\emph{left} chart is the mean-speedup view (section "
            r"intro above); the \emph{right} chart is the win-rate "
            r"view (same intro, different bar definition). Combo "
            r"columns occupy the same x position in both charts of a "
            r"row, so a column can be scanned down the page to follow "
            r"one combo across perturbation types in both metrics.")
        # Shared per-arch column order from the union of abbrevs.
        arch_abbrevs = set()
        for pert in ordered_perts:
            for rec in by_pert_sp.get(pert, []):
                arch_abbrevs.add(rec["abbrev"])
            for rec in by_pert_wr.get(pert, []):
                arch_abbrevs.add(rec["abbrev"])
        position_map = {
            abbrev: i for i, abbrev in enumerate(sorted(arch_abbrevs))
        }
        # One paired row per perturbation: [sp chart | wr chart].
        for pert in ordered_perts:
            sp_recs = by_pert_sp.get(pert, [])
            wr_recs = by_pert_wr.get(pert, [])
            arch_out.append(r"\par\medskip\noindent")
            arch_out.append(r"\begin{minipage}[t]{0.48\linewidth}")
            if sp_recs:
                arch_out.append(_render_perturbation_histogram(
                    arch, pert, sp_recs, color_map, position_map))
            else:
                arch_out.append(r"\strut")
            arch_out.append(r"\end{minipage}")
            arch_out.append(r"\hfill")
            arch_out.append(r"\begin{minipage}[t]{0.48\linewidth}")
            if wr_recs:
                arch_out.append(_render_perturbation_winrate_histogram(
                    arch, pert, wr_recs, color_map, position_map))
            else:
                arch_out.append(r"\strut")
            arch_out.append(r"\end{minipage}")
            arch_out.append("")
        total_pairs += len(ordered_perts)
        arch_blocks.append("\n".join(arch_out))

    if not arch_blocks:
        filter_repr = _format_combination_filter(combination_filter)
        return ("% no paired per-perturbation histograms"
                + (f" (combo_filter={filter_repr})"
                   if filter_repr else "")
                + "\n")

    header = []
    header.append(r"\clearpage")
    header.append(
        r"\section*{Per-perturbation histograms: "
        r"$\overline{\text{sp}}$ paired with win-rate}")
    intro = (
        r"The figures below answer ``which advstd combination should "
        r"I pick on this perturbation $\times$ architecture?'' from "
        r"\emph{two} angles at once. Each row of paired charts shows "
        r"a single $\langle\text{arch}, \text{pert}\rangle$ slice "
        r"twice: the \textbf{left} chart reports the combo's mean "
        r"speed-up (how fast it is on average), the \textbf{right} "
        r"chart reports its win rate (how often it actually beats "
        r"standard). Combos occupy a \textbf{stable x position} "
        r"across both charts of a row \emph{and} across every row of "
        r"an arch (alphabetical on the combo abbreviation), so a "
        r"reader can scan one column down the page and watch a "
        r"combo's behaviour change as the perturbation type changes "
        r"in either metric.\par\medskip "
        r"\noindent\textbf{Left chart} --- bar height $=$ the "
        r"arithmetic mean of $\overline{\text{sp}}$ across every "
        r"$c_\mathrm{src}\times c_\mathrm{tgt}\times$ seed cell across "
        r"all p\_sizes of the perturbation type. \textbf{Each bar is "
        r"vertically split} into a \emph{solid bottom} and a "
        r"\emph{faded $+$ diagonally hatched top} in the same combo "
        r"colour: the solid fraction equals $\textsf{wins}/"
        r"\textsf{p\_sizes}$, where a p\_size's \textsf{win} is "
        r"defined by a \emph{hybrid} rule that mirrors the orange "
        r"gap-comparison table's treatment of the two cell "
        r"populations --- a p\_size in which every cell finished (no "
        r"\texttt{TIME\_LIMIT}) wins iff $\overline{\text{sp}}>1$; a "
        r"p\_size in which every cell hit \texttt{TIME\_LIMIT} wins "
        r"iff $\overline{\text{gap}}_\mathrm{adv}<"
        r"\overline{\text{gap}}_\mathrm{std}$ across its timeout "
        r"cells; a p\_size with cells of both kinds is judged "
        r"\emph{per cell} and wins iff a majority of cells win on "
        r"their own type (each non-timeout cell votes \emph{yes} when "
        r"$\text{sp}>1$, each timeout cell votes \emph{yes} when its "
        r"$u-l$ gap on advstd is tighter than on standard). Bars "
        f"above ${PERT_HIST_YMAX:.1f}$ are clipped at that height and "
        r"the true value is printed in \textbf{bold} above the bar; "
        r"the dashed line marks the $\overline{\text{sp}}=1$ "
        r"breakeven.\par\medskip "
        r"\noindent\textbf{Right chart} --- each column encodes three "
        r"quantities by nested rectangles on an absolute-count $y$-axis "
        r"(scale $0\to n_{\max}$, where $n_{\max}$ is the largest "
        r"\textsf{n\_cells} of any combo on the chart, marked by a "
        r"dashed reference line): a faint \emph{envelope} of height "
        r"$n_{\max}$ (maximum coverage anywhere on the chart); a "
        r"mid-grey \emph{tests} rectangle of height \textsf{n\_cells} "
        r"(this combo's actual coverage on the slice); and a coloured "
        r"\emph{wins} bar inside, split into a \emph{solid bottom} "
        r"$=$ $\textsf{finished wins}$ (cells where the run finished "
        r"and advstd was faster, $\text{sp}>1$) and a \emph{faded $+$ "
        r"hatched top} $=$ $\textsf{timeout wins}$ (cells where both "
        r"modes hit \texttt{TIME\_LIMIT} and advstd's "
        r"$\overline{\text{gap}}_\mathrm{adv}<"
        r"\overline{\text{gap}}_\mathrm{std}$). The combo's "
        r"\textbf{win rate} is therefore the fraction of the mid-grey "
        r"rectangle that is coloured; an under-tested combo's column "
        r"is visibly shorter than its envelope. A combo with no "
        r"timeout cells shows a fully solid coloured bar, and a "
        r"combo whose wins all came from tighter-gap timeout cells "
        r"shows only a hatched top.\par\medskip "
        r"\noindent The \textcolor{orange!85!black}{$\bigstar$} above "
        r"each chart marks the best combo on that chart (highest "
        r"$\overline{\text{sp}}$ on the left, highest win rate on the "
        r"right --- the two stars may pick different combos when the "
        r"orderings disagree). A best-to-worst ranking summary line "
        r"sits above each chart with the per-combo metric value and "
        r"the relevant breakdown ($\textsf{wins}/\textsf{p\_sizes}$ "
        r"for sp, $(\textsf{finished}+\textsf{timeout})/\textsf{n}$ "
        r"for win rate). Cells with mismatched \texttt{--timout} caps "
        r"between the two modes are excluded (same rule as the "
        r"per-arch perturbation tables and the blue overall ranking).")
    if combination_filter:
        n_combos_filter = len(combination_filter)
        combo_list_tex = _format_combination_filter_tex(combination_filter)
        intro += (
            "\n\n"
            r"\noindent\textbf{Note:} these histograms are restricted "
            f"to the {n_combos_filter} "
            f"{'combination' if n_combos_filter == 1 else 'combinations'} "
            + combo_list_tex
            + r" (via \texttt{--combination\_table}).")
    intro += (
        "\n\n"
        f"% auto-generated: {total_pairs} paired sp+win-rate rows "
        f"across {len(arch_blocks)} archs, seed={seed}, tau={tau}"
        + (f", combo_filter={_format_combination_filter(combination_filter)}"
           if combination_filter else "")
        + "\n")
    header.append(intro)
    return "\n".join(header) + "\n\n" + "\n\n".join(arch_blocks) + "\n"


def render_perturbation_combined_section_all_archs(
        per_arch_combos, seed, tau, combination_filter=None):
    """Same paired sp/win-rate layout as the per-arch section, but
    cells are pooled across every architecture first — one paired row
    per perturbation type, with combo bars aggregating evidence from
    every arch we ran. Architecture is reported as ``ALL`` in the
    per-chart caption."""
    pooled = []
    for _arch, all_combos in per_arch_combos:
        pooled.extend(all_combos)
    if not pooled:
        return ""
    color_map = _build_combo_color_map(
        per_arch_combos, combination_filter=combination_filter)
    by_pert_sp = _perturbation_combo_data(
        pooled, combination_filter=combination_filter)
    by_pert_wr = _perturbation_winrate_data(
        pooled, combination_filter=combination_filter)
    if not by_pert_sp and not by_pert_wr:
        return ""
    all_perts = set(by_pert_sp.keys()) | set(by_pert_wr.keys())
    ordered_perts = sorted(
        all_perts,
        key=lambda p: (PERT_ORDER.index(p) if p in PERT_ORDER else 99, p))
    # Stable per-section column order from the union of abbrevs across
    # every paired row, so combos line up across perturbations.
    sec_abbrevs = set()
    for pert in ordered_perts:
        for rec in by_pert_sp.get(pert, []):
            sec_abbrevs.add(rec["abbrev"])
        for rec in by_pert_wr.get(pert, []):
            sec_abbrevs.add(rec["abbrev"])
    position_map = {a: i for i, a in enumerate(sorted(sec_abbrevs))}
    lines = []
    lines.append(r"\clearpage")
    lines.append(
        r"\section*{Per-perturbation histograms across all architectures: "
        r"$\overline{\text{sp}}$ paired with win-rate}")
    lines.append(
        r"This section pools every $(c_\mathrm{src}, c_\mathrm{tgt}, "
        r"\text{seed})$ cell across \emph{all} measured architectures, "
        r"so each paired row reports a combo's behaviour on one "
        r"perturbation type aggregated over every arch we ran. Bar "
        r"definitions, the $\star$ best-combo marker, and the rules used to "
        r"split each bar into solid + hatched segments are exactly "
        r"the same as in the per-arch section above. Architecture is "
        r"shown as \textbf{ALL} in each chart's caption.")
    if combination_filter:
        n_combos_filter = len(combination_filter)
        combo_list_tex = _format_combination_filter_tex(combination_filter)
        lines.append(
            r"\noindent\textbf{Note:} these histograms are restricted "
            f"to the {n_combos_filter} "
            f"{'combination' if n_combos_filter == 1 else 'combinations'} "
            + combo_list_tex
            + r" (via \texttt{--combination\_table}).")
    for pert in ordered_perts:
        sp_recs = by_pert_sp.get(pert, [])
        wr_recs = by_pert_wr.get(pert, [])
        lines.append(r"\par\medskip\noindent")
        lines.append(r"\begin{minipage}[t]{0.48\linewidth}")
        if sp_recs:
            lines.append(_render_perturbation_histogram(
                "ALL", pert, sp_recs, color_map, position_map))
        else:
            lines.append(r"\strut")
        lines.append(r"\end{minipage}")
        lines.append(r"\hfill")
        lines.append(r"\begin{minipage}[t]{0.48\linewidth}")
        if wr_recs:
            lines.append(_render_perturbation_winrate_histogram(
                "ALL", pert, wr_recs, color_map, position_map))
        else:
            lines.append(r"\strut")
        lines.append(r"\end{minipage}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_perturbation_combined_section_all(
        per_arch_combos, seed, tau, combination_filter=None):
    """Same paired sp/win-rate layout, but cells are pooled across
    every architecture \\emph{and} every perturbation type — one
    paired row total, one bar per combo summarising every cell we
    have. Architecture and perturbation are both reported as ``ALL``
    in the per-chart caption."""
    pooled = []
    for _arch, all_combos in per_arch_combos:
        pooled.extend(all_combos)
    if not pooled:
        return ""
    color_map = _build_combo_color_map(
        per_arch_combos, combination_filter=combination_filter)
    by_pert_sp = _perturbation_combo_data(
        pooled, combination_filter=combination_filter)
    by_pert_wr = _perturbation_winrate_data(
        pooled, combination_filter=combination_filter)
    if not by_pert_sp and not by_pert_wr:
        return ""

    # Collapse per-pert -> per-combo by summing/weighting the existing
    # aggregates. n_cells is the per-pert cell count (from
    # `_perturbation_combo_data`), so a weighted mean of `mean_sp`
    # reproduces the all-cell arithmetic mean of sp.
    sp_by_key = {}
    for pert_records in by_pert_sp.values():
        for rec in pert_records:
            agg = sp_by_key.setdefault(rec["key"], {
                "key": rec["key"],
                "label": rec["label"],
                "abbrev": rec["abbrev"],
                "sum_sp": 0.0,
                "n_cells": 0,
                "n_psizes_tested": 0,
                "n_psizes_won": 0,
                "psize_details": [],
            })
            agg["sum_sp"] += rec["mean_sp"] * rec["n_cells"]
            agg["n_cells"] += rec["n_cells"]
            agg["n_psizes_tested"] += rec["n_psizes_tested"]
            agg["n_psizes_won"] += rec["n_psizes_won"]
            agg["psize_details"].extend(rec["psize_details"])
    sp_records = []
    for agg in sp_by_key.values():
        if agg["n_cells"] == 0:
            continue
        sp_records.append({
            "key": agg["key"],
            "label": agg["label"],
            "abbrev": agg["abbrev"],
            "mean_sp": agg["sum_sp"] / agg["n_cells"],
            "n_cells": agg["n_cells"],
            "n_psizes_tested": agg["n_psizes_tested"],
            "n_psizes_won": agg["n_psizes_won"],
            "psize_details": agg["psize_details"],
        })
    sp_records.sort(key=lambda r: r["abbrev"])

    wr_by_key = {}
    for pert_records in by_pert_wr.values():
        for rec in pert_records:
            agg = wr_by_key.setdefault(rec["key"], {
                "key": rec["key"],
                "label": rec["label"],
                "abbrev": rec["abbrev"],
                "n_cells": 0,
                "n_finished_cells": 0,
                "n_timeout_cells": 0,
                "n_finished_wins": 0,
                "n_timeout_wins": 0,
                "psizes": set(),
            })
            agg["n_cells"] += rec["n_cells"]
            agg["n_finished_cells"] += rec["n_finished_cells"]
            agg["n_timeout_cells"] += rec["n_timeout_cells"]
            agg["n_finished_wins"] += rec["n_finished_wins"]
            agg["n_timeout_wins"] += rec["n_timeout_wins"]
            agg["psizes"].update(rec["psizes"])
    wr_records = []
    for agg in wr_by_key.values():
        n = agg["n_cells"]
        if n == 0:
            continue
        n_wins = agg["n_finished_wins"] + agg["n_timeout_wins"]
        wr_records.append({
            "key": agg["key"],
            "label": agg["label"],
            "abbrev": agg["abbrev"],
            "n_cells": n,
            "n_finished_cells": agg["n_finished_cells"],
            "n_timeout_cells": agg["n_timeout_cells"],
            "n_finished_wins": agg["n_finished_wins"],
            "n_timeout_wins": agg["n_timeout_wins"],
            "n_wins": n_wins,
            "finished_win_frac": agg["n_finished_wins"] / n,
            "timeout_win_frac": agg["n_timeout_wins"] / n,
            "win_rate": n_wins / n,
            "psizes": sorted(agg["psizes"]),
        })
    wr_records.sort(key=lambda r: r["abbrev"])

    sec_abbrevs = {r["abbrev"] for r in sp_records}
    sec_abbrevs.update(r["abbrev"] for r in wr_records)
    if not sec_abbrevs:
        return ""
    position_map = {a: i for i, a in enumerate(sorted(sec_abbrevs))}

    lines = []
    lines.append(r"\clearpage")
    lines.append(
        r"\section*{Overall histograms across all architectures and "
        r"perturbations: $\overline{\text{sp}}$ paired with win-rate}")
    lines.append(
        r"This section pools every $(c_\mathrm{src}, c_\mathrm{tgt}, "
        r"\text{seed})$ cell across \emph{all} architectures \emph{and} "
        r"\emph{all} perturbation types into a single paired row---one "
        r"bar per combo summarising every cell on record. Bar "
        r"definitions are the same as in the sections above; both the "
        r"architecture and the perturbation are shown as \textbf{ALL} "
        r"in each chart's caption.")
    if combination_filter:
        n_combos_filter = len(combination_filter)
        combo_list_tex = _format_combination_filter_tex(combination_filter)
        lines.append(
            r"\noindent\textbf{Note:} these histograms are restricted "
            f"to the {n_combos_filter} "
            f"{'combination' if n_combos_filter == 1 else 'combinations'} "
            + combo_list_tex
            + r" (via \texttt{--combination\_table}).")
    lines.append(r"\par\medskip\noindent")
    lines.append(r"\begin{minipage}[t]{0.48\linewidth}")
    if sp_records:
        lines.append(_render_perturbation_histogram(
            "ALL", "ALL", sp_records, color_map, position_map))
    else:
        lines.append(r"\strut")
    lines.append(r"\end{minipage}")
    lines.append(r"\hfill")
    lines.append(r"\begin{minipage}[t]{0.48\linewidth}")
    if wr_records:
        lines.append(_render_perturbation_winrate_histogram(
            "ALL", "ALL", wr_records, color_map, position_map))
    else:
        lines.append(r"\strut")
    lines.append(r"\end{minipage}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_merged_combo_tables(arch, arch_label, by_ps, combination_filter,
                                seed, tau, has_timeout_table):
    """Emit one \\begin{table} per (pert, psize) for the filtered combos.

    All c_src groups for that (pert, psize) are stacked into a single
    table; each c_src group is rendered as its own block via
    `_emit_block_rows` with `row_color=_csrc_color(c_src)`, so c_src=0
    rows stay green, c_src=1 yellow, c_src=2 magenta, etc. — the same
    palette used in the unfiltered tables. Block ordering inside a
    table is by ascending c_src. When `combination_filter` carries
    multiple combos, every matching combo's rows appear together in
    the same c_src block (each combo printed as its own combo group
    via `_emit_block_rows`, which already shows per-combo flag values
    on the first cell of each combo).
    """
    by_pp = defaultdict(lambda: defaultdict(list))
    for ps, sub_combos in by_ps.items():
        pert, psize, c_src, _bucket = ps
        # Multiple tau_buckets may collapse here (a single combo
        # always pins one bucket, but with a multi-combo filter,
        # several buckets may legitimately land here). Merge them
        # into the same c_src block so the output is one table per
        # (pert, psize).
        by_pp[(pert, psize)][c_src].extend(sub_combos)
    ordered_pp = sorted(
        by_pp.keys(),
        key=lambda pp: (PERT_ORDER.index(pp[0]) if pp[0] in PERT_ORDER
                        else 99, pp[1]))

    combo_tex = _format_combination_filter_tex(combination_filter)
    n_combos_filter = (len(combination_filter)
                       if combination_filter else 0)
    combo_phrase = ("combo " if n_combos_filter <= 1
                    else f"{n_combos_filter} combos ")

    tables = []
    anchor_issued = False
    sub_idx = 0
    for (pert, psize) in ordered_pp:
        csrc_groups = by_pp[(pert, psize)]
        out = []
        # No forced \clearpage here: merged-combo tables are short, so
        # let LaTeX float them onto the same page when they fit.
        _emit_table_header(out, clear_page=False)
        first_csrc = True
        block_cells = 0
        n_combos = 0
        for c_src in sorted(csrc_groups.keys()):
            block = sorted(csrc_groups[c_src], key=lambda r: -r[3])
            if not first_csrc:
                out.append(r"\hline")
            _emit_block_rows(out, (pert, psize), block,
                             head_first=first_csrc,
                             row_color=_csrc_color(c_src))
            first_csrc = False
            block_cells += sum(len(e[14]) for e in block)
            n_combos += len(block)
        out.append(r"\hline")
        out.append(r"\end{tabular}%")
        out.append(r"\end{adjustbox}")
        is_anchor = not anchor_issued
        if is_anchor:
            cross_ref = (
                f" Timeout-cell gaps: Table~\\ref{{tab:timeout_gap_"
                f"{arch_label}}}." if has_timeout_table else "")
            caption = (
                f"\\caption{{Architecture \\textbf{{{arch}}} --- "
                f"merged per-perturbation table for {combo_phrase}"
                f"{combo_tex} at "
                f"{_seed_tau_phrase(seed, tau)}; perturbation "
                f"\\textbf{{{pert}}}(\\texttt{{{psize}}}). All "
                f"$c_\\mathrm{{src}}$ groups are stacked into one "
                f"table; rows keep their original $c_\\mathrm{{src}}$ "
                f"tint (green=0, yellow=1, magenta=2, \\ldots) so "
                f"groups remain visually distinct. \\textsf{{sp}}$>\\!1$ "
                f"(advstd faster) in \\textbf{{bold}}; cells within each "
                f"$c_\\mathrm{{src}}$ block sorted by $c_\\mathrm{{tgt}}$. "
                f"{block_cells} cell rows.{cross_ref}}}")
            anchor_issued = True
        else:
            caption = (
                f"\\caption{{Architecture \\textbf{{{arch}}} --- "
                f"perturbation \\textbf{{{pert}}}(\\texttt{{{psize}}}), "
                f"{combo_phrase}{combo_tex} ({block_cells} cell rows); "
                f"continuation of Table~\\ref{{tab:safe_{arch_label}}}. "
                f"Column meanings: see "
                f"Table~\\ref{{tab:safe_{arch_label}}}.}}")
        out.append(caption)
        labels = [f"\\label{{tab:safe_{arch_label}_{sub_idx}}}"]
        if is_anchor:
            labels.insert(0, f"\\label{{tab:safe_{arch_label}}}")
        out.extend(labels)
        out.append(r"\end{table}")
        out.append("")
        tables.append("\n".join(out))
        sub_idx += 1
    return ("\n".join(tables), "")


def render_table(arch, all_combos, timeout_all, seed, tau,
                 combination_filter=None):
    # When --combination_table is set, restrict the per-arch perturbation
    # tables (the c_src-tinted blocks) to only the matching combos. The
    # overall summary and the timeout-gap table are also filtered (see
    # render_overall_combo_summary / render_timeout_table).
    if combination_filter is not None:
        all_combos = [e for e in all_combos
                      if _combo_label(e[0][0]) in combination_filter]
    if not all_combos:
        filter_repr = _format_combination_filter(combination_filter)
        return (f"% no rows for arch={arch} at seed={seed}, "
                f"tau={tau}"
                + (f", combo={filter_repr}" if filter_repr else "")
                + "\n", "")
    has_timeout_table = bool(timeout_all)
    arch_label = arch.replace("=", "_")
    n_total_cells = sum(len(e[14]) for e in all_combos)

    # Split each combo by c_src and bucket into (pert, size, c_src,
    # tau_bucket). The tau-bucket split produces two green sub-tables per
    # (pert, psize, c_src): one for the "aggressive" tau values
    # (off / 0.5 / 1.0) and one for the intermediate values
    # (0.01 / 0.05 / 0.1), keeping each sub-table readable on one page.
    by_ps = defaultdict(list)
    for e in all_combos:
        grp = e[0]
        pert, psize = grp[1], grp[2]
        # relax_threshold is at index 7 of the 9-tuple grp[0] (sibling_gate
        # took over the last slot). Index from the start to stay correct
        # whether the legacy 8-tuple or new 9-tuple key is used.
        rt = grp[0][7]
        bucket = _tau_bucket(rt)
        cells = e[14]
        cells_by_csrc = defaultdict(list)
        for cell in cells:
            cells_by_csrc[cell[0]].append(cell)
        for c_src_key, sub_cells in cells_by_csrc.items():
            sps = [c[2] / c[3] for c in sub_cells
                   if c[3] > 0 and not c[10]]
            sub_mean_sp = (sum(sps) / len(sps)
                           if sps else float("-inf"))
            sub_combo = (e[0], e[1], e[2], sub_mean_sp,
                         e[4], e[5], e[6], e[7], e[8], e[9],
                         e[10], e[11], e[12], e[13], sub_cells)
            by_ps[(pert, psize, c_src_key, bucket)].append(sub_combo)
    ordered_ps = sorted(
        by_ps.keys(),
        key=lambda ps: (PERT_ORDER.index(ps[0]) if ps[0] in PERT_ORDER
                        else 99, ps[1], ps[2],
                        _TAU_BUCKET_ORDER.index(ps[3])))

    if combination_filter is not None:
        # Merged-table mode: with the table restricted to a single combo,
        # collapse the per-(c_src, tau_bucket) sub-tables into one
        # \begin{table} per (pert, psize). Each c_src group inside the
        # merged table keeps its original tint (_csrc_color) so the
        # reader can still match rows back to the c_src palette used in
        # the unfiltered green/yellow/pink tables.
        return _render_merged_combo_tables(
            arch, arch_label, by_ps, combination_filter, seed, tau,
            has_timeout_table)

    if n_total_cells <= MAX_ROWS_SINGLE_TABLE:
        # Single combined table — one block per midrule.
        out = []
        _emit_table_header(out)
        first = True
        for ps in ordered_ps:
            if not first:
                out.append(r"\hline")
            first = False
            block = sorted(by_ps[ps], key=lambda r: -r[3])
            _emit_block_rows(out, (ps[0], ps[1]), block, head_first=True)
        out.append(r"\hline")
        out.append(r"\end{tabular}%")
        out.append(r"\end{adjustbox}")
        out.append(
            f"\\caption{{Architecture \\textbf{{{arch}}} --- every "
            f"recorded advstd combination at {_seed_tau_phrase(seed, tau)}, "
            f"broken out by perturbation type, size, and $c_\\mathrm{{src}}$. "
            + _full_caption_body(arch, n_total_cells, has_timeout_table)
            + "}")
        out.append(f"\\label{{tab:safe_{arch_label}}}")
        out.append(r"\end{table}")
        out.append("")
        return ("\n".join(out), "")

    # Per-block split mode — one \begin{table} per
    # (perturbation, size, c_src, tau_bucket). Agg-bucket tables
    # (green) stay in the main flow; mid-bucket tables (cyan) are
    # pushed to the end of the document so the shorter, headline
    # agg-bucket tables aren't interleaved with the denser mid-bucket
    # ones.
    agg_tables = []
    mid_tables = []
    agg_anchor_issued = False
    sub_idx = 0
    for ps in ordered_ps:
        full_block = sorted(by_ps[ps], key=lambda r: -r[3])
        # Chunk the sorted combos so each emitted sub-table fits on a
        # page. Since combos are already sorted by descending mean sp,
        # chunk k contains combos whose rank falls in a contiguous
        # slice of the ranking.
        n = len(full_block)
        n_chunks = max(1,
                       (n + MAX_COMBOS_PER_SUBTABLE - 1)
                       // MAX_COMBOS_PER_SUBTABLE)
        # Spread combos evenly across chunks rather than last-chunk
        # leftovers (e.g. 20 → 10+10, 11 → 6+5, not 10+1).
        chunk_size = (n + n_chunks - 1) // n_chunks
        chunks = [full_block[k:k + chunk_size]
                  for k in range(0, n, chunk_size)]
        for chunk_i, block in enumerate(chunks):
            out = []
            _emit_table_header(out)
            pert, psize, c_src, bucket = ps
            row_color = _csrc_color(c_src)
            _emit_block_rows(out, (pert, psize), block,
                             head_first=True, row_color=row_color)
            out.append(r"\hline")
            out.append(r"\end{tabular}%")
            out.append(r"\end{adjustbox}")
            tau_phrase = _tau_bucket_label(bucket)
            block_cells = sum(len(e[14]) for e in block)
            if len(chunks) > 1:
                part_phrase = (
                    f" (part {chunk_i + 1}/{len(chunks)}, "
                    f"combos ranked {chunk_i * chunk_size + 1}"
                    f"--{chunk_i * chunk_size + len(block)} "
                    f"of {n} by mean \\textsf{{sp}})")
            else:
                part_phrase = ""
            is_anchor = (bucket == "agg" and chunk_i == 0
                         and not agg_anchor_issued)
            if is_anchor:
                caption = (
                    f"\\caption{{Architecture \\textbf{{{arch}}} --- every "
                    f"recorded advstd combination at "
                    f"{_seed_tau_phrase(seed, tau)}, "
                    f"perturbation \\textbf{{{pert}}}"
                    f"(\\texttt{{{psize}}}), $c_\\mathrm{{src}}={c_src}$, "
                    f"{tau_phrase}{part_phrase} "
                    f"({len(block)} combos, {block_cells} cell rows; "
                    f"see Tables labelled \\texttt{{tab:safe\\_{arch_label}\\_*}}"
                    f" for the other perturbations / $c_\\mathrm{{src}}$ / "
                    f"$\\tau$-buckets of this arch). "
                    + _full_caption_body(arch, n_total_cells,
                                          has_timeout_table)
                    + "}")
                agg_anchor_issued = True
            else:
                caption = (
                    f"\\caption{{Architecture \\textbf{{{arch}}} --- "
                    f"perturbation \\textbf{{{pert}}}(\\texttt{{{psize}}}), "
                    f"$c_\\mathrm{{src}}={c_src}$, {tau_phrase}{part_phrase}; "
                    f"continuation of Table~\\ref{{tab:safe_{arch_label}}} "
                    f"({len(block)} combos, {block_cells} cell rows). "
                    f"Column meanings: see Table~\\ref{{tab:safe_{arch_label}}}.}}")
            out.append(caption)
            labels = [f"\\label{{tab:safe_{arch_label}_{sub_idx}}}"]
            if is_anchor:
                labels.insert(0, f"\\label{{tab:safe_{arch_label}}}")
            out.extend(labels)
            out.append(r"\end{table}")
            out.append("")
            rendered = "\n".join(out)
            if bucket == "agg":
                agg_tables.append(rendered)
            else:
                mid_tables.append(rendered)
            sub_idx += 1
    return ("\n".join(agg_tables), "\n".join(mid_tables))


def render_timeout_table(arch, timeout_all, seed, tau,
                         combination_filter=None):
    # When --combination_table is set, restrict the orange
    # timeout-gap table to the matching combos too. The combo key on
    # each timeout entry is `e[0][0]` (same shape as the per-arch
    # combo grp), so we can apply `_combo_label` exactly like in
    # `render_table`.
    if combination_filter is not None:
        timeout_all = [e for e in timeout_all
                       if _combo_label(e[0][0]) in combination_filter]
    if not timeout_all:
        filter_repr = _format_combination_filter(combination_filter)
        return (f"% no timeout cells for arch={arch} at seed={seed}, "
                f"tau={tau}"
                + (f", combo={filter_repr}" if filter_repr else "")
                + "\n")

    by_ps = defaultdict(list)
    for e in timeout_all:
        by_ps[(e[0][1], e[0][2])].append(e)
    ordered_ps = sorted(
        by_ps.keys(),
        key=lambda ps: (PERT_ORDER.index(ps[0]) if ps[0] in PERT_ORDER
                        else 99, ps[1]))

    out = []
    out.append(r"\begin{table}[!htbp]")
    out.append(r"\centering")
    out.append(r"\scriptsize")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\begin{adjustbox}{max width=\textwidth,center}%")
    out.append(r"\begin{tabular}{@{}l l l l l | r r r r r r@{}}")
    out.append(r"\hline")
    out.append(r"p\_type & p\_size & \tech{bound\_tight} & "
               r"\tech{varHint} & $\tau$ & c\_src & "
               r"n\_timeout & n\_advstd\_tighter & "
               r"avg\_gap\_std & avg\_gap\_adv & "
               r"$|\overline{\text{gap}}_\mathrm{std}-\overline{\text{gap}}_\mathrm{adv}|$ \\")
    out.append(r"\hline")

    first_row = True
    for ps in ordered_ps:
        # Sort by tighter-count desc, then by c_src for stable readout.
        block = sorted(by_ps[ps], key=lambda r: (-r[3], r[1]))
        head = True
        for (grp, c_src, n_timeout, n_advstd_tighter, avg_gap_std,
             avg_gap_adv) in block:
            if not first_row:
                out.append(r"\hline")
            first_row = False
            if len(grp[0]) == 9:
                _ms, bp, _lb, bt, vh, zb, np_, rt, sg = grp[0]
            else:
                _ms, bp, _lb, bt, vh, zb, np_, rt = grp[0]
                sg = "no"
            if bt != "yes":
                bt_label = "none"
            else:
                base = "zono" if zb == "yes" else "interval"
                bt_label = base + ("+lp" if np_ == "lp" else "")
            pert, psize = ps
            pert_c = pert if head else ""
            size_c = f"\\texttt{{{psize}}}" if head else ""
            ag_s_c = (f"{avg_gap_std:8.4f}"
                      if avg_gap_std is not None else "    ----")
            if avg_gap_adv is None:
                ag_a_c = "    ----"
            elif (avg_gap_std is not None and avg_gap_adv < avg_gap_std):
                ag_a_c = f"\\textbf{{{avg_gap_adv:.4f}}}"
            else:
                ag_a_c = f"{avg_gap_adv:8.4f}"
            # |avg_gap_std − avg_gap_adv|: absolute difference of the
            # two per-c_src means already shown in the previous columns.
            # Note this is NOT the same as the per-cell mean of
            # |gap_std − gap_adv| — the column-difference can hide
            # offsetting per-cell swings, but it gives a quick read on
            # how separated the two methods' bound estimates are at
            # TIME_LIMIT for this (combo, c_src).
            if avg_gap_std is not None and avg_gap_adv is not None:
                diff_c = f"{abs(avg_gap_std - avg_gap_adv):8.4f}"
            else:
                diff_c = "    ----"
            prefix = r"\rowcolor{orange!20} " if np_ != "lp" else ""
            vh_tex = vh.replace("_", r"\_")
            rt_label = str(rt).strip() + ("+sg" if sg == "yes" else "")
            out.append(
                f"{prefix}{pert_c:11s} & {size_c:20s} & {bt_label:11s} & "
                f"{vh_tex:3s} & {rt_label:4s} & {c_src:>5s} & "
                f"{n_timeout:9d} & {n_advstd_tighter:16d} & "
                f"{ag_s_c} & {ag_a_c} & {diff_c} \\\\")
            head = False
    out.append(r"\hline")
    out.append(r"\end{tabular}%")
    out.append(r"\end{adjustbox}")
    out.append(
        f"\\caption{{Architecture \\textbf{{{arch}}} --- gap comparison "
        f"on cells where both modes hit \\texttt{{TIME\\_LIMIT}}. "
        f"One row per (combo, $c_\\mathrm{{src}}$), aggregated over all "
        f"tested $c_\\mathrm{{tgt}}$. \\textsf{{n\\_advstd\\_tighter}}: "
        f"count where advstd's $u-l$ gap $<$ standard's. "
        f"\\textsf{{avg\\_gap\\_*}}: mean $u-l$ per method. "
        f"\\textsf{{avg\\_gap\\_adv}}$<$\\textsf{{avg\\_gap\\_std}} in "
        f"\\textbf{{bold}}. Includes every combo with at least one such "
        f"cell (may include combos absent from "
        f"Table~\\ref{{tab:safe_{arch.replace('=', '_')}}}).}}")
    label = arch.replace("=", "_")
    out.append(f"\\label{{tab:timeout_gap_{label}}}")
    out.append(r"\end{table}")
    out.append("")
    return "\n".join(out)


def _breakable_commas(s):
    # Insert a zero-width break opportunity after each comma so long
    # \texttt{<...>} items can wrap inside narrow p{} columns without
    # triggering adjustbox to scale the whole table down to fit.
    return s.replace(",", r",\discretionary{}{}{}")


def render_overall_combo_summary(per_arch_combos, seed, tau,
                                 combination_filter=None,
                                 per_arch_timeouts=None):
    """One-row-per-combo table averaging sp across arch, perturbation,
    c_src, c_tgt — i.e., over every cell contributing to the per-arch
    tables. Combos are identified by the tech-flag tuple (bound_tight,
    bp, varHint, zono, n1Probe, tau) and sorted by descending mean sp.
    A trailing column lists the (arch, perturbation, size, c_src)
    sub-groups whose per-subgroup mean sp falls below 1 (i.e., where
    advstd was on average slower than standard)."""
    # Per-(arch, combo-key, pert, psize, c_src) → (n_t, n_at, ags, aga)
    # lookup built from the orange timeout-gap aggregations. Used by the
    # winner-column logic to upgrade a subgroup with sp ≤ 1 to a "tied"
    # win when both modes' average wall-clock sat at the subgroup's
    # observed TIME_LIMIT cap and advstd's u−l gap is no worse than
    # standard's (or within GAP_TIE_TOL).
    GAP_TIE_TOL = 0.05
    TIMEOUT_AT_CAP_TOL_SEC = 60.0
    timeout_lookup = {}
    if per_arch_timeouts:
        for arch, timeout_all in per_arch_timeouts:
            for grp, cs, n_t, n_at, ags, aga in timeout_all:
                tkey = grp[0]
                tpert = grp[1]
                tpsize = grp[2]
                timeout_lookup[(arch, tkey, tpert, tpsize, cs)] = (
                    n_t, n_at, ags, aga)

    # Per-(arch, combo-key, pert, psize, c_src) → (mean_t_std, mean_t_adv,
    # T_observed). T_observed is the largest wall-clock seen in any
    # TIME_LIMIT cell of the subgroup; mean_t_* is the mean wall-clock
    # over all (mismatch-free) cells in the subgroup. The tied-win rule
    # fires only when both means are within TIMEOUT_AT_CAP_TOL_SEC of
    # T_observed --- i.e., on \emph{average} both modes sat at the cap,
    # so any sp ≤ 1 verdict is attributable to the timeout, not to
    # advstd actually losing on cells that did finish.
    mean_t_lookup = {}
    for arch, combos in per_arch_combos:
        for e in combos:
            ekey = e[0][0]
            epert = e[0][1]
            epsize = e[0][2]
            per_cs = defaultdict(list)
            for cell in e[14]:
                if cell[10]:
                    continue
                per_cs[cell[0]].append(cell)
            for cs, cells in per_cs.items():
                ts_vals = [c[2] for c in cells]
                ta_vals = [c[3] for c in cells]
                t_at_cap = [max(c[2], c[3]) for c in cells if c[6]]
                T_obs = max(t_at_cap) if t_at_cap else None
                mts = sum(ts_vals) / len(ts_vals) if ts_vals else None
                mta = sum(ta_vals) / len(ta_vals) if ta_vals else None
                mean_t_lookup[(arch, ekey, epert, epsize, cs)] = (
                    mts, mta, T_obs)

    def _is_timeout_tied_win(arch, key, pert, psize, cs):
        # Step 1: average wall-clock at the cap on both sides? If not,
        # any sp ≤ 1 is genuine slowdown on cells that finished, not a
        # timeout artefact --- so no rescue.
        mt_info = mean_t_lookup.get((arch, key, pert, psize, cs))
        if mt_info is None:
            return False
        mts, mta, T_obs = mt_info
        if T_obs is None or mts is None or mta is None:
            return False
        if mts < T_obs - TIMEOUT_AT_CAP_TOL_SEC:
            return False
        if mta < T_obs - TIMEOUT_AT_CAP_TOL_SEC:
            return False
        # Step 2: gap rule on the timeout cells (orange aggregation).
        info = timeout_lookup.get((arch, key, pert, psize, cs))
        if info is None:
            return False
        _n_t, _n_at, ags, aga = info
        if ags is None or aga is None:
            return False
        return aga <= ags or abs(ags - aga) < GAP_TIE_TOL

    agg = defaultdict(lambda: {"sps": [], "n_cells": 0,
                               "sub_sps": defaultdict(list),
                               "arch_universe": defaultdict(set),
                               "csrc_perts": defaultdict(set)})
    # Global compatibility map: which c_srcs ever appear paired with
    # each (pert, psize) anywhere in the dataset. A (pert, psize, cs)
    # combo that is never observed is presumed structurally
    # inapplicable (e.g. translation shift sizes that skip certain
    # c_srcs by design), so its absence should NOT trigger the
    # partial-coverage asterisk.
    pp_csrcs_global = defaultdict(set)
    # Global set of (arch, pert, psize) triples observed by ANY combo
    # anywhere in the dataset. Used to compute the "missing triples"
    # column: per-combo gaps relative to this union.
    arch_pp_global = set()
    for arch, combos in per_arch_combos:
        for e in combos:
            key = e[0][0]  # (ms, bp, lb, bt, vh, zb, np_, rt)
            # Honour --combination_table on the overall ranking too:
            # when one or more combos were requested, drop every other
            # combo from this aggregation so the row count, win-rate,
            # and loser/winner classifications all reflect only the
            # requested combo(s).
            if (combination_filter is not None
                    and _combo_label(key) not in combination_filter):
                continue
            pert = e[0][1]
            psize = e[0][2]
            for cell in e[14]:
                c_src = cell[0]
                ts, ta = cell[2], cell[3]
                # Skip mismatched-timeout cells from every aggregation
                # the overall (blue) ranking displays — `sps`, `n_cells`,
                # `n_wins`, `sub_sps` (loser/winner classification), and
                # the coverage sets. Their per-cell `sp` is meaningless
                # under different `--timout` caps, so including them
                # would skew win_rate / mean_sp and could mislead the
                # ranking. The next genuine attempt for that pair will
                # repopulate the slot with a comparable measurement.
                if cell[10]:
                    continue
                if ta > 0 and ts > 0:
                    ratio = ts / ta
                    agg[key]["sps"].append(ratio)
                    # Store (t_std, t_adv) pairs so the per-c_src
                    # aggregate below matches the
                    # $\overline{\text{sp}}_{c\_\mathrm{src}}$ column
                    # rendered in the per-arch tables, which uses
                    # $\sum t_\mathrm{std} / \sum t_\mathrm{adv}$ (see
                    # `mean_sp_by_csrc` above) rather than the mean of
                    # per-cell ratios. Keeps the loser / winner verdict
                    # consistent with the number a reader sees in the
                    # per-arch table.
                    agg[key]["sub_sps"][
                        (arch, pert, psize, c_src)].append((ts, ta))
                    # Track what (pert, psize) each arch was exercised
                    # on (universe) and what each (arch, c_src) actually
                    # got measured on, so the winner column can flag
                    # c_srcs that didn't cover the full perturbation
                    # set for that arch.
                    agg[key]["arch_universe"][arch].add((pert, psize))
                    agg[key]["csrc_perts"][
                        (arch, c_src)].add((pert, psize))
                    pp_csrcs_global[(pert, psize)].add(c_src)
                    arch_pp_global.add((arch, pert, psize))
                agg[key]["n_cells"] += 1
    rows = []
    for key, d in agg.items():
        n = len(d["sps"])
        mean_sp = (sum(d["sps"]) / n) if n else float("-inf")
        n_wins = sum(1 for r in d["sps"] if r > 1.0)
        # Group per-c_src means by (arch, pert, psize); an (arch,
        # pert, psize) triple qualifies as a loser iff *every* tested
        # c_src within it averaged sp < 1 \emph{and} that c_src isn't
        # rescued by the timeout-tied predicate (both averaged at the
        # cap with the gap rule passing on the timeout cells). The
        # rescue applies the same rule the winner column uses, so a
        # subgroup that the winner column credits as "tied at
        # TIME\_LIMIT" must not be simultaneously flagged as a loss.
        # Then collapse qualifying triples sharing (pert, psize) into a
        # single entry listing the set of archs, so the column doesn't
        # repeat a (pert, psize) on consecutive rows.
        by_triple = defaultdict(list)
        for (a, p, ps, cs), sub_sps in d["sub_sps"].items():
            if sub_sps:
                sum_ts = sum(ts for ts, _ in sub_sps)
                sum_ta = sum(ta for _, ta in sub_sps)
                if sum_ta > 0:
                    by_triple[(a, p, ps)].append(
                        (cs, sum_ts / sum_ta))
        by_pert_size = defaultdict(list)  # (p, ps) -> [(arch, worst_sp)]
        for (a, p, ps), cs_means in by_triple.items():
            if not cs_means:
                continue
            qualifies = all(
                sm < 1.0
                and not _is_timeout_tied_win(a, key, p, ps, cs)
                for cs, sm in cs_means)
            if qualifies:
                worst_sp = min(sm for _, sm in cs_means)
                by_pert_size[(p, ps)].append((a, worst_sp))
        losers = []
        for (p, ps), items in by_pert_size.items():
            archs = sorted({a for a, _ in items})
            worst_sp = min(sp for _, sp in items)
            losers.append(((archs, p, ps), worst_sp))
        # Worst (smallest worst-c_src mean across contributing archs)
        # first.
        losers.sort(key=lambda x: x[1])

        # Winners: (arch, c_src) pairs where *every* tested (pert,
        # psize) averaged sp > 1. Mirror of the losers logic; collapse
        # qualifying pairs that share the same c_src (and the same
        # partial-coverage status) into a single entry listing the
        # set of archs. The `partial` flag is true when that
        # (arch, c_src) wasn't tested on every perturbation that the
        # arch was otherwise exercised on, so the winner verdict
        # rests on only a subset of perturbations.
        by_pair = defaultdict(list)  # (arch, c_src) -> [(p, ps, mean_sp)]
        for (a, p, ps, cs), sub_sps in d["sub_sps"].items():
            if sub_sps:
                sum_ts = sum(ts for ts, _ in sub_sps)
                sum_ta = sum(ta for _, ta in sub_sps)
                if sum_ta > 0:
                    by_pair[(a, cs)].append(
                        (p, ps, sum_ts / sum_ta))
        # Bucket key now also tracks `tied`: True when at least one
        # contributing (pert, psize) subgroup had sp ≤ 1 but qualified
        # via the TIME_LIMIT-tied predicate (every cell timed out and
        # advstd's u−l gap was no worse than standard's, or within
        # GAP_TIE_TOL). The displayed entries get a † superscript so
        # the reader can tell the verdict relied on the relaxed rule.
        by_csrc = defaultdict(list)  # (c_src, partial, tied) -> [(arch, min_sp)]
        for (a, cs), pp_means in by_pair.items():
            if not pp_means:
                continue
            qualifies = True
            any_tied = False
            for p, ps, sm in pp_means:
                if sm > 1.0:
                    continue
                if _is_timeout_tied_win(a, key, p, ps, cs):
                    any_tied = True
                    continue
                qualifies = False
                break
            if not qualifies:
                continue
            min_sp = min(sm for _, _, sm in pp_means)
            tested = d["csrc_perts"][(a, cs)]
            # Expected universe: every (pert, psize) this combo
            # exercised on this arch --- regardless of whether that
            # c_src was ever observed globally for that (pert, psize).
            # "Never tested anywhere" is treated as a coverage gap,
            # so it triggers *.
            expected = set(d["arch_universe"][a])
            partial = tested != expected
            by_csrc[(cs, partial, any_tied)].append((a, min_sp))
        winners = []
        for (cs, partial, tied), items in by_csrc.items():
            archs = sorted({a for a, _ in items})
            min_sp = min(sp for _, sp in items)
            winners.append(((archs, cs, partial, tied), min_sp))
        # Best (highest guaranteed-minimum mean sp across contributing
        # archs) first.
        winners.sort(key=lambda x: -x[1])

        # Missing (arch, pert, psize) triples: global universe minus
        # what this combo actually exercised. Collapse same-(pert,psize)
        # across archs into a single entry so the column doesn't repeat
        # a (pert, psize) on consecutive rows.
        tested_triples = set()
        for a, pps in d["arch_universe"].items():
            for (p, ps) in pps:
                tested_triples.add((a, p, ps))
        missing_triples = arch_pp_global - tested_triples
        by_pp_missing = defaultdict(list)
        for (a, p, ps) in missing_triples:
            by_pp_missing[(p, ps)].append(a)
        missing = []
        for (p, ps), archs in by_pp_missing.items():
            missing.append((sorted(set(archs)), p, ps))
        missing.sort(key=lambda m: (m[1], m[2], m[0]))

        rows.append((key, mean_sp, d["n_cells"], losers, winners,
                     n_wins, missing))
    rows.sort(key=lambda r: (-(r[5] / r[2]) if r[2] else 0, -r[1]))
    if not rows:
        return ""
    n_rows = len(rows)
    n_chunks = max(1, (n_rows + MAX_OVERALL_ROWS_PER_CHUNK - 1)
                   // MAX_OVERALL_ROWS_PER_CHUNK)
    chunk_size = (n_rows + n_chunks - 1) // n_chunks
    all_out = []
    for chunk_i in range(n_chunks):
        start = chunk_i * chunk_size
        chunk_rows = rows[start:start + chunk_size]
        if not chunk_rows:
            break
        out = []
        out.append(r"\begin{table}[!htbp]")
        out.append(r"\centering")
        out.append(r"\scriptsize")
        out.append(r"\setlength{\tabcolsep}{5pt}")
        out.append(r"\begin{adjustbox}{max width=\textwidth,center}%")
        out.append(r"\begin{tabular}{@{}r l l l | r r r r "
                   r">{\raggedright\arraybackslash}p{3.2cm} "
                   r">{\raggedright\arraybackslash}p{6.2cm}@{}}")
        out.append(r"\hline")
        out.append(r"\# & \tech{bound\_tight} & \tech{varHint} & "
                   r"$\tau$ & $\overline{\text{sp}}$ & n\_wins & n\_cells & "
                   r"win\_rate & "
                   r"$\langle$arch, pert, p\_size$\rangle$ where standard "
                   r"beats advstd on every $c_\mathrm{src}$ & "
                   r"$\langle$arch, $c_\mathrm{src}\rangle$ where, on "
                   r"\emph{every} tested (pert, p\_size), advstd was "
                   r"either faster on average ($\overline{\text{sp}}>1$), "
                   r"or both methods averaged at the wall-clock cap "
                   r"($\overline{t_\mathrm{std}}\!\approx\!"
                   r"\overline{t_\mathrm{adv}}\!\approx\!T$, so the "
                   r"$\overline{\text{sp}}\!\le\!1$ verdict is a timeout "
                   r"artefact) AND advstd's $u\!-\!l$ gap on the timeout "
                   r"cells was no worse than standard's "
                   r"(entries needing this timeout-tied clause for at "
                   r"least one (pert, p\_size) are tagged "
                   r"\textit{(tied at TIME\_LIMIT)}) \\")
        out.append(r"\hline")
        for local_i, (key, mean_sp, n_cells, losers, winners, n_wins,
                      missing) in enumerate(chunk_rows):
            i = start + local_i + 1
            if local_i > 0:
                out.append(r"\hline")
            if len(key) == 9:
                _ms, bp, _lb, bt, vh, zb, np_, rt, sg = key
            else:
                _ms, bp, _lb, bt, vh, zb, np_, rt = key
                sg = "no"
            if bt != "yes":
                bt_label = "none"
            else:
                base = "zono" if zb == "yes" else "interval"
                bt_label = base + ("+lp" if np_ == "lp" else "")
            rt_label = str(rt).strip() + ("+sg" if sg == "yes" else "")
            prefix = r"\rowcolor{blue!12} " if np_ != "lp" else ""
            if mean_sp == mean_sp and mean_sp > 1.0:
                sp_c = f"\\textbf{{{mean_sp:.3f}}}"
            elif mean_sp == mean_sp:
                sp_c = f"{mean_sp:.3f}"
            else:
                sp_c = " --- "
            if losers:
                parts = []
                for (archs, p, ps), _sm in losers:
                    archs_tex = [a.replace("_", r"\_") for a in archs]
                    p_tex = p.replace("_", r"\_")
                    ps_tex = ps.replace("_", r"\_")
                    if len(archs_tex) == 1:
                        arch_field = archs_tex[0]
                    else:
                        arch_field = "{" + ",".join(archs_tex) + "}"
                    body = _breakable_commas(
                        f"<{arch_field},{p_tex},{ps_tex}>")
                    parts.append(f"\\texttt{{{body}}}")
                losers_c = "; ".join(parts)
            else:
                losers_c = r"\textit{none}"
            if winners:
                parts = []
                for (archs, cs, partial, tied), _sm in winners:
                    archs_tex = [a.replace("_", r"\_") for a in archs]
                    if len(archs_tex) == 1:
                        arch_field = archs_tex[0]
                    else:
                        arch_field = "{" + ",".join(archs_tex) + "}"
                    marker = ""
                    if partial:
                        marker += r"\textsuperscript{*}"
                    suffix = (r"~\textit{(tied at TIME\_LIMIT)}"
                              if tied else "")
                    body = _breakable_commas(f"<{arch_field},{cs}>")
                    parts.append(f"\\texttt{{{body}}}{marker}{suffix}")
                winners_c = "; ".join(parts)
            else:
                winners_c = r"\textit{none}"
            win_rate = (n_wins / n_cells) if n_cells else 0.0
            wr_c = (f"\\textbf{{{win_rate:.2%}}}" if win_rate > 0.5
                    else f"{win_rate:.2%}").replace("%", r"\%")
            vh_tex = vh.replace("_", r"\_")
            out.append(
                f"{prefix}{i:3d} & {bt_label:11s} & {vh_tex:3s} & "
                f"{rt_label:4s} & {sp_c} & {n_wins:5d} & {n_cells:6d} & "
                f"{wr_c} & {losers_c} & {winners_c} \\\\")
        out.append(r"\hline")
        out.append(r"\end{tabular}%")
        out.append(r"\end{adjustbox}")
        if n_chunks > 1:
            part_phrase = (
                f" (part {chunk_i + 1}/{n_chunks}, combos ranked "
                f"{start + 1}--{start + len(chunk_rows)} of {n_rows} "
                f"by \\textsf{{win\\_rate}})")
        else:
            part_phrase = ""
        if chunk_i == 0:
            out.append(
                f"\\caption{{Overall advstd combination ranking at "
                f"{_seed_tau_phrase(seed, tau)}{part_phrase} --- one row "
                f"per flag combination, aggregated over every "
                f"(arch, pert, size, $c_\\mathrm{{src}}$, $c_\\mathrm{{tgt}}$) "
                f"cell in the per-arch tables below. "
                f"$\\overline{{\\text{{sp}}}}$: mean per-cell "
                f"$\\text{{sp}}=\\textsf{{t\\_std}}/\\textsf{{t\\_adv}}$; "
                f"\\textsf{{n\\_wins}} counts cells with $\\text{{sp}}>1$; "
                f"\\textsf{{win\\_rate}}=\\textsf{{n\\_wins}}/\\textsf{{n\\_cells}} "
                f"(bold when $>\\!50\\%$). \\textsf{{losers}}: "
                f"$\\langle$arch, pert, p\\_size$\\rangle$ triples where every "
                f"tested $c_\\mathrm{{src}}$ averaged $\\text{{sp}}<1$. "
                f"\\textsf{{winners}}: $\\langle$arch, $c_\\mathrm{{src}}\\rangle$ "
                f"pairs where every tested (pert, p\\_size) averaged "
                f"$\\text{{sp}}>1$, or "
                f"$\\overline{{t_\\mathrm{{std}}}}$ and "
                f"$\\overline{{t_\\mathrm{{adv}}}}$ both sat within 60\\,s "
                f"of the observed \\texttt{{TIME\\_LIMIT}} cap "
                f"\\emph{{and}} the gap rule "
                f"($\\overline{{\\text{{gap}}}}_\\mathrm{{adv}}\\le"
                f"\\overline{{\\text{{gap}}}}_\\mathrm{{std}}$ or "
                f"$|\\overline{{\\text{{gap}}}}_\\mathrm{{std}}-"
                f"\\overline{{\\text{{gap}}}}_\\mathrm{{adv}}|<0.05$) "
                f"holds on the timeout cells "
                f"(such entries are tagged "
                f"\\textit{{(tied at TIME\\_LIMIT)}}); "
                f"\\textsuperscript{{*}} flags partial "
                f"coverage (some (pert, p\\_size) for this arch were not "
                f"exercised by this combo). "
                f"Sorted by descending \\textsf{{win\\_rate}}, then "
                f"$\\overline{{\\text{{sp}}}}$; $\\overline{{\\text{{sp}}}}>1$ "
                f"in \\textbf{{bold}}. Rows are shaded blue.}}")
            out.append(r"\label{tab:safe_overall_combo_ranking}")
        else:
            out.append(
                f"\\caption{{Overall advstd combination ranking"
                f"{part_phrase}; continuation of "
                f"Table~\\ref{{tab:safe_overall_combo_ranking}}. Column "
                f"meanings: see Table~\\ref{{tab:safe_overall_combo_ranking}}.}}")
            out.append(
                f"\\label{{tab:safe_overall_combo_ranking_{chunk_i}}}")
        out.append(r"\end{table}")
        out.append("")
        all_out.append("\n".join(out))
    return "\n".join(all_out)


def render_all(archs, rows, seed, tau, combination_filter=None):
    sections_main = []
    sections_end = []
    summary = []
    per_arch_combos = []
    per_arch_timeouts = []
    for arch in archs:
        all_combos, timeout_all = collect_arch(rows, arch, seed, tau)
        per_arch_combos.append((arch, all_combos))
        per_arch_timeouts.append((arch, timeout_all))
        agg_content, mid_content = render_table(
            arch, all_combos, timeout_all, seed, tau,
            combination_filter=combination_filter)
        sections_main.append(agg_content)
        sections_main.append(render_timeout_table(
            arch, timeout_all, seed, tau,
            combination_filter=combination_filter))
        if mid_content:
            sections_end.append(mid_content)
        n_cells = sum(len(e[14]) for e in all_combos)
        summary.append(
            f"{arch}={len(all_combos)}combos/{n_cells}cells"
            f"(timeout-combos={len(timeout_all)})")
    overall_table = render_overall_combo_summary(
        per_arch_combos, seed, tau,
        combination_filter=combination_filter,
        per_arch_timeouts=per_arch_timeouts)
    pert_hist_section = render_perturbation_combined_section(
        per_arch_combos, seed, tau,
        combination_filter=combination_filter)
    pert_hist_all_archs_section = render_perturbation_combined_section_all_archs(
        per_arch_combos, seed, tau,
        combination_filter=combination_filter)
    pert_hist_all_section = render_perturbation_combined_section_all(
        per_arch_combos, seed, tau,
        combination_filter=combination_filter)

    intro = (
        r"Tables below list \emph{every} recorded advstd combination at "
        rf"{_seed_tau_phrase(seed, tau)}, "
        r"evaluated per-architecture and broken out by perturbation type "
        r"and size. Each row corresponds to a single "
        r"$(c_\mathrm{src}, c_\mathrm{tgt})$ cell of one combo (rows for "
        r"the same combo are grouped, with the flag columns shown only on "
        r"the first cell of the combo). We report wall-clock "
        r"\textsf{t\_std} and \textsf{t\_adv} per cell, the per-cell "
        r"speedup \textsf{sp}$=\textsf{t\_std}/\textsf{t\_adv}$ (higher is "
        r"better; $>\!1$ means advstd finished faster than the "
        r"with-perturbed-intervals baseline), "
        r"$\overline{\text{sp}}_{c\_\mathrm{src}}$ (mean \textsf{sp} over "
        r"all tested $c_\mathrm{tgt}$ for the same $c_\mathrm{src}$ within "
        r"a combo, shown once per $c_\mathrm{src}$ group), and \textsf{n\_relax} "
        r"(per-cell number of N2 binary variables removed by "
        r"Technique~3). The \tech{bound\_tight} column labels which "
        r"Technique~1 variant is used (\texttt{interval} / \texttt{zono} / "
        r"\texttt{none}; \tech{n1Probe} is always off), and the $\tau$ column is the "
        r"Technique~3 threshold. Within each (perturbation, size) block "
        r"combos are sorted by descending mean \textsf{sp} (fastest advstd first); cells within a "
        r"combo are sorted by $(c_\mathrm{src}, c_\mathrm{tgt})$. "
        r"\tech{mipStart} and \tech{lpBasis} are always off and omitted "
        r"for brevity. Cells where both modes hit Gurobi's "
        r"\texttt{TIME\_LIMIT} are shown with their wall-clock times; for "
        r"those cells, a separate per-arch gap-comparison table reports "
        r"how often advstd reached a tighter $u - l$ bound on the "
        r"objective than standard, alongside the mean gap of each method.")
    if combination_filter:
        combo_list_tex = _format_combination_filter_tex(combination_filter)
        n_combos_filter = len(combination_filter)
        if n_combos_filter == 1:
            restriction_phrase = (
                r"is restricted to the single combination "
                + combo_list_tex)
        else:
            restriction_phrase = (
                rf"is restricted to the {n_combos_filter} combinations "
                + combo_list_tex)
        intro += (
            "\n\n"
            r"\noindent\textbf{Note:} every table below --- the overall "
            r"combination ranking, the per-arch perturbation tables, "
            r"and the per-arch \texttt{TIME\_LIMIT} gap-comparison "
            r"tables --- " + restriction_phrase + " "
            r"(via \texttt{--combination\_table}).")
    intro += (
        "\n\n"
        + f"% auto-generated: archs=[{', '.join(summary)}], seed={seed}, "
          f"tau={tau}"
        + (f", combo_filter="
           f"{_format_combination_filter(combination_filter)}"
           if combination_filter else "")
        + "\n")
    mid_section = ""
    if sections_end:
        mid_section = (
            "\n\\clearpage\n"
            r"\section*{Mid-$\tau$ detail tables ($\tau\in\{0.01,\,0.05,"
            r"\,0.1\}$)}"
            "\n"
            r"The following tables break out the same per-architecture "
            r"advstd combinations as the green tables above, but for the "
            r"intermediate Technique-3 thresholds $\tau\in\{0.01,\,0.05,"
            r"\,0.1\}$. They are grouped at the end of the paper because "
            r"these mid-range thresholds are denser and more exploratory "
            r"than the $\tau\in\{0.0,\,0.5,\,1.0\}$ headline tables. "
            r"Rows are tinted by $c_\mathrm{src}$ (same palette as the "
            r"headline tables above); "
            "column meanings match the tables above.\n\n"
            + "\n".join(sections_end))
    return (intro + overall_table + pert_hist_section
            + pert_hist_all_archs_section
            + pert_hist_all_section
            + "\n".join(sections_main) + mid_section)


def update_tex(tex_path, new_body):
    with open(tex_path) as f:
        text = f.read()
    if BEGIN_MARK not in text or END_MARK not in text:
        raise SystemExit(
            f"markers not found in {tex_path}; expected lines containing "
            f"'{BEGIN_MARK}' and '{END_MARK}'")
    pre, rest = text.split(BEGIN_MARK, 1)
    _body, post = rest.split(END_MARK, 1)
    updated = f"{pre}{BEGIN_MARK}\n{new_body}\n{END_MARK}{post}"
    if updated == text:
        print(f"[update_advstd_tex_tables] no changes to {tex_path}")
        return
    with open(tex_path, "w") as f:
        f.write(updated)
    print(f"[update_advstd_tex_tables] wrote {tex_path}")


# ── nn1 Safe-to-Use Combinations (Boosting Standard Mode, sec:safe_nn1) ──
NN1_BEGIN_MARK = "% BEGIN AUTO: nn1_safe_tables"
NN1_END_MARK   = "% END AUTO: nn1_safe_tables"


def _classify_stdboost_filename(name):
    """Map a standard-mode boost result filename to a combo tuple.

    Returns a dict with the flag fields the nn1 tables key on:
      zono_bounds: yes / no
      relax_threshold: stringified τ or 'off'
      sibling_gate: yes / no
      perturbed_intervals: yes / no
      seed: gurobi seed (default '0')
    A None return means the filename does not encode an nn1-boosting run.
    The presence of '_stdBoost_' is the discriminator — files written by
    main_standard() in run.jl carry that prefix when any of the three
    nn1-boost flags is active.
    """
    if "_stdBoost_" not in name:
        return None
    seed_match = re.search(r"_seed(\d+)(?=_)(?!_itr)", name)
    btpr_match = re.search(r"_BTPR([-0-9.]+)", name)
    return {
        "zono_bounds":         "yes" if "_stdBoost_zono" in name or "_zono_" in name else "no",
        "relax_threshold":     btpr_match.group(1) if btpr_match else "off",
        "sibling_gate":        "yes" if "_SibGate" in name else "no",
        "perturbed_intervals": "yes" if "_PertruebedIntervals" in name else "no",
        "seed":                seed_match.group(1) if seed_match else "0",
    }


def _role_of_stdboost_dir(cd):
    """Map a cell directory to the model role (N1 or N2) it stores results
    for. Directories use these conventions:
      * vagharWithPerturbed_{arch}_{tag}                   → N1 baseline
      * vagharWithPerturbed_{arch}_{tag}_sgd_itr*          → N2 baseline
      * N1stdBoost_{arch}_{tag}                            → N1 boost
      * N2stdBoost_{arch}_{tag}_sgd_itr*                   → N2 boost
    Falls back to N1 for legacy 'vaghar_*' dirs.
    """
    base = os.path.basename(cd.rstrip(os.sep))
    if base.startswith("N2stdBoost_") or "_sgd_itr" in base:
        return "N2"
    return "N1"


def _is_baseline_stdname(name):
    """Match the with-perturbed-intervals baseline file used as the
    nn1-boost comparison reference. The baseline carries _PertruebedIntervals
    (sic — the run.jl tag is misspelled and we mirror it) and NO _stdBoost_
    tag. _HyperAttackHints and _VagharDeps may decorate either."""
    return "_PertruebedIntervals" in name and "_stdBoost_" not in name


def _collect_stdboost_cells(arch_runs, cwd, dataset, parse_result_file,
                            seeds_filter=None):
    """Walk each arch's results dir, pair each _stdBoost_* cell with the
    matching baseline cell, and return per-cell rows ready for tabulation.

    Mirrors the parse_result_file / glob loop in
    find_advstd_faster_than_standard but for the nn1-boost vs
    with-perturbed baseline comparison rather than advstd vs standard.
    """
    import glob
    rows = []
    seeds_filter = set(str(s) for s in seeds_filter) if seeds_filter else None

    pert_subdirs = ["patch", "occ", "translation", "rotation",
                    "brightness", "linf", "contrast"]
    for arch, _ in arch_runs:
        exp_base = os.path.join(cwd, "paper_experiments", dataset,
                                f"{arch}_exp")
        if not os.path.isdir(exp_base):
            continue
        for pert_dir in pert_subdirs:
            pert_path = os.path.join(exp_base, pert_dir)
            if not os.path.isdir(pert_path):
                continue
            for eps_dir in sorted(glob.glob(os.path.join(pert_path,
                                                        "eps_*"))):
                # Standard-mode result directories are keyed by the
                # 'vaghar*' prefix run_experiment.py uses. Both the
                # baseline run and the boost runs live under that tree.
                # "vagharWithPerturbed_*" matches both the N1 baseline
                # (no _sgd_itr suffix) and the N2 baseline (_sgd_itr*).
                # "vaghar_*" is kept for legacy dirs that begin with
                # 'vaghar_' but are not 'vagharWithPerturbed*'. dedupe via
                # set() since the two patterns can both match the same dir.
                cell_dirs = sorted(set(
                    glob.glob(os.path.join(eps_dir,
                                           "vagharWithPerturbed_*"))
                    + glob.glob(os.path.join(eps_dir,
                                             "vaghar_*"))
                    + glob.glob(os.path.join(eps_dir,
                                             "N1stdBoost_*"))
                    + glob.glob(os.path.join(eps_dir,
                                             "N2stdBoost_*"))
                ))
                # baseline_by_cell is (role, cs, ct) -> (info, fname) so the
                # N1 baseline (vagharWithPerturbed_{arch}_{tag}) does not
                # collide with the N2 baseline (..._sgd_itr*).
                baseline_by_cell = {}
                # combo_key carries role and perturbed_intervals so N1/N2
                # boost rows and pi=true / pi=false boost rows stay distinct.
                boost_by_cell = {}
                for cd in cell_dirs:
                    role = _role_of_stdboost_dir(cd)
                    for tf in glob.glob(os.path.join(cd, "*.txt")):
                        fname = os.path.basename(tf)
                        if _is_baseline_stdname(fname):
                            parsed = parse_result_file(tf)
                            for key, val in parsed.items():
                                cs, ct = key
                                baseline_by_cell.setdefault(
                                    (role, cs, ct), (val, fname))
                            continue
                        combo = _classify_stdboost_filename(fname)
                        if combo is None:
                            continue
                        if seeds_filter and combo["seed"] not in seeds_filter:
                            continue
                        combo_key = (role,
                                     combo["zono_bounds"],
                                     combo["relax_threshold"],
                                     combo["sibling_gate"],
                                     combo["perturbed_intervals"])
                        parsed = parse_result_file(tf)
                        for key, val in parsed.items():
                            cs, ct = key
                            boost_by_cell.setdefault(
                                combo_key, {})[(cs, ct)] = (val, fname, combo)

                for combo_key, cells in boost_by_cell.items():
                    role = combo_key[0]
                    for (cs, ct), (b_info, b_name, combo) in cells.items():
                        base = baseline_by_cell.get((role, cs, ct))
                        if base is None:
                            continue
                        s_info, s_name = base
                        t_boost = b_info.get("total_time", 0.0) or 0.0
                        t_base  = s_info.get("total_time", 0.0) or 0.0
                        if t_boost <= 0 or t_base <= 0:
                            continue
                        rows.append({
                            "arch": arch,
                            "role": role,
                            "perturbation": pert_dir,
                            "perturbation_size":
                                os.path.basename(eps_dir).replace(
                                    "eps_", ""),
                            "c_source": cs, "c_target": ct,
                            "zono_bounds": combo["zono_bounds"],
                            "relax_threshold": combo["relax_threshold"],
                            "sibling_gate": combo["sibling_gate"],
                            "perturbed_intervals":
                                combo["perturbed_intervals"],
                            "seed": combo["seed"],
                            "t_baseline": t_base,
                            "t_boost": t_boost,
                            "sp": t_base / t_boost,
                            "gap_baseline": (s_info.get("upper_bound", 0.0)
                                             - s_info.get("lower_bound", 0.0)),
                            "gap_boost": (b_info.get("upper_bound", 0.0)
                                          - b_info.get("lower_bound", 0.0)),
                            "boost_file": b_name,
                            "baseline_file": s_name,
                        })
    return rows


def _render_nn1_body(rows, archs):
    """Render the LaTeX body that fills the % BEGIN AUTO: nn1_safe_tables
    block. One overall ranking table + one per-arch summary table.

    The schema is intentionally simpler than the advstd safe-combos block
    (Section sec:safe). Standard-mode boosts only have three combinatoric
    flags (zono, BTPR, SibGate), so an exhaustive cross-product is small
    enough that a single ranking table covers it.
    """
    if not rows:
        return ("\\noindent\\textit{No \\texttt{\\_stdBoost\\_*} result "
                "files were found under \\texttt{paper\\_experiments/<arch>"
                "\\_exp/}. Re-run the sweep with the standard-mode boost "
                "flags enabled to populate this section.}")

    # Aggregate by combo: mean sp, win count, n_cells.
    # Combo key spans (role, zono, τ, sg, pi) — role and pi are kept
    # separate so N1 vs N2 boost stats don't collapse, and the same combo
    # with perturbed_intervals on vs off doesn't either.
    from collections import defaultdict
    by_combo = defaultdict(list)
    for r in rows:
        key = (r.get("role", "N1"),
               r["zono_bounds"], r["relax_threshold"], r["sibling_gate"],
               r.get("perturbed_intervals", "yes"))
        by_combo[key].append(r)

    def _combo_label(key):
        role, z, t, sg, pi = key
        parts = [role]
        parts.append("zono" if z == "yes" else "no-zono")
        parts.append(f"$\\tau$={t}" if t != "off" else "no-BTPR")
        if sg == "yes":
            parts.append("SibGate")
        if pi == "no":
            parts.append("no-PI")
        return ", ".join(parts)

    overall = []
    for key, cells in by_combo.items():
        sps = [c["sp"] for c in cells if c["sp"] > 0]
        if not sps:
            continue
        mean_sp = sum(sps) / len(sps)
        n_wins  = sum(1 for s in sps if s > 1.0)
        overall.append({
            "role": key[0],
            "label": _combo_label(key),
            "mean_sp": mean_sp,
            "n_wins": n_wins,
            "n_cells": len(sps),
            "win_rate": n_wins / len(sps),
        })
    overall.sort(key=lambda r: (r["role"], -r["win_rate"], -r["mean_sp"]))

    lines = []
    lines.append("Tables below summarise every recorded standard-mode "
                 "boost combination across all sweep seeds, broken out "
                 "by architecture. \\tech{sp}$=\\textsf{t\\_base}/"
                 "\\textsf{t\\_boost}$ is the per-cell speedup over the "
                 "with-perturbed-intervals baseline (higher is better; "
                 "$>\\!1$ means the boost was faster). The boost ``combo'' "
                 "is the triple "
                 "(\\tech{nn1\\_zono\\_bounds}, $\\tau$=\\tech{nn1\\_relax"
                 "\\_threshold}, \\tech{nn1\\_sibling\\_gate}). Rows are "
                 "sorted by descending win-rate, then mean \\textsf{sp}.")
    lines.append("")
    lines.append(f"% auto-generated: archs={archs}, "
                 f"total_cells={sum(c['n_cells'] for c in overall)}")
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}r l r r r r@{}}")
    lines.append("\\hline")
    lines.append("\\# & combo & $\\overline{\\text{sp}}$ & "
                 "n\\_wins & n\\_cells & win\\_rate \\\\")
    lines.append("\\hline")
    for idx, c in enumerate(overall, 1):
        sp_str = f"{c['mean_sp']:.3f}"
        if c["mean_sp"] > 1.0:
            sp_str = "\\textbf{" + sp_str + "}"
        wr_str = f"{100*c['win_rate']:.1f}\\%"
        if c["win_rate"] > 0.5:
            wr_str = "\\textbf{" + wr_str + "}"
        # Escape underscores for LaTeX
        label = c["label"].replace("_", "\\_")
        lines.append(f"{idx} & {label} & {sp_str} & "
                     f"{c['n_wins']} & {c['n_cells']} & {wr_str} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Overall nn1-boost combination ranking, "
                 "aggregated over every (arch, pert, size, $c_\\mathrm{src}$, "
                 "$c_\\mathrm{tgt}$) cell. $\\overline{\\text{sp}}$: mean "
                 "per-cell $\\text{sp}=\\textsf{t\\_base}/\\textsf{t\\_boost}$ "
                 "vs the with-perturbed-intervals baseline. \\textsf{n\\_wins}: "
                 "cells with $\\text{sp}>1$. Sorted by descending "
                 "\\textsf{win\\_rate}, then $\\overline{\\text{sp}}$.}")
    lines.append("\\label{tab:safe_nn1_overall_combo_ranking}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def update_nn1_tex(tex_path, body):
    """Rewrite the % BEGIN AUTO: nn1_safe_tables block in tex_path.
    Mirrors update_tex() but for the nn1 markers."""
    with open(tex_path) as f:
        text = f.read()
    if NN1_BEGIN_MARK not in text or NN1_END_MARK not in text:
        raise SystemExit(
            f"nn1 markers not found in {tex_path}; expected lines "
            f"containing '{NN1_BEGIN_MARK}' and '{NN1_END_MARK}'")
    pre, rest = text.split(NN1_BEGIN_MARK, 1)
    _body, post = rest.split(NN1_END_MARK, 1)
    updated = f"{pre}{NN1_BEGIN_MARK}\n{body}\n{NN1_END_MARK}{post}"
    if updated == text:
        print("[update_advstd_tex_tables] no changes to nn1 block")
        return
    with open(tex_path, "w") as f:
        f.write(updated)
    print(f"[update_advstd_tex_tables] wrote nn1 block in {tex_path}")


def regenerate_nn1_section(tex_path, cwd, dataset, arch_runs,
                            parse_result_file, seeds_filter=None):
    """End-to-end: scan _stdBoost_* results, render the nn1-safe body,
    rewrite the AUTO block in tex_path. Called from run_relaxation_sweep.py
    immediately after the advstd safe-tables block is regenerated."""
    try:
        rows = _collect_stdboost_cells(arch_runs, cwd, dataset,
                                       parse_result_file,
                                       seeds_filter=seeds_filter)
        archs = [a for a, _ in arch_runs]
        body = _render_nn1_body(rows, archs)
        update_nn1_tex(tex_path, body)
    except SystemExit as exc:
        print(f"[update_advstd_tex_tables] nn1 block skipped: {exc}")
    except Exception as exc:
        print(f"[update_advstd_tex_tables] nn1 block error: {exc}")


# `re` is used by _classify_stdboost_filename above; import it locally so
# the existing module's top-of-file imports stay minimal.
import re


# ===== Per-architecture wide comparison section =====
# Mirrors the structure of the nn1_safe_tables block but reports the raw
# per-cell wall-clock time of every standard-mode configuration produced by
# the boost sweep, side-by-side. Each table covers one architecture; rows
# are one per (arch, role, pert, p_size, c_src) with the displayed time
# being the mean over c_tgt and Gurobi seed.

WIDE_BEGIN_MARK = "% BEGIN AUTO: wide_perarch_tables"
WIDE_END_MARK   = "% END AUTO: wide_perarch_tables"


def _wide_combo_of_dir(cd):
    """Classify a cell directory by which timing column it contributes to.
    Returns 'vaghar' (no-PI baseline), 'PI' (with-PI baseline), 'boost'
    (a stdBoost variant whose flags are read from the filename), or None.
    """
    base = os.path.basename(cd.rstrip(os.sep))
    if base.startswith("vagharNoPerturbed_"):
        return "vaghar"
    if base.startswith("vagharWithPerturbed_"):
        return "PI"
    if base.startswith("N1stdBoost_") or base.startswith("N2stdBoost_"):
        return "boost"
    return None


def _wide_boost_key(combo):
    """4-tuple (z, SG, PI, tau) key for a stdBoost cell. z/SG/PI are '0'/'1'
    bits; tau is a string like '0' or '0.5' ('off' if the threshold tag was
    absent). Each (z, SG, PI, tau) tuple becomes its own table column so that
    a same-MIP-but-different-Gurobi-seed boost cell can be compared
    side-by-side against its baseline."""
    z  = "1" if combo["zono_bounds"]         == "yes" else "0"
    sg = "1" if combo["sibling_gate"]        == "yes" else "0"
    pi = "1" if combo["perturbed_intervals"] == "yes" else "0"
    tau = combo["relax_threshold"]
    return (z, sg, pi, tau)


# Build the ordered column list. Layout: PI=off block first (baseline +
# 4 boost variants at tau=0; the sweep filter at run_relaxation_sweep.py:
# 2854 drops PI=off + tau>0, so those 4 cells don't exist), then PI=on
# block (baseline + 4 boost variants x 2 tau values = 8). Cells where
# the relaxation does not actually fire (tau=0 across all combos, plus
# any (z=F, SG=F, PI=*, tau=*) cell where no binary is dropped) are
# left in so the reader can see the seed-induced wall-clock variance.
_WIDE_TAU_VALUES = ("0.0", "0.5")
_WIDE_ZSG_VARIANTS = (("0", "0"), ("1", "0"), ("0", "1"), ("1", "1"))


# Two advstd N2 combos surfaced as extra rightmost columns. Each entry is
# (column_key, zono_bounds, var_hint, relax_threshold, sibling_gate); we
# match against the per-cell CSVs produced by --find_advstd_faster_than
# _standard. role is always N2 because advstd is a transfer-mode (N1->N2)
# technique.
_ADVSTD_WIDE_COMBOS = (
    ("adv_zono_prevpgd_0.0+sg", "yes", "prev_pgd", "0.0", "yes"),
    ("adv_zono_prevpgd_0.5+sg", "yes", "prev_pgd", "0.5", "yes"),
)


def _wide_column_order():
    """Yield the columns in the order they appear in the table.
    A column is either a baseline tag ('vaghar' or 'PI') or a tuple
    (z, sg, pi, tau). The 'all-off' boost cells (z=0, SG=0, PI=*, tau=0)
    are pooled into the corresponding baseline column at collection time
    (see _collect_wide_perarch_cells), so they don't appear as separate
    columns here."""
    yield "vaghar"
    for z, sg in _WIDE_ZSG_VARIANTS:
        if (z, sg) == ("0", "0"):
            continue  # merged into 'vaghar'
        yield (z, sg, "0", "0.0")  # PI=off, tau=0
    yield "PI"
    for z, sg in _WIDE_ZSG_VARIANTS:
        for tau in _WIDE_TAU_VALUES:
            if (z, sg, tau) == ("0", "0", "0.0"):
                continue  # merged into 'PI'
            yield (z, sg, "1", tau)
    # advstd N2 columns (transfer-mode, zono + prev_pgd + SibGate)
    for label, *_ in _ADVSTD_WIDE_COMBOS:
        yield label


def _wide_column_header(col):
    """LaTeX label for a column key, formatted as a `+`-joined feature
    list (e.g. \"zono+pi+tau=0.5\") rather than a subscripted/super-
    scripted $t^{\\tau=..}_..$ glyph."""
    if col == "vaghar":
        return r"\textsf{vaghar}"
    if col == "PI":
        return r"\textsf{pi}"
    if isinstance(col, str) and col.startswith("adv_"):
        for label, _zb, _vh, rt, _sg in _ADVSTD_WIDE_COMBOS:
            if label == col:
                return (r"\textsf{adv+zono+prev\_pgd+sg+tau="
                        + rt + r"}")
        return col
    z, sg, pi, tau = col
    parts = []
    if z  == "1": parts.append("zono")
    if sg == "1": parts.append("sg")
    if pi == "1": parts.append("pi")
    parts.append(f"tau={tau}")
    return r"\textsf{" + "+".join(parts) + r"}"


def _wide_column_short_label(col):
    """Compact label for a column key, suitable for inline use inside the
    \"missing c\\_targets\" cell where space is tight. Uses single-letter
    flag mnemonics (z/sg/pi) and a bare tau= value; does not wrap in
    \\textsf.
    """
    if col == "vaghar":
        return "vaghar"
    if col == "PI":
        return "pi"
    if isinstance(col, str) and col.startswith("adv_"):
        for label, _zb, _vh, rt, _sg in _ADVSTD_WIDE_COMBOS:
            if label == col:
                return f"adv{rt}+sg"
        return col
    z, sg, pi, tau = col
    parts = []
    if z  == "1": parts.append("z")
    if sg == "1": parts.append("sg")
    if pi == "1": parts.append("pi")
    parts.append(f"t={tau}")
    return "+".join(parts)


def _collect_wide_perarch_cells(arch_runs, cwd, dataset, parse_result_file,
                                 seeds_filter=None):
    """Walk each arch's results dir and yield per-cell timing rows for the
    per-arch wide comparison section. One row per (.txt file, cs, ct) cell.
    The combo field is 'vaghar', 'PI', or one of the 8 boost masks.
    """
    import glob
    rows = []
    seeds_filter = set(str(s) for s in seeds_filter) if seeds_filter else None

    pert_subdirs = ["patch", "occ", "translation", "rotation",
                    "brightness", "linf", "contrast"]
    for arch, _ in arch_runs:
        exp_base = os.path.join(cwd, "paper_experiments", dataset,
                                f"{arch}_exp")
        if not os.path.isdir(exp_base):
            continue
        for pert_dir in pert_subdirs:
            pert_path = os.path.join(exp_base, pert_dir)
            if not os.path.isdir(pert_path):
                continue
            for eps_dir in sorted(glob.glob(os.path.join(pert_path,
                                                         "eps_*"))):
                p_size = os.path.basename(eps_dir).replace("eps_", "")
                cell_dirs = sorted(set(
                    glob.glob(os.path.join(eps_dir, "vagharNoPerturbed_*"))
                    + glob.glob(os.path.join(eps_dir, "vagharWithPerturbed_*"))
                    + glob.glob(os.path.join(eps_dir, "N1stdBoost_*"))
                    + glob.glob(os.path.join(eps_dir, "N2stdBoost_*"))
                ))
                for cd in cell_dirs:
                    src_kind = _wide_combo_of_dir(cd)
                    if src_kind is None:
                        continue
                    role = _role_of_stdboost_dir(cd)
                    for tf in glob.glob(os.path.join(cd, "*.txt")):
                        fname = os.path.basename(tf)
                        if src_kind == "boost":
                            combo = _classify_stdboost_filename(fname)
                            if combo is None:
                                continue
                            if seeds_filter and combo["seed"] not in seeds_filter:
                                continue
                            combo_label = _wide_boost_key(combo)
                            seed_val = combo["seed"]
                            # An 'all-off' boost cell (z=F, SG=F, tau=0)
                            # encodes a MIP that is identical to the
                            # corresponding baseline (vaghar if PI=F,
                            # PI if PI=T); the only wall-clock
                            # difference is Gurobi's seed. Pool it into
                            # the baseline bucket so the user can read
                            # the baseline column directly without
                            # remembering which boost cell is also
                            # the baseline.
                            z, sg, pi, tau = combo_label
                            if z == "0" and sg == "0" and tau in ("0", "0.0"):
                                combo_label = "PI" if pi == "1" else "vaghar"
                        else:
                            # Baseline files (vagharNoPerturbed_*,
                            # vagharWithPerturbed_*) generally do not carry
                            # an explicit Gurobi-seed tag in the filename,
                            # so the seeds_filter from --combo_ranking_seeds
                            # would otherwise filter them out. Treat the
                            # baselines as always passing seed_filter; their
                            # purpose is as a same-MIP reference for each
                            # boost row regardless of seed.
                            if "_stdBoost_" in fname:
                                continue
                            combo_label = src_kind
                            sm = re.search(r"_seed(\d+)(?=_)(?!_itr)", fname)
                            seed_val = sm.group(1) if sm else "0"
                        parsed = parse_result_file(tf)
                        for key, val in parsed.items():
                            cs, ct = key
                            t = val.get("total_time", 0.0) or 0.0
                            if t <= 0:
                                continue
                            # Skip externally-killed runs (e.g., Julia
                            # process killed by the sweep supervisor or
                            # OOM-killer). Their wall-clock is below the
                            # Gurobi time cap but the bounds are
                            # partial, so including them would pull the
                            # mean `t` under 60 min while leaving the
                            # mean lb/ub gap wide. The CSV builder in
                            # run_relaxation_sweep.py applies the same
                            # filter at lines 1000 and 1274.
                            if (val.get("solve_status", "") or "").upper() == "INTERRUPTED":
                                continue
                            rows.append({
                                "arch": arch,
                                "role": role,
                                "perturbation": pert_dir,
                                "perturbation_size": p_size,
                                "c_source": cs, "c_target": ct,
                                "seed": seed_val,
                                "combo": combo_label,
                                "t_total": t,
                                "lb_total": val.get("lower_bound"),
                                "ub_total": val.get("upper_bound"),
                                "solve_status": val.get("solve_status", ""),
                                # geometric_intervals twin (same cell, same
                                # delta, faster/slower solve): paired as base,geom
                                # in the time column by the renderer.
                                "geom": ("_geomInt" in fname),
                            })
    return rows


def _load_delta_max_values(cwd, dataset, archs):
    """Scan paper_experiments/{dataset}/{arch}_exp/delta_max/ for the
    pre-phase delta_max result files and return
        {(arch, role, c_src_0indexed): {'lower': lb, 'upper': ub,
                                        'status': str}}.

    delta_max files are written by run_relaxation_sweep.py's Phase 0.5
    via `julia run.jl --perturbation max --ctag <c_src>` and contain
    standard c_source=,c_target=,lower_bound=,upper_bound=,... lines.
    For the "max" perturbation the c_target is a dummy unused inside
    the MIP, so we key only by c_source. When multiple files cover the
    same (arch, role, c_src) we keep the one with the tightest gap.
    """
    import glob
    out = {}
    for arch in archs:
        base = os.path.join(cwd, "paper_experiments", dataset,
                            f"{arch}_exp", "delta_max")
        if not os.path.isdir(base):
            continue
        for role in ("N1", "N2"):
            for d in glob.glob(os.path.join(base,
                                            f"delta_max_{arch}_{role}_*")):
                for fpath in glob.glob(os.path.join(d, "*.txt")):
                    fname = os.path.basename(fpath)
                    if fname == "_filename_legend.txt":
                        continue
                    try:
                        f = open(fpath)
                    except OSError:
                        continue
                    with f:
                        for line in f:
                            line = line.strip()
                            if not line or "c_source=" not in line:
                                continue
                            fields = {}
                            for pair in line.split(","):
                                if "=" in pair:
                                    k, v = pair.split("=", 1)
                                    fields[k] = v
                            try:
                                cs = int(fields["c_source"])
                                lb = float(fields.get("lower_bound", "nan"))
                                ub = float(fields.get("upper_bound", "nan"))
                            except (KeyError, ValueError):
                                continue
                            status = fields.get("solve_status", "")
                            key = (arch, role, cs)
                            entry = {"lower": lb, "upper": ub,
                                     "status": status}
                            prev = out.get(key)
                            # Keep the file whose [lb, ub] interval is
                            # tightest. nan-safe via math.isfinite.
                            def _gap(e):
                                if not (math.isfinite(e["lower"])
                                        and math.isfinite(e["upper"])):
                                    return float("inf")
                                return e["upper"] - e["lower"]
                            if prev is None or _gap(entry) < _gap(prev):
                                out[key] = entry
    return out


def _format_delta_max_for_cell(entry):
    """Render a delta_max entry as a short LaTeX fragment for the model cell."""
    if entry is None:
        return r"$\delta_{\max}$=?"
    lb = entry.get("lower")
    ub = entry.get("upper")
    if not (lb is not None and math.isfinite(lb)
            and ub is not None and math.isfinite(ub)):
        return r"$\delta_{\max}$=?"
    # Reported value is the sound upper bound on the maximum.
    # Show as "<=" prefix when the optimality gap is non-trivial,
    # else just the bare value.
    gap_abs = abs(ub - lb)
    denom = max(1e-9, abs(ub))
    rel_gap = gap_abs / denom
    if gap_abs < 1e-3 or rel_gap < 1e-2:
        return r"$\delta_{\max}$=" + f"{ub:.2f}"
    return r"$\delta_{\max}\!\le\!" + f"{ub:.2f}$"


# Architecture -> PyTorch model class for the dataset-empirical delta_d
# computation. Mirrors utils/train.py's dispatch.
_ARCH_TO_MODEL_CLASS_NAME = {
    "3x10": "FNN_3_10",
    "4x10": "FNN_4_10",
    "5x10": "FNN_5_10",
    "10x10": "FNN_10_10",
    "3x50": "FNN_3_50",
    "5x50": "FNN_5_50",
    "3x100": "FNN_3_100",
    "6x100": "FNN_6_100",
    "9x200": "FNN_9_200",
    "cnn0": "CNN0",
    "cnn1": "CNN1",
    "cnn2": "CNN2",
    "cnn3": "CNN3",
}


def _find_mnist_data_root(cwd):
    """Return a directory containing MNIST/raw/{train,t10k}-*-ubyte so
    torchvision.datasets.MNIST(root=..., download=False) succeeds.
    torchvision expects the layout {root}/MNIST/raw/*."""
    cands = [
        os.path.dirname(os.path.dirname(cwd)),  # for_dana/
        cwd,
        os.path.join(cwd, "data"),
        os.path.join(cwd, "..", "MNIST"),
    ]
    for c in cands:
        if os.path.isfile(os.path.join(c, "MNIST", "raw",
                                       "train-images-idx3-ubyte")):
            return c
    return None


def _compute_delta_d_for_arch_role(arch, model_pth_path, c_srcs, cwd):
    """Forward-pass MNIST train+test through the .pth model and return
        {c_src_0indexed: max_x (N(x)[c_src] - max_{k!=c_src} N(x)[k])}
    over both dataset splits. Returns None if PyTorch/MNIST/the model
    file isn't available — caller treats that as 'unknown' and renders '?'.
    """
    try:
        import torch
        import torchvision.datasets as dsets
        import torchvision.transforms as transforms
    except ImportError:
        return None
    if not os.path.isfile(model_pth_path):
        return None
    cls_name = _ARCH_TO_MODEL_CLASS_NAME.get(arch)
    if cls_name is None:
        return None
    utils_dir = os.path.join(cwd, "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    try:
        import models as _models_mod
    except ImportError:
        return None
    cls = getattr(_models_mod, cls_name, None)
    if cls is None:
        return None
    data_root = _find_mnist_data_root(cwd)
    if data_root is None:
        return None
    try:
        state = torch.load(model_pth_path, map_location="cpu")
        model = cls()
        model.load_state_dict(state)
        model.eval()
    except Exception:
        return None
    tx = transforms.Compose([transforms.ToTensor()])
    try:
        train_ds = dsets.MNIST(root=data_root, train=True,
                                transform=tx, download=False)
        test_ds = dsets.MNIST(root=data_root, train=False,
                               transform=tx, download=False)
    except Exception:
        return None
    deltas = {c: -float("inf") for c in c_srcs}
    with torch.no_grad():
        for ds in (train_ds, test_ds):
            loader = torch.utils.data.DataLoader(
                ds, batch_size=2048, shuffle=False)
            for imgs, _ in loader:
                out = model(imgs)
                for c_src in c_srcs:
                    own = out[:, c_src]
                    if out.shape[1] > 1:
                        mask = torch.ones(out.shape[1], dtype=torch.bool)
                        mask[c_src] = False
                        other_max = out[:, mask].max(dim=1).values
                    else:
                        other_max = torch.zeros_like(own)
                    m = (own - other_max).max().item()
                    if m > deltas[c_src]:
                        deltas[c_src] = m
    return {c: v for c, v in deltas.items() if math.isfinite(v)}


def _load_delta_d_values(cwd, dataset, archs, c_srcs=range(10)):
    """For each (arch, role, tag) discovered under the delta_max sibling
    dir, compute (or load from JSON cache) the dataset-empirical
        delta_d(c_src) = max over MNIST(train+test) of
                         N(x)[c_src] - max_{k!=c_src} N(x)[k]
    using the model.pth at
        paper_experiments/{dataset}/{arch}_exp/{tag}/model.pth.
    Cache lives at
        paper_experiments/{dataset}/{arch}_exp/delta_d/
            delta_d_{arch}_{role}_{tag}.json.
    Returns {(arch, role, c_src_0indexed): float}.
    """
    import glob, json
    out = {}
    c_srcs_list = list(c_srcs)
    for arch in archs:
        dm_base = os.path.join(cwd, "paper_experiments", dataset,
                               f"{arch}_exp", "delta_max")
        if not os.path.isdir(dm_base):
            continue
        cache_base = os.path.join(cwd, "paper_experiments", dataset,
                                  f"{arch}_exp", "delta_d")
        os.makedirs(cache_base, exist_ok=True)
        for role in ("N1", "N2"):
            for d in glob.glob(os.path.join(dm_base,
                                            f"delta_max_{arch}_{role}_*")):
                tag = os.path.basename(d)[len(f"delta_max_{arch}_{role}_"):]
                if not tag:
                    continue
                cache_fp = os.path.join(
                    cache_base, f"delta_d_{arch}_{role}_{tag}.json")
                deltas = None
                if os.path.isfile(cache_fp):
                    try:
                        with open(cache_fp) as f:
                            raw = json.load(f)
                        deltas = {int(k): float(v) for k, v in raw.items()}
                    except (OSError, ValueError):
                        deltas = None
                if deltas is None:
                    # Two on-disk layouts in this repo:
                    #   cnn1: {arch}_exp/{tag}/model.pth
                    #     (tag is e.g. model_seed42_itr20)
                    #   3x10/3x50: {arch}_exp/model_seed42_itr20/{tag}/model.pth
                    #     (tag is an SGD iteration like 19 or 19_sgd_itr1)
                    # Try the flat layout first, then the nested fallback,
                    # then any model.pth one level deeper as a last resort.
                    import glob as _glob
                    arch_root = os.path.join(
                        cwd, "paper_experiments", dataset, f"{arch}_exp")
                    candidates = [
                        os.path.join(arch_root, tag, "model.pth"),
                        os.path.join(arch_root, "model_seed42_itr20",
                                     tag, "model.pth"),
                    ]
                    candidates += _glob.glob(os.path.join(
                        arch_root, "*", tag, "model.pth"))
                    model_pth = next(
                        (c for c in candidates if os.path.isfile(c)),
                        candidates[0])
                    deltas = _compute_delta_d_for_arch_role(
                        arch, model_pth, c_srcs_list, cwd)
                    if deltas is not None:
                        try:
                            with open(cache_fp, "w") as f:
                                json.dump({str(k): v
                                           for k, v in deltas.items()}, f)
                        except OSError:
                            pass
                if deltas is None:
                    continue
                for cs, v in deltas.items():
                    out[(arch, role, cs)] = v
    return out


def _format_delta_d_for_cell(value, delta_max_entry=None):
    """Render delta_d as a percentage of delta_max: (delta_d/delta_max)*100.
    Label is "$\\delta_d$" (the ratio scale is implicit in the % unit).
    delta_max is read from the delta_max_entry's upper bound (the sound
    upper bound). Falls back to '?' if either value is missing or the
    ratio cannot be computed."""
    if value is None or not math.isfinite(value):
        return r"$\delta_d$=?"
    dmax = None
    if delta_max_entry is not None:
        ub = delta_max_entry.get("upper")
        if ub is not None and math.isfinite(ub) and abs(ub) > 1e-9:
            dmax = ub
    if dmax is None:
        return r"$\delta_d$=?"
    pct = (value / dmax) * 100.0
    return r"$\delta_d$=" + f"{pct:.1f}\\%"


def _render_wide_perarch_body(rows, archs, dataset, delta_max_by_key=None,
                               delta_d_by_key=None):
    """One table per architecture. Rows: (arch, role, pert, p_size, c_src).
    Displayed time per cell: mean of t_total over c_tgt and Gurobi seed.

    The model column groups by (arch, role, c_src) — each block shows
    arch / dataset / role / c_src / delta_max for that c_src — instead
    of (arch, role) alone, so the per-c_src delta_max from the Phase 0.5
    pre-phase can be displayed alongside the timings. delta_max_by_key
    is the dict returned by _load_delta_max_values (may be empty/None,
    in which case the delta_max line shows '?')."""
    if not rows:
        return ("\\noindent\\textit{No \\texttt{stdBoost} or "
                "\\texttt{vaghar*} result files were found under "
                "\\texttt{paper\\_experiments/<arch>\\_exp/}. Re-run the "
                "sweep with \\texttt{--include\\_nn1\\_boost "
                "--include\\_nn2\\_boost} and the "
                "\\texttt{--sweep\\_stdboost\\_*} flags.}")

    from collections import defaultdict
    buckets = defaultdict(lambda: defaultdict(
        lambda: {"t": [], "lb": [], "ub": [], "status": [],
                  "c_targets": set()}))
    for r in rows:
        key = (r["arch"], r["role"], r["perturbation"],
               r["perturbation_size"], r["c_source"])
        cell = buckets[key][r["combo"]]
        cell["t"].append(r["t_total"])
        lb = r.get("lb_total")
        if lb is not None and math.isfinite(lb):
            cell["lb"].append(lb)
        ub = r.get("ub_total")
        if ub is not None and math.isfinite(ub):
            cell["ub"].append(ub)
        cell["status"].append(str(r.get("solve_status", "") or ""))
        ct = r.get("c_target")
        if ct is not None:
            try:
                cell["c_targets"].add(int(ct))
            except (TypeError, ValueError):
                pass

    columns = list(_wide_column_order())

    lines = []
    lines.append(f"% auto-generated: archs={archs}, dataset={dataset}, "
                 f"total_cell_rows={len(rows)}")

    for arch in archs:
        # Sort by (arch, role, c_src, pert, p_size) so all (pert, p_size)
        # rows for one (role, c_src) are consecutive — the multirow span
        # in the model column relies on this contiguity.
        arch_keys = sorted(
            (k for k in buckets if k[0] == arch),
            key=lambda k: (k[0], k[1], k[4], k[2], k[3]),
        )
        if not arch_keys:
            continue
        lines.append("\\begin{table}[!htbp]")
        lines.append("\\centering")
        lines.append("\\scriptsize")
        lines.append("\\setlength{\\tabcolsep}{3pt}")
        lines.append("\\begin{adjustbox}{max width=\\textwidth,center}%")
        # Each technique-combo column is now a group of 3 sub-columns:
        # delta_l (%), delta_u (%), time. Separate groups with a "|".
        # An extra "missing c_targets" column sits between "pert (size)"
        # and the combo groups; it lists any c_target the sweep should
        # have run for this (arch, role, pert, p_size, c_src) row but
        # didn't (union across all combos). Per-combo gaps are signalled
        # by a `*` on the combo's $t$ sub-cell.
        col_spec = ("@{}l l l | "
                    + " | ".join(["r r r"] * len(columns))
                    + "@{}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\hline")
        # Two-row header: top row carries the combo title spanning 3
        # sub-columns; bottom row labels the sub-columns themselves.
        top = ["\\multirow{2}{*}{model}",
               "\\multirow{2}{*}{pert (size)}",
               "\\multirow{2}{*}{missing c\\_targets}"]
        for c in columns:
            top.append("\\multicolumn{3}{c"
                       + ("|" if c is not columns[-1] else "")
                       + "}{" + _wide_column_header(c) + "}")
        lines.append(" & ".join(top) + r" \\")
        sub = ["", "", ""]
        for _ in columns:
            sub += [r"$\delta_l$\%", r"$\delta_u$\%", r"$t$"]
        lines.append(" & ".join(sub) + r" \\")
        lines.append("\\hline")

        # Pre-group keys by (role, c_src) so we know each block's row
        # count up front — the model label is spread across the block's
        # consecutive rows (line 1 = arch/role, line 2 = c_src,
        # line 3 = delta_max) instead of stacked in a \shortstack on
        # the first row, which had left rows 2-3 of the model column
        # next to empty whitespace in every other column. If a block
        # has fewer rows than label lines, the remaining label parts
        # are collapsed onto the last available row as a \shortstack.
        from itertools import groupby
        block_groups = []
        for role_csrc, gkeys in groupby(arch_keys,
                                         key=lambda k: (k[1], k[4])):
            block_groups.append((role_csrc, list(gkeys)))

        for bi, (role_csrc, block_keys) in enumerate(block_groups):
            role, c_src = role_csrc
            if bi > 0:
                # Horizontal separator between every (role, c_src) block.
                lines.append("\\hline")
            dm_entry = None
            if delta_max_by_key is not None:
                dm_entry = delta_max_by_key.get((arch, role, c_src))
            dm_str = _format_delta_max_for_cell(dm_entry)
            dd_val = None
            if delta_d_by_key is not None:
                dd_val = delta_d_by_key.get((arch, role, c_src))
            dd_str = _format_delta_d_for_cell(dd_val, dm_entry)
            # delta_max + delta_d share one line (per user); dataset on
            # its own line below.
            label_parts = [
                r"\textbf{" + arch + r"/" + role + r"}",
                r"$c_s$=" + str(c_src),
                dm_str + r", " + dd_str,
                r"dataset: \texttt{" + dataset.replace("_", r"\_") + r"}",
            ]
            n_rows = len(block_keys)
            # Distribute label_parts across rows. If we have more rows
            # than parts, later rows have an empty model cell. If we
            # have fewer rows than parts (very small blocks), pack the
            # leftover parts into the last available row via \shortstack.
            row_labels = [""] * n_rows
            for i in range(min(n_rows, len(label_parts))):
                if i == n_rows - 1 and n_rows < len(label_parts):
                    leftover = label_parts[i:]
                    row_labels[i] = (r"\shortstack[l]{"
                                     + r"\\".join(leftover) + r"}")
                else:
                    row_labels[i] = label_parts[i]

            # Expected c_targets per row: all class indices in the
            # dataset except c_src. The sweep is supposed to fan out
            # over every other class, so anything missing is a gap
            # worth surfacing. Class counts are dataset-specific.
            _DATASET_NUM_CLASSES = {"mnist": 10, "cifar10": 10,
                                     "fashion_mnist": 10}
            n_classes = _DATASET_NUM_CLASSES.get(dataset, 10)

            prev_pert_key = None  # re-emit pert on (pert,p_size) change
            for ri, key in enumerate(block_keys):
                _, role, pert, p_size, c_src = key
                cell_dict = buckets[key]
                model_str = row_labels[ri]
                pert_key = (role, c_src, pert, p_size)
                if pert_key != prev_pert_key:
                    pert_str = (f"{pert} ({p_size})").replace("_", "\\_")
                    prev_pert_key = pert_key
                else:
                    pert_str = ""
                # Row-level missing-c_target set: a c_target appears iff
                # some combo on the row has a partial result (data on
                # this row) that didn't include that c_target. Combos
                # with no data at all on the row don't contribute — they
                # simply weren't run, which is different from "started
                # but skipped this c_target". Per-combo gaps are also
                # signalled via a `*` on the combo's $t$ sub-cell below.
                expected_cts = {ct for ct in range(n_classes)
                                if ct != c_src}
                per_combo_missing = {}
                for c in columns:
                    cell = cell_dict.get(c)
                    if not cell or not cell.get("t"):
                        continue
                    seen = cell.get("c_targets", set())
                    per_combo_missing[c] = expected_cts - seen
                row_missing = set()
                for miss in per_combo_missing.values():
                    row_missing |= miss
                if row_missing:
                    missing_str = ("(" + ",".join(str(ct)
                                                   for ct in sorted(row_missing))
                                   + ")")
                else:
                    missing_str = ""
                row = [model_str, pert_str, missing_str]
                # delta_max upper-bound for this (arch, role, c_src) cell
                # — divisor for the delta_l / delta_u percentages.
                dmax_ub = None
                if dm_entry is not None:
                    ub_dm = dm_entry.get("upper")
                    if (ub_dm is not None and math.isfinite(ub_dm)
                            and abs(ub_dm) > 1e-9):
                        dmax_ub = ub_dm
                # Per-combo aggregates for this row: mean time (min),
                # mean delta_l/delta_u (% of delta_max), whether every
                # underlying cell hit TIME_LIMIT (so the displayed time
                # is not a real finish — disqualifies the combo from
                # time-based wins per the user's rule).
                stats = {}
                for c, cell in cell_dict.items():
                    s = {}
                    if cell["t"]:
                        s["t"] = sum(cell["t"]) / len(cell["t"]) / 60.0
                    if cell["lb"] and dmax_ub is not None:
                        s["lb_pct"] = (sum(cell["lb"]) / len(cell["lb"])
                                       / dmax_ub) * 100.0
                    if cell["ub"] and dmax_ub is not None:
                        s["ub_pct"] = (sum(cell["ub"]) / len(cell["ub"])
                                       / dmax_ub) * 100.0
                    # all-timeout if every recorded status is TIME_LIMIT
                    # (case-insensitive, also matches the "TIME LIMIT"
                    # spelling). Empty status strings are treated as
                    # "unknown, not timeout" so a missing solve_status
                    # field doesn't accidentally disqualify a combo.
                    if cell["status"]:
                        norm = [st.upper().replace(" ", "_")
                                for st in cell["status"] if st]
                        s["all_timeout"] = (bool(norm) and all(
                            "TIME_LIMIT" in n for n in norm))
                    else:
                        s["all_timeout"] = False
                    stats[c] = s
                # Pick winners: shortest time among combos whose time
                # is real (not all-timeout). Ties are decided at the
                # same 1-decimal precision the reader sees, so any
                # combo whose `f"{t:.1f}"` equals the row minimum's
                # also wins. If no combo qualifies, fall back to
                # combos sharing the smallest delta_u-delta_l gap
                # among those with both bounds.
                # Only combos that actually map to a rendered column are
                # eligible to win — otherwise a "ghost" combo whose key
                # never appears in the table could carry the minimum
                # time/gap and silently leave the row with no green cells.
                columns_set = set(columns)
                winners = set()
                time_candidates = [
                    (c, s["t"]) for c, s in stats.items()
                    if c in columns_set and "t" in s
                    and not s.get("all_timeout")]
                if time_candidates:
                    min_t = min(t for _, t in time_candidates)
                    min_t_disp = f"{min_t:.1f}"
                    winners = {c for c, t in time_candidates
                               if f"{t:.1f}" == min_t_disp}
                else:
                    gap_candidates = [
                        (c, s["ub_pct"] - s["lb_pct"])
                        for c, s in stats.items()
                        if c in columns_set
                        and "lb_pct" in s and "ub_pct" in s]
                    if gap_candidates:
                        min_g = min(g for _, g in gap_candidates)
                        min_g_disp = f"{min_g:.1f}"
                        winners = {c for c, g in gap_candidates
                                   if f"{g:.1f}" == min_g_disp}
                # Emit the 3 sub-cells per combo. Every winning combo's
                # 3 cells get \cellcolor{green!25}; everyone else is
                # plain.
                def _paint(val, is_winner):
                    if is_winner:
                        return "\\cellcolor{green!25}" + val
                    return val
                for c in columns:
                    is_win = (c in winners)
                    s = stats.get(c)
                    if s is None:
                        # No data at all for this (row, combo) — the
                        # row-level "missing c_targets" column already
                        # reflects this; the `*` would just add noise.
                        row += [_paint("---", is_win)] * 3
                        continue
                    combo_gap = bool(per_combo_missing.get(c))
                    star = "$^*$" if combo_gap else ""
                    # delta_l is a percentage of delta_max, so a negative
                    # value (margin below 0) is clamped to 0; delta_u above
                    # 100% is clamped to 100 for display.
                    row.append(_paint(
                        f"{max(0.0, s['lb_pct']):.1f}" if "lb_pct" in s else "---",
                        is_win))
                    row.append(_paint(
                        f"{min(100.0, s['ub_pct']):.1f}" if "ub_pct" in s else "---",
                        is_win))
                    t_str = f"{s['t']:.1f}" if "t" in s else "---"
                    row.append(_paint(t_str + star, is_win))
                lines.append(" & ".join(row) + r" \\")
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{adjustbox}")
        cap = (f"Architecture \\textbf{{{arch}}} (dataset {dataset}) "
               f"--- the \\textsf{{missing c\\_targets}} column lists "
               f"any $c_\\mathrm{{tgt}}$ value (every class index except "
               f"$c_\\mathrm{{src}}$) that \\emph{{some}} combo with a "
               f"partial result on this row didn't cover. Equivalently: "
               f"the union of per-combo missing sets restricted to "
               f"combos that actually ran something on this row. Combos "
               f"that never ran on the row (all-\\textsf{{---}}) "
               f"contribute nothing here. An empty cell therefore means "
               f"every combo that has data on the row covered every "
               f"expected $c_\\mathrm{{tgt}}$. Per-combo partial "
               f"coverage is also marked with a \\textsuperscript{{*}} "
               f"after that combo's $t$ sub-cell: \\texttt{{t.t$^*$}} "
               f"means this particular combo has data on this row but "
               f"was not exercised on every expected "
               f"$c_\\mathrm{{tgt}}$ (so its mean time / $\\delta$ "
               f"aggregates over fewer cells than its un-starred "
               f"peers). Combos with no data on a row show plain "
               f"\\textsf{{---}} with no star. "
               f"Per technique-combo, each cell is a 3-tuple "
               f"$(\\delta_l, \\delta_u, t)$: "
               f"$\\delta_l = 100 \\cdot \\mathrm{{lower\\_bound}}/"
               f"\\delta_{{\\max}}$ and "
               f"$\\delta_u = 100 \\cdot \\mathrm{{upper\\_bound}}/"
               f"\\delta_{{\\max}}$ are the MIP-returned bounds rescaled "
               f"into per-cent units of the $\\delta_{{\\max}}$ in the "
               f"model column; $t$ is the wall-clock time (minutes), "
               f"mean over $c_\\mathrm{{tgt}}$ and Gurobi seed. "
               f"Each model cell groups by ($c_\\mathrm{{src}}$) and reports "
               f"$\\delta_{{\\max}}$ for that source class, as computed "
               f"by Phase~0.5 of the sweep "
               f"(\\texttt{{run.jl --perturbation max --ctag $c_\\mathrm{{src}}$}} "
               f"per network) and cached under "
               f"\\texttt{{paper\\_experiments/{dataset}/<arch>\\_exp/delta\\_max/}}. "
               f"$\\delta_{{\\max}}$ is the sound upper bound on "
               f"$\\max_x N(x)[c_s] - \\max_{{k \\ne c_s}} N(x)[k]$ "
               f"over the clean input box; rows where the pre-phase "
               f"finished optimally show a bare value, rows where it "
               f"hit \\texttt{{TIME\\_LIMIT}} show "
               f"$\\delta_{{\\max}} \\le Y$ (Gurobi upper bound). "
               f"The two baselines "
               f"\\textsf{{vaghar}} (PI=off, no boosts) and "
               f"\\textsf{{pi}} (PI=on, no boosts) each pool together "
               f"the standalone vaghar baseline runs "
               f"(\\texttt{{vagharNoPerturbed\\_*}} / "
               f"\\texttt{{vagharWithPerturbed\\_*}}) \\emph{{and}} the "
               f"\\texttt{{stdBoost}} cells with all options off "
               f"(zono=off, sg=off, tau=0 and matching pi value), "
               f"since those cells "
               f"encode an identical MIP and only differ in Gurobi seed. "
               f"The remaining columns are the actual boosts. In the "
               f"column labels, \\textsf{{zono}} stands for "
               f"\\tech{{nn1\\_zono\\_bounds}}, \\textsf{{sg}} for "
               f"\\tech{{nn1\\_sibling\\_gate}}, \\textsf{{pi}} for "
               f"\\cli{{use\\_perturbed\\_intervals}}, and "
               f"\\textsf{{tau=$x$}} is \\tech{{nn1\\_relax\\_threshold}}. "
               f"Combinations the sweep filter excludes ((pi=off, "
               f"tau$>0$); see "
               f"\\texttt{{run\\_relaxation\\_sweep.py:2854}}) appear as "
               f"\\textsf{{---}}. The two rightmost columns "
               f"\\textsf{{adv+zono+prev\\_pgd+sg+tau=0.0}} and "
               f"\\textsf{{adv+zono+prev\\_pgd+sg+tau=0.5}} "
               f"report the advstd N2 transfer-mode runs for the two "
               f"\\texttt{{zono:prev\\_pgd:0.0+sg}} and "
               f"\\texttt{{zono:prev\\_pgd:0.5+sg}} combinations (loaded "
               f"directly from the per-cell advstd CSVs that back "
               f"Section~\\ref{{sec:safe}}); these rows only populate on "
               f"the $N_2$ block since advstd is a transfer technique. "
               f"In each row, every winning combo's three sub-cells "
               f"are shaded \\cellcolor{{green!25}}~light~green: a "
               f"combo wins if its mean $t$ ties the row-minimum (at "
               f"the displayed 0.1-min precision) \\emph{{and}} its "
               f"underlying cells did \\emph{{not}} all hit "
               f"\\texttt{{TIME\\_LIMIT}}; if every combo on the row "
               f"is timed out, winners are instead all combos sharing "
               f"the smallest $\\delta_u - \\delta_l$ gap.")
        lines.append(f"\\caption{{{cap}}}")
        safe_arch = arch.replace("_", "")
        lines.append(f"\\label{{tab:safe_wide_{safe_arch}}}")
        lines.append("\\end{table}")
        lines.append("")

    return "\n".join(lines)


def update_wide_perarch_tex(tex_path, body):
    """Rewrite the % BEGIN AUTO: wide_perarch_tables block in tex_path."""
    with open(tex_path) as f:
        text = f.read()
    if WIDE_BEGIN_MARK not in text or WIDE_END_MARK not in text:
        raise SystemExit(
            f"wide_perarch markers not found in {tex_path}; expected lines "
            f"containing '{WIDE_BEGIN_MARK}' and '{WIDE_END_MARK}'")
    pre, rest = text.split(WIDE_BEGIN_MARK, 1)
    _body, post = rest.split(WIDE_END_MARK, 1)
    updated = f"{pre}{WIDE_BEGIN_MARK}\n{body}\n{WIDE_END_MARK}{post}"
    if updated == text:
        print("[update_advstd_tex_tables] no changes to wide_perarch block")
        return
    with open(tex_path, "w") as f:
        f.write(updated)
    print(f"[update_advstd_tex_tables] wrote wide_perarch block in {tex_path}")


## ---------------------------------------------------------------------------
## AAAI slim variant: same data, but only the four safe-combo columns and
## table* (full-width) floats so it fits the AAAI 2026 two-column layout.
## ---------------------------------------------------------------------------

AAAI_WIDE_BEGIN_MARK = "% BEGIN AUTO: aaai_safe_wide_tables"
AAAI_WIDE_END_MARK   = "% END AUTO: aaai_safe_wide_tables"

# Source-network (N1) per-cell tables. These live in the appendix
# (sec_appendix_percell.tex); the target-network (N2) tables stay in the
# evaluation body between the AAAI_WIDE marks above.
AAAI_WIDE_N1_BEGIN_MARK = "% BEGIN AUTO: aaai_safe_wide_n1_tables"
AAAI_WIDE_N1_END_MARK   = "% END AUTO: aaai_safe_wide_n1_tables"

# Body-section summary table (Table 2 in the paper): one row per architecture
# with the transfer-mode bound gap vs. the exact VHAGaR bound and the
# transfer-mode speedup over VHAGaR. Lives between these markers in
# sec_evaluation.tex; derived from the SAME per-cell rows as the appendix.
AAAI_SUMMARY_BEGIN_MARK = "% BEGIN AUTO: aaai_summary_table"
AAAI_SUMMARY_END_MARK   = "% END AUTO: aaai_summary_table"

# 3 columns: the original VHAGaR baseline (PI=off, no boosts), the
# standard-mode safe combo zono+SibGate+PI at tau=0.5, and the
# transfer-mode safe combo adv+zono+prev_pgd+SibGate at tau=0.5.
# Each entry: (column_key, multicolumn header label).
_AAAI_WIDE_COLUMNS = (
    ("vaghar",                   r"vaghar"),
    (("1", "1", "1", "0.5"),     r"standard ($\tau{=}0.5$)"),
    ("adv_zono_prevpgd_0.5+sg",  r"transfer ($\tau{=}0.5$)"),
)


_AAAI_TIMEOUT_STATUSES = frozenset({
    "TIME_LIMIT", "TIME LIMIT", "USER_OBJ_LIMIT", "USER_LIMIT",
    "ITERATION_LIMIT", "NODE_LIMIT", "SOLUTION_LIMIT",
    "MEMORY_LIMIT", "WORK_LIMIT",
})


def _aaai_is_timeout_mismatch(row, force_timeout, eps):
    """True iff this row's Gurobi run hit a termination limit at a wall-clock
    cap different from `force_timeout` (within ±eps seconds). Non-timeout
    rows always return False (they're included regardless)."""
    if force_timeout is None:
        return False
    status = str(row.get("solve_status", "") or "").upper().replace(" ", "_")
    if not any(tag in status for tag in _AAAI_TIMEOUT_STATUSES):
        return False
    t = row.get("t_total")
    if t is None:
        return True  # timeout with unknown wall-clock; conservatively drop
    try:
        return abs(float(t) - float(force_timeout)) > float(eps)
    except (TypeError, ValueError):
        return True


def _fmt_trim(x):
    """Format a number with one decimal, dropping a trailing '.0' so a
    whole value renders as '180' rather than '180.0' while '29.5' stays
    '29.5'. Used for the delta_max and time cells in the neta_s_paper
    tables."""
    s = f"{x:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _fmt_sig(x, sig=2):
    """Format x to `sig` significant figures, never in scientific
    notation, with trailing zeros and a bare trailing '.' removed. This
    mirrors the VHAGaR bound columns where the digit count tracks the
    magnitude: 99.2->'99', 28.8->'29', 1.6->'1.6', 0.6->'0.6',
    100->'100', 2.0->'2'. Used for the delta_l / delta_u / delta_d
    columns in the neta_s_paper tables."""
    if not math.isfinite(x) or x == 0:
        return "0"
    d = math.floor(math.log10(abs(x)))
    decimals = max(0, sig - 1 - d)
    s = f"{x:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _render_aaai_wide_perarch_body(rows, archs, dataset,
                                    delta_max_by_key=None,
                                    delta_d_by_key=None,
                                    force_timeout=None,
                                    rerun_timeout_eps=30.0,
                                    roles=None,
                                    label_suffix=""):
    """Slim version of _render_wide_perarch_body: only the four safe-combo
    columns and table* (two-column-spanning) floats, suitable for inclusion
    in the AAAI 2026 paper's evaluation section.

    `roles`, when given, keeps only blocks for those network roles (e.g.
    {"N2"} for the target-network tables in the body or {"N1"} for the
    source-network tables in the appendix). `label_suffix` is appended to
    each table's \\label so the body and appendix tables get distinct
    labels (the body uses "", the N1 appendix uses "-n1"). The shared
    header macro is emitted with \\providecommand so both the body and the
    appendix block can define it without a redefinition clash.

    Data shape and column semantics are identical to the full table; only
    the column subset and the surrounding LaTeX shell differ. When
    `force_timeout` is set (in seconds), rows whose Gurobi run hit a
    termination limit at a different wall-clock cap (outside
    ±rerun_timeout_eps seconds) are dropped before bucketing.
    """
    if not rows:
        return ("\\noindent\\textit{No data available.}")
    if force_timeout is not None:
        before = len(rows)
        rows = [r for r in rows
                if not _aaai_is_timeout_mismatch(r, force_timeout,
                                                  rerun_timeout_eps)]
        dropped = before - len(rows)
        if dropped:
            print(f"[update_advstd_tex_tables] aaai_safe_wide: dropped "
                  f"{dropped}/{before} timeout-mismatched cells "
                  f"(force_timeout={force_timeout}s "
                  f"\\pm {rerun_timeout_eps}s)")

    from collections import defaultdict
    buckets = defaultdict(lambda: defaultdict(
        lambda: {"t_base": [], "t_geom": [], "lb_base": [], "lb_geom": [],
                  "ub_base": [], "ub_geom": [], "status": [],
                  "c_targets": set(), "c_targets_geom": set()}))
    for r in rows:
        key = (r["arch"], r["role"], r["perturbation"],
               r["perturbation_size"], r["c_source"])
        cell = buckets[key][r["combo"]]
        # Split the solve time AND the delta bounds by the geometric_intervals
        # twin flag, so every cell renders "base/geom" for delta_l, delta_u and
        # t. status / c_targets (the partial-asterisk coverage) come from the
        # BASELINE runs only, so the asterisk reflects baseline completeness.
        is_geom = bool(r.get("geom"))
        t_total = r.get("t_total")
        if t_total is not None:
            (cell["t_geom"] if is_geom else cell["t_base"]).append(t_total)
        lb = r.get("lb_total")
        if lb is not None and math.isfinite(lb):
            (cell["lb_geom"] if is_geom else cell["lb_base"]).append(lb)
        ub = r.get("ub_total")
        if ub is not None and math.isfinite(ub):
            (cell["ub_geom"] if is_geom else cell["ub_base"]).append(ub)
        # c_target coverage tracked PER SIDE, so base and geom each get their
        # own partial "*" when that side missed an expected class-pair.
        ct = r.get("c_target")
        if ct is not None:
            try:
                (cell["c_targets_geom"] if is_geom
                 else cell["c_targets"]).add(int(ct))
            except (TypeError, ValueError):
                pass
        if is_geom:
            continue
        cell["status"].append(str(r.get("solve_status", "") or ""))

    columns = [col for col, _hdr in _AAAI_WIDE_COLUMNS]

    lines = []
    lines.append(f"% auto-generated: archs={archs}, dataset={dataset}, "
                 f"total_cell_rows={len(rows)}")
    # Shared header macro so the three per-arch tables stay in sync.
    header_cells = ["\\multirow{2}{*}{model}",
                    "\\multirow{2}{*}{pert (size)}"]
    for _col, hdr in _AAAI_WIDE_COLUMNS:
        sep = "|" if (_col, hdr) != _AAAI_WIDE_COLUMNS[-1] else ""
        header_cells.append("\\multicolumn{3}{c" + sep + "}{" + hdr + "}")
    sub_cells = ["", ""]
    for _ in _AAAI_WIDE_COLUMNS:
        sub_cells += [r"$\delta_l$\%", r"$\delta_u$\%", r"$t$"]
    lines.append(r"\providecommand{\aaaisafewideheader}{%")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(" & ".join(sub_cells) + r" \\}")
    lines.append("")

    for arch in archs:
        arch_keys = sorted(
            (k for k in buckets
             if k[0] == arch and (roles is None or k[1] in roles)),
            key=lambda k: (k[1], k[4], k[2], k[3]),  # role, c_src, pert, p_size
        )
        if not arch_keys:
            continue
        # Drop rows where every one of the 4 columns has no data (base or geom)
        arch_keys = [
            k for k in arch_keys
            if any(buckets[k].get(c)
                   and (buckets[k][c].get("t_base") or buckets[k][c].get("t_geom"))
                   for c in columns)
        ]
        if not arch_keys:
            continue
        safe_arch = arch.replace("_", "")
        lines.append(r"\begin{table*}[!tbp]")
        lines.append(r"\centering")
        lines.append(r"\scriptsize")
        lines.append(r"\setlength{\tabcolsep}{3pt}")
        lines.append(r"\begin{adjustbox}{max width=\textwidth,center}%")
        col_spec = ("@{}l l | " + " | ".join(["r r r"] * len(columns))
                    + "@{}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")
        lines.append(r"\aaaisafewideheader")
        lines.append(r"\midrule")

        # Group by (role, c_src) for the multicolumn section heading
        from itertools import groupby
        for (role, c_src), gkeys in groupby(arch_keys,
                                             key=lambda k: (k[1], k[4])):
            dm_entry = None
            if delta_max_by_key is not None:
                dm_entry = delta_max_by_key.get((arch, role, c_src))
            dmax_ub = None
            dmax_str = "?"
            if dm_entry is not None:
                ub_dm = dm_entry.get("upper")
                if (ub_dm is not None and math.isfinite(ub_dm)
                        and abs(ub_dm) > 1e-9):
                    dmax_ub = ub_dm
                    dmax_str = _fmt_trim(ub_dm)
            # delta_d is the dataset-empirical max margin as a percentage
            # of the sound delta_max upper bound: (delta_d / delta_max)*100.
            # Same definition as the full advstd_techniques.tex tables.
            dd_str = "?"
            dd_val = None
            if delta_d_by_key is not None:
                dd_val = delta_d_by_key.get((arch, role, c_src))
            if (dd_val is not None and math.isfinite(dd_val)
                    and dmax_ub is not None):
                dd_str = _fmt_sig((dd_val / dmax_ub) * 100.0) + "\\%"
            # role is stored as "N1"/"N2"; render with a subscript so
            # the model cell reads "cnn1 / N_1" / "cnn1 / N_2" rather
            # than the italic-bare "N1"/"N2".
            role_tex = "N_" + role[1:] if role.startswith("N") else role
            # The model value is distributed down the first column, one
            # part per row, across the block's rows (rather than a wide
            # full-width heading that overflows into the data columns).
            label_parts = [
                r"\textbf{" + arch.replace("_", r"\_")
                + r" / $" + role_tex + r"$}",
                r"$c_s{=}" + str(c_src) + r"$",
                r"$\delta_{\max}{=}" + dmax_str + r"$",
                r"$\delta_d{=}" + dd_str + r"$",
            ]

            # Expected c_targets for every row in this block: every class
            # index except c_src. A combo whose actually-recorded set of
            # c_targets is a strict subset of this gets a red asterisk
            # next to its time cell so the reader knows the mean was
            # taken over a partial sweep.
            _DATASET_NUM_CLASSES = {"mnist": 10, "cifar10": 10,
                                     "fashion_mnist": 10}
            n_classes = _DATASET_NUM_CLASSES.get(dataset, 10)
            expected_cts_block = {ct for ct in range(n_classes)
                                  if ct != int(c_src)}

            # First pass: build the data cells (everything except the
            # model column) for every row that actually has data, so the
            # model label can be spread across exactly the rendered rows.
            block_rows = []
            for key in list(gkeys):
                _, role, pert, p_size, c_src = key
                cell_dict = buckets[key]
                # Pre-compute stats for each rendered column
                stats = {}
                partial = {}
                partial_geom = {}
                for c in columns:
                    cell = cell_dict.get(c)
                    if not cell or not (cell.get("t_base") or cell.get("t_geom")):
                        continue
                    s = {}
                    if cell.get("t_base"):
                        s["t_base"] = (sum(cell["t_base"])
                                       / len(cell["t_base"]) / 60.0)
                    if cell.get("t_geom"):
                        s["t_geom"] = (sum(cell["t_geom"])
                                       / len(cell["t_geom"]) / 60.0)
                    if dmax_ub is not None:
                        if cell["lb_base"]:
                            s["lb_pct_base"] = (sum(cell["lb_base"])
                                / len(cell["lb_base"]) / dmax_ub) * 100.0
                        if cell["lb_geom"]:
                            s["lb_pct_geom"] = (sum(cell["lb_geom"])
                                / len(cell["lb_geom"]) / dmax_ub) * 100.0
                        if cell["ub_base"]:
                            s["ub_pct_base"] = (sum(cell["ub_base"])
                                / len(cell["ub_base"]) / dmax_ub) * 100.0
                        if cell["ub_geom"]:
                            s["ub_pct_geom"] = (sum(cell["ub_geom"])
                                / len(cell["ub_geom"]) / dmax_ub) * 100.0
                    stats[c] = s
                    # A side is partial iff at least one expected c_target
                    # wasn't run on that side. Don't penalise extra c_targets
                    # outside expected_cts_block (e.g. dataset-specific quirks).
                    # Tracked per side so base and geom each get their own "*".
                    partial[c] = bool(expected_cts_block
                                      - cell.get("c_targets", set()))
                    partial_geom[c] = bool(expected_cts_block
                                           - cell.get("c_targets_geom", set()))
                # Drop the row if no rendered column has data
                if not stats:
                    continue
                pert_str = f"{pert} ({p_size})".replace("_", r"\_")
                # Replace "linf" textual key with the math glyph
                pert_str = pert_str.replace("linf", r"$\ell_\infty$")
                data_cells = [pert_str]
                for c in columns:
                    s = stats.get(c)
                    if s is None:
                        data_cells += ["---", "---", "---"]
                        continue
                    # Every sub-column renders "base/geom" (baseline run vs its
                    # --geometric_intervals twin). A side with no runs shows "?"
                    # (geometric_intervals is inapplicable to linf/brightness/
                    # occ/patch, or its run is absent); the whole sub-column is
                    # "---" when neither side has data. delta is a sound
                    # tightening, so base/geom match except where a run timed out
                    # at a different gap or the baseline/geom c_target coverage
                    # differs.
                    def _pair(bk, gk, fmt):
                        hb, hg = bk in s, gk in s
                        if not hb and not hg:
                            return "---"
                        return ((fmt(s[bk]) if hb else "?") + "/"
                                + (fmt(s[gk]) if hg else "?"))
                    # Clamp for display: delta_l (% of delta_max) below 0 shows
                    # 0, delta_u above 100 shows 100.
                    data_cells.append(_pair("lb_pct_base", "lb_pct_geom",
                                            lambda x: _fmt_sig(max(0.0, x))))
                    data_cells.append(_pair("ub_pct_base", "ub_pct_geom",
                                            lambda x: _fmt_sig(min(100.0, x))))
                    # Time column reads "base/geom": the baseline solve time
                    # and the --geometric_intervals twin's solve time (same
                    # delta, so this is a pure speed comparison). A side with
                    # no runs renders "?" (e.g. linf/brightness/occ/patch,
                    # where geometric_intervals is structurally inapplicable,
                    # or a translation/rotation cell whose geom run is absent);
                    # the whole cell is "---" when neither side has data.
                    # Time is in minutes; when --force_timeout is set the rows
                    # that survived the timeout-mismatch filter all ran under
                    # that cap, so a mean above it is overhead beyond the
                    # solver limit and is clamped (force_timeout is in seconds).
                    # Each side carries its OWN partial "*": base is starred
                    # when the baseline sweep missed a class-pair, geom when the
                    # geometric_intervals sweep did. A "?" side (no runs) is
                    # never starred; the whole cell is "---" when both are "?".
                    STAR = r"\textcolor{red}{$^*$}"
                    def _t_side(key, is_partial):
                        if key not in s:
                            return "?"
                        v = s[key]
                        if force_timeout is not None:
                            v = min(v, force_timeout / 60.0)
                        return _fmt_trim(v) + (STAR if is_partial else "")
                    base_side = _t_side("t_base", partial.get(c))
                    geom_side = _t_side("t_geom", partial_geom.get(c))
                    if base_side == "?" and geom_side == "?":
                        t_str = "---"
                    else:
                        t_str = base_side + "/" + geom_side
                    data_cells.append(t_str)
                block_rows.append(data_cells)

            if not block_rows:
                continue

            # Distribute the model label parts down the first column.
            # More rows than parts -> trailing rows get an empty model
            # cell. Fewer rows than parts -> pack the leftover parts into
            # the last row via a left-aligned \shortstack so nothing is
            # dropped.
            n_rows = len(block_rows)
            row_labels = [""] * n_rows
            for i in range(min(n_rows, len(label_parts))):
                if i == n_rows - 1 and n_rows < len(label_parts):
                    leftover = label_parts[i:]
                    row_labels[i] = (r"\shortstack[l]{"
                                     + r"\\".join(leftover) + r"}")
                else:
                    row_labels[i] = label_parts[i]

            for lbl, data_cells in zip(row_labels, block_rows):
                lines.append(" & ".join([lbl] + data_cells) + r" \\")
            lines.append(r"\midrule")

        # Trailing \midrule -> \bottomrule
        if lines[-1].rstrip() == r"\midrule":
            lines[-1] = r"\bottomrule"
        lines.append(r"\end{tabular}%")
        lines.append(r"\end{adjustbox}")
        if roles is not None and set(roles) == {"N2"}:
            role_note = (r"This table reports the target network $N_2$; "
                         r"per-cell results for the source network $N_1$ "
                         r"are in Table~\ref{tab:safe-wide-"
                         + safe_arch + r"-n1}.")
        elif roles is not None and set(roles) == {"N1"}:
            role_note = (r"This table reports the source network $N_1$, "
                         r"for which transfer-mode does not apply (transfer "
                         r"is $N_2$-only), so the transfer columns are "
                         r"blank; the target network $N_2$ is in "
                         r"Table~\ref{tab:safe-wide-" + safe_arch + r"}.")
        else:
            role_note = (r"Transfer columns are blank on $N_1$ rows since "
                         r"transfer-mode is $N_2$-only.")
        cap = (
            f"{arch} --- per-cell comparison of the original VHAGaR "
            r"baseline against our standard- and transfer-mode safe "
            r"combinations at $\tau{=}0.5$. \emph{vaghar} is the "
            r"VHAGaR baseline (no boosts). \emph{Standard} is the "
            r"\emph{zono + SibGate} combo with perturbed-interval "
            r"constraints. \emph{Transfer} adds the \emph{adv + "
            r"prev\_pgd} variable-hint pipeline that reuses $N_1$'s "
            r"solver state. " + role_note + r" Each cell is "
            r"$(\delta_l, \delta_u, t)$ with $\delta_l$ and $\delta_u$ "
            r"in per-cent of $\delta_{\max}$ and $t$ in wall-clock "
            r"minutes (mean over $c_t$ and Gurobi seeds). Each block "
            r"heading reports $\delta_{\max}$, the sound upper bound on "
            r"the maximum margin, and $\delta_d$, the dataset-empirical "
            r"maximum margin over MNIST as a per-cent of $\delta_{\max}$. "
            r"A dash means "
            r"the combo was not exercised on the cell; a red "
            r"\textcolor{red}{$^*$} on the time means the cell aggregates "
            r"over only some of the expected target classes."
        )
        lines.append(f"\\caption{{{cap}}}")
        lines.append(f"\\label{{tab:safe-wide-{safe_arch}{label_suffix}}}")
        lines.append(r"\end{table*}")
        lines.append("")

    return "\n".join(lines)


def update_aaai_wide_perarch_tex(tex_path, body,
                                 begin_mark=AAAI_WIDE_BEGIN_MARK,
                                 end_mark=AAAI_WIDE_END_MARK):
    """Rewrite the block between begin_mark and end_mark in tex_path."""
    with open(tex_path) as f:
        text = f.read()
    if begin_mark not in text or end_mark not in text:
        raise SystemExit(
            f"aaai_safe_wide markers not found in {tex_path}; expected lines "
            f"containing '{begin_mark}' and "
            f"'{end_mark}'")
    pre, rest = text.split(begin_mark, 1)
    _body, post = rest.split(end_mark, 1)
    updated = (f"{pre}{begin_mark}\n{body}\n"
               f"{end_mark}{post}")
    if updated == text:
        print("[update_advstd_tex_tables] no changes to aaai_safe_wide block")
        return
    with open(tex_path, "w") as f:
        f.write(updated)
    print(f"[update_advstd_tex_tables] wrote aaai_safe_wide block "
          f"in {tex_path}")


def regenerate_aaai_wide_perarch_section(tex_path, cwd, dataset, arch_runs,
                                          parse_result_file,
                                          seeds_filter=None,
                                          force_timeout=None,
                                          rerun_timeout_eps=30.0,
                                          roles=None,
                                          label_suffix="",
                                          begin_mark=AAAI_WIDE_BEGIN_MARK,
                                          end_mark=AAAI_WIDE_END_MARK):
    """Mirror regenerate_wide_perarch_section, but emit the slim 4-column
    AAAI variant into the neta_s_paper evaluation section. When
    `force_timeout` is set (in seconds), cells whose Gurobi run hit a
    termination limit under a different wall-clock cap are excluded.

    `roles`/`label_suffix`/`begin_mark`/`end_mark` select which network
    role to emit and where: the target-network (N2) tables go to the body
    with the default marks, and the source-network (N1) tables go to the
    appendix with roles={"N1"}, label_suffix="-n1", and the N1 marks."""
    try:
        rows = _collect_wide_perarch_cells(arch_runs, cwd, dataset,
                                            parse_result_file,
                                            seeds_filter=seeds_filter)
        archs = [a for a, _ in arch_runs]
        rows += _load_advstd_rows_for_wide(cwd, dataset, archs,
                                            seeds_filter=seeds_filter)
        delta_max_by_key = _load_delta_max_values(cwd, dataset, archs)
        delta_d_by_key = _load_delta_d_values(cwd, dataset, archs)
        body = _render_aaai_wide_perarch_body(
            rows, archs, dataset,
            delta_max_by_key=delta_max_by_key,
            delta_d_by_key=delta_d_by_key,
            force_timeout=force_timeout,
            rerun_timeout_eps=rerun_timeout_eps,
            roles=roles,
            label_suffix=label_suffix)
        update_aaai_wide_perarch_tex(tex_path, body,
                                     begin_mark=begin_mark,
                                     end_mark=end_mark)
    except SystemExit as exc:
        print(f"[update_advstd_tex_tables] aaai_safe_wide block skipped: "
              f"{exc}")
    except Exception as exc:
        print(f"[update_advstd_tex_tables] aaai_safe_wide block error: "
              f"{exc}")


# Combo keys for the summary aggregation. Must match _AAAI_WIDE_COLUMNS:
# 'vaghar' is the exact VHAGaR baseline, the transfer column is the
# adv+zono+prev_pgd+SibGate combo at tau=0.5.
_AAAI_SUMMARY_VAGHAR_COMBO   = "vaghar"
_AAAI_SUMMARY_TRANSFER_COMBO = "adv_zono_prevpgd_0.5+sg"
# A VHAGaR cell counts as "exact" (so its bound can anchor the bound gap)
# only when its MIP closed to within this relative gap. MIPGap is 0.01;
# 0.05 leaves margin for the mean over seeds/targets.
_AAAI_SUMMARY_EXACT_REL_GAP = 0.05

# Display names: internal arch key -> the name used in the paper body.
_AAAI_ARCH_DISPLAY = {
    "cnn1": r"\emph{conv1}",
    "cnn2": r"\emph{conv2}",
    "3x50": r"3$\times$50",
    "3x10": r"3$\times$10",
}


_AAAI_SUMMARY_NUM_CLASSES = {"mnist": 10, "cifar10": 10, "fashion_mnist": 10}


def _aaai_summary_stats(rows, archs, dataset="mnist", force_timeout=None,
                        rerun_timeout_eps=30.0):
    """Aggregate the per-cell rows into per-architecture summary metrics.

    Returns {arch: {"gap": float|None, "mean_sp": float|None,
                    "max_sp": float|None, "n_sp": int, "n_gap": int,
                    "star": bool}} plus an "__overall__" entry pooled across
    all archs.

    Bound gap (per cell) = 100*(transfer_ub - vaghar_ub)/vaghar_ub, kept only
    where the VHAGaR cell is exact (converged). Speedup (per cell) =
    vaghar_time / transfer_time, kept where both ran. Per arch the gap is the
    arithmetic mean and the speedup is the geometric mean (max is the max).

    "star" marks incomplete results: an architecture is starred when any of
    its three metrics has no data, or when a contributing cell did not cover
    every expected target class (same partial-coverage notion the per-cell
    appendix tables flag with a red asterisk on the time).
    """
    import math as _math
    from collections import defaultdict

    if force_timeout is not None:
        rows = [r for r in rows
                if not _aaai_is_timeout_mismatch(r, force_timeout,
                                                 rerun_timeout_eps)]

    n_classes = _AAAI_SUMMARY_NUM_CLASSES.get(dataset, 10)

    # Same bucketing as _render_aaai_wide_perarch_body, plus c_targets so we
    # can detect partial target-class coverage.
    buckets = defaultdict(lambda: defaultdict(
        lambda: {"t": [], "lb": [], "ub": [], "c_targets": set()}))
    for r in rows:
        key = (r["arch"], r["role"], r["perturbation"],
               r["perturbation_size"], r["c_source"])
        cell = buckets[key][r["combo"]]
        if r.get("t_total"):
            cell["t"].append(r["t_total"])
        lb = r.get("lb_total")
        if lb is not None and _math.isfinite(lb):
            cell["lb"].append(lb)
        ub = r.get("ub_total")
        if ub is not None and _math.isfinite(ub):
            cell["ub"].append(ub)
        ct = r.get("c_target")
        if ct is not None:
            try:
                cell["c_targets"].add(int(ct))
            except (TypeError, ValueError):
                pass

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    per_arch_sp = defaultdict(list)
    per_arch_gap = defaultdict(list)
    per_arch_partial = defaultdict(bool)
    for key, combos in buckets.items():
        arch, c_src = key[0], key[4]
        vag = combos.get(_AAAI_SUMMARY_VAGHAR_COMBO)
        tra = combos.get(_AAAI_SUMMARY_TRANSFER_COMBO)
        if not vag or not tra:
            continue
        try:
            expected = {ct for ct in range(n_classes) if ct != int(c_src)}
        except (TypeError, ValueError):
            expected = set()

        vt, tt = _mean(vag["t"]), _mean(tra["t"])
        contributes = False
        if vt and tt and tt > 0:
            per_arch_sp[arch].append(vt / tt)
            contributes = True
        vub, vlb, tub = _mean(vag["ub"]), _mean(vag["lb"]), _mean(tra["ub"])
        if (vub and vub > 0 and tub is not None and vlb is not None
                and (vub - vlb) / vub <= _AAAI_SUMMARY_EXACT_REL_GAP):
            per_arch_gap[arch].append(max(0.0, 100.0 * (tub - vub) / vub))
            contributes = True
        # A contributing cell is partial if either combo it used missed an
        # expected target class.
        if contributes and (
                (expected - vag["c_targets"]) or (expected - tra["c_targets"])):
            per_arch_partial[arch] = True

    def _geomean(xs):
        return _math.exp(sum(_math.log(x) for x in xs) / len(xs)) if xs else None

    out = {}
    all_sp, all_gap = [], []
    any_star = False
    for arch in archs:
        sp, gap = per_arch_sp.get(arch, []), per_arch_gap.get(arch, [])
        all_sp += sp
        all_gap += gap
        gap_m = _mean(gap)
        msp = _geomean(sp)
        xsp = max(sp) if sp else None
        star = (gap_m is None or msp is None or xsp is None
                or per_arch_partial.get(arch, False))
        any_star = any_star or star
        out[arch] = {"gap": gap_m, "mean_sp": msp, "max_sp": xsp,
                     "n_sp": len(sp), "n_gap": len(gap), "star": star}
    ov_gap, ov_msp = _mean(all_gap), _geomean(all_sp)
    ov_xsp = max(all_sp) if all_sp else None
    # Overall is starred if any architecture is incomplete, or it has no data.
    out["__overall__"] = {"gap": ov_gap, "mean_sp": ov_msp, "max_sp": ov_xsp,
                          "n_sp": len(all_sp), "n_gap": len(all_gap),
                          "star": any_star or ov_gap is None
                          or ov_msp is None or ov_xsp is None}
    return out


def _render_aaai_summary_body(stats, archs):
    """Render the data rows (conv1/3x50/3x10/Overall) for the body summary
    table from _aaai_summary_stats output. Missing values render as a red
    dash so an incomplete sweep is visible."""
    miss = r"\textcolor{red}{--}"

    def _gap(v):
        return _fmt_sig(v) if v is not None else miss

    def _sp(v):
        return (_fmt_sig(v) + r"$\times$") if v is not None else (miss + r"$\times$")

    def _row(label, s):
        # A red asterisk on the network name marks incomplete results.
        if s.get("star"):
            label = label + r"\textcolor{red}{$^*$}"
        return (f"{label} & {_gap(s['gap'])} & {_sp(s['mean_sp'])} "
                f"& {_sp(s['max_sp'])} \\\\")

    lines = []
    for arch in archs:
        label = _AAAI_ARCH_DISPLAY.get(arch, arch.replace("_", r"\_"))
        lines.append(_row(label, stats[arch]))
    lines.append(r"\midrule")
    lines.append(_row("Overall", stats["__overall__"]))
    return "\n".join(lines)


def update_aaai_summary_tex(tex_path, body):
    """Rewrite the % BEGIN AUTO: aaai_summary_table block in tex_path."""
    with open(tex_path) as f:
        text = f.read()
    if (AAAI_SUMMARY_BEGIN_MARK not in text
            or AAAI_SUMMARY_END_MARK not in text):
        raise SystemExit(
            f"aaai_summary markers not found in {tex_path}; expected lines "
            f"containing '{AAAI_SUMMARY_BEGIN_MARK}' and "
            f"'{AAAI_SUMMARY_END_MARK}'")
    pre, rest = text.split(AAAI_SUMMARY_BEGIN_MARK, 1)
    _body, post = rest.split(AAAI_SUMMARY_END_MARK, 1)
    updated = (f"{pre}{AAAI_SUMMARY_BEGIN_MARK}\n{body}\n"
               f"{AAAI_SUMMARY_END_MARK}{post}")
    if updated == text:
        print("[update_advstd_tex_tables] no changes to aaai_summary block")
        return
    with open(tex_path, "w") as f:
        f.write(updated)
    print(f"[update_advstd_tex_tables] wrote aaai_summary block "
          f"in {tex_path}")


def regenerate_aaai_summary_section(tex_path, cwd, dataset, arch_runs,
                                    parse_result_file, seeds_filter=None,
                                    force_timeout=None,
                                    rerun_timeout_eps=30.0):
    """Compute the per-architecture summary (Table 2) from the same per-cell
    rows as the appendix tables and rewrite the aaai_summary AUTO block."""
    try:
        rows = _collect_wide_perarch_cells(arch_runs, cwd, dataset,
                                           parse_result_file,
                                           seeds_filter=seeds_filter)
        archs = [a for a, _ in arch_runs]
        rows += _load_advstd_rows_for_wide(cwd, dataset, archs,
                                           seeds_filter=seeds_filter)
        stats = _aaai_summary_stats(rows, archs, dataset=dataset,
                                    force_timeout=force_timeout,
                                    rerun_timeout_eps=rerun_timeout_eps)
        body = _render_aaai_summary_body(stats, archs)
        update_aaai_summary_tex(tex_path, body)
    except SystemExit as exc:
        print(f"[update_advstd_tex_tables] aaai_summary block skipped: "
              f"{exc}")
    except Exception as exc:
        print(f"[update_advstd_tex_tables] aaai_summary block error: "
              f"{exc}")


def _load_advstd_rows_for_wide(cwd, dataset, archs, seeds_filter=None,
                                vs_with_perturbed=True):
    """Read the advstd per-cell CSVs and return rows in the same shape as
    _collect_wide_perarch_cells, filtered to the two advstd N2 combos
    listed in _ADVSTD_WIDE_COMBOS. role is always 'N2' because advstd is
    transfer mode (N1 -> N2)."""
    suffix = "_vs_withPerturbed" if vs_with_perturbed else ""
    combined_base = os.path.join(cwd, "paper_experiments", dataset)
    csv_paths = [
        os.path.join(combined_base,
                     f"advstd_faster_than_standard{suffix}.csv"),
        os.path.join(combined_base,
                     f"standard_faster_than_advstd{suffix}.csv"),
        os.path.join(combined_base,
                     f"advstd_tighter_at_timeout{suffix}.csv"),
        os.path.join(combined_base,
                     f"standard_tighter_at_timeout_vs_advstd{suffix}.csv"),
    ]
    seeds_filter = set(str(s) for s in seeds_filter) if seeds_filter else None
    arch_set = set(archs) if archs else None
    rows = []
    for p in csv_paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if not r.get("arch"):
                    continue
                if arch_set is not None and r["arch"] not in arch_set:
                    continue
                if seeds_filter and r.get("seed") not in seeds_filter:
                    continue
                if (r.get("bound_tightening") != "yes"
                        or r.get("branch_priorities") != "off"
                        or r.get("n1_probe", "off") != "off"):
                    continue
                combo_label = None
                for label, zb, vh, rt, sg in _ADVSTD_WIDE_COMBOS:
                    if (r.get("zono_bounds") == zb
                            and r.get("var_hint") == vh
                            and r.get("relax_threshold") == rt
                            and r.get("sibling_gate", "no") == sg):
                        combo_label = label
                        break
                if combo_label is None:
                    continue
                try:
                    t = float(r["time_advstd"])
                except (KeyError, ValueError, TypeError):
                    continue
                if t <= 0:
                    continue
                # Skip INTERRUPTED runs (same rationale as
                # _collect_wide_perarch_cells).
                if (r.get("solve_status_advstd", "") or "").upper() == "INTERRUPTED":
                    continue
                try:
                    cs_val = int(r["c_source"])
                    ct_val = int(r["c_target"])
                except (KeyError, ValueError, TypeError):
                    continue
                def _f(k):
                    try:
                        x = float(r.get(k, ""))
                        return x if math.isfinite(x) else None
                    except (ValueError, TypeError):
                        return None
                rows.append({
                    "arch": r["arch"],
                    "role": "N2",
                    "perturbation": r["perturbation"],
                    "perturbation_size": r["perturbation_size"],
                    "c_source": cs_val,
                    "c_target": ct_val,
                    "seed": r.get("seed", "0"),
                    "combo": combo_label,
                    "t_total": t,
                    "lb_total": _f("delta_advstd_lower_bound"),
                    "ub_total": _f("delta_advstd_upper_bound"),
                    "solve_status": r.get("solve_status_advstd", ""),
                    # geometric_intervals twin flag from the advstd CSV's
                    # geom column ("yes"/"no") -> paired as base,geom in time.
                    "geom": (str(r.get("geom", "no")) == "yes"),
                })
    return rows


def regenerate_wide_perarch_section(tex_path, cwd, dataset, arch_runs,
                                     parse_result_file, seeds_filter=None):
    """End-to-end: scan vaghar* + stdBoost results, layer in two advstd
    N2 combos from the per-cell CSVs, render the per-arch wide
    comparison body, rewrite the AUTO block in tex_path."""
    try:
        rows = _collect_wide_perarch_cells(arch_runs, cwd, dataset,
                                            parse_result_file,
                                            seeds_filter=seeds_filter)
        archs = [a for a, _ in arch_runs]
        rows += _load_advstd_rows_for_wide(cwd, dataset, archs,
                                            seeds_filter=seeds_filter)
        delta_max_by_key = _load_delta_max_values(cwd, dataset, archs)
        delta_d_by_key = _load_delta_d_values(cwd, dataset, archs)
        body = _render_wide_perarch_body(rows, archs, dataset,
                                          delta_max_by_key=delta_max_by_key,
                                          delta_d_by_key=delta_d_by_key)
        update_wide_perarch_tex(tex_path, body)
    except SystemExit as exc:
        print(f"[update_advstd_tex_tables] wide_perarch block skipped: "
              f"{exc}")
    except Exception as exc:
        print(f"[update_advstd_tex_tables] wide_perarch block error: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=DEFAULT_TEX)
    ap.add_argument("--csv_dir", default=DEFAULT_CSV_DIR)
    ap.add_argument("--seed", default=None,
                    help="filter to a single seed (default: all seeds)")
    ap.add_argument("--tau", default=None,
                    help="filter to a single relax_threshold "
                         "(default: all thresholds)")
    ap.add_argument("--archs", nargs="+", default=None,
                    help="archs to emit tables for (default: infer from CSV)")
    ap.add_argument("--vs_with_perturbed", action="store_true", default=True)
    ap.add_argument("--no_vs_with_perturbed", dest="vs_with_perturbed",
                    action="store_false")
    ap.add_argument("--combination_table", default=None,
                    metavar="BT:VH:TAU[,BT:VH:TAU,...]",
                    help="Restrict every advstd table (overall ranking, "
                         "per-arch perturbation tables, and timeout-gap "
                         "tables) to one or more combos. Format "
                         "'<bound_tight>:<varHint>:<tau>' per combo, "
                         "comma-separated. Examples: 'zono:prev_pgd:0.5' "
                         "or 'interval:prev_pgd:0.5+sg,"
                         "zono:prev_pgd:0.5+sg'. Use the '+sg' suffix on "
                         "tau to select SibGate (Technique 4) rows; use "
                         "'off' as a varHint alias for 'no'.")
    args = ap.parse_args()

    suffix = "_vs_withPerturbed" if args.vs_with_perturbed else ""
    rows = load_rows(args.csv_dir, suffix)
    if not rows:
        raise SystemExit(
            f"no rows loaded from {args.csv_dir} (suffix='{suffix}')")

    archs = args.archs
    if archs is None:
        archs = sorted({r["arch"] for r in rows if r.get("arch")})
    combination_filter = parse_combination_spec(args.combination_table)
    body = render_all(archs, rows, args.seed, args.tau,
                      combination_filter=combination_filter)
    update_tex(args.tex, body)


if __name__ == "__main__":
    main()
