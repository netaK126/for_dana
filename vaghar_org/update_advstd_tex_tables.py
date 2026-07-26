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

    Two accepted forms:
      - '<bt>:<varHint>[+sg]'        — tau-free. The threshold is supplied
                                       separately (--paper_taus), so the two
                                       flags stay orthogonal: this one picks
                                       the technique, that one picks the
                                       thresholds. Expanded to one combo per
                                       tau by expand_combination_spec_taus.
      - '<bt>:<varHint>:<tau>[+sg]'  — explicit tau, as before.
    Examples:
      - 'zono:prev_pgd+sg'                              (tau from --paper_taus)
      - 'interval:prev_pgd:0.5+sg,zono:prev_pgd:0.5+sg' (explicit)
    Per-combo fields:
      - bt   ∈ {none, interval, zono, interval+lp, zono+lp} — matches the
               bound_tight column rendered in the per-arch tables.
      - vh   ∈ {no, off, prev, direct, direct_pgd, prev_pgd} — matches the
               CSV var_hint column verbatim ('off' is accepted as an alias
               for 'no').
      - tau  ∈ {off, 0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0}, optionally with
               '+sg' suffix for SibGate (Technique 4). In the tau-free form
               '+sg' rides on varHint instead.
    Returns None if spec is empty / None; otherwise a list of
    (bt, vh, tau) tuples suitable for membership testing against
    _combo_label. A tau-free combo carries tau '*' (or '*+sg'), a wildcard
    that expand_combination_spec_taus resolves before any matching happens.
    """
    if not spec:
        return None
    combos = []
    for combo_str in str(spec).split(","):
        combo_str = combo_str.strip()
        if not combo_str:
            continue
        parts = [p.strip() for p in combo_str.split(":")]
        if len(parts) == 2:
            # Tau-free form: '+sg' is attached to varHint, so move it onto the
            # wildcard tau where the rest of the pipeline expects to find it.
            bt, vh = parts
            sg = "+sg" if vh.endswith("+sg") else ""
            vh = vh[:-3] if sg else vh
            tau = "*" + sg
        elif len(parts) == 3:
            bt, vh, tau = parts
        else:
            raise SystemExit(
                "--combination_table: expected "
                "'bt:varHint[+sg]' or 'bt:varHint:tau[+sg]' "
                "(comma-separated), "
                f"got {spec!r}")
        if vh.lower() == "off":
            vh = "no"
        combo = (bt, vh, tau)
        if combo not in combos:
            combos.append(combo)
    if not combos:
        return None
    return combos


def expand_combination_spec_taus(combos, taus):
    """Resolve tau-wildcard combos ('zono:prev_pgd+sg') into one concrete combo
    per tau, so every downstream matcher keeps doing a plain membership test
    against _combo_label output and needs no wildcard awareness.

    This is what keeps --combination_table and --paper_taus from overlapping:
    the spec names the technique, the tau list names the thresholds, and the
    admitted set is their cross product. Combos with an explicit tau pass
    through untouched."""
    if not combos:
        return combos
    out = []
    for bt, vh, tau in combos:
        if not str(tau).startswith("*"):
            if (bt, vh, tau) not in out:
                out.append((bt, vh, tau))
            continue
        sg = "+sg" if str(tau).endswith("+sg") else ""
        for t in taus:
            combo = (bt, vh, f"{t}{sg}")
            if combo not in out:
                out.append(combo)
    return out


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


def _suffix_block_labels(body, label_suffix):
    r"""Append `label_suffix` to every ``\label{...}`` defined in `body`, and
    to every ``\ref``/``\cref``/``\Cref``/``\autoref``/``\eqref`` that points at
    one of those same labels. This lets an auto-generated block be duplicated
    per dataset (MNIST + Fashion-MNIST) without LaTeX "multiply-defined label"
    collisions. No-op when the suffix is empty, so the MNIST/default path stays
    byte-identical."""
    if not label_suffix:
        return body
    import re as _re
    labels = set(_re.findall(r"\\label\{([^}]*)\}", body))
    if not labels:
        return body

    def _repl(m):
        cmd, name = m.group(1), m.group(2)
        if name in labels:
            return f"\\{cmd}{{{name}{label_suffix}}}"
        return m.group(0)

    return _re.sub(r"\\(label|ref|cref|Cref|autoref|eqref)\{([^}]*)\}",
                   _repl, body)


def update_tex(tex_path, new_body, begin_mark=BEGIN_MARK, end_mark=END_MARK,
               label_suffix=""):
    with open(tex_path) as f:
        text = f.read()
    if begin_mark not in text or end_mark not in text:
        raise SystemExit(
            f"markers not found in {tex_path}; expected lines containing "
            f"'{begin_mark}' and '{end_mark}'")
    pre, rest = text.split(begin_mark, 1)
    _body, post = rest.split(end_mark, 1)
    new_body = _suffix_block_labels(new_body, label_suffix)
    updated = f"{pre}{begin_mark}\n{new_body}\n{end_mark}{post}"
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
    # Leave-one-out ablation runs (--stdboost_ablations, filename marker
    # _ablation) are excluded from the nn-boost tables: their flag combos can
    # coincide with legitimate grid columns (e.g. zono+pi+tau=0), which would
    # silently mix ablation rows into paper cells.
    if "_ablation" in name:
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
    # _sgd_itr is the image pipeline's N2 tag; _int8 is the benchmark nets'
    # (acas/har), whose N2 comes from reduced weight precision, not extra SGD.
    if base.startswith("N2stdBoost_") or any(s in base for s in ("_sgd_itr", "_int8")):
        return "N2"
    return "N1"


def _is_baseline_stdname(name):
    """Match the with-perturbed-intervals baseline file used as the
    nn1-boost comparison reference. The baseline carries _PertruebedIntervals
    (sic — the run.jl tag is misspelled and we mirror it) and NO _stdBoost_
    tag. _HyperAttackHints and _VagharDeps may decorate either."""
    return "_PertruebedIntervals" in name and "_stdBoost_" not in name


# The real perturbation directory names under paper_experiments/{ds}/{arch}_exp.
# Every paper-table/chart loader discovers perturbations by scanning the
# filesystem for these dirs (then globbing their eps_* subdirs), so the tables
# reflect whatever results exist on disk. This is intentionally DECOUPLED from
# run_relaxation_sweep.PERTURBATIONS, which only schedules which jobs to launch:
# commenting a perturbation out of that list (e.g. to run fewer jobs at once)
# must never drop already-computed results from the tables/charts. Anything not
# in this list (deprecated/, patch_old/, delta_max/, model_seed*/, ...) is
# ignored.
_WIDE_PERT_SUBDIRS = ["patch", "occ", "translation", "rotation",
                      "brightness", "linf", "contrast"]


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

    pert_subdirs = _WIDE_PERT_SUBDIRS
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


def update_nn1_tex(tex_path, body, begin_mark=NN1_BEGIN_MARK,
                   end_mark=NN1_END_MARK, label_suffix=""):
    """Rewrite the % BEGIN AUTO: nn1_safe_tables block in tex_path.
    Mirrors update_tex() but for the nn1 markers."""
    with open(tex_path) as f:
        text = f.read()
    if begin_mark not in text or end_mark not in text:
        raise SystemExit(
            f"nn1 markers not found in {tex_path}; expected lines "
            f"containing '{begin_mark}' and '{end_mark}'")
    pre, rest = text.split(begin_mark, 1)
    _body, post = rest.split(end_mark, 1)
    body = _suffix_block_labels(body, label_suffix)
    updated = f"{pre}{begin_mark}\n{body}\n{end_mark}{post}"
    if updated == text:
        print("[update_advstd_tex_tables] no changes to nn1 block")
        return
    with open(tex_path, "w") as f:
        f.write(updated)
    print(f"[update_advstd_tex_tables] wrote nn1 block in {tex_path}")


def regenerate_nn1_section(tex_path, cwd, dataset, arch_runs,
                            parse_result_file, seeds_filter=None,
                            begin_mark=NN1_BEGIN_MARK, end_mark=NN1_END_MARK,
                            ds_label_suffix=""):
    """End-to-end: scan _stdBoost_* results, render the nn1-safe body,
    rewrite the AUTO block in tex_path. Called from run_relaxation_sweep.py
    immediately after the advstd safe-tables block is regenerated."""
    try:
        rows = _collect_stdboost_cells(arch_runs, cwd, dataset,
                                       parse_result_file,
                                       seeds_filter=seeds_filter)
        archs = [a for a, _ in arch_runs]
        body = _render_nn1_body(rows, archs)
        update_nn1_tex(tex_path, body, begin_mark=begin_mark,
                       end_mark=end_mark, label_suffix=ds_label_suffix)
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


def _advstd_wide_combos():
    """_ADVSTD_WIDE_COMBOS widened to every tau the per-cell tables currently
    render (--paper_taus), so a threshold beyond the built-in 0.0 / 0.5 pair
    (e.g. 0.25) is admitted by the .txt loader instead of being filtered out
    before it can reach a row. Evaluated per call because --paper_taus is
    applied at runtime; a tau with no result files just contributes nothing."""
    combos = list(_ADVSTD_WIDE_COMBOS)
    known = {c[3] for c in combos}
    # Union of the table taus and the chart taus: a threshold drawn in a figure
    # must be loadable even if it is not rendered as a table row, and vice
    # versa.
    for tau in tuple(_AAAI_WIDE_TAUS) + tuple(_AAAI_CHART_TAUS):
        if tau not in known:
            combos.append((f"adv_zono_prevpgd_{tau}+sg",
                           "yes", "prev_pgd", tau, "yes"))
    return tuple(combos)


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


def _wide_row_file_rank(row):
    """Sort key deciding WHICH result file wins when several of them carry a
    line for the same (column, c_source, c_target) cell. Prefer a file from a
    stdBoost_*/advStd_* dir over one from a vaghar{No,With}Perturbed_*
    baseline dir, then the most recent run (the leading timestamp in the file
    name), then the name itself so the choice never depends on glob order."""
    fname = row.get("src_file") or ""
    m = re.match(r"(\d+)", fname)
    stamp = int(m.group(1)) if m else -1
    return (1 if row.get("src_baseline_dir") else 0, -stamp, fname)


def _dedupe_wide_rows(rows):
    """Keep ONE row per (arch, role, perturbation, size, combo, c_source,
    c_target, geom) cell.

    A c_target that was solved more than once -- a re-run after a timeout, or
    a cell that exists both in a vaghar{No,With}Perturbed_* dir and as the
    pooled 'all-off' stdBoost file -- otherwise contributes one row per file.
    The column mean is then taken over an unbalanced multiset: the repeated
    target carries several times the weight of its siblings while the other
    columns still average one run per target, so the columns no longer
    describe the same set of experiments. That is what let delta_l of a
    column whose slow target was run three times exceed delta_u of a column
    that ran each target once (mnist / conv1 / patch, c_s=0).

    `seed` is deliberately NOT part of the key: the baseline dirs bypass
    --combo_ranking_seeds, so a seed-0 baseline file and the seed-filtered
    boost file describe the SAME cell and must not both survive."""
    best = {}
    for r in rows:
        key = (r.get("arch"), r.get("role"), r.get("perturbation"),
               r.get("perturbation_size"), r.get("combo"),
               r.get("c_source"), r.get("c_target"), bool(r.get("geom")))
        cur = best.get(key)
        if cur is None or _wide_row_file_rank(r) < _wide_row_file_rank(cur):
            best[key] = r
    return list(best.values())


def _collect_wide_perarch_cells(arch_runs, cwd, dataset, parse_result_file,
                                 seeds_filter=None, stale_fn=None):
    """Walk each arch's results dir and yield per-cell timing rows for the
    per-arch wide comparison section. One row per (.txt file, cs, ct) cell.
    The combo field is 'vaghar', 'PI', or one of the 8 boost masks.

    `stale_fn` (run_relaxation_sweep._is_pre_fix_dropped) drops files made
    unsound by the perturbation-dependency fix -- a pre-fix file that relaxed
    >=1 binary. None disables the gate (keeps the prior behavior). Applies to
    the relaxed combos (e.g. "ours" at tau=0.5); all-off/vaghar files dropped
    no binary so they pass through unchanged.
    """
    import glob
    rows = []
    seeds_filter = set(str(s) for s in seeds_filter) if seeds_filter else None

    pert_subdirs = _WIDE_PERT_SUBDIRS
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
                        # Drop pre-fix files that relaxed >=1 binary (unsound
                        # under the perturbation-dependency fix). all-off/vaghar
                        # files dropped nothing, so they pass.
                        if stale_fn is not None and stale_fn(fname):
                            continue
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
                                # Gurobi solver wall-clock only (no hyper-attack
                                # overhead); used for the timeout-cap dedup.
                                "t_opt": val.get("optimization_time"),
                                "lb_total": val.get("lower_bound"),
                                "ub_total": val.get("upper_bound"),
                                "solve_status": val.get("solve_status", ""),
                                # Binaries the BoundTightPertRelax gate dropped,
                                # per copy. Constant across c_targets within a
                                # file (the gate reads only the bounds, which do
                                # not depend on the target class); carried per
                                # row so the relaxation table pairs them under
                                # the same filters every other table applies.
                                "relaxed_org": _as_int(
                                    val.get("n2_org_relaxed_binaries")),
                                "relaxed_pert": _as_int(
                                    val.get("n2_pert_relaxed_binaries")),
                                # geometric_intervals twin (same cell, same
                                # delta, faster/slower solve): paired as base,geom
                                # in the time column by the renderer.
                                "geom": ("_geomInt" in fname),
                                # Which file this line came from, and whether
                                # that file sits in a vaghar{No,With}Perturbed_*
                                # baseline dir. Both feed _dedupe_wide_rows.
                                "src_file": fname,
                                "src_baseline_dir": (src_kind != "boost"),
                            })
    return _dedupe_wide_rows(rows)


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
    "cnn5": "CNN5",   # paper "conv4"
}


_FASHION_DATASET_ALIASES = frozenset(
    {"fashion-mnist", "fashion_mnist", "fashion", "fmnist"})

_CIFAR_DATASET_ALIASES = frozenset({"cifar", "cifar10", "cifar-10"})


# Output-class count per dataset, for the "expected c_target" set behind the
# red partial-row "*". The image datasets all have 10; the pretrained tabular
# benchmark nets do not (ACAS Xu has 5 advisories, HAR 6 activities), and a
# missing key used to fall back to 10 -- which marked every acas/har row
# partial and, under _AAAI_WIDE_DROP_PARTIAL_ROWS, dropped it from the
# appendix entirely. Mirrors _num_classes_for in run_relaxation_sweep.py.
_DATASET_NUM_CLASSES = {"mnist": 10, "cifar10": 10, "cifar": 10,
                        "fashion_mnist": 10, "fashion-mnist": 10,
                        "acas": 5, "har": 6}


def _num_classes_for_dataset(dataset):
    """Output-class count for `dataset`, defaulting to 10 for image datasets."""
    return _DATASET_NUM_CLASSES.get(dataset, 10)


def _dataset_display_name(dataset):
    """Human-readable dataset name for captions, tolerant of the
    hyphen/underscore spellings used across this repo."""
    return {"mnist": "MNIST",
            "fashion_mnist": "Fashion-MNIST", "fashion-mnist": "Fashion-MNIST",
            "fashion": "Fashion-MNIST", "fmnist": "Fashion-MNIST",
            "cifar10": "CIFAR-10", "cifar": "CIFAR-10",
            "acas": "ACAS Xu", "har": "HAR"}.get(dataset, dataset)


def _torchvision_dataset_for(dataset):
    """(torchvision folder name, torchvision dataset class name) for the
    given (possibly aliased) dataset name. Fashion-MNIST lives under
    {root}/FashionMNIST/raw/*; plain MNIST under {root}/MNIST/raw/*;
    CIFAR-10 under {root}/cifar-10-batches-py/*."""
    if dataset in _FASHION_DATASET_ALIASES:
        return "FashionMNIST", "FashionMNIST"
    if dataset in _CIFAR_DATASET_ALIASES:
        return "cifar-10-batches-py", "CIFAR10"
    return "MNIST", "MNIST"


def _dataset_input_dims(dataset):
    """(channels, width, height) of the dataset's images, used to build a model
    with the correct input shape before loading its state_dict. MNIST and
    Fashion-MNIST are 1x28x28 (the model default); CIFAR-10 is 3x32x32."""
    if dataset in _CIFAR_DATASET_ALIASES:
        return 3, 32, 32
    return 1, 28, 28


def _find_mnist_data_root(cwd, dataset="mnist"):
    """Return a directory containing {Folder}/raw/{train,t10k}-*-ubyte so
    torchvision.datasets.{MNIST,FashionMNIST}(root=..., download=False)
    succeeds. torchvision expects the layout {root}/{Folder}/raw/* where
    Folder is 'MNIST' for mnist and 'FashionMNIST' for Fashion-MNIST."""
    folder, _cls = _torchvision_dataset_for(dataset)
    cands = [
        os.path.join(cwd, "data"),
        os.path.dirname(os.path.dirname(cwd)),  # for_dana/
        cwd,
        os.path.join(cwd, "..", folder),
    ]
    # CIFAR-10 uses torchvision's {root}/cifar-10-batches-py/ layout (pickled
    # batches), not the MNIST {Folder}/raw/*-ubyte layout.
    if dataset in _CIFAR_DATASET_ALIASES:
        for c in cands:
            if os.path.isfile(os.path.join(c, "cifar-10-batches-py",
                                           "batches.meta")):
                return c
        return None
    for c in cands:
        if os.path.isfile(os.path.join(c, folder, "raw",
                                       "train-images-idx3-ubyte")):
            return c
    return None


def _compute_delta_d_har_tabular(model_pth_path, c_srcs, cwd):
    """delta_d for the HAR tabular benchmark net. HAR is not a torchvision image
    dataset -- it is the UCI "Human Activity Recognition Using Smartphones" set
    (Anguita 2013), 561 numeric features already scaled to [-1,1] -- so it has
    its own forward pass over data/UCI HAR Dataset/{train,test}/X_*.txt instead
    of the ToTensor/image-DataLoader path. Returns
        {c_src_0indexed: max_x (f(x)[c_src] - max_{k!=c_src} f(x)[k])}
    over train+test, or None if torch / the data / the model is unavailable.
    The net is fc1(561->500) -> ReLU -> fc2(500->6), reconstructed straight from
    the state_dict (no model class needed)."""
    try:
        import numpy as np
        import torch
    except ImportError:
        return None
    data_dir = os.path.join(cwd, "data", "UCI HAR Dataset")
    x_parts = []
    for rel in ("train/X_train.txt", "test/X_test.txt"):
        p = os.path.join(data_dir, rel)
        if os.path.isfile(p):
            try:
                x_parts.append(np.loadtxt(p))
            except ValueError:
                return None
    if not x_parts:
        return None
    x = np.vstack(x_parts)
    try:
        sd = torch.load(model_pth_path, map_location="cpu")
        w1, b1 = sd["fc1.weight"].cpu().numpy(), sd["fc1.bias"].cpu().numpy()
        w2, b2 = sd["fc2.weight"].cpu().numpy(), sd["fc2.bias"].cpu().numpy()
    except Exception:
        return None
    if x.shape[1] != w1.shape[1]:
        return None
    out = np.maximum(0.0, x @ w1.T + b1) @ w2.T + b2   # (N, n_classes)
    n_cls = out.shape[1]
    deltas = {}
    for c in c_srcs:
        if 0 <= c < n_cls:
            other_max = np.delete(out, c, axis=1).max(axis=1)
            deltas[c] = float((out[:, c] - other_max).max())
    return deltas


def _compute_delta_d_for_arch_role(arch, model_pth_path, c_srcs, cwd,
                                   dataset="mnist"):
    """Forward-pass the dataset's train+test through the .pth model and return
        {c_src_0indexed: max_x (N(x)[c_src] - max_{k!=c_src} N(x)[k])}
    over both dataset splits. Returns None if PyTorch/the data/the model
    file isn't available — caller treats that as 'unknown' and renders '?'.
    """
    if not os.path.isfile(model_pth_path):
        return None
    # HAR is a tabular benchmark net (UCI HAR 561-feature vectors), not a
    # torchvision image dataset, so it takes the tabular forward-pass path.
    if dataset == "har" or arch == "har":
        return _compute_delta_d_har_tabular(model_pth_path, c_srcs, cwd)
    try:
        import torch
        import torchvision.datasets as dsets
        import torchvision.transforms as transforms
    except ImportError:
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
    data_root = _find_mnist_data_root(cwd, dataset)
    if data_root is None:
        return None
    try:
        state = torch.load(model_pth_path, map_location="cpu")
        # Build with the dataset's real input shape: MNIST/Fashion default to
        # 1x28x28, but CIFAR-10 is 3x32x32, so cls() (the MNIST default) would
        # size-mismatch the CIFAR state_dict. All arch classes share the
        # (k, w, h, output_size) signature, so this is safe for every arch.
        k, w, h = _dataset_input_dims(dataset)
        model = cls(k=k, w=w, h=h)
        model.load_state_dict(state)
        model.eval()
    except Exception:
        return None
    tx = transforms.Compose([transforms.ToTensor()])
    _folder, ds_cls_name = _torchvision_dataset_for(dataset)
    ds_cls = getattr(dsets, ds_cls_name, None)
    if ds_cls is None:
        return None
    try:
        train_ds = ds_cls(root=data_root, train=True,
                          transform=tx, download=False)
        test_ds = ds_cls(root=data_root, train=False,
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


def _discover_model_role_tags(arch_root):
    """Fallback (role, tag) discovery for delta_d when the delta_max sibling
    dir is absent (e.g. CIFAR, whose delta_max pre-phase has not run). Scans
    model dirs holding a model.pth directly under {arch}_exp and classifies
    each by the repo's N1/N2 naming: an "_sgd_itr" suffix marks the
    SGD-boosted target network N2, the base model is the source N1. Skips the
    _collapsed_bak / _stale / _try scratch dirs. Returns [(role, tag), ...]."""
    import glob as _glob
    out = []
    for d in sorted(_glob.glob(os.path.join(arch_root, "model_*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if any(s in name for s in
               ("_collapsed_bak", "_stale", "_try", "_bak")):
            continue
        # Flat layout {tag}/model.pth, or nested {tag}/*/model.pth.
        if not (os.path.isfile(os.path.join(d, "model.pth"))
                or _glob.glob(os.path.join(d, "*", "model.pth"))):
            continue
        role = "N2" if any(s in name for s in ("_sgd_itr", "_int8")) else "N1"
        out.append((role, name))
    return out


def _load_delta_d_values(cwd, dataset, archs, c_srcs=range(10)):
    """For each (arch, role, tag), compute (or load from JSON cache) the
    dataset-empirical
        delta_d(c_src) = max over dataset(train+test) of
                         N(x)[c_src] - max_{k!=c_src} N(x)[k]
    using the model.pth at
        paper_experiments/{dataset}/{arch}_exp/{tag}/model.pth.
    Cache lives at
        paper_experiments/{dataset}/{arch}_exp/delta_d/
            delta_d_{arch}_{role}_{tag}.json.
    Returns {(arch, role, c_src_0indexed): float}.

    (role, tag) pairs come from the delta_max sibling dir when it exists (the
    cached path used for MNIST/Fashion). delta_d itself needs only the model
    .pth -- not delta_max -- so when that dir is absent (e.g. CIFAR, whose
    delta_max pre-phase has not run) we fall back to discovering the model dirs
    directly, letting the tables show a real delta_d even where delta_max is
    still '?'.
    """
    import glob, json
    out = {}
    c_srcs_list = list(c_srcs)
    for arch in archs:
        arch_root = os.path.join(cwd, "paper_experiments", dataset,
                                 f"{arch}_exp")
        dm_base = os.path.join(arch_root, "delta_max")
        role_tags = []
        if os.path.isdir(dm_base):
            for role in ("N1", "N2"):
                for d in glob.glob(os.path.join(
                        dm_base, f"delta_max_{arch}_{role}_*")):
                    tag = os.path.basename(d)[len(f"delta_max_{arch}_{role}_"):]
                    if tag:
                        role_tags.append((role, tag))
        if not role_tags:
            role_tags = _discover_model_role_tags(arch_root)
        if not role_tags:
            continue
        cache_base = os.path.join(arch_root, "delta_d")
        os.makedirs(cache_base, exist_ok=True)
        for role, tag in role_tags:
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
                #   cnn1/cnn2: {arch}_exp/{tag}/model.pth
                #     (tag is e.g. model_seed42_itr19_sgd_itr1)
                #   3x10/3x50: {arch}_exp/model_seed42_itr20/{tag}/model.pth
                #     (tag is an SGD iteration like 19 or 19_sgd_itr1)
                # Try the flat layout first, then the nested fallback, then any
                # model.pth one level deeper as a last resort.
                import glob as _glob
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
                    arch, model_pth, c_srcs_list, cwd, dataset)
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
        _ro, _rp = r.get("relaxed_org"), r.get("relaxed_pert")
        if _ro is not None or _rp is not None:
            cell["relaxed"].append((_ro or 0) + (_rp or 0))
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
            # The benchmark nets are not 10-class: HAR classifies 6 activities
            # and ACAS Xu 5 advisories, so assuming 10 would call a complete
            # sweep partial.
            _DATASET_NUM_CLASSES = {"mnist": 10, "cifar10": 10,
                                     "fashion_mnist": 10,
                                     "har": 6, "acas": 5}
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
                    # Both delta_l and delta_u are percentages of delta_max, so
                    # neither should render above 100% (delta_max is the max) or
                    # below 0. Cap BOTH to [0, 100]; capping only delta_u (as
                    # before) let a cell whose bounds both sit just above
                    # delta_max show delta_l > delta_u. Display-only; the gap
                    # winner logic above uses the pre-clamp lb_pct/ub_pct.
                    row.append(_paint(
                        f"{min(100.0, max(0.0, s['lb_pct'])):.1f}" if "lb_pct" in s else "---",
                        is_win))
                    row.append(_paint(
                        f"{min(100.0, max(0.0, s['ub_pct'])):.1f}" if "ub_pct" in s else "---",
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
               f"the $N$ block since advstd is a transfer technique. "
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


def update_wide_perarch_tex(tex_path, body, begin_mark=WIDE_BEGIN_MARK,
                            end_mark=WIDE_END_MARK, label_suffix=""):
    """Rewrite the % BEGIN AUTO: wide_perarch_tables block in tex_path."""
    with open(tex_path) as f:
        text = f.read()
    if begin_mark not in text or end_mark not in text:
        raise SystemExit(
            f"wide_perarch markers not found in {tex_path}; expected lines "
            f"containing '{begin_mark}' and '{end_mark}'")
    pre, rest = text.split(begin_mark, 1)
    _body, post = rest.split(end_mark, 1)
    body = _suffix_block_labels(body, label_suffix)
    updated = f"{pre}{begin_mark}\n{body}\n{end_mark}{post}"
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
# (sec_appendix_percell.tex).
AAAI_WIDE_N1_BEGIN_MARK = "% BEGIN AUTO: aaai_safe_wide_n1_tables"
AAAI_WIDE_N1_END_MARK   = "% END AUTO: aaai_safe_wide_n1_tables"

# Target-network (N2) per-cell tables relocated to the appendix
# (sec_appendix_percell.tex), alongside the N1 tables. The evaluation
# body shows the N2 time charts (below) in their place; the full per-cell
# N2 bounds live here.
AAAI_WIDE_N2_APPENDIX_BEGIN_MARK = "% BEGIN AUTO: aaai_safe_wide_n2_appendix_tables"
AAAI_WIDE_N2_APPENDIX_END_MARK   = "% END AUTO: aaai_safe_wide_n2_appendix_tables"

# When True, regenerate_aaai_wide_perarch_section DROPS every row that would
# carry the red partial-coverage "*" in any of its time sub-cells, so the
# rendered table shows only rows averaged over every expected c_target. The
# rows are discarded before the model label is distributed, so the left-hand
# label column re-flows over exactly the surviving rows (no dangling
# delta_max / delta_d line, no empty block).
#
# The paper appendix renders with this ON; the standalone
# table_full_results.tex renders the SAME tables with it OFF, so the full,
# starred data is still reported somewhere. run_relaxation_sweep.py flips it
# around the two calls -- do not set it globally.
_AAAI_WIDE_DROP_PARTIAL_ROWS = False


def set_aaai_wide_drop_partial_rows(flag):
    """Set the drop-starred-rows toggle; returns the PREVIOUS value so the
    caller can restore it."""
    global _AAAI_WIDE_DROP_PARTIAL_ROWS
    prev = _AAAI_WIDE_DROP_PARTIAL_ROWS
    _AAAI_WIDE_DROP_PARTIAL_ROWS = bool(flag)
    return prev

# Target-network (N2) per-perturbation solve-time charts. These live in
# the evaluation body (sec_evaluation.tex) and replace the N2 per-cell
# tables there; one grouped-bar figure per (architecture, source class).
AAAI_N2_CHARTS_BEGIN_MARK = "% BEGIN AUTO: aaai_n2_time_charts"
AAAI_N2_CHARTS_END_MARK   = "% END AUTO: aaai_n2_time_charts"

# Body-section summary table (Table 2 in the paper): one row per architecture
# with the transfer-mode bound gap vs. the exact VHAGaR bound and the
# transfer-mode speedup over VHAGaR. Lives between these markers in
# sec_evaluation.tex; derived from the SAME per-cell rows as the appendix.
AAAI_SUMMARY_BEGIN_MARK = "% BEGIN AUTO: aaai_summary_table"
AAAI_SUMMARY_END_MARK   = "% END AUTO: aaai_summary_table"

# 3 columns: the original VHAGaR baseline (PI=off, no boosts), the
# standard-mode safe combo zono+SibGate+PI, and the transfer-mode safe combo
# adv+zono+prev_pgd+SibGate. The relaxation threshold tau is NOT baked into
# the column keys: it is its own table column, so one (arch, role, pert,
# c_source) cell renders one row PER tau value that has data. The header
# labels below are therefore tau-free.
_AAAI_WIDE_COLUMN_HEADERS = (
    r"\baseline",
    r"ours",
    r"ours with transfer",
)


def _wide_group_width(hdr):
    """Sub-columns a method group spans in the per-cell tables: 4 for the two
    \tool modes (delta_l, delta_u, t, #relaxed) and 3 for \baseline, which
    relaxes no binary and would only ever print 0."""
    return 3 if hdr.strip() == r"\baseline" else 4

# Tau values the per-cell tables render, in row order. This is the DEFAULT;
# --paper_taus overrides it via set_aaai_wide_taus() so an extra threshold
# (e.g. 0.25) can be pulled into the tables without editing the source.
_AAAI_WIDE_TAUS_DEFAULT = ("0.0", "0.5")
_AAAI_WIDE_TAUS = _AAAI_WIDE_TAUS_DEFAULT


def set_aaai_wide_taus(taus):
    """Set which tau values the per-cell tables render as rows (--paper_taus).

    Each tau gets its own row per cell, so this only widens the table
    vertically. A tau with no data on a given cell simply yields no row for
    that cell. Passing None/empty restores the default."""
    global _AAAI_WIDE_TAUS
    if not taus:
        _AAAI_WIDE_TAUS = _AAAI_WIDE_TAUS_DEFAULT
        return _AAAI_WIDE_TAUS
    # Normalize ("0.50" -> "0.5") so the strings match the tau tags parsed out
    # of the result filenames, and keep the caller's order (row order).
    seen, norm = set(), []
    for t in taus:
        t = str(t).strip()
        if not t:
            continue
        try:
            t = repr(float(t)) if float(t) != int(float(t)) else f"{float(t):.1f}"
        except ValueError:
            pass
        if t not in seen:
            seen.add(t)
            norm.append(t)
    _AAAI_WIDE_TAUS = tuple(norm) or _AAAI_WIDE_TAUS_DEFAULT
    return _AAAI_WIDE_TAUS


def _aaai_wide_columns_for_tau(tau):
    """The three column keys for one tau. The \\baseline column is
    tau-independent (the baseline runs no relaxation), so it repeats on every
    tau row; only ours / ours-with-transfer are tau-specific."""
    return ["vaghar", ("1", "1", "1", tau), f"adv_zono_prevpgd_{tau}+sg"]


def _aaai_wide_cell_has_data(cell_dict, col):
    """True iff this cell has at least one run (base or geom twin) for `col`."""
    cell = cell_dict.get(col)
    return bool(cell and (cell.get("t_base") or cell.get("t_geom")))


def _aaai_wide_taus_for_cell(cell_dict):
    """The tau values this cell actually has ours / ours-with-transfer data
    for, in _AAAI_WIDE_TAUS order. Empty when only the baseline ran, in which
    case the caller still emits one row so the baseline stays visible."""
    return [t for t in _AAAI_WIDE_TAUS
            if any(_aaai_wide_cell_has_data(cell_dict, c)
                   for c in _aaai_wide_columns_for_tau(t)[1:])]


# The tau=0.5 instance, kept as a module-level constant because the Evaluation
# relaxation/precision table (which has no tau column) pins itself to it below.
_AAAI_WIDE_COLUMNS = tuple(
    zip(_aaai_wide_columns_for_tau("0.5"), _AAAI_WIDE_COLUMN_HEADERS))


def _as_int(v):
    """Parse a result-file count into an int, or None when absent/unparsable.
    The .txt fields arrive as strings ('' when the run predates the field)."""
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# AAAI relaxation / precision-cost table (Evaluation body)
# ---------------------------------------------------------------------------
AAAI_RELAX_PREC_BEGIN_MARK = "% BEGIN AUTO: aaai_relax_precision_table"
AAAI_RELAX_PREC_END_MARK = "% END AUTO: aaai_relax_precision_table"

# The three columns the relaxation table compares, as combo keys. They must
# match _AAAI_WIDE_COLUMNS so this table describes exactly the runs the per-cell
# tables display.
_AAAI_RELAX_BASE_COMBO = _AAAI_WIDE_COLUMNS[0][0]
_AAAI_RELAX_OURS_COMBO = _AAAI_WIDE_COLUMNS[1][0]
_AAAI_RELAX_TRANSFER_COMBO = _AAAI_WIDE_COLUMNS[2][0]


def _norm_dataset_key(dataset):
    """Canonical dataset key for _AAAI_TAB1_NEURONS, tolerant of the
    hyphen/underscore spellings used across this repo."""
    return {"fashion_mnist": "fashion-mnist", "fashion": "fashion-mnist",
            "fmnist": "fashion-mnist", "cifar10": "cifar"}.get(dataset, dataset)


# Row highlight for the per-cell tables: fires only when the [delta_l, delta_u]
# intervals of _AAAI_WIDE_COLUMNS[1] ("ours") and [2] ("ours with transfer") are
# disjoint AND transfer landed below ours, which its relaxed superset of ours'
# neurons should make impossible. Kept light so the black cell text stays
# readable.
_AAAI_WIDE_DISJOINT_COLOR = "red!25"


_AAAI_TIMEOUT_STATUSES = frozenset({
    "TIME_LIMIT", "TIME LIMIT", "USER_OBJ_LIMIT", "USER_LIMIT",
    "ITERATION_LIMIT", "NODE_LIMIT", "SOLUTION_LIMIT",
    "MEMORY_LIMIT", "WORK_LIMIT",
})


def _ft_for(force_timeout, arch):
    """Resolve the effective wall-clock cap for one architecture.

    `force_timeout` is either a plain scalar in seconds (the same cap for every
    arch, from --force_timeout) or a dict {arch: seconds} carrying a different
    cap per architecture (from --arch_timeouts). None -- or an arch missing from
    the dict -- disables the cap for that arch."""
    if isinstance(force_timeout, dict):
        return force_timeout.get(arch)
    return force_timeout


def _benchmark_provenance_clause(dataset):
    """A caption sentence naming a benchmark dataset's source network, the work
    it was taken from, the dataset it was trained on, and how the target network
    was derived from it. Empty string for the image datasets, so their captions
    are unchanged.

    Mirrors the phrasing the evaluation section already uses for the image
    models ("$\\Npre$ is trained using Adam, and $N$ is obtained by fine-tuning
    $\\Npre$"), since neither benchmark net has training data and both pairs are
    built by quantization instead.

    Callers pass either the raw dataset key or its display name, so "ACAS Xu"
    and "acas" both have to resolve.
    """
    key = str(dataset).strip().lower().replace(" ", "")
    if key in ("acas", "acasxu"):
        return (r" $\Npre$ is the ACAS Xu network \texttt{ACASXU\_run2a\_1\_1} "
                r"of \citet{katz2017reluplex}, and $N$ is obtained by quantizing its "
                r"weights to 8-bit integers, one scale per output channel.")
    if key == "har":
        # Same wording as the Table 1 caption, so the two agree on where the
        # network and its data come from.
        return (r" $\Npre$ is the network \texttt{HAR.nnet} of "
                r"ReluDiff~\cite{paulsen2020reludiff}, originally trained on "
                r"the UCI \emph{Human Activity Recognition Using Smartphones} "
                r"dataset~\cite{anguita2013har}, and $N$ is obtained by "
                r"quantizing its weights to 8-bit integers, one scale per "
                r"output channel.")
    return ""


def _timeout_caption_clause(force_timeout, arch):
    """A caption sentence stating this architecture's solver timeout, always in
    hours. Empty string when no cap is set for the arch (so the caption is
    unchanged). Example: ' Each run is given a solver timeout of $3$ hours.'"""
    ft = _ft_for(force_timeout, arch)
    if ft is None:
        return ""
    hrs = ft / 3600.0
    hrs_str = (f"{hrs:.0f}" if abs(hrs - round(hrs)) < 1e-9
               else _fmt_trim(hrs))
    unit = "hour" if hrs_str == "1" else "hours"
    return f" Each run is given a solver timeout of ${hrs_str}$ {unit}."


def _ct_for(requested_c_targets, arch):
    """Resolve the requested target-class set (0-indexed) for one architecture.

    `requested_c_targets` is either a set (the same --ct for every arch), a dict
    {arch: set|None} (a different --ct per architecture, from the per-group ct
    lists of --arch_timeouts), or None (no restriction). A dict entry of None --
    or an arch absent from the dict -- means no restriction for that arch."""
    if isinstance(requested_c_targets, dict):
        return requested_c_targets.get(arch)
    return requested_c_targets


def _cs_for(requested_c_sources, arch):
    """Resolve the requested source-class set (0-indexed) for one architecture.

    The c_source (a.k.a. c_tag / Cs) analogue of `_ct_for`: `requested_c_sources`
    is a set (the same --cs for every arch), a dict {arch: set|None} (a different
    --cs per architecture, from the per-group #CS lists of --arch_timeouts), or
    None (no restriction). A dict entry of None -- or an arch absent from the
    dict -- means no restriction for that arch."""
    if isinstance(requested_c_sources, dict):
        return requested_c_sources.get(arch)
    return requested_c_sources


def _aaai_is_timeout_mismatch(row, force_timeout, eps):
    """True iff this row's Gurobi run hit a termination limit at a wall-clock
    cap different from `force_timeout` (within ±eps seconds). Non-timeout
    rows always return False (they're included regardless).

    `force_timeout` may be a per-arch dict, in which case this row's own arch
    selects the cap; a scalar is applied to every arch."""
    force_timeout = _ft_for(force_timeout, row.get("arch"))
    if force_timeout is None:
        return False
    status = str(row.get("solve_status", "") or "").upper().replace(" ", "_")
    if not any(tag in status for tag in _AAAI_TIMEOUT_STATUSES):
        return False
    # Compare the Gurobi SOLVER wall-clock (optimization_time) against the cap,
    # NOT total_time. total_time = optimization_time + hyper_attack_time, and
    # the PGD hyper-attack warm-up adds ~50-70s (unbounded, grows with the net).
    # A run that genuinely hit the cap has optimization_time ~= force_timeout
    # but total_time = force_timeout + hyper_attack_time, so testing total_time
    # would push a legitimate at-cap timeout past `eps` and wrongly drop it as a
    # cross-cap re-run (emptying the cell to "---"). Fall back to t_total only
    # when the solver time wasn't recorded.
    t = row.get("t_opt")
    if t is None:
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
                                    label_suffix="",
                                    requested_c_targets=None):
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
                  # Binaries the relaxation dropped, summed over the two MIP
                  # copies (clean N(x) and perturbed N(x')) -- the same
                  # convention the relaxation/precision table uses.
                  "relaxed": [],
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
        _ro, _rp = r.get("relaxed_org"), r.get("relaxed_pert")
        if _ro is not None or _rp is not None:
            cell["relaxed"].append((_ro or 0) + (_rp or 0))

    columns = [col for col, _hdr in _AAAI_WIDE_COLUMNS]

    lines = []
    lines.append(f"% auto-generated: archs={archs}, dataset={dataset}, "
                 f"total_cell_rows={len(rows)}")
    # Two shared header macros -- the tables differ ONLY in the last two
    # columns: the "solved" tables carry SPEEDUP columns (t_vaghar / t_method,
    # a "x" ratio), while the "all-timeout" tables carry BOUND-DIFFERENCE
    # columns (how many percentage points of delta_max tighter than the baseline
    # each method's remaining gap is), since the solve times there are all pinned
    # at the cap and carry no signal.
    def _wide_header_macro(name, trailing_top):
        top = ["\\multirow{2}{*}{model}", "\\multirow{2}{*}{pert (size)}",
               "\\multirow{2}{*}{$\\tau$}"]
        for hdr in _AAAI_WIDE_COLUMN_HEADERS:
            top.append("\\multicolumn{%d}{c|}{%s}"
                       % (_wide_group_width(hdr), hdr))
        top += trailing_top
        sub = ["", "", ""]
        for hdr in _AAAI_WIDE_COLUMN_HEADERS:
            sub += [r"$\delta_l$\%", r"$\delta_u$\%", r"$t$"]
            if _wide_group_width(hdr) == 4:
                sub += [r"\#rel"]
        sub += ["", ""]  # the two trailing columns span both header rows
        return [f"\\providecommand{{\\{name}}}{{%",
                " & ".join(top) + r" \\",
                " & ".join(sub) + r" \\}"]

    lines += _wide_header_macro(
        "aaaisafewideheader",
        [r"\multirow{2}{*}{\shortstack{speedup\\single-net}}",
         r"\multirow{2}{*}{\shortstack{speedup\\transfer}}"])
    lines += _wide_header_macro(
        "aaaisafewideheaderbd",
        [r"\multirow{2}{*}{\shortstack{gap ratio\\ours}}",
         r"\multirow{2}{*}{\shortstack{gap ratio\\transfer}}"])
    lines.append("")

    for arch in archs:
        arch_keys = sorted(
            (k for k in buckets
             if k[0] == arch and (roles is None or k[1] in roles)),
            key=lambda k: (k[1], k[4], k[2], k[3]),  # role, c_src, pert, p_size
        )
        if not arch_keys:
            continue
        # Drop cells where no column at ANY tau has data (base or geom).
        _all_cols = {c for t in _AAAI_WIDE_TAUS
                     for c in _aaai_wide_columns_for_tau(t)}
        arch_keys = [
            k for k in arch_keys
            if any(_aaai_wide_cell_has_data(buckets[k], c) for c in _all_cols)
        ]
        if not arch_keys:
            continue
        safe_arch = arch.replace("_", "")
        # Displayed architecture name (conv1/conv3/3x50/...); the \label ids keep
        # the raw key via safe_arch so cross-refs stay stable.
        arch_disp = _AAAI_ARCH_DISPLAY.get(arch, arch.replace("_", r"\_"))
        _ft = _ft_for(force_timeout, arch)
        _cap_min = (_ft / 60.0) if _ft is not None else None
        _eps_min = rerun_timeout_eps / 60.0
        # l l l = model, pert (size), tau; then one "r r r" group per method.
        col_spec = ("@{}l l l | "
                    + " | ".join("r r r" if _wide_group_width(h) == 3
                                 else "r r r r"
                                 for h in _AAAI_WIDE_COLUMN_HEADERS)
                    + " | r r@{}")
        # Collect every (role, c_src) block's rows, each tagged with all_timeout,
        # so this architecture can be split into TWO tables below: one for cells
        # where every method hit the solver timeout (no finisher), one for cells
        # where at least one method finished.
        arch_blocks = []  # (label_parts, [(data_cells, all_timeout), ...])

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
            if dd_val is not None and math.isfinite(dd_val):
                if dmax_ub is not None:
                    dd_str = _fmt_sig((dd_val / dmax_ub) * 100.0) + "\\%"
                else:
                    # delta_max unavailable: show the RAW delta_d (the absolute
                    # margin, not a percentage) flagged with a blue "*", matching
                    # the raw delta_l/delta_u fallback in the bound cells. This
                    # label is emitted INSIDE math ($\delta_d{=}...$), so the star
                    # must be a math superscript ({}^{*}); a text-mode $^*$ here
                    # would close the surrounding math and break the build.
                    dd_str = _fmt_sig(dd_val) + r"\textcolor{blue!60!cyan}{{}^{*}}"
            # role is stored as "N1"/"N2"; render per the paper's notation:
            # the target network N2 is written N, and the source network N1
            # is written N_{\mathrm{pre}} (so the model cell reads
            # "cnn1 / N" or "cnn1 / N_{pre}", matching \Npre in the body).
            _ROLE_TEX = {"N2": "N", "N1": r"N_{\mathrm{pre}}"}
            role_tex = _ROLE_TEX.get(
                role, "N_" + role[1:] if role.startswith("N") else role)
            # The model value is distributed down the first column, one
            # part per row, across the block's rows (rather than a wide
            # full-width heading that overflows into the data columns).
            label_parts = [
                r"\textbf{" + arch_disp
                + r" / $" + role_tex + r"$}",
                r"$c_s{=}" + str(c_src) + r"$",
                r"$\delta_{\max}{=}" + dmax_str + r"$",
                r"$\delta_d{=}" + dd_str + r"$",
            ]

            # Expected c_targets for every row in this block: every class
            # index except c_src. A combo whose actually-recorded set of
            # c_targets is a strict subset of this gets a red asterisk
            # next to its time cell so the reader knows the mean was
            # taken over a partial sweep. When the caller restricts the
            # sweep to specific target classes (--ct / requested_c_targets,
            # 0-indexed), the "expected" set is exactly that request (minus
            # the self-pair c_src), so the "*" flags a requested c_target
            # with no data rather than every unrun class.
            # The benchmark nets are not 10-class: HAR classifies 6 activities
            # and ACAS Xu 5 advisories, so assuming 10 would call a complete
            # sweep partial.
            _DATASET_NUM_CLASSES = {"mnist": 10, "cifar10": 10,
                                     "fashion_mnist": 10,
                                     "har": 6, "acas": 5}
            n_classes = _DATASET_NUM_CLASSES.get(dataset, 10)
            _req_cts = _ct_for(requested_c_targets, arch)
            if _req_cts is not None:
                expected_cts_block = {ct for ct in _req_cts
                                      if ct != int(c_src)}
            else:
                expected_cts_block = {ct for ct in range(n_classes)
                                      if ct != int(c_src)}

            # First pass: build the data cells (everything except the
            # model column) for every row that actually has data, so the
            # model label can be spread across exactly the rendered rows.
            block_rows = []
            # One rendered row per (cell, tau): a cell that ran ours /
            # ours-with-transfer at several thresholds gets one row per
            # threshold, with the tau column naming which one. `columns` is
            # rebound per row to that tau's three column keys, so every
            # downstream lookup (stats, speedups, gap ratios, highlight,
            # all_timeout) is automatically tau-scoped. A cell with baseline
            # data only still yields a single row so the baseline stays
            # visible; its tau cell reads "---".
            _cell_tau_rows = []
            for key in list(gkeys):
                _taus = _aaai_wide_taus_for_cell(buckets[key])
                for _t in (_taus or [None]):
                    _cell_tau_rows.append((key, _t))
            for key, tau in _cell_tau_rows:
                _, role, pert, p_size, c_src = key
                cell_dict = buckets[key]
                columns = _aaai_wide_columns_for_tau(
                    tau if tau is not None else _AAAI_WIDE_TAUS[-1])
                tau_cell = "---" if tau is None else f"${tau}$"
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
                    else:
                        # delta_max unavailable for this (arch, role, c_src): we
                        # cannot normalize the bounds to a percentage, so keep the
                        # RAW (un-normalized) delta_l/delta_u means. The renderer
                        # shows these real values flagged with a blue "*" instead
                        # of a bare dash, so no data is hidden just because the
                        # delta_max pre-phase has not been run for this cell.
                        if cell["lb_base"]:
                            s["lb_raw_base"] = (sum(cell["lb_base"])
                                / len(cell["lb_base"]))
                        if cell["lb_geom"]:
                            s["lb_raw_geom"] = (sum(cell["lb_geom"])
                                / len(cell["lb_geom"]))
                        if cell["ub_base"]:
                            s["ub_raw_base"] = (sum(cell["ub_base"])
                                / len(cell["ub_base"]))
                        if cell["ub_geom"]:
                            s["ub_raw_geom"] = (sum(cell["ub_geom"])
                                / len(cell["ub_geom"]))
                    if cell["relaxed"]:
                        s["relaxed"] = (sum(cell["relaxed"])
                                        / len(cell["relaxed"]))
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
                data_cells = [pert_str, tau_cell]
                STAR = r"\textcolor{red}{$^*$}"
                # Blue "*" flags a RAW (un-normalized) bound shown because
                # delta_max was unavailable, distinct from the red "*" (partial
                # c_target coverage). See the raw-bound fallback above.
                BLUESTAR = r"\textcolor{blue!60!cyan}{$^*$}"
                for _ci, c in enumerate(columns):
                    _w = _wide_group_width(
                        _AAAI_WIDE_COLUMN_HEADERS[_ci]
                        if _ci < len(_AAAI_WIDE_COLUMN_HEADERS) else "")
                    s = stats.get(c)
                    if s is None:
                        data_cells += ["---"] * _w
                        continue
                    # Each sub-column renders a SINGLE value, preferring the
                    # geometric-range (--geometric_intervals, "geom") run and
                    # falling back to the base run when no geom twin exists.
                    # vaghar and the non-geometric perturbations (linf /
                    # brightness / occ / patch) have no geom run, so they always
                    # show their base value; only translation/rotation
                    # ours/transfer cells switch to geom. (Same side preference
                    # as the body N2 solve-time charts.) "---" when neither side
                    # has data.
                    def _pick(bk, gk):
                        if gk in s:
                            return s[gk], True
                        if bk in s:
                            return s[bk], False
                        return None, False
                    # Clamp for display: both bounds are percentages of
                    # delta_max, so nothing should ever render above 100%
                    # (delta_max is by definition the max) or below 0. Cap
                    # BOTH to [0, 100]. delta_u already had the 100 cap; delta_l
                    # needs it too, otherwise a transfer cell whose bounds both
                    # sit just above delta_max shows delta_l rounding up to 101
                    # while delta_u is pinned to 100 -- a spurious delta_l >
                    # delta_u. The raw incumbent <= best_bound invariant holds;
                    # this is purely a display cap. The diff/gap column is
                    # computed from the pre-clamp values, so it is unaffected.
                    def _bound_cell(pct_bk, pct_gk, raw_bk, raw_gk):
                        # Prefer the percentage-of-delta_max value (clamped to
                        # [0, 100] as before). When delta_max was unavailable no
                        # percentage exists, so fall back to the RAW bound mean
                        # (un-clamped -- it is not a percentage) flagged with a
                        # blue "*". "---" only when the cell has no bound at all.
                        v, _ = _pick(pct_bk, pct_gk)
                        if v is not None:
                            return _fmt_sig(min(100.0, max(0.0, v)))
                        raw, _ = _pick(raw_bk, raw_gk)
                        if raw is not None:
                            return _fmt_sig(raw) + BLUESTAR
                        return "---"
                    data_cells.append(_bound_cell(
                        "lb_pct_base", "lb_pct_geom", "lb_raw_base", "lb_raw_geom"))
                    data_cells.append(_bound_cell(
                        "ub_pct_base", "ub_pct_geom", "ub_raw_base", "ub_raw_geom"))
                    # Solve time (minutes), same geom-preferred side. When
                    # --force_timeout is set, a mean above the cap is overhead
                    # beyond the solver limit and is clamped. The partial "*"
                    # follows the chosen side: geom-partial when the geom sweep
                    # missed a class-pair, else base-partial.
                    t_v, t_is_geom = _pick("t_base", "t_geom")
                    if t_v is None:
                        data_cells.append("---")
                    else:
                        _ft = _ft_for(force_timeout, arch)
                        if _ft is not None:
                            t_v = min(t_v, _ft / 60.0)
                        t_partial = (partial_geom.get(c) if t_is_geom
                                     else partial.get(c))
                        data_cells.append(_fmt_trim(t_v)
                                          + (STAR if t_partial else ""))
                    # #relaxed: mean binaries the relaxation dropped over the
                    # two copies of N. Only the two \tool groups carry it.
                    if _w == 4:
                        _rv = s.get("relaxed")
                        data_cells.append("---" if _rv is None
                                          else f"{_rv:.0f}")

                # Trailing diff column. The compared method is the transfer
                # column on N2 rows and the standard ("ours") column on N1 rows
                # (transfer is N2-only). Two regimes, chosen per row:
                #
                #  * Solve-time reduction (usual case): the signed gap versus
                #    the vaghar baseline as a fraction of the baseline time,
                #    (t_vaghar - t_method) / t_vaghar (positive => the method is
                #    faster). The representative time is the baseline ("base")
                #    solve time, falling back to the geom twin, clamped to
                #    force_timeout to match the rendered minutes.
                #
                #  * Bound-gap reduction (fallback): when EVERY method that ran
                #    on the cell hit the timeout on average, the solve times are
                #    all pinned to the cap and carry no signal, so compare the
                #    remaining MIP optimality gap g = delta_u - delta_l instead:
                #    (g_vaghar - g_method) / g_vaghar (positive => a tighter
                #    bound). These entries are flagged with a dagger.
                #
                # Either way, a cell with partial c_target coverage (the red
                # "*") is skipped ("---"): its mean is over an incomplete /
                # mismatched class set, so the comparison is not
                # apples-to-apples. This value both fills the column and orders
                # the block's rows.
                def _repr_time(c):
                    # Geom-preferred, base fallback -- same side the cell shows.
                    s2 = stats.get(c)
                    if not s2:
                        return None
                    v = s2.get("t_geom")
                    if v is None:
                        v = s2.get("t_base")
                    if v is None:
                        return None
                    _ft = _ft_for(force_timeout, arch)
                    if _ft is not None:
                        v = min(v, _ft / 60.0)
                    return v

                def _disp_time(c):
                    # The solve time EXACTLY as the cell prints it: geom-
                    # preferred and clamped (via _repr_time), then rounded to
                    # the one decimal _fmt_trim shows, so the time-diff is
                    # reproducible from the printed minutes.
                    v = _repr_time(c)
                    return None if v is None else float(f"{v:.1f}")

                def _repr_partial(c):
                    # Partial "*" of the side the cell uses (geom if a geom run
                    # exists for this column, else base).
                    s2 = stats.get(c)
                    if not s2:
                        return False
                    if "t_geom" in s2:
                        return bool(partial_geom.get(c))
                    return bool(partial.get(c))

                # A cell carries the partial "*" on its chosen side when that
                # sweep missed an expected c_target. If ANY column in the row
                # is starred, the row has a missing run, so BOTH speedups are
                # left blank -- the means are not over a comparable class set.
                row_partial = any(_repr_partial(c) for c in columns)

                def _speedup_for(cmp_c):
                    """(speedup, cell) = t_vaghar / t_cmp, the baseline's solve
                    time over column cmp_c's, rendered as an "x" ratio (higher =
                    faster; <1 means slower). Even when every method is pinned at
                    the timeout cap the raw ratio (~1x) is shown -- no special
                    case. Built from the DISPLAYED (rounded) times so it is
                    reproducible from the printed minutes; blank ("---") when the
                    row is partial, an input time is missing, or cmp_c's printed
                    time is 0."""
                    if row_partial:
                        return None, "---"
                    tv = _disp_time(columns[0])
                    tc = _disp_time(cmp_c)
                    if tv is None or tc is None or tc <= 0:
                        return None, "---"
                    s = tv / tc
                    return s, _fmt_trim(s) + r"$\times$"

                # Two speedup columns, left to right: speedup-single-net (vaghar
                # over \emph{ours}) then speedup-transfer (vaghar over \emph{ours
                # with transfer}). On N1 the transfer column has no data, so
                # speedup-transfer renders "---" for every row. These are the
                # trailing columns for the SOLVED tables (kept OUT of data_cells;
                # the emitter appends the right trailing pair per table kind).
                std_frac, std_cell = _speedup_for(columns[1])
                trans_frac, trans_cell = _speedup_for(columns[2])
                speedup_cells = [std_cell, trans_cell]

                # Bound-tightness columns for the ALL-TIMEOUT tables: the RATIO
                # of the \baseline's remaining gap to each method's,
                #   ratio = g_baseline / g_method,  g = delta_u - delta_l
                # so a value > 1 means the method's bound is tighter than the
                # baseline's and < 1 means the baseline is tighter. Same gap the
                # bounds-difference graph plots; computed from the geom-preferred,
                # [0,100]-clamped bounds so it matches the delta_l/delta_u shown.
                def _bounds_pct(c):
                    # The DISPLAYED (rounded, [0,100]-clamped, geom-preferred)
                    # delta_l/delta_u pair, so both the gap columns and the row
                    # highlight below are reproducible from the printed bounds --
                    # same reproducibility rule the speedup columns use for the
                    # printed times.
                    s2 = stats.get(c)
                    if not s2:
                        return None
                    lb = s2.get("lb_pct_geom", s2.get("lb_pct_base"))
                    ub = s2.get("ub_pct_geom", s2.get("ub_pct_base"))
                    if lb is None or ub is None:
                        return None
                    lb = float(_fmt_sig(min(100.0, max(0.0, lb))))
                    ub = float(_fmt_sig(min(100.0, max(0.0, ub))))
                    return lb, max(lb, ub)

                def _gap_pct(c):
                    b = _bounds_pct(c)
                    return None if b is None else b[1] - b[0]

                def _bounddiff_cell(cmp_c):
                    gb = _gap_pct(columns[0])   # baseline gap
                    gc = _gap_pct(cmp_c)        # method gap
                    # Ratio g_baseline / g_method: >1 => method tighter, <1 =>
                    # baseline tighter. Blank when a gap is missing or the method
                    # gap is 0 (ratio undefined / unbounded).
                    if gb is None or gc is None or gc <= 0:
                        return "---"
                    return _fmt_trim(gb / gc) + r"$\times$"

                bounddiff_cells = [_bounddiff_cell(columns[1]),
                                   _bounddiff_cell(columns[2])]

                # Row highlight from how the two INTERVALS [delta_l, delta_u] of
                # \emph{ours} and \emph{ours with transfer} sit relative to each
                # other. Both bracket the same true delta, so they are expected
                # to overlap; a row where they do not is worth the reader's eye:
                #
                #   * disjoint (they share no point at all) AND transfer sits
                #     BELOW ours -> red. Transfer relaxes a superset of ours'
                #     neurons, so its delta can only be the higher of the two;
                #     landing below ours contradicts that and marks a row worth
                #     a second look.
                #   * anything else -> no highlight. Transfer sitting ABOVE ours
                #     is the expected direction (the extra relaxed neurons
                #     enlarge its feasible set), and overlapping or nested
                #     intervals are simply consistent.
                #
                # Compared on the DISPLAYED bounds (via _bounds_pct), so the
                # color always matches what the printed cells show. A row where
                # either column has no bounds ("---", or the raw-bound fallback
                # when delta_max is unavailable) stays uncolored.
                def _overlap_color():
                    a = _bounds_pct(columns[1])   # ours
                    b = _bounds_pct(columns[2])   # ours with transfer
                    if a is None or b is None:
                        return None
                    if max(a[0], b[0]) <= min(a[1], b[1]):
                        return None             # they meet: consistent
                    if b[0] > a[0]:
                        return None             # transfer above ours: expected
                    return _AAAI_WIDE_DISJOINT_COLOR

                row_color = _overlap_color()
                # Order rows by the transfer speedup on N2 (the headline method,
                # as before); N1 has no transfer, so fall back to the single-net
                # speedup there. A blank ranking value sinks the row to the end
                # of its cluster ((0, -s) sorts descending; (1, 0.0) parks
                # blanks).
                sort_frac = std_frac if role == "N1" else trans_frac
                row_sort = ((0, -sort_frac) if sort_frac is not None
                            else (1, 0.0))
                # all_timeout: ALL THREE methods are present AND pinned at the
                # timeout (per the user's rule "all values -- vaghar, ours and
                # transfer -- reached timeout"). A row where any method finishes
                # earlier OR was not run ("---") goes to the "solved" table.
                # (_repr_time is geom-preferred and already clamped to the cap.)
                if _cap_min is None:
                    all_timeout = False
                else:
                    all_timeout = all(
                        (_repr_time(c) is not None
                         and _repr_time(c) >= _cap_min - _eps_min)
                        for c in columns)
                # Partial-coverage filter: drop the row entirely when any
                # rendered cell carries the red "*" (a mean taken over only
                # some of the expected c_targets). Only the RED star counts --
                # the blue "*" flags a raw, un-normalized bound, which is a
                # complete measurement and stays. See
                # _AAAI_WIDE_DROP_PARTIAL_ROWS.
                if _AAAI_WIDE_DROP_PARTIAL_ROWS and any(
                        STAR in str(cell) for cell in
                        list(data_cells) + list(speedup_cells)
                        + list(bounddiff_cells)):
                    continue
                # Incomplete-comparison filter: a row only earns a place in the
                # paper if ALL THREE methods ran on it. data_cells is
                # [pert, tau] followed by the (delta_l, delta_u, t) triple of
                # each rendered column, so a "---" anywhere past the first two
                # means some method has no data and the row is not a full
                # three-way comparison. The trailing speedup / gap-ratio cells
                # are NOT checked: they legitimately read "---" when the regime
                # makes them meaningless.
                if _AAAI_WIDE_DROP_PARTIAL_ROWS and any(
                        "---" in str(cell) for cell in data_cells[2:]):
                    continue
                block_rows.append((row_sort, pert, p_size, data_cells,
                                   all_timeout, speedup_cells, bounddiff_cells,
                                   row_color))

            if not block_rows:
                continue

            # Two-level ordering that keeps each perturbation TYPE contiguous:
            #  (1) cluster the rows by perturbation name and order the clusters
            #      by their best (largest) speedup -- a cluster whose rows all
            #      lack a speedup sinks below every cluster that has one;
            #  (2) within a cluster, order rows by that speedup (largest first)
            #      with the speedup-less ("---") rows kept at the END of THAT
            #      cluster (not the block);
            #  (3) ties broken by perturbation size for a stable order.
            # row_sort is (0, -speedup) when a speedup exists and (1, 0.0)
            # otherwise, so min(row_sort) over a cluster is its best rank and
            # smaller sorts earlier.
            best_by_pert = {}
            for row_sort, pert_name, _ps, _dc, _at, _sp, _bd, _rc in block_rows:
                cur = best_by_pert.get(pert_name)
                if cur is None or row_sort < cur:
                    best_by_pert[pert_name] = row_sort
            block_rows.sort(key=lambda r: (best_by_pert[r[1]], r[1],
                                           r[0], r[2]))
            # Keep each row as (data_cells, all_timeout, speedup_cells,
            # bounddiff_cells, row_color) in sorted order; the split, the
            # trailing-column choice, and label distribution happen per emitted
            # table.
            arch_blocks.append(
                (label_parts,
                 [(r[3], r[4], r[5], r[6], r[7]) for r in block_rows]))

        if not arch_blocks:
            continue

        # Emit TWO tables for this architecture: first the cells where all three
        # methods hit the timeout, then the cells where at least one finished.
        # Each block's model label is distributed over only the rows that land in
        # the current table. The bare `tab:safe-wide-<arch>` label goes on the
        # first table emitted (so older \refs resolve), plus a kind-specific
        # label; empty tables are skipped.
        base_label = f"tab:safe-wide-{safe_arch}{label_suffix}"
        bare_emitted = False
        ds_disp = _dataset_display_name(dataset)
        is_n1 = roles is not None and set(roles) == {"N1"}
        for want_all_timeout, extra_label in ((True, "timeout"),
                                              (False, "solved")):
            # Each rendered row = data_cells + the trailing pair for this table:
            # the bound-difference cells for the all-timeout table, the speedup
            # cells for the solved table.
            kind = [(lp, [(dc + (bd if want_all_timeout else sp), rc)
                          for (dc, at, sp, bd, rc) in rows
                          if at == want_all_timeout])
                    for (lp, rows) in arch_blocks]
            kind = [(lp, r) for (lp, r) in kind if r]
            if not kind:
                continue
            lines.append(r"\begin{table*}[!tbp]")
            lines.append(r"\centering")
            lines.append(r"\scriptsize")
            lines.append(r"\setlength{\tabcolsep}{3pt}")
            lines.append(r"\begin{adjustbox}{max width=\textwidth,center}%")
            lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
            lines.append(r"\toprule")
            lines.append(r"\aaaisafewideheaderbd" if want_all_timeout
                         else r"\aaaisafewideheader")
            lines.append(r"\midrule")
            for label_parts, krows in kind:
                # Distribute the model label parts down the first column. More
                # rows than parts -> trailing rows get an empty model cell. Fewer
                # rows than parts -> pack the leftover parts into the last row via
                # a left-aligned \shortstack so nothing is dropped.
                n_rows = len(krows)
                row_labels = [""] * n_rows
                for i in range(min(n_rows, len(label_parts))):
                    if i == n_rows - 1 and n_rows < len(label_parts):
                        row_labels[i] = (r"\shortstack[l]{"
                                         + r"\\".join(label_parts[i:]) + r"}")
                    else:
                        row_labels[i] = label_parts[i]
                for lbl, (data_cells, row_color) in zip(row_labels, krows):
                    # \rowcolor must open the row, before the model label.
                    prefix = (f"\\rowcolor{{{row_color}}} " if row_color
                              else "")
                    lines.append(prefix + " & ".join([lbl] + data_cells)
                                 + r" \\")
                lines.append(r"\midrule")
            if lines[-1].rstrip() == r"\midrule":
                lines[-1] = r"\bottomrule"
            lines.append(r"\end{tabular}%")
            lines.append(r"\end{adjustbox}")
            if is_n1:
                cap = (
                    f"\\baseline and \\emph{{ours}} on {ds_disp} for architecture "
                    f"\\textbf{{{arch_disp}}}, source network $N_{{\\mathrm{{pre}}}}$. "
                    r"Each cell gives $(\delta_l, \delta_u, t)$ for the \baseline "
                    r"baseline and \emph{ours} (the \emph{ours "
                    r"with transfer} columns are blank, as transfer applies to "
                    r"$N$ only). The target network $N$ is in "
                    r"Table~\ref{tab:safe-wide-" + safe_arch + r"}.")
            else:
                cap = (
                    r"\tool vs. \baseline on "
                    + ds_disp + r" for architecture \textbf{" + arch_disp
                    + r"}, target network $N$")
                if want_all_timeout:
                    cap += (r", over the cells where all three methods "
                            r"(\baseline, \emph{ours}, and \emph{ours with "
                            r"transfer}) reach the solver timeout. As the solve "
                            r"times carry no signal there, the last two columns "
                            r"give the ratio of the baseline's remaining bound "
                            r"gap to each method's, "
                            r"$\dfrac{\delta_u^{\baseline}-\delta_l^{\baseline}}"
                            r"{\delta_u-\delta_l}$ (with $\delta_u-\delta_l$ the "
                            r"gap in percentage points of $\delta_{\max}$): above "
                            r"$1$ the method's bound is tighter than \baseline's, "
                            r"below $1$ \baseline is tighter.")
                else:
                    cap += (r", over the cells where at least one method "
                            r"finishes before the solver timeout (or was not "
                            r"run).")
            cap += _timeout_caption_clause(force_timeout, arch)
            cap += _benchmark_provenance_clause(dataset)
            lines.append(f"\\caption{{{cap}}}")
            if not bare_emitted:
                lines.append(f"\\label{{{base_label}}}")
                bare_emitted = True
            lines.append(f"\\label{{{base_label}-{extra_label}}}")
            lines.append(r"\end{table*}")
            lines.append("")

    return "\n".join(lines)


def update_aaai_wide_perarch_tex(tex_path, body,
                                 begin_mark=AAAI_WIDE_BEGIN_MARK,
                                 end_mark=AAAI_WIDE_END_MARK,
                                 label_suffix=""):
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
    body = _suffix_block_labels(body, label_suffix)
    updated = (f"{pre}{begin_mark}\n{body}\n"
               f"{end_mark}{post}")
    if updated == text:
        print("[update_advstd_tex_tables] no changes to aaai_safe_wide block")
        return
    with open(tex_path, "w") as f:
        f.write(updated)
    print(f"[update_advstd_tex_tables] wrote aaai_safe_wide block "
          f"in {tex_path}")


def _filter_rows_by_c_targets(rows, requested_c_targets):
    """Keep only rows whose (0-indexed) c_target is in `requested_c_targets`.

    Used by the --paper_tables_from_txt path to restrict every generated
    table/chart to a caller-chosen subset of target classes (--ct). Rows with
    no c_target (e.g. the 'max' perturbation's dummy) are kept as-is. A None
    request is a no-op, preserving the default full-sweep behavior. Note
    `requested_c_targets` is 0-indexed here to match the c_target field the
    result files store (Julia's 1-indexed --ct is converted by the caller).

    `requested_c_targets` may be a per-arch dict (from the per-group ct lists of
    --arch_timeouts); each row is then filtered against its own arch's set, and
    an arch with no restriction keeps all its rows."""
    if requested_c_targets is None:
        return rows
    out = []
    for r in rows:
        req = _ct_for(requested_c_targets, r.get("arch"))
        ct = r.get("c_target")
        if req is None or ct is None:
            out.append(r)
            continue
        try:
            if int(ct) in req:
                out.append(r)
        except (TypeError, ValueError):
            out.append(r)
    return out


def _filter_rows_by_c_sources(rows, requested_c_sources):
    """Keep only rows whose (0-indexed) c_source is in `requested_c_sources`.

    The c_source (Cs / c_tag) analogue of `_filter_rows_by_c_targets`: used by
    the --paper_tables_from_txt path to restrict every generated table/chart to a
    caller-chosen subset of SOURCE classes (--cs / #CS groups). A row with no
    c_source is kept as-is; a None request is a no-op (full sweep).
    `requested_c_sources` is 0-indexed (the caller converts Julia's 1-indexed
    --cs), matching the c_source field the result files store, and may be a
    per-arch dict so each row is filtered against its own arch's set."""
    if requested_c_sources is None:
        return rows
    out = []
    for r in rows:
        req = _cs_for(requested_c_sources, r.get("arch"))
        cs = r.get("c_source")
        if req is None or cs is None:
            out.append(r)
            continue
        try:
            if int(cs) in req:
                out.append(r)
        except (TypeError, ValueError):
            out.append(r)
    return out


def _relax_gap_loss(m_row, b_row, dmax):
    """Precision loss of method row `m_row` against baseline row `b_row`: the
    DISTANCE BETWEEN THE TWO BOUND INTERVALS, as a percentage of delta_max.

        loss = 100 * max(0, max(dl_m, dl_b) - min(du_m, du_b)) / delta_max

    This is the standard distance (gap) between two sets, inf{|a-b| : a in A,
    b in B} (Boyd and Vandenberghe 2004), written out for two intervals. Its
    defining property is the one we want: it is 0 exactly when the intervals
    overlap, and positive only when they are disjoint.

    Why it is sound on EVERY cell, including the ones where the baseline times
    out: the baseline relaxes no binary, so its [dl, du] soundly contains the
    exact delta whether it proves optimality or returns a wide anytime interval.
    A method interval disjoint from it therefore differs from the exact delta by
    AT LEAST this distance, making the gap a certified lower bound on the error;
    a method interval that overlaps it is consistent with the same delta, so no
    error is provable and the loss is 0. A wide (timed-out) baseline overlaps
    more easily, so it proves less loss -- the measure is conservative, never
    wrong. Contrast the Hausdorff distance, which is symmetric and would score a
    method that converged TIGHTER than a timed-out baseline as a large loss.

    Normalized by delta_max, the scale the per-cell tables render dl/du on.
    Returns None when a bound or delta_max is missing."""
    ml, mu = m_row.get("lb_total"), m_row.get("ub_total")
    bl, bu = b_row.get("lb_total"), b_row.get("ub_total")
    vals = (ml, mu, bl, bu, dmax)
    if any(v is None for v in vals):
        return None
    if not all(math.isfinite(v) for v in vals) or abs(dmax) < 1e-9:
        return None
    return max(0.0, max(ml, bl) - min(mu, bu)) / abs(dmax) * 100.0


def collect_aaai_relax_precision_rows(cwd, dataset, arch_runs,
                                      parse_result_file,
                                      seeds_filter=None,
                                      force_timeout=None,
                                      rerun_timeout_eps=30.0,
                                      advstd_meta_fn=None,
                                      perts=None,
                                      combination_filter=None,
                                      requested_c_targets=None,
                                      requested_c_sources=None,
                                      stale_fn=None):
    """Aggregate one dataset's N2 runs into per-network relaxation/precision
    rows for the Evaluation relaxation table.

    Assembles rows exactly as regenerate_aaai_wide_perarch_section does, so this
    table describes the same runs the per-cell tables display (same seed, combo,
    timeout, c_target/c_source and staleness filters, and the same
    geometric-range side preference). Returns a list of dicts, one per network:

        {"dataset", "arch",
         "relaxed_ours", "relaxed_transfer",   # mean binaries dropped, N+N'
         "loss_ours", "loss_transfer",         # mean interval-gap loss, % dmax
         "cells_ours", "cells_transfer"}       # runs behind each mean
    """
    rows = _collect_wide_perarch_cells(arch_runs, cwd, dataset,
                                       parse_result_file,
                                       seeds_filter=seeds_filter,
                                       stale_fn=stale_fn)
    archs = sorted((a for a, _ in arch_runs), key=_tab1_arch_sort_key)
    if advstd_meta_fn is not None and perts is not None:
        rows += _load_advstd_rows_for_wide_from_txt(
            cwd, dataset, archs, perts, parse_result_file, advstd_meta_fn,
            seeds_filter=seeds_filter,
            combination_filter=combination_filter,
            force_timeout=force_timeout,
            rerun_timeout_eps=rerun_timeout_eps,
            stale_fn=stale_fn)
    rows = _filter_rows_by_c_targets(rows, requested_c_targets)
    rows = _filter_rows_by_c_sources(rows, requested_c_sources)
    # Drop timeout cells recorded under a different wall-clock cap, the same
    # dedup the per-cell tables apply to the vaghar/ours rows.
    if force_timeout is not None:
        rows = [r for r in rows
                if not _aaai_is_timeout_mismatch(r, force_timeout,
                                                 rerun_timeout_eps)]

    wanted = {_AAAI_RELAX_BASE_COMBO, _AAAI_RELAX_OURS_COMBO,
              _AAAI_RELAX_TRANSFER_COMBO}
    # One slot per (arch, pert, size, c_src, c_tgt) x column. Geom-preferred with
    # a base fallback, matching the per-cell tables' _pick, so the runs counted
    # here are the runs those tables print.
    by_cell = {}
    for r in rows:
        if r.get("role") != "N2" or r.get("combo") not in wanted:
            continue
        key = (r["arch"], r["perturbation"], r["perturbation_size"],
               r["c_source"], r["c_target"])
        slot = by_cell.setdefault(key, {})
        prev = slot.get(r["combo"])
        if prev is None or (r.get("geom") and not prev.get("geom")):
            slot[r["combo"]] = r

    ds_key = _norm_dataset_key(dataset)
    # delta_max per (arch, role, c_src): the scale the loss is normalized by,
    # the same one the per-cell tables render delta_l/delta_u against.
    delta_max_by_key = _load_delta_max_values(cwd, dataset, archs)
    from collections import defaultdict
    acc = defaultdict(lambda: {"relax_ours": [], "relax_transfer": [],
                               "loss_ours": [], "loss_transfer": []})
    for (arch, _p, _s, cs, _ct), slot in by_cell.items():
        base = slot.get(_AAAI_RELAX_BASE_COMBO)
        dm_entry = (delta_max_by_key or {}).get((arch, "N2", cs))
        dmax = dm_entry.get("upper") if dm_entry else None
        # Only cells where ALL THREE methods ran, the same rule the N2 bar charts
        # use for a cluster and the per-cell tables use for a full row. Without
        # it the two method columns would average over different cells (on MNIST
        # conv1, 72 for ours against 42 for transfer), so their losses would not
        # be comparable to each other.
        if any(slot.get(c) is None for c in (_AAAI_RELAX_BASE_COMBO,
                                             _AAAI_RELAX_OURS_COMBO,
                                             _AAAI_RELAX_TRANSFER_COMBO)):
            continue
        # Every such cell counts whether or not the baseline proved optimality:
        # its interval is sound either way, so the interval distance below is a
        # certified lower bound on the method's error. A timed-out baseline is
        # simply wide, which makes the gap harder to prove, not invalid. See
        # _relax_gap_loss.
        for combo, rk, lk in ((_AAAI_RELAX_OURS_COMBO, "relax_ours", "loss_ours"),
                              (_AAAI_RELAX_TRANSFER_COMBO, "relax_transfer",
                               "loss_transfer")):
            m = slot[combo]
            # Binaries the relaxation dropped, summed over BOTH MIP copies: the
            # clean N(x) and the perturbed N(x') each drop their own.
            org, pert = m.get("relaxed_org"), m.get("relaxed_pert")
            if org is not None and pert is not None:
                acc[arch][rk].append(org + pert)
            loss = _relax_gap_loss(m, base, dmax)
            if loss is not None:
                acc[arch][lk].append(loss)

    def _mean(v):
        return (sum(v) / len(v)) if v else None

    out = []
    for arch in archs:
        a = acc.get(arch)
        if not a or not (a["relax_ours"] or a["relax_transfer"]
                         or a["loss_ours"] or a["loss_transfer"]):
            continue
        out.append({
            "dataset": ds_key,
            "arch": arch,
            "relaxed_ours": _mean(a["relax_ours"]),
            "relaxed_transfer": _mean(a["relax_transfer"]),
            "loss_ours": _mean(a["loss_ours"]),
            "loss_transfer": _mean(a["loss_transfer"]),
            "cells_ours": len(a["loss_ours"]),
            "cells_transfer": len(a["loss_transfer"]),
        })
    return out


_AAAI_RELAX_DATASET_ORDER = ["mnist", "fashion-mnist", "cifar"]


def _tab1_neuron_counts(sections_dir):
    """{(dataset_display, network_display): neurons} parsed from the frozen
    full Table 1 master (tab_networks_full.tex), the paper's own statement of
    each network's ReLU count. The relaxed-share denominator of the
    relaxation/precision table is derived from these, so the two tables can
    never disagree about a network's size; a hard-coded dict here could.

    A data row is '<Dataset> & <Network> & <Architecture> & <#Neurons> & ...';
    the header (\textbf) and rule lines don't match the shape and fall
    through."""
    out = {}
    path = os.path.join(sections_dir, "tab_networks_full.tex")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return out
    for line in text.splitlines():
        if not line.rstrip().endswith(r"\\") or "textbf" in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 4:
            continue
        digits = re.sub(r"\D", "", cells[3])
        if digits:
            out[(cells[0], cells[1])] = int(digits)
    return out


def _render_aaai_relax_precision_body(model_rows, neuron_counts=None):
    """Render the single Evaluation relaxation/precision table from the pooled
    per-network rows of collect_aaai_relax_precision_rows (all datasets).

    `neuron_counts` ({(dataset_display, network_display): neurons}, from
    _tab1_neuron_counts) turns the relaxed columns into SHARES: dropped
    binaries over the total binaries in \baseline's encoding, which has one
    binary per ReLU neuron in each of the TWO copies of $N$ it encodes, i.e.
    2 * #Neurons of Table 1. A network absent from the counts falls back to
    the absolute number (and says so on the console) rather than guessing."""
    if not model_rows:
        return "% auto-generated: aaai_relax_precision -- no data"

    def _ds_key(r):
        try:
            return (0, _AAAI_RELAX_DATASET_ORDER.index(r["dataset"]))
        except ValueError:
            return (1, r["dataset"])

    rows = sorted(model_rows,
                  key=lambda r: (_ds_key(r), _tab1_arch_sort_key(r["arch"])))

    def _relaxed(v, denom):
        # Share of the encoding's binaries (user request): the dropped count,
        # summed over the two copies of $N$, over 2 * #Neurons. One decimal,
        # so 0 reads as exactly none relaxed and small shares keep a digit.
        if v is None:
            return "---"
        if denom:
            return f"{100.0 * v / denom:.1f}\\%"
        return f"{int(round(v))}"

    def _loss(v):
        if v is None:
            return "---"
        # Two decimals throughout, so a 0 reads as an exact 0 rather than a
        # rounded-down small loss, and a leading zero is always present.
        return f"{v:.2f}\\%"

    lines = [f"% auto-generated: aaai_relax_precision, networks={len(rows)}"]
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    # Fitting one column: the dataset and the network share a single model cell,
    # and the adjustbox only shrinks if a longer network name overruns, so the
    # table can never bleed into the neighbouring column the way a bare tabular
    # does.
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{adjustbox}{max width=\columnwidth,center}%")
    lines.append(r"\begin{tabular}{@{}l r r r r@{}}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Model}} & "
                 r"\multicolumn{2}{c}{\textbf{\emph{ours}}} & "
                 r"\multicolumn{2}{c}{\textbf{\emph{transfer}}} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(r" & \textbf{relaxed} & \textbf{loss} & "
                 r"\textbf{relaxed} & \textbf{loss} \\")
    lines.append(r"\midrule")
    for r in rows:
        ds_disp = _dataset_display_name(r["dataset"])
        arch_disp = _AAAI_ARCH_DISPLAY.get(r["arch"],
                                           r["arch"].replace("_", r"\_"))
        model = ds_disp + " " + arch_disp
        neurons = (neuron_counts or {}).get((ds_disp, arch_disp))
        denom = 2 * neurons if neurons else None
        if denom is None:
            print(f"[relax-precision] {model}: not in Table 1's master; "
                  f"relaxed shown as an absolute count, not a share")
        lines.append(" & ".join([
            model,
            _relaxed(r["relaxed_ours"], denom), _loss(r["loss_ours"]),
            _relaxed(r["relaxed_transfer"], denom), _loss(r["loss_transfer"]),
        ]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"\end{adjustbox}")
    # AAAI puts the caption UNDER the table, so it follows the tabular.
    lines.append(
        r"\caption{The share of ReLU binaries the relaxation drops (relaxed) "
        r"and the precision it costs (loss), averaged over all experiments. "
        r"A run's relaxed share is the number of binaries the relaxation "
        r"drops, summed over the two copies of $N$ that \baseline's MIP "
        r"encodes, divided by the total number of binaries in that encoding: "
        r"one binary per ReLU neuron in each of the two copies, that is, "
        r"twice the \#Neurons column of Table~\ref{tab:networks}.}")
    lines.append(r"\label{tab:relax-precision}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def regenerate_aaai_relax_precision_section(tex_path, model_rows):
    """Rewrite the aaai_relax_precision AUTO block in tex_path with the single
    table pooled over every dataset. Run once, after the per-dataset
    collect_aaai_relax_precision_rows calls."""
    try:
        # tab_networks_full.tex sits beside tex_path (both under sections/).
        body = _render_aaai_relax_precision_body(
            model_rows,
            neuron_counts=_tab1_neuron_counts(
                os.path.dirname(os.path.abspath(tex_path))))
        update_aaai_wide_perarch_tex(tex_path, body,
                                     begin_mark=AAAI_RELAX_PREC_BEGIN_MARK,
                                     end_mark=AAAI_RELAX_PREC_END_MARK)
    except SystemExit as exc:
        print(f"[update_advstd_tex_tables] aaai_relax_precision block "
              f"skipped: {exc}")
    except Exception as exc:
        print(f"[update_advstd_tex_tables] aaai_relax_precision block "
              f"error: {exc}")


def regenerate_aaai_wide_perarch_section(tex_path, cwd, dataset, arch_runs,
                                          parse_result_file,
                                          seeds_filter=None,
                                          force_timeout=None,
                                          rerun_timeout_eps=30.0,
                                          roles=None,
                                          label_suffix="",
                                          begin_mark=AAAI_WIDE_BEGIN_MARK,
                                          end_mark=AAAI_WIDE_END_MARK,
                                          ds_label_suffix="",
                                          advstd_meta_fn=None,
                                          perts=None,
                                          combination_filter=None,
                                          requested_c_targets=None,
                                          requested_c_sources=None,
                                          stale_fn=None):
    """Mirror regenerate_wide_perarch_section, but emit the slim 4-column
    AAAI variant into the neta_s_paper evaluation section. When
    `force_timeout` is set (in seconds), cells whose Gurobi run hit a
    termination limit under a different wall-clock cap are excluded.

    `roles`/`label_suffix`/`begin_mark`/`end_mark` select which network
    role to emit and where: the target-network (N2) tables go to the body
    with the default marks, and the source-network (N1) tables go to the
    appendix with roles={"N1"}, label_suffix="-n1", and the N1 marks.

    The transfer (advstd N2) rows come from the CSVs by default; when
    `advstd_meta_fn` and `perts` are supplied they instead come DIRECTLY from
    the advStd .txt files (no CSV, no baseline pairing) via
    _load_advstd_rows_for_wide_from_txt, optionally filtered by
    `combination_filter` (--combination_table). `stale_fn`
    (_is_pre_fix_dropped) drops pre-fix files that relaxed >=1 binary, applied
    uniformly to the vaghar/ours AND transfer rows."""
    try:
        rows = _collect_wide_perarch_cells(arch_runs, cwd, dataset,
                                            parse_result_file,
                                            seeds_filter=seeds_filter,
                                            stale_fn=stale_fn)
        # Emit archs in Table-1 order so the per-cell tables follow the paper's
        # Networks table rather than the --arch_timeouts command order.
        archs = sorted((a for a, _ in arch_runs), key=_tab1_arch_sort_key)
        if advstd_meta_fn is not None and perts is not None:
            rows += _load_advstd_rows_for_wide_from_txt(
                cwd, dataset, archs, perts, parse_result_file, advstd_meta_fn,
                seeds_filter=seeds_filter,
                combination_filter=combination_filter,
                force_timeout=force_timeout,
                rerun_timeout_eps=rerun_timeout_eps,
                stale_fn=stale_fn)
        else:
            rows += _load_advstd_rows_for_wide(cwd, dataset, archs,
                                                seeds_filter=seeds_filter)
        rows = _filter_rows_by_c_targets(rows, requested_c_targets)
        rows = _filter_rows_by_c_sources(rows, requested_c_sources)
        delta_max_by_key = _load_delta_max_values(cwd, dataset, archs)
        delta_d_by_key = _load_delta_d_values(cwd, dataset, archs)
        body = _render_aaai_wide_perarch_body(
            rows, archs, dataset,
            delta_max_by_key=delta_max_by_key,
            delta_d_by_key=delta_d_by_key,
            force_timeout=force_timeout,
            rerun_timeout_eps=rerun_timeout_eps,
            roles=roles,
            label_suffix=label_suffix,
            requested_c_targets=requested_c_targets)
        update_aaai_wide_perarch_tex(tex_path, body,
                                     begin_mark=begin_mark,
                                     end_mark=end_mark,
                                     label_suffix=ds_label_suffix)
    except SystemExit as exc:
        print(f"[update_advstd_tex_tables] aaai_safe_wide block skipped: "
              f"{exc}")
    except Exception as exc:
        print(f"[update_advstd_tex_tables] aaai_safe_wide block error: "
              f"{exc}")


# ---------------------------------------------------------------------------
# AAAI N2 per-perturbation charts (Evaluation body)
# ---------------------------------------------------------------------------
# Per (architecture, source class c_s) we emit up to two grouped-bar
# figures, each with one bar per perturbation for the three methods (the
# VHAGaR baseline, \tool's standard mode, \tool's transfer mode):
#   * a SOLVE-TIME figure (y = minutes) for the perturbations on which at
#     least one method finishes before the three-hour timeout; and
#   * a BOUND-GAP figure (y = delta_u - delta_l, in percentage points of
#     delta_max) for the perturbations on which ALL three methods hit the
#     timeout -- there the time bars are all flat at the cap and carry no
#     signal, so the remaining MIP optimality gap is the informative metric
#     (smaller = tighter).
# For translation/rotation the standard and transfer bars use the
# geometric-range twin (the "with geom" variant); for every other
# perturbation the geom twin does not exist so they fall back to the base
# run. The full per-cell bounds behind these charts live in the appendix
# N2 tables.
#
# (combo_key, legend label, pgfplots per-series style). The styles use a
# light->dark spread so the three series stay distinguishable in grayscale
# as well as in colour (cf. the AAAI grayscale-legibility guideline).
# Each series: (combo_key, legend label, fill+draw style, fill-pattern). The
# series are distinguished THREE ways at once so they stay readable in colour,
# in grayscale, and for every type of colour-blindness: (1) HUE (red baseline;
# the two "ours" methods are a related light-green / dark-green pair), (2)
# distinct LUMINANCE, and (3) a distinct fill PATTERN/texture (horizontal lines
# / diagonal lines / vertical lines). Pattern colours contrast with each fill
# (white on the red and dark-green bars, dark on the light-green bar).
_AAAI_CHART_SERIES = (
    ("vaghar",                  r"\baseline",               "fill=vagharred, draw=vagharred!60!black",
        ""),
    (("1", "1", "1", "0.5"),    r"ours",                    "fill=oursgreen!42, draw=oursgreen!60!black",
        "pattern=north east lines, pattern color=oursgreen!70!black"),
    ("adv_zono_prevpgd_0.5+sg", r"ours with transfer",      "fill=oursgreen!48!black, draw=oursgreen!25!black",
        "pattern=vertical lines, pattern color=white"),
)

# Palette for the "bounds difference" charts (clusters where all three methods
# hit the timeout). Same series KEYS, same three fill PATTERNS/textures as
# _AAAI_CHART_SERIES (none / north-east lines / vertical lines), but a DISTINCT
# colour family (amber baseline + blue light/dark "ours" pair) so a
# bounds-difference figure never reads as a solve-time figure. Colours are
# defined in main.tex (bddiffbase, bddiffours).
_AAAI_CHART_SERIES_GAP = (
    ("vaghar",                  r"\baseline",               "fill=bddiffbase, draw=bddiffbase!60!black",
        ""),
    (("1", "1", "1", "0.5"),    r"ours",                    "fill=bddiffours!38, draw=bddiffours!60!black",
        "pattern=north east lines, pattern color=bddiffours!70!black"),
    ("adv_zono_prevpgd_0.5+sg", r"ours with transfer",      "fill=bddiffours!55!black, draw=bddiffours!25!black",
        "pattern=vertical lines, pattern color=white"),
)

# Which relaxation thresholds get their own PAIR of series (ours + ours with
# transfer) in the charts. Default is the single headline threshold, which
# reproduces the three-series figures above byte for byte; --paper_chart_taus
# adds more. Kept SEPARATE from _AAAI_WIDE_TAUS (the appendix-table rows)
# because a chart cluster only fits a few bars before the per-bar value labels
# stop fitting, whereas a table simply grows another row.
_AAAI_CHART_TAUS_DEFAULT = ("0.5",)
_AAAI_CHART_TAUS = _AAAI_CHART_TAUS_DEFAULT

# Per-tau bar styling, as (ours fill/draw, ours pattern, transfer fill/draw,
# transfer pattern). Within one figure the METHOD is carried by the pattern
# family and the THRESHOLD by luminance, so every bar stays separable in
# colour, in grayscale, and under colour-blindness (the same three-way rule the
# base palette follows). The '0.5' entries are verbatim the styles used above,
# so a default single-tau run emits identical TikZ.
_AAAI_CHART_TAU_STYLES = {
    "0.5": ("fill=oursgreen!42, draw=oursgreen!60!black",
            "pattern=north east lines, pattern color=oursgreen!70!black",
            "fill=oursgreen!48!black, draw=oursgreen!25!black",
            "pattern=vertical lines, pattern color=white"),
    "0.25": ("fill=oursgreen!18, draw=oursgreen!55!black",
             "pattern=crosshatch, pattern color=oursgreen!75!black",
             "fill=oursgreen!72!black, draw=oursgreen!20!black",
             "pattern=horizontal lines, pattern color=white"),
    "0.0": ("fill=oursgreen!8, draw=oursgreen!50!black",
            "pattern=dots, pattern color=oursgreen!80!black",
            "fill=oursgreen!88!black, draw=oursgreen!15!black",
            "pattern=grid, pattern color=white"),
}
_AAAI_CHART_TAU_STYLES_GAP = {
    "0.5": ("fill=bddiffours!38, draw=bddiffours!60!black",
            "pattern=north east lines, pattern color=bddiffours!70!black",
            "fill=bddiffours!55!black, draw=bddiffours!25!black",
            "pattern=vertical lines, pattern color=white"),
    "0.25": ("fill=bddiffours!16, draw=bddiffours!55!black",
             "pattern=crosshatch, pattern color=bddiffours!75!black",
             "fill=bddiffours!75!black, draw=bddiffours!20!black",
             "pattern=horizontal lines, pattern color=white"),
    "0.0": ("fill=bddiffours!6, draw=bddiffours!50!black",
            "pattern=dots, pattern color=bddiffours!80!black",
            "fill=bddiffours!90!black, draw=bddiffours!15!black",
            "pattern=grid, pattern color=white"),
}


def _aaai_build_chart_series(baseline_entry, style_table):
    """Series tuple for one figure family: the \\baseline bar, then an
    (ours, ours with transfer) pair per tau in _AAAI_CHART_TAUS. The tau is
    named in the legend label only when more than one is drawn, so the
    single-tau default keeps the original unqualified 'ours' / 'ours with
    transfer' labels."""
    show_tau = len(_AAAI_CHART_TAUS) > 1
    out = [baseline_entry]
    for tau in _AAAI_CHART_TAUS:
        styles = style_table.get(tau)
        if styles is None:
            # Unknown tau: fall back to the headline styling rather than
            # dropping the series, so a new threshold still renders.
            styles = style_table["0.5"]
        o_fill, o_pat, t_fill, t_pat = styles
        suffix = (r" ($\tau{=}" + tau + r"$)") if show_tau else ""
        out.append((("1", "1", "1", tau), r"ours" + suffix, o_fill, o_pat))
        out.append((f"adv_zono_prevpgd_{tau}+sg",
                    r"ours with transfer" + suffix, t_fill, t_pat))
    return tuple(out)


def set_aaai_chart_taus(taus):
    """Set which thresholds the charts draw (--paper_chart_taus) and rebuild
    both series tables. Passing None/empty restores the single-tau default."""
    global _AAAI_CHART_TAUS, _AAAI_CHART_SERIES, _AAAI_CHART_SERIES_GAP
    norm = []
    for t in (taus or ()):
        t = str(t).strip()
        if not t:
            continue
        try:
            t = repr(float(t)) if float(t) != int(float(t)) else f"{float(t):.1f}"
        except ValueError:
            pass
        if t not in norm:
            norm.append(t)
    _AAAI_CHART_TAUS = tuple(norm) or _AAAI_CHART_TAUS_DEFAULT
    _AAAI_CHART_SERIES = _aaai_build_chart_series(
        _AAAI_CHART_SERIES[0], _AAAI_CHART_TAU_STYLES)
    _AAAI_CHART_SERIES_GAP = _aaai_build_chart_series(
        _AAAI_CHART_SERIES_GAP[0], _AAAI_CHART_TAU_STYLES_GAP)
    return _AAAI_CHART_TAUS


def _aaai_chart_time(cell, force_timeout, prefer_geom):
    """Mean solve time in minutes for one method's cell, clamped to the
    timeout, plus which side ('base'/'geom') the value came from.
    `prefer_geom` picks the geometric-range twin's time when it exists and
    falls back to the base run otherwise (standard/transfer); vaghar has no
    geom twin so it always reads the base side. Returns (None, None) when
    the cell has no timing on the chosen side."""
    if cell is None:
        return None, None
    tb = cell.get("t_base")
    tg = cell.get("t_geom")
    if prefer_geom:
        chosen, side = (tg, "geom") if tg else ((tb, "base") if tb else (None, None))
    else:
        chosen, side = (tb, "base") if tb else ((tg, "geom") if tg else (None, None))
    if not chosen:
        return None, None
    v = sum(chosen) / len(chosen) / 60.0
    if force_timeout is not None:
        v = min(v, force_timeout / 60.0)
    return v, side


def _aaai_chart_times_list(cell, force_timeout, prefer_geom):
    """The individual per-target solve times (minutes, each clamped to the
    timeout) on the SAME base/geom side `_aaai_chart_time` averages, plus that
    side. These per-(c_s, c_t) values are the samples the confidence-interval
    I-beam pools across a cluster's source classes, so the interval reflects
    every class pair rather than only the handful of source-class means. Returns
    ([], None) when the chosen side has no timing."""
    if cell is None:
        return [], None
    tb = cell.get("t_base")
    tg = cell.get("t_geom")
    if prefer_geom:
        chosen, side = (tg, "geom") if tg else ((tb, "base") if tb else (None, None))
    else:
        chosen, side = (tb, "base") if tb else ((tg, "geom") if tg else (None, None))
    if not chosen:
        return [], None
    cap = (force_timeout / 60.0) if force_timeout is not None else None
    out = []
    for t in chosen:
        m = t / 60.0
        out.append(min(m, cap) if cap is not None else m)
    return out, side


def _aaai_chart_time_by_ct(cell, force_timeout, prefer_geom):
    """Per-target mean solve time (minutes, clamped to the timeout) as a dict
    {c_t: minutes} on the SAME base/geom side `_aaai_chart_time` averages. Lets
    the solve-time figure draw ONE cluster per (c_s, c_t) pair instead of a
    single bar averaged over all targets. Returns ({}, None) when the chosen side
    has no timing."""
    if cell is None:
        return {}, None
    tb = cell.get("t_base_by_ct")
    tg = cell.get("t_geom_by_ct")
    if prefer_geom:
        chosen, side = (tg, "geom") if tg else ((tb, "base") if tb else (None, None))
    else:
        chosen, side = (tb, "base") if tb else ((tg, "geom") if tg else (None, None))
    if not chosen:
        return {}, None
    cap = (force_timeout / 60.0) if force_timeout is not None else None
    out = {}
    for ct, times in chosen.items():
        if not times:
            continue
        m = sum(times) / len(times) / 60.0
        out[ct] = min(m, cap) if cap is not None else m
    return out, side


def _aaai_chart_gap(cell, dmax_ub, prefer_geom):
    """delta_u - delta_l (percentage points of delta_max) for one method's
    cell, plus which side ('base'/'geom') it came from, using the same
    base/geom side preference as the time bars. The bounds are clamped to
    [0, 100]% of delta_max exactly as the per-cell table renders them.
    Returns (None, None) when the chosen side lacks a finite lower/upper
    bound, or when delta_max is unknown."""
    if cell is None or not dmax_ub:
        return None, None

    def _side(lb_key, ub_key):
        lb = cell.get(lb_key)
        ub = cell.get(ub_key)
        if not lb or not ub:
            return None
        lb_pct = max(0.0, (sum(lb) / len(lb)) / dmax_ub * 100.0)
        ub_pct = min(100.0, (sum(ub) / len(ub)) / dmax_ub * 100.0)
        return max(0.0, ub_pct - lb_pct)

    order = (("geom", "lb_geom", "ub_geom"), ("base", "lb_base", "ub_base"))
    if not prefer_geom:
        order = order[::-1]
    for side, lbk, ubk in order:
        g = _side(lbk, ubk)
        if g is not None:
            return g, side
    return None, None


def _aaai_chart_gaps_list(cell, dmax_ub, prefer_geom):
    """The individual per-target bound gaps delta_u - delta_l (percentage points
    of delta_max, each clamped to [0, 100]% exactly as _aaai_chart_gap clamps the
    cell mean) on the SAME base/geom side _aaai_chart_gap picks. These
    per-(c_s, c_t) values are the samples the confidence-interval I-beam on the
    bounds-difference bars pools across a cluster's source classes -- the gap
    analogue of _aaai_chart_times_list for the time bars. Returns ([], None) when
    the chosen side lacks paired lower/upper bounds or delta_max is unknown."""
    if cell is None or not dmax_ub:
        return [], None

    def _side(lb_key, ub_key):
        lb = cell.get(lb_key)
        ub = cell.get(ub_key)
        if not lb or not ub:
            return None
        out = []
        for lb_i, ub_i in zip(lb, ub):
            lb_pct = max(0.0, lb_i / dmax_ub * 100.0)
            ub_pct = min(100.0, ub_i / dmax_ub * 100.0)
            out.append(max(0.0, ub_pct - lb_pct))
        return out

    order = (("geom", "lb_geom", "ub_geom"), ("base", "lb_base", "ub_base"))
    if not prefer_geom:
        order = order[::-1]
    for side, lbk, ubk in order:
        g = _side(lbk, ubk)
        if g is not None:
            return g, side
    return [], None


def _aaai_chart_gap_by_ct(cell, dmax_ub, prefer_geom):
    """Per-target bound gap delta_u - delta_l (percentage points of delta_max,
    clamped like _aaai_chart_gap) as a dict {c_t: gap}, on the SAME base/geom
    side preference. Lets the bounds-difference figure draw ONE cluster per
    all-timeout (c_s, c_t) pair instead of a bar averaged over targets.
    Returns ({}, None) when the chosen side lacks per-target bounds or
    delta_max is unknown."""
    if cell is None or not dmax_ub:
        return {}, None

    def _side(lb_key, ub_key):
        lbd = cell.get(lb_key)
        ubd = cell.get(ub_key)
        if not lbd or not ubd:
            return None
        out = {}
        for ct, lbs in lbd.items():
            ubs = ubd.get(ct)
            if not lbs or not ubs:
                continue
            lb_pct = max(0.0, (sum(lbs) / len(lbs)) / dmax_ub * 100.0)
            ub_pct = min(100.0, (sum(ubs) / len(ubs)) / dmax_ub * 100.0)
            out[ct] = max(0.0, ub_pct - lb_pct)
        return out or None

    order = (("geom", "lb_geom_by_ct", "ub_geom_by_ct"),
             ("base", "lb_base_by_ct", "ub_base_by_ct"))
    if not prefer_geom:
        order = order[::-1]
    for side, lbk, ubk in order:
        g = _side(lbk, ubk)
        if g is not None:
            return g, side
    return {}, None


def _aaai_partial(cell, side, expected_cts):
    """True iff the bar's chosen side aggregates over only SOME of the
    expected target classes (same partial-coverage notion the per-cell
    tables flag with a red asterisk). `side` is 'base' or 'geom'."""
    if cell is None or side is None:
        return False
    seen = cell.get("c_targets_geom" if side == "geom" else "c_targets", set())
    return bool(expected_cts - seen)


# Bar heights are drawn through a transform f(v)=v**POWER. POWER=1.0 is a
# plain LINEAR axis showing the real values (current choice, per user); a
# square-root scale (POWER=0.5) compresses a few very large bars so the
# small ones stay visible, at the cost of non-real spacing. With POWER<1 the
# tick LABELS still show the true values placed at f(value).
_AAAI_YAXIS_POWER = 1.0

# A solve-time panel gets a dashed horizontal line at the solver timeout when
# its tallest bar comes within this many MINUTES of the cap -- close enough
# that the reader could mistake a timed-out (capped) bar for a finished run.
# Panels whose bars all sit far below the cap draw no line (it would only
# squash the bars).
_AAAI_TIMEOUT_LINE_SLACK_MIN = 20.0

# A pooled solve-time row draws its dashed timeout level once its tallest bar
# passes this many MINUTES (user request). Below it no run is near the cap and
# the line would only squash the bars.
_AAAI_TIME_LINE_MIN = 150.0

# A bounds-difference panel draws its dashed 100 line once its tallest bar comes
# within this many percentage points of delta_max, the level the gap cannot
# pass.
_AAAI_BD_CEIL_SLACK = 10.0

# Width a row reserves on its RIGHT for the shared bound-difference axis: its
# tick numbers plus the rotated "bounds diff" label. Only the row's last panel
# draws them, so the row pays this once.
_AAAI_RIGHT_DECO_PT = 34.0

# Width a LATER panel of a row reserves on its left. It keeps its own time
# scale and so prints its own tick numbers, but NOT the rotated unit name
# (written once per row), which is all that _AAAI_PANEL_DECO_NEXT_PT budgets
# for. Three digits at \small plus the tick and a little air.
_AAAI_PANEL_TICKS_PT = 22.0

# A (c_s, c_t) pair EVERY method solves within this many MINUTES is left out of
# the solve-time charts (user request): such easy pairs carry no acceleration
# story and only crowd the panels. The cut is on the whole cluster, not on the
# baseline alone, so a pair where any one of \baseline, ours, or ours with
# transfer needs longer stays on the chart, whichever method that is. They stay
# in the appendix per-cell tables, and the chart captions state the cut.
_AAAI_TIME_CHART_MIN_SOLVE_MIN = 15.0


def _aaai_yaxis(vmax):
    """Return (ymax, ytick_clause) for the y-axis. On a LINEAR axis
    (_AAAI_YAXIS_POWER == 1) pgfplots auto-ticks at real values and we just
    cap the axis a little above the tallest bar. On a power/sqrt axis we
    place curated round ticks at f(value) but label them with the true
    values, and `ytick_clause` carries the explicit ytick/yticklabels."""
    power = _AAAI_YAXIS_POWER
    if vmax <= 0:
        return 1.0, ""
    if power >= 1.0:
        return vmax * 1.10, ""
    yt = _aaai_yaxis_ticks(vmax)
    ypos = ",".join(f"{(t ** power):.4g}" for t in yt)
    ylab = ",".join("{" + f"{t:g}" + "}" for t in yt)
    ymax = (yt[-1] ** power) * 1.10 if yt[-1] > 0 else 1.0
    return ymax, f"ytick={{{ypos}}}, yticklabels={{{ylab}}}, "


def _aaai_yaxis_ticks(vmax):
    """Curated round tick values in [0, vmax] for the power-scaled y-axis,
    with the last tick the first candidate >= vmax (the axis top). Denser
    at the low end so the many small bars get resolution once the sqrt
    spreads them out."""
    if vmax <= 0:
        return [0.0]
    if vmax <= 1.5:
        cand = [0, 0.1, 0.2, 0.5, 1, 1.5]
    elif vmax <= 6:
        cand = [0, 0.5, 1, 2, 3, 5, 6]
    elif vmax <= 15:
        cand = [0, 1, 2, 5, 10, 15]
    elif vmax <= 60:
        cand = [0, 1, 5, 10, 20, 40, 60]
    else:
        cand = [0, 1, 10, 30, 60, 90, 120, 180, 240]
    ticks = []
    for c in cand:
        ticks.append(float(c))
        if c >= vmax:
            break
    return ticks


# Bar width in x data-units. EVERY bar is exactly _AAAI_BAR_W wide regardless of
# how many methods (k) a group holds, so all bars in the figure render at the
# same physical width (combined with `scale only axis` + a figure-wide shared
# x-range in _aaai_arch_typegrid, which makes every panel's plot box -- and
# hence its data->cm scale -- identical). Bars within a perturbation-size
# cluster are drawn ADJACENT (touching); clusters sit 1.0 unit apart, so a full
# 3-bar cluster spans 3*0.30=0.90 and still leaves a ~0.10-unit gap before the
# next cluster. The per-bar value labels are printed HORIZONTALLY at one
# figure-wide point size (from _aaai_hlabel_pt) chosen so the widest label fits
# the bar WIDTH with a small gap each side -- see _aaai_hlabel_pt / _draw_group.
_AAAI_BAR_W = 0.30
_AAAI_GROUP_SPAN = 0.88

# HARD-CODED physical bar width (pt) shared by EVERY evaluation bar figure --
# the solve-time panels AND the bounds-difference figures (user request: all
# bars across the evaluation render at one width). The value equals the bar
# width the MNIST bounds-difference figure (fig 6) had when its 23 clusters
# spanned the text width at _AAAI_BAR_W=0.30 data units, so that reference
# figure is visually unchanged and every other figure narrows to match it.
# Each panel converts this to its own data units via its plot-box width and
# x-range (see _aaai_bar_data_w); _AAAI_BAR_W remains the fallback for chart
# types that never set an explicit physical width.
_AAAI_BAR_PT = 6.0


def _aaai_bar_data_w(span, plotbox_pt):
    """Convert the hard-coded physical bar width (_AAAI_BAR_PT) into DATA
    units for one panel: `span` is the panel's x-range in data units and
    `plotbox_pt` its plot-box width in pt (exact under `scale only axis`).
    Capped so a full 3-bar cluster can never overrun its 1.0-unit slot (a
    panel dense enough to hit the cap draws slightly narrower bars instead
    of overlapping its neighbour)."""
    if plotbox_pt <= 0:
        return _AAAI_BAR_W
    return min(_AAAI_BAR_PT * span / plotbox_pt, 0.98 / 3.0)


def _aaai_size_run_layout(groups, sort_by_label=False, run_gap_extra=0.0,
                          group_gap_extra=None):
    """Position one panel's clusters for the SIZE-RUN layout: clusters sit at
    a 1.0-slot pitch (the tight hard-coded 27pt) carrying only their
    "$c_t{=}N$" line; within a source group, same-size clusters are sorted
    ADJACENT (size-major, then mean) and each maximal same-size RUN gets its
    size written ONCE, centred under the run. Extra air is inserted between
    two runs only when their size labels would otherwise collide (and at
    source boundaries, at least the source-gap extra), so the panel stays as
    narrow as the labels permit.

    `groups` is [(src_label, "ct_line\\\\size", bars)] with the source groups
    already in the desired order. Returns (items, runs, srcs, xmax) where
    items = [(x, ct_line, bars)], runs = [(x_center, size_text)],
    srcs = [(x_center, src_label)], and xmax is the last cluster's x."""
    import re as _re
    from collections import OrderedDict
    slot = _AAAI_SLOT_PT
    # Air at a GROUP boundary, when it differs from the air between two runs
    # of one group. The solve-time figures group by MODEL and run by
    # perturbation size, so only the model boundary earns the wide gap; None
    # keeps both the same, which is what the bounds-difference figure wants.
    if group_gap_extra is None:
        group_gap_extra = run_gap_extra

    def _mean(bars):
        vs = [pair[0] for pair in bars.values()]
        return sum(vs) / len(vs) if vs else 0.0

    def _w_pt(text):
        plain = _re.sub(r"\\[a-zA-Z]+|[${}]", "", text)
        return _aaai_label_em(plain) * 9.0 + 3.0 if plain else 0.0

    by_src = OrderedDict()
    for (src, lbl, bars) in groups:
        parts = str(lbl).split(r"\\", 1)
        ct_line = parts[0]
        size = parts[1] if len(parts) > 1 else ""
        by_src.setdefault(src, []).append((size, _mean(bars), ct_line, bars))
    items, runs, srcs, subruns = [], [], [], []
    runs_raw = []     # (x_first, x_last, label) per run, merged below
    x = 0.0           # last placed cluster (0 = none yet; first lands at 1.0)
    prev_c = None     # previous run's centre (data units)
    prev_w = 0.0      # previous run's size-label width (pt)
    prev_lbl = None   # previous run's label, to spot a repeat
    for gi, (src, its) in enumerate(by_src.items()):
        # sort_by_label orders clusters inside a run by their tick-label line
        # (e.g. the bounds-difference "$c_s{=}N$" ascending) instead of by
        # bar mean (the solve-time default).
        its.sort(key=(lambda t: (t[0], t[2])) if sort_by_label
                 else (lambda t: (t[0], t[1])))
        gfirst = None
        i = 0
        while i < len(its):
            j = i
            while j < len(its) and its[j][0] == its[i][0]:
                j += 1
            k = j - i
            w = _w_pt(its[i][0])
            # Base extra before this run: `run_gap_extra` between two runs of
            # the SAME group (e.g. the bounds-difference model blocks, which
            # want visibly bigger air, user request), the source extra at a
            # group boundary, none before the very first run.
            if i == 0 and gi > 0:
                base = max(_AAAI_TYPE_GAP, group_gap_extra)
            elif i > 0:
                base = run_gap_extra
            else:
                base = 0.0
            extra = base
            if prev_c is not None and its[i][0] != prev_lbl:
                # Keep this run's size label clear of the previous run's:
                # centre distance (in pt) must cover both half-widths plus
                # the (halved) run-label air -- within a "c=" group this is
                # the ONLY spacing between different-size runs, so they sit
                # as close as their labels physically allow.
                need = ((prev_w + w) / 2.0
                        + _AAAI_RUN_LABEL_AIR_PT) / slot
                cand = x + 1.0 + extra + (k - 1) / 2.0  # this run's centre
                if cand - prev_c < need:
                    extra += need - (cand - prev_c)
            x0 = x + 1.0 + extra
            for t in range(k):
                xc = x0 + t
                items.append((xc, its[i + t][2], its[i + t][3]))
                if gfirst is None:
                    gfirst = xc
            x = x0 + k - 1
            centre = (x0 + x) / 2.0
            runs_raw.append((x0, x, its[i][0]))
            # SUB-RUNS: consecutive clusters of this run sharing the same
            # tick-label line (e.g. the bounds-difference "$c_s{=}N$"), so a
            # repeated c_s can be written once under its stretch.
            t0 = 0
            for t in range(1, k + 1):
                if t == k or its[i + t][2] != its[i + t0][2]:
                    subruns.append(((x0 + t0 + x0 + t - 1) / 2.0,
                                    its[i + t0][2]))
                    t0 = t
            prev_c, prev_w, prev_lbl = centre, w, its[i][0]
            i = j
        srcs.append(((gfirst + x) / 2.0, src))
    # One size label per STRETCH of adjacent runs carrying it, not one per run
    # (user request): consecutive models whose clusters share a perturbation
    # size get the size written once, centred over the whole stretch. Runs
    # inside one group are maximal same-size stretches already, so this only
    # ever joins runs across a group boundary.
    for (a, b, lbl) in runs_raw:
        if runs and runs[-1][1] == lbl and lbl:
            runs[-1] = (runs[-1][0], lbl, runs[-1][2], b)
        else:
            runs.append((a, lbl, a, b))
    runs = [((first + last) / 2.0, lbl) for (_a, lbl, first, last) in runs
            if lbl]
    return items, runs, srcs, (x if items else 1.0), subruns


def _aaai_panel_span(groups, sort_by_label=False, run_gap_extra=0.0,
                     group_gap_extra=None):
    """x-extent (data units) of one CLUSTERED panel under the size-run
    layout -- the last cluster's position from _aaai_size_run_layout, so the
    packing and the axes sized from this exactly fit their content."""
    return _aaai_size_run_layout(groups, sort_by_label=sort_by_label,
                                 run_gap_extra=run_gap_extra,
                                 group_gap_extra=group_gap_extra)[3]


# Packed-row solve-time layout: horizontal room reserved LEFT of a panel's
# plot box for its rotated ylabel and y-tick numbers at \small. The FIRST
# panel of a row carries a 2-line ylabel (perturbation + the gray
# "time (minutes)" unit -- written once per row, user request) and needs
# 2.0cm (verified; 1.6cm clipped); every LATER panel in the row carries only
# the 1-line perturbation name, so it reserves less. _AAAI_PANEL_HSEP_PT is
# pure AIR between a panel's plot box and the next panel's decorations, so
# adjacent panels can never touch.
_AAAI_PANEL_DECO_PT = 56.91
_AAAI_PANEL_DECO_NEXT_PT = 40.0
_AAAI_PANEL_HSEP_PT = 22.0

# HARD-CODED cluster geometry (user request: the bar width and EVERY gap are
# explicit pt constants). Within a cluster the three method bars touch
# (3 * _AAAI_BAR_PT = 18pt of ink). _AAAI_CLUSTER_GAP_PT is the air between
# adjacent clusters of the SAME source group; _AAAI_SOURCE_GAP_PT the air
# across a source-group boundary -- larger than the cluster gap, as required.
# 4.5pt/13pt (the within-source gap halved again on user request; clusters
# carry no tick label of their own -- the caption says each cluster is a
# different c_t -- so nothing collides at this pitch). The perturbation size
# is written ONCE per RUN of adjacent same-size clusters, centred under the
# run (see _aaai_size_run_layout). Where two runs' size labels would still
# collide, the layout inserts just enough extra air between those runs --
# the gaps never grow globally.
_AAAI_CLUSTER_GAP_PT = 4.5
_AAAI_SOURCE_GAP_PT = 13.0
# Air kept between the size labels of two adjacent same-source runs (the
# only structural spacing between different-size runs of one "c=" group;
# halved from 4 on user request -- the labels themselves are the floor).
_AAAI_RUN_LABEL_AIR_PT = 2.0
# Bounds-difference figures: air between two MODEL blocks (arch+dataset
# runs) inside one type+size group -- visibly bigger than the cluster gap
# (user request).
_AAAI_MODEL_GAP_PT = 24.0
_AAAI_SLOT_PT = 3 * _AAAI_BAR_PT + _AAAI_CLUSTER_GAP_PT  # 55pt cluster pitch

# Most packed rows one rendered solve-time figure may hold: a panel row
# (title + 2.5cm box + the two-line labels + source line) is ~130pt, so 4
# rows plus the legend (~545pt) stay under the 0.88\textheight (~572pt)
# import cap and render 1:1 (9pt text); a 5th row would trigger
# keepaspectratio downscaling. Figures with more rows are split across
# floats (see _aaai_combined_time_figure).
_AAAI_MAX_ROWS_PER_FIG = 4



# I-beam (error-bar-style) gap marker drawn ON each time bar. Its height encodes
# the remaining bound gap delta_u - delta_l (in percentage points of delta_max,
# the same number the appendix N2 tables and the timed-out gap figure report),
# on a FIXED visual scale: a full 100-pp gap draws an I-beam whose total height
# is _AAAI_IBEAM_PANEL_FRAC of the panel's data range (no numeric secondary
# axis). The marker is CENTRED on the bar top -- its lower cap sits inside the
# bar fill, its upper cap floats above the bar -- exactly like the error bars in
# arXiv:2511.10576 fig 5. The rotated value label is lifted to start above the
# upper cap so the two never overlap. Gaps below _AAAI_IBEAM_MIN_PP (a finished,
# provably-optimal run) draw no marker. The short caps span _AAAI_IBEAM_CAP_FRAC
# of the bar width to each side, so the marker reads as a narrow capital "I"
# (clearly thinner than the bar) whose length is the gap.
# NOTE: this is the visual density of the implicit gap axis -- a larger value
# maps the same gap to a LONGER I-beam. Raised from 0.30 so the (typically
# few-pp) gaps read as visibly long markers rather than short stubs.
_AAAI_IBEAM_PANEL_FRAC = 0.60
_AAAI_IBEAM_CAP_FRAC = 0.20
_AAAI_IBEAM_MIN_PP = 0.05


# Student-t two-sided 0.975 critical values keyed by degrees of freedom (df =
# n-1), the 95% confidence multiplier for a mean estimated from n observations.
# The per-bar I-beam sample is small (df ~ 2-17), so the flat 1.96 Gaussian value
# understates the interval; the exact multiplier is the 95% point of the
# Student-t density with n-1 df (heavier tails -> larger than 1.96, approaching
# 1.96 as n grows). We look it up here and fall back to 1.96 only for df > 30.
_T975_BY_DF = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _aaai_ci_halfwidth(vals):
    """Half-width of the 95% confidence interval of the MEAN of `vals` -- the
    individual c_s-against-c_t experiment solve times (minutes) under ONE bar --
    i.e. the I-beam's half-height in the bar's own units. Multiplies the standard
    error of the mean s/sqrt(n) by the Student-t 95% critical value
    t_{0.975, n-1} (`_T975_BY_DF`), the multiplier that fits a small sample whose
    spread is estimated from the data. Returns None when n < 3 (a 1- or 2-point
    interval is degenerate/misleading); returns 0.0 when every value is identical
    (e.g. all runs pinned at the timeout cap)."""
    n = len(vals)
    if n < 3:
        return None
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    if var <= 0.0:
        return 0.0
    sem = math.sqrt(var) / math.sqrt(n)
    tcrit = _T975_BY_DF.get(n - 1, 1.96)
    return tcrit * sem


def _aaai_draw_color(sty):
    """Pull the `draw=` colour out of a series style string (e.g.
    'fill=teal!55, draw=teal!65!black' -> 'teal!65!black')."""
    if "draw=" in sty:
        return sty.split("draw=", 1)[1].split(",", 1)[0].strip()
    return "black"


# Extra x-gap (data units, i.e. fractions of a cluster slot) inserted between
# GROUPS in the cluster layout -- the source-class groups of the solve-time
# panels. DERIVED from the hard-coded pt gaps so that the air across a source
# boundary is exactly _AAAI_SOURCE_GAP_PT while the air within a group is
# _AAAI_CLUSTER_GAP_PT (boundary pitch = slot + this extra).
_AAAI_TYPE_GAP = ((_AAAI_SOURCE_GAP_PT - _AAAI_CLUSTER_GAP_PT)
                  / _AAAI_SLOT_PT)


def _aaai_label_em(s):
    """Rough horizontal advance of a label string in em (font-size units):
    digits ~0.52em, '.'/',' ~0.28em. Used to size the value-label font so the
    widest label in a panel still fits over its bar."""
    return sum(0.28 if ch in ".," else 0.52 for ch in s)


def _aaai_fmt_bar_value(v):
    """Compact text for the value printed above a bar: an integer at >=10,
    else up to two decimals -- but with trailing zeros (and a trailing dot)
    dropped, since they don't change the value and only widen the label
    (e.g. 144, 7.94, 0.3, 5)."""
    if v >= 10:
        return f"{v:.0f}"
    s = f"{v:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# The per-bar value labels are printed HORIZONTALLY (not rotated), so their
# WIDTH -- not their height -- must fit inside a bar. We pick one figure-wide
# point size such that the widest label a cell can print ("XX.XX", ~2.36 em)
# spans at most _AAAI_LABEL_FIT_FRAC of a bar's physical width, leaving a small
# gap on each side so adjacent labels stay visually separated; the size is then
# clamped to a legible range. The AAAI 2026 (letterpaper) template has
# \textwidth = 505.89pt, which the N2 figures span (figure*); a bar's physical
# width follows the panel geometry in _aaai_arch_typegrid: panel width =
# (\textwidth - overhead)/ncol_max, the panel's x-range is `nmax` slots, and a
# bar is _AAAI_BAR_W data-units wide.
_AAAI_TEXTWIDTH_PT = 505.89
_AAAI_PT_PER_CM = 28.4527
_AAAI_LABEL_FIT_FRAC = 0.86
# AAAI guideline 3: text inside figures must render at >=9pt. The per-bar
# value labels are currently disabled everywhere, but if they are ever
# re-enabled the clamp must not go below the floor -- a label that no longer
# fits its bar width is a layout problem to solve by design (fewer panels,
# wider bars), never by shrinking the font under 9pt.
_AAAI_LABEL_PT_MIN = 9.0
_AAAI_LABEL_PT_MAX = 9.0


# ---------------------------------------------------------------------------
# Standalone chart rendering.
#
# AAAI forbids pgfplots in the SUBMITTED paper source, and directs authors to
# pre-generate such figures, export them as PDF, and import them with
# \includegraphics. pgfplots still draws the charts -- it just runs in the
# separate build below instead of in main.tex, so the chart code, geometry, and
# palettes are unchanged and only the delivery differs.
#
# The standalone document pins \textwidth to _AAAI_TEXTWIDTH_PT (the same value
# the panel geometry above is computed against) and preview/tightpage crops the
# page to the content. The emitted PDF is therefore exactly \textwidth wide, so
# \includegraphics[width=\textwidth] scales it by 1.0 and every font reaches the
# page at the size it was authored at -- scaling here would shrink the figure
# text and push it under the 9pt floor.
_AAAI_FIGDIR = "figures"

_AAAI_STANDALONE_DOC = r"""%% AUTO-GENERATED by update_advstd_tex_tables.py -- do not edit by hand.
%% Standalone build of one paper figure. This file is NOT part of the paper
%% source: it exists so pgfplots runs here rather than in main.tex (AAAI bans
%% pgfplots in submitted source). Only the PDF it produces is imported.
\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage{xcolor}
\usepackage{xspace}
\usepackage{tikz}
\usetikzlibrary{patterns,positioning}
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}
\usepgfplotslibrary{groupplots}
%% \tool/\baseline and the bar palette come from the paper's figure_defs.tex --
%% the SAME file main.tex \inputs -- so the chart bodies and the prose can never
%% disagree about the tool name or a colour. This file sits in <paper>/figures/.
\input{../figure_defs}
\setlength{\textwidth}{%(width).2fpt}
\setlength{\parindent}{0pt}
%% tightpage crops the page to the minipage. PreviewBorder is zeroed so the PDF
%% comes out exactly \textwidth wide and \includegraphics[width=\textwidth]
%% scales it by exactly 1.0 -- any border would force a slight rescale, and
%% shrinking a chart shrinks its text toward the 9pt floor.
\usepackage[active,tightpage]{preview}
\setlength\PreviewBorder{0pt}
\PreviewEnvironment{minipage}
\begin{document}
\begin{minipage}{\textwidth}
\centering
%(body)s
\end{minipage}
\end{document}
"""


def _aaai_render_chart_pdf(body_lines, basename, tex_dir):
    """Compile the tikz/pgfplots `body_lines` of ONE figure into a standalone
    PDF under `tex_dir`/_AAAI_FIGDIR and return the \\includegraphics line that
    replaces them in the paper.

    `basename` names the .tex/.pdf pair and must be stable across runs so the
    figure keeps one filename. Raises RuntimeError if pdflatex fails, rather
    than leaving the paper pointing at a stale or missing PDF."""
    import subprocess

    figdir = os.path.join(tex_dir, _AAAI_FIGDIR)
    os.makedirs(figdir, exist_ok=True)
    body = "\n".join(body_lines)
    src = _AAAI_STANDALONE_DOC % {"width": _AAAI_TEXTWIDTH_PT, "body": body}
    tex_file = os.path.join(figdir, basename + ".tex")
    with open(tex_file, "w") as fh:
        fh.write(src)
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         basename + ".tex"],
        cwd=figdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pdf_file = os.path.join(figdir, basename + ".pdf")
    if proc.returncode != 0 or not os.path.exists(pdf_file):
        tail = proc.stdout.decode("utf8", "replace").splitlines()[-25:]
        raise RuntimeError(
            "pdflatex failed for figure %s:\n%s" % (basename, "\n".join(tail)))
    # Drop the per-figure build droppings; keep only .tex (the source of record)
    # and .pdf (what the paper imports).
    for ext in (".aux", ".log"):
        try:
            os.remove(os.path.join(figdir, basename + ext))
        except OSError:
            pass
    # Cap the height too (keepaspectratio): a wide/short figure still binds on
    # width (unchanged), but a tall one -- e.g. the 1-plot-per-row per-(c_s,c_t)
    # solve-time figure -- scales DOWN to fit a float page instead of overrunning
    # it ("Float too large for page").
    return ("\\includegraphics[width=\\textwidth,height=0.88\\textheight,"
            "keepaspectratio]{%s/%s}" % (_AAAI_FIGDIR, basename))


def _aaai_hlabel_pt(nmax, ncol_max):
    """Figure-wide point size for the HORIZONTAL per-bar value labels, chosen so
    the widest possible label ('XX.XX') fits within one bar's physical width
    with a small gap on each side, then clamped to [_AAAI_LABEL_PT_MIN,
    _AAAI_LABEL_PT_MAX]. Mirrors the panel-width geometry in
    _aaai_arch_typegrid (panel_w = (\\textwidth - overhead)/ncol_max, x-range =
    `nmax` slots, bar = _AAAI_BAR_W data-units)."""
    ncol_max = max(int(ncol_max), 1)
    slots = max(int(nmax), 1)
    overhead_cm = 1.5 + 0.2 * max(ncol_max - 1, 0)
    panel_w_pt = (max(_AAAI_TEXTWIDTH_PT - overhead_cm * _AAAI_PT_PER_CM, 1.0)
                  / ncol_max)
    bar_w_pt = _AAAI_BAR_W * panel_w_pt / slots
    avail_pt = bar_w_pt * _AAAI_LABEL_FIT_FRAC
    widest_em = _aaai_label_em("88.88")  # 'XX.XX' upper bound (~2.36 em)
    pt = avail_pt / widest_em if widest_em > 0 else _AAAI_LABEL_PT_MAX
    return max(_AAAI_LABEL_PT_MIN, min(_AAAI_LABEL_PT_MAX, pt))


def _aaai_bar_content(groups, power, cluster=False, x_offset=0.0,
                      value_labels=False, label_font=6.5,
                      series=None, bar_w=None,
                      group_order="mean", group_label_yshift_pt=None,
                      cluster_value_key=None, stagger_labels=False,
                      size_runs=False, run_tick_labels=False,
                      runs_above=False, run_sort_by_label=False,
                      run_gap_extra=0.0, cs_subrun_labels=False,
                      group_gap_extra=None, line_out=None):
    """Build the in-axis drawing commands for one plot. `groups` is a list of
    (type_disp, item_label, {sk:(value, partial[, ci_half[, gap_pp]])}). An
    optional third element `ci_half` (the solve-time bars supply it) is the
    confidence-interval half-width in the bar's units and draws a black I-beam
    centred on the bar top; when absent or None no I-beam is drawn. An optional
    FOURTH element `gap_pp` is that bar's remaining bound gap
    delta_u - delta_l in percentage points of delta_max. Nothing draws it at
    present; it is carried so a chosen encoding of the remaining gap has the
    number to hand. `series` is the per-method
    style table (_AAAI_CHART_SERIES for the solve-time figures,
    _AAAI_CHART_SERIES_GAP for the bounds-difference figures) and fixes the
    per-bar fill/draw colour, fill pattern, and left-to-right order.

    When `cluster` is set, the bar-groups are CLUSTERED by `type_disp` (the
    grouping key, e.g. "source = 2"): the groups are ordered by mean (or, with
    group_order="given", kept in first-seen order so the caller controls the
    order), the clusters within a group are ordered by mean, an extra gap
    separates the groups, and a small bold group label is placed below each
    group (so the per-cluster x labels show only the short `item_label`).
    `group_label_yshift_pt` places that label: None keeps the legacy formula
    sized for ROTATED item labels; a number is the yshift (pt, negative =
    below the axis) for HORIZONTAL item labels. When `cluster` is False, all
    bar-groups are flattened into one mean-ordered list (type ignored) and
    `item_label` is the full x label; `x_offset` shifts the first cluster
    right in both modes (used to centre a sparse panel).

    Within every bar-group the bars are drawn in the FIXED order transfer,
    ours, vaghar (left to right), ADJACENT, coloured AND textured per method
    (each gets its `series` fill pattern overlaid); a partial bar
    gets a red `*`. Heights pass through f(v)=v**power. Returns (lines, xmax,
    xtick_str, xticklabels_str); each item_label is wrapped in braces here.

    `bar_w` is the bar width in DATA units (from _aaai_bar_data_w, so every
    figure's bars share ONE physical width); None keeps the legacy fixed
    _AAAI_BAR_W data-unit width.

    `cluster_value_key` names ONE series (e.g. "vaghar"): each cluster then
    prints that series' value once, centred ABOVE the cluster's tallest
    decoration (bar top or I-beam cap), at \\small -- the AAAI caption size --
    so the baseline's number anchors the whole cluster without per-bar
    clutter. None (default) prints nothing."""
    # Resolve against the LIVE global (set_aaai_chart_taus may have
    # rebuilt it); a default arg would have frozen it at import time.
    if series is None:
        series = _AAAI_CHART_SERIES
    style_of = {k: sty for k, _l, sty, _pat in series}
    pattern_of = {k: pat for k, _l, _s, pat in series}
    # Fixed within-group left-to-right order: transfer, ours, vaghar.
    bar_rank = {k: i for i, (k, _l, _s, _p)
                in enumerate(reversed(series))}
    lines = []
    xticks = []  # (xpos, item_label)

    def _bars_mean(bars):
        vs = [pair[0] for pair in bars.values()]
        return sum(vs) / len(vs) if vs else 0.0

    # One width for every bar in the figure (see docstring); resolved once so
    # the closure below and the group geometry agree.
    bar_w_data = bar_w if bar_w is not None else _AAAI_BAR_W

    def _draw_group(x, bars):
        items = sorted(bars.items(), key=lambda kv: bar_rank.get(kv[0], 99))
        k = len(items)
        # Every bar is the SAME width regardless of how many methods the group
        # holds, so all bars in the figure render identically. Bars within a
        # group touch (stride == width); a full 3-bar group spans
        # 3*bar_w_data < 1.0 and so still fits inside its cluster slot.
        bar_w = bar_w_data
        stride = bar_w
        center0 = x - (k - 1) * stride / 2.0
        group_top = 0.0  # tallest decoration (bar/I-beam) of this cluster
        # Points for the overlaid line graph: this cluster's per-method bound
        # gap at each bar's x. Collected here because the x positions are only
        # known inside this closure; the caller draws them in a SECOND axis
        # carrying its own right-hand scale.
        cluster_pts = []
        for bi, (sk, pair) in enumerate(items):
            v, p = pair[0], pair[1]
            ci = pair[2] if len(pair) > 2 else None
            xc = center0 + bi * stride
            xl = xc - bar_w / 2.0
            xr = xc + bar_w / 2.0
            h = v ** power
            rect = (f"(axis cs:{xl:.4g},0) rectangle "
                    f"(axis cs:{xr:.4g},{h:.4g})")
            lines.append(
                f"\\filldraw[{style_of[sk]}, line width=0.2pt] {rect};")
            if pattern_of[sk]:
                lines.append(f"\\fill[{pattern_of[sk]}] {rect};")
            # Confidence-interval I-beam, CENTRED on the bar top (the mean): the
            # stem runs mean+-ci through the axis transform, the two short caps
            # span _AAAI_IBEAM_CAP_FRAC of the bar width to each side. Drawn in
            # black (not the bar colour) so the lower half stays legible inside
            # the fill and the marker survives grayscale printing. `ci` is None
            # when n<3 (see _aaai_ci_halfwidth) -> no marker.
            deco_top = h
            if ci is not None:
                # Drawn 1:1 on the minute axis: the whisker reaches the full 95%
                # CI half-width above and below the bar mean.
                ci_draw = ci
                y_hi = (v + ci_draw) ** power
                y_lo = max(v - ci_draw, 0.0) ** power
                cap = bar_w * _AAAI_IBEAM_CAP_FRAC
                lines.append(
                    f"\\draw[black, line width=0.5pt] "
                    f"(axis cs:{xc:.4g},{y_lo:.4g}) -- "
                    f"(axis cs:{xc:.4g},{y_hi:.4g});")
                for yy in (y_hi, y_lo):
                    lines.append(
                        f"\\draw[black, line width=0.5pt] "
                        f"(axis cs:{xc - cap:.4g},{yy:.4g}) -- "
                        f"(axis cs:{xc + cap:.4g},{yy:.4g});")
                deco_top = y_hi
            if value_labels:
                # Value label printed HORIZONTALLY, centred just above the bar's
                # decoration stack -- the bar top, or the I-beam's upper cap when
                # a CI is drawn -- with anchor=south. The figure-wide point size
                # (label_font, from _aaai_hlabel_pt) is sized so even the widest
                # label fits the bar WIDTH with a small gap each side, so
                # horizontal numbers stay separated and readable.
                ly = deco_top
                lines.append(
                    f"\\node[anchor=south,"
                    f" font=\\fontsize{{{label_font:.3g}}}"
                    f"{{{label_font * 1.15:.3g}}}\\selectfont,"
                    f" inner sep=1pt, text=black, yshift=1.5pt] at "
                    f"(axis cs:{xc:.4g},{ly:.4g}) {{{_aaai_fmt_bar_value(v)}}};")
            if p:
                # \small (9pt): AAAI's floor for rendered figure text applies
                # to this marker too.
                lines.append(
                    r"\node[font=\small\bfseries, text=red,"
                    r" anchor=south, inner sep=1pt] at "
                    f"(axis cs:{xc:.4g},{h:.4g}) {{*}};")
            gap_pp = pair[3] if len(pair) > 3 else None
            if line_out is not None and gap_pp is not None:
                cluster_pts.append((xc, float(gap_pp)))
            group_top = max(group_top, deco_top)
        if line_out is not None and len(cluster_pts) > 1:
            line_out.append(cluster_pts)
        # One value per CLUSTER (user request): the named series' value --
        # \baseline, the reference every other bar is judged against --
        # centred above the cluster's tallest decoration, at \small (the
        # AAAI caption size, like all other figure text).
        if cluster_value_key is not None and cluster_value_key in bars:
            lines.append(
                f"\\node[anchor=south, font=\\small, inner sep=1pt,"
                f" text=black, yshift=1.5pt] at "
                f"(axis cs:{x:.4g},{group_top:.4g}) "
                f"{{{_aaai_fmt_bar_value(bars[cluster_value_key][0])}}};")

    if cluster and size_runs:
        # SIZE-RUN layout (user request): tight 1-slot pitch; NO per-cluster
        # tick labels (the caption states that each cluster is a different
        # $c_t$), one size label per run of adjacent same-size clusters just
        # below the axis (-2), the "$c{=}N$" source label below it (-15).
        # Positions, runs, and collision-driven extra air all come from
        # _aaai_size_run_layout -- the same walk the row packing measures
        # panels with. Tick MARKS stay at the cluster positions; the caller
        # sets xticklabels=\empty.
        items, run_lbls, srcs, xmax_pos, subruns = _aaai_size_run_layout(
            groups, sort_by_label=run_sort_by_label,
            run_gap_extra=run_gap_extra,
            group_gap_extra=group_gap_extra)
        for (xc, ct_line, bars) in items:
            # run_tick_labels: print the per-cluster line on every cluster;
            # cs_subrun_labels supersedes it (the line is written once per
            # stretch of adjacent clusters sharing it instead); the
            # solve-time panels use neither (their caption explains each
            # cluster is a different c_t).
            show = run_tick_labels and not cs_subrun_labels
            xticks.append((xc, ct_line if show else ""))
            _draw_group(xc, bars)
        if cs_subrun_labels:
            for (xc, lbl) in subruns:
                if not lbl:
                    # No per-cluster line on this figure (e.g. the solve-time
                    # figures name only c_s, once per block); skip the node.
                    continue
                lines.append(
                    f"\\node[anchor=north, font=\\small, inner sep=1pt,"
                    f" yshift=-2pt] at (axis cs:{xc:.4g},0) {{{lbl}}};")
        # With a first label row below the axis (per-cluster ticks or the
        # once-per-stretch line), the lower rows drop one level (~12pt) so
        # nothing overlaps. runs_above puts the run labels ABOVE the plot box
        # instead (e.g. the bounds-difference "arch, dataset" line, user
        # request), freeing the level below.
        has_row0 = run_tick_labels or cs_subrun_labels
        if runs_above:
            run_y, src_y = None, (-14.0 if has_row0 else -2.0)
        else:
            run_y, src_y = ((-14.0, -27.0) if has_row0
                            else (-2.0, -15.0))
        for (xc, size) in run_lbls:
            if not size:
                # A figure with a single model writes no per-block model line
                # (it names the model in its caption instead); skip the node
                # rather than emit an empty one.
                continue
            if runs_above:
                # `(A |- B)` is plain tikz (no `calc` needed): x of the data
                # coordinate, y of the axis top.
                lines.append(
                    f"\\node[anchor=south, font=\\small, align=center,"
                    f" inner sep=1pt, yshift=2pt] at "
                    f"({{axis cs:{xc:.4g},0}}|-{{rel axis cs:0,1}}) "
                    f"{{{size}}};")
            else:
                lines.append(
                    f"\\node[anchor=north, font=\\small, inner sep=1pt,"
                    f" yshift={run_y:.4g}pt] at (axis cs:{xc:.4g},0) "
                    f"{{{size}}};")
        for (xc, src) in srcs:
            if not src:
                # Single-model figures name the model in the caption instead.
                continue
            lines.append(
                f"\\node[anchor=north, font=\\small\\bfseries,"
                f" align=center, inner sep=1pt,"
                f" yshift={src_y:.4g}pt] at (axis cs:{xc:.4g},0) {{{src}}};")
        xtick = ",".join(f"{xp:.4g}" for xp, _l in xticks)
        if run_tick_labels and not cs_subrun_labels:
            xticklabels = ",".join(
                "{" + lbl.replace(",", "{,}") + "}" for _xp, lbl in xticks)
            return lines, xmax_pos, xtick, xticklabels
        return lines, xmax_pos, xtick, ""

    if cluster:
        from collections import OrderedDict
        by_type = OrderedDict()
        for (type_disp, item_label, bars) in groups:
            by_type.setdefault(type_disp, []).append((item_label, bars))
        if group_order == "given":
            # First-seen order: the caller pre-sorted the groups (e.g. the
            # solve-time panels append source classes ascending).
            ordered_types = list(by_type.items())
        else:
            ordered_types = sorted(
                by_type.items(),
                key=lambda kv: sum(_bars_mean(b)
                                   for _l, b in kv[1]) / len(kv[1]))
        if group_label_yshift_pt is not None:
            type_yshift = group_label_yshift_pt
        else:
            # How far below the axis the type labels sit: clear the longest
            # (rotated) item label. item labels are short sizes -> char count
            # is a reliable proxy for their rotated drop.
            maxlen = max((len(lbl) for _t, items in ordered_types
                          for lbl, _b in items), default=1)
            type_yshift = -(maxlen * 3.8 + 16.0)
        x = 1.0 + x_offset
        for ci, (type_disp, items) in enumerate(ordered_types):
            if ci > 0:
                x += _AAAI_TYPE_GAP
            cstart = x
            for item_label, bars in sorted(items,
                                           key=lambda li: _bars_mean(li[1])):
                xticks.append((x, item_label))
                _draw_group(x, bars)
                x += 1.0
            cx = (cstart + (x - 1.0)) / 2.0
            lines.append(
                r"\node[anchor=north, font=\small\bfseries,"
                f" yshift={type_yshift:.4g}pt] at "
                f"(axis cs:{cx:.4g},0) {{{type_disp}}};")
        xmax_pos = x - 1.0
    else:
        # x_offset shifts the first group right (used to CENTER a panel that
        # has fewer groups than the figure-wide maximum, so every panel keeps
        # the same data->cm scale and therefore the same physical bar width).
        # group_order="given" keeps the caller's cluster order (the pooled
        # bounds-difference figures pre-sort by perturbation type+size);
        # the default remains mean-ordered.
        ordered = (groups if group_order == "given" else
                   sorted(groups, key=lambda g: _bars_mean(g[2])))
        x = 1.0 + x_offset
        for (_type_disp, item_label, bars) in ordered:
            xticks.append((x, item_label))
            _draw_group(x, bars)
            x += 1.0
        xmax_pos = x - 1.0

    xtick = ",".join(f"{xp:.4g}" for xp, _l in xticks)
    if stagger_labels:
        # The hard-coded 30pt cluster pitch is narrower than a two-line
        # cluster label (~50pt), so side-by-side tick labels would overlap.
        # Instead the labels are drawn as manual nodes ALTERNATING between
        # two depths below the axis -- neighbours on the same level are two
        # pitches (60pt) apart, which fits the widest label. The tick MARKS
        # stay (xtick returned as usual); the caller must set
        # xticklabels=\empty.
        for i, (xp, lbl) in enumerate(xticks):
            depth = -3.0 if i % 2 == 0 else -26.0
            lines.append(
                f"\\node[anchor=north, font=\\small, align=center,"
                f" inner sep=1pt, yshift={depth:.4g}pt] at "
                f"(axis cs:{xp:.4g},0) {{{lbl}}};")
        return lines, xmax_pos, xtick, ""
    # Commas inside a label are wrapped as {,}: with a SINGLE label the value
    # {label} is fully braced, pgfkeys strips the braces (repeatedly), and a
    # bare comma then splits the label at the list level -- "(1,1)" rendered
    # as "(1". The {,} group renders as a plain comma but is never a list
    # separator, whatever the label count.
    xticklabels = ",".join(
        "{" + lbl.replace(",", "{,}") + "}" for _xp, lbl in xticks)
    return lines, xmax_pos, xtick, xticklabels


def _aaai_bar_figure(title, label, ylabel, caption, groups, wide=False):
    """Emit one bar figure (the three vaghar/ours/ours-with-transfer methods)
    as a list of LaTeX lines. When `wide` is set the figure spans both columns
    (figure*, full text width) -- used for the merged timed-out-cells chart.
    `groups` is a list of (type, label, {sk:(value, partial)}); see
    _aaai_bar_content for the sort/draw rules (bars are coloured AND textured
    per method). The y-axis shows the real values (linear). A light shaded plot
    background keeps this (timed-out) figure visually distinct from the time
    figures."""
    power = _AAAI_YAXIS_POWER
    allvals = [pair[0] for _t, _l, bars in groups for pair in bars.values()]
    vmax = max(allvals) if allvals else 1.0
    ymax, ytick_clause = _aaai_yaxis(vmax)
    # A bar within _AAAI_BD_CEIL_SLACK of 100 is effectively at the ceiling:
    # delta_u - delta_l cannot exceed delta_max, so a run that close has
    # learnt almost nothing. Mark the level so the reader sees the bar is
    # pinned rather than merely tall (user request).
    ceil_line = (100.0 - vmax) < _AAAI_BD_CEIL_SLACK
    if ceil_line:
        ymax = max(ymax, 100.0 ** power * 1.05)
    content, n, xtick, xticklabels = _aaai_bar_content(groups, power)
    fig_env = "figure*" if wide else "figure"
    plot_width = r"\textwidth" if wide else r"\columnwidth"
    out = []
    out.append("\\begin{" + fig_env + "}[!tbp]")
    out.append(r"\centering")
    out += _aaai_standalone_legend()
    out.append(r"\par\smallskip")
    out.append(r"\begin{tikzpicture}")
    out.append(
        r"\begin{axis}[" "\n"
        r"  width=" + plot_width + r", height=6cm," "\n"
        f"  ymin=0, ymax={ymax:.4g}, xmin=0.4, xmax={n + 0.6:g}, "
        f"{ytick_clause}" "\n"
        f"  ylabel={{{ylabel}}}," "\n"
        r"  ylabel style={font=\small}," "\n"
        f"  title={{{title}}}, title style={{font=\\small}}," "\n"
        f"  xtick={{{xtick}}}, xticklabels={{{xticklabels}}}," "\n"
        r"  x tick label style={rotate=90, anchor=east, font=\small},"
        "\n"
        r"  y tick label style={font=\small}," "\n"
        r"  axis background/.style={fill=black!4}," "\n"
        r"  ymajorgrids, major grid style={gray!25}]")
    out += content
    out.append(r"\end{axis}")
    out.append(r"\end{tikzpicture}")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{" + fig_env + "}")
    out.append("")
    return out


def _aaai_bd_single_figure(groups, force_timeout=None,
                           label_base="fig:n2-bounddiff", dataset_disp=None,
                           tex_dir=None, basename=None,
                           ds_phrase=None, extra_labels=()):
    """Emit the BOUNDS-DIFFERENCE chart as ONE flat grouped-bar figure (not a
    per-architecture grid) for a SINGLE dataset (the pooled-across-datasets
    figure was split per dataset on user request -- one call per dataset, see
    regenerate_aaai_bounddiff_section): every all-three-timeout cluster of that
    dataset sits on a single axis, ordered by magnitude, and each HORIZONTAL x
    label names the perturbation type + size and the architecture (plus any
    "missing Cs=k" note). The dataset (`dataset_disp`, when given) names the
    figure via its title, caption, and label suffix rather than being repeated
    on every x label. Bars are the mean delta_u-delta_l (% of delta_max,
    shorter = tighter) in the distinct bounds-difference palette, keeping the
    same per-method textures as the time figure. `groups` is a list of
    (dataset_disp, full_x_label, {sk:(gap, partial, ci_half)})."""
    power = _AAAI_YAXIS_POWER
    # NO confidence-interval I-beams here (user request): the CI half-widths
    # collected upstream are stripped from every bar, so only plain bars are
    # drawn and the axis top just clears the tallest bar.
    groups = [(t, l, {sk: (pair[0], pair[1]) for sk, pair in bars.items()})
              for (t, l, bars) in groups]
    allvals = [pair[0]
               for _t, _l, bars in groups for pair in bars.values()]
    vmax = max(allvals) if allvals else 1.0
    ymax, ytick_clause = _aaai_yaxis(vmax)
    # A bar within _AAAI_BD_CEIL_SLACK of 100 is effectively at the ceiling:
    # delta_u - delta_l cannot exceed delta_max, so a run that close has
    # learnt almost nothing. Mark the level so the reader sees the bar is
    # pinned rather than merely tall (user request).
    ceil_line = (100.0 - vmax) < _AAAI_BD_CEIL_SLACK
    if ceil_line:
        ymax = max(ymax, 100.0 ** power * 1.05)
    # Per-bar value labels removed (user request): bars carry no text above
    # them; heights read off the y-axis alone.
    # The plot box is pinned explicitly (`scale only axis`, \textwidth minus a
    # 1.5cm allowance for the ylabel + y-tick numbers) so the data->pt scale is
    # exact and the bars can take the hard-coded physical width every
    # evaluation figure shares. The x-range is 0.4..n+0.6 (span n+0.2).
    bd_overhead_cm = 1.5
    plotbox_pt = max(_AAAI_TEXTWIDTH_PT - bd_overhead_cm * _AAAI_PT_PER_CM, 1.0)
    # SIZE-RUN layout (same machinery as the solve-time panels): clusters at
    # the tight hard-coded pitch, "$c_s{=}N$" per cluster (tick labels ON),
    # "arch, dataset" once per adjacent sub-run, "type (size)" once per
    # block. The axis width DERIVES from the layout's span so the collision
    # spacing computed in pt holds exactly; if a chunk ever outgrew the text
    # width the slot would compress (labels may then crowd -- split further
    # upstream instead).
    model_extra = ((_AAAI_MODEL_GAP_PT - _AAAI_CLUSTER_GAP_PT)
                   / _AAAI_SLOT_PT)
    span = _aaai_panel_span(groups, sort_by_label=True,
                            run_gap_extra=model_extra)
    slot = _AAAI_SLOT_PT
    w = span * slot
    if w > plotbox_pt:
        slot, w = plotbox_pt / span, plotbox_pt
    content, n, xtick, xticklabels = _aaai_bar_content(
        groups, power, cluster=True, size_runs=True,
        runs_above=True, run_sort_by_label=True,
        run_gap_extra=model_extra, cs_subrun_labels=True,
        group_order="given",
        series=_AAAI_CHART_SERIES_GAP, bar_w=_AAAI_BAR_PT / slot)
    if ceil_line:
        y100 = 100.0 ** power
        content.append(
            f"\\draw[dashed, gray, line width=0.6pt] "
            f"(axis cs:0.5,{y100:.4g}) -- (axis cs:{n + 0.5:.4g},{y100:.4g});")
    # As in _aaai_combined_time_figure: the pgfplots graphic is pre-rendered to
    # a PDF (AAAI bans pgfplots in the paper source) while the caption/label stay
    # in the paper so the AAAI style typesets them.
    graphic = []
    graphic.append(r"\begin{tikzpicture}")
    graphic.append(
        r"\begin{axis}[" "\n"
        r"  scale only axis, "
        f"width={w:.2f}pt, "
        # Same plot-box height as the solve-time combination chart (user
        # request), so a bar reads at the same scale in both figure families.
        r"height=2.2cm, clip=false," "\n"
        f"  ymin=0, ymax={ymax:.4g}, xmin=0.5, xmax={n + 0.5:.4g}, "
        f"{ytick_clause}" "\n"
        r"  ylabel={$\delta_u-\delta_l$ (\%$\delta_{\max}$)}," "\n"
        r"  ylabel style={font=\small}," "\n"
        + (f"  xtick={{{xtick}}}, xticklabels={{{xticklabels}}},\n"
           if xticklabels else
           f"  xtick={{{xtick}}}, xticklabels=\\empty,\n")
        # HORIZONTAL labels, \small (9pt, the AAAI floor): the "$c_s{=}N$"
        # line (once per stretch), the "arch, dataset" line (above), and the
        # bold "type (size)" line are all drawn as nodes by the size-run
        # branch; pgfplots itself shows only the tick marks.
        + r"  x tick label style={align=center, font=\small}," "\n"
        r"  y tick label style={font=\small}," "\n"
        r"  axis background/.style={fill=black!4}," "\n"
        r"  ymajorgrids, major grid style={gray!25}]")
    graphic += content
    graphic.append(r"\end{axis}")
    graphic.append(r"\end{tikzpicture}")
    # Just this chunk's picture: the caller stacks EVERY chunk into ONE
    # figure (user request), adds the legend once, and renders the lot to a
    # single PDF.
    return graphic

def _aaai_standalone_legend(series=None):
    """A centred, self-contained legend -- one box + label per method, drawn
    manually with tikz nodes (chained left-to-right via the `positioning`
    library) so it sits once at the top of a figure. Each box shows the
    method's fill colour AND its fill pattern (overlaid), matching the bars.
    Entries are in REVERSED `series` order so the legend reads
    left-to-right in the same order the bars are drawn within a group
    (ours with transfer, ours, vaghar). Pass _AAAI_CHART_SERIES_GAP to match
    the bounds-difference figures' palette."""
    # Resolve against the LIVE global (set_aaai_chart_taus may have
    # rebuilt it); a default arg would have frozen it at import time.
    if series is None:
        series = _AAAI_CHART_SERIES
    # \small = 9pt, the AAAI caption size: every piece of figure text renders
    # at least as big as the caption (and clears the 9pt floor).
    out = [r"\begin{tikzpicture}[font=\small]"]
    prev = None
    for i, (_k, lbl, sty, pat) in enumerate(tuple(reversed(series))):
        box, txt = f"lb{i}", f"lt{i}"
        pos = "" if prev is None else f", right=14pt of {prev}"
        out.append(f"\\node[{sty}, line width=0.2pt, minimum width=0.32cm,"
                   f" minimum height=0.18cm, inner sep=0pt{pos}] ({box}) {{}};")
        if pat:
            out.append(f"\\fill[{pat}] ({box}.south west) rectangle "
                       f"({box}.north east);")
        out.append(f"\\node[right=2pt of {box}, inner sep=1pt] "
                   f"({txt}) {{{lbl}}};")
        prev = txt
    out.append(r"\end{tikzpicture}")
    return out


# Plots per row in the FLAT (packed) solve-time strip. ONE per row (full text
# width each) so the many per-(c_s, c_t) pair clusters have horizontal room; the
# x labels are rotated vertical (see the flat branch) so they never collide.
_AAAI_FLAT_MAX_COLS = 1


def _aaai_arch_typegrid(c_cols, types_order, cell_map, ylabel, nmax,
                        ncol_max, series=None,
                        unit_label="time (minutes)", col_titles=None,
                        flat_panels=None, timeout_min_of=None,
                        return_rows=False):
    """A 2-D groupplot whose COLUMNS are `c_cols` (left-to-right) and ROWS are
    the perturbation types `types_order` (top-to-bottom), so every
    (type, column) gets its own little graph. When `col_titles` is given (a dict
    column_key -> title string) the top-row title is that string -- used for the
    merged solve-time grid whose columns are architectures (titled by arch +
    dataset); otherwise the columns are source classes and the title is the c_s
    value (or "mean over c_s" for the merged column).
    `cell_map[(type_disp, column_key)]` is a list of (type, size, bars) for
    that panel (missing -> blank panel). Each GRAPH (panel) carries its OWN
    standalone linear (real-value) y-axis, fitted to that panel's tallest bar
    and printing its OWN y-tick numbers (user request: the y-axis stands on its
    own per graph rather than being shared/labelled once per row). Each row is
    still labelled on the left by its perturbation type, each column titled by
    its c_s (top row only), and the left-column axis label carries -- once per
    row -- the perturbation name plus, on an inner vertical line to its right,
    the y-axis unit "time (minutes)"; within a panel the bars are the per-size
    groups (mean-sorted).

    `timeout_min_of` marks the solver timeout on the SOLVE-TIME panels: either
    a dict {column_key: timeout in minutes} (the merged grid, one architecture
    per column) or a scalar in minutes (the per-arch figures, one cap for the
    whole figure). A panel whose tallest bar comes within
    _AAAI_TIMEOUT_LINE_SLACK_MIN minutes of its cap gets a dashed horizontal
    line at the timeout (and its y-axis is raised to keep the line inside the
    box); panels whose bars sit far below their cap stay unchanged, so the
    line only appears where a bar could be mistaken for a finished run."""
    # Resolve against the LIVE global (set_aaai_chart_taus may have
    # rebuilt it); a default arg would have frozen it at import time.
    if series is None:
        series = _AAAI_CHART_SERIES
    power = _AAAI_YAXIS_POWER

    def _panel_timeout_line(groups, col_key, slots):
        """(extra draw commands, min ymax) for one panel's timeout marker.
        Returns ([], 0) when no cap is set for the column or the tallest bar
        is more than _AAAI_TIMEOUT_LINE_SLACK_MIN minutes below it."""
        tm = (timeout_min_of.get(col_key)
              if isinstance(timeout_min_of, dict) else timeout_min_of)
        vmax = max((pair[0] for _t, _l, bars in groups
                    for pair in bars.values()), default=0.0)
        if tm is None or not groups or tm - vmax > _AAAI_TIMEOUT_LINE_SLACK_MIN:
            return [], 0.0
        y = tm ** power
        line = (f"\\draw[dashed, black!85, line width=0.6pt] "
                f"(axis cs:0.5,{y:.4g}) -- (axis cs:{slots + 0.5:g},{y:.4g});")
        # Name the level ON the chart: "timeout" just below the line at its
        # left end, where the mean-sorted clusters leave the panel empty (the
        # shortest bars sit leftmost). \small = 9pt, the same size the AAAI
        # style gives the figure caption (user request). Near-black (not the
        # frame's plain black) so the level reads dark yet still distinct
        # from the axis frame.
        word = (f"\\node[anchor=north west, font=\\small, text=black!85, "
                f"inner sep=1.5pt] at (axis cs:0.5,{y:.4g}) {{timeout}};")
        # Keep the dashed line strictly inside the plot box (a small margin
        # above it) so it reads as a level, not as the axis frame.
        return [line, word], y * 1.05

    # Per-PANEL (per graph) y-axis: each (type, c_s) graph is scaled to its OWN
    # tallest bar and shows its OWN y-tick numbers, so every graph reads on a
    # standalone axis. (This trades the old within-row comparability -- where a
    # row shared one scale -- for each graph being sized to its own data, as
    # requested.)
    def _panel_axis(groups, label_pt):
        # Cap the y-axis just above the DECORATION STACK of the tallest bar --
        # its top, plus (for a timed-out bar) the upper half of its I-beam, plus
        # the horizontal value label -- rather than by a blanket multiple of the
        # tallest bar. The old flat multiplier (~1.6x) wasted a wide band of
        # white space above tall bars: the value label is a FIXED pt height (not
        # proportional to the bar), so a tall bar needs proportionally little
        # headroom for it. We size each bar's requirement individually and take
        # the max, so a panel whose tallest bar is 2.66 tops out near ~3 instead
        # of ~4.7.
        bars_all = [pair for _t, _l, bars in groups for pair in bars.values()]
        vals = [pair[0] for pair in bars_all]
        vmax = max(vals) if vals else 1.0
        base_ymax, pytick = _aaai_yaxis(vmax)
        if power < 1.0 or vmax <= 0:
            # A power/sqrt axis keeps its curated ticks (base_ymax already sits
            # at the top tick); retain the generous headroom there.
            return base_ymax * (1.32 + _AAAI_IBEAM_PANEL_FRAC / 2.0), pytick
        # Value-label vertical extent (pt) as a fraction of the 2.5cm panel box:
        # font ascent + leading (label_pt*1.15) + the node's yshift/inner sep.
        panel_h_pt = 2.5 * _AAAI_PT_PER_CM
        label_frac = (label_pt * 1.15 + 2.5) / panel_h_pt
        # For each bar: top h, plus I-beam upper half (fraction of pymax), plus
        # the label band (fraction of pymax) must fit under pymax. Solve for the
        # smallest pymax per bar and take the max.
        need = vmax
        for pair in bars_all:
            v = pair[0]
            ci = pair[2] if len(pair) > 2 else None
            # Top of the bar's decoration stack: the bar top, or the I-beam's
            # upper cap (mean + full CI half-width, matching _draw_group)
            # when a confidence interval is drawn. The value label then needs
            # `label_frac` above that.
            top = ((v + ci) ** power) if ci is not None \
                else (v ** power)
            denom = max(1.0 - label_frac, 0.35)
            need = max(need, top / denom)
        # Cap at the previous blanket headroom so this only ever SHRINKS the
        # gap, never grows it (a rare panel whose tallest bar is a timed-out bar
        # with a very tall I-beam would otherwise want slightly more room; the
        # old rendering clipped that too, so matching it is safe).
        old_blanket = base_ymax * (1.32 + _AAAI_IBEAM_PANEL_FRAC / 2.0)
        return min(need * 1.03, old_blanket), pytick

    if flat_panels is not None:
        # PACKED-ROW layout (user request): each panel is only as wide as its
        # OWN content (its clusters at a per-panel fitted slot width, see
        # _aaai_panel_slot_pt), and consecutive panels share a row while they
        # fit \textwidth -- so almost-empty panels no longer burn a full row
        # each. Fewer rows keep every figure under the 0.88\textheight import
        # cap, so all figures render 1:1 and their \small text stays a true
        # 9pt (AAAI floor / "same size as the MNIST figure"). Each panel still
        # carries the architecture in its title and the perturbation name (+
        # y unit) rotated on its left. `flat_panels` is a list of
        # (col_key, col_title, type_disp, groups).
        #
        # Row-width bookkeeping is exact: a panel consumes its plot box plus
        # _AAAI_PANEL_DECO_PT for the rotated ylabel + y-tick numbers to its
        # left, plus _AAAI_PANEL_HSEP_PT between panels; rows are packed
        # greedily against \textwidth minus a small margin for the outermost
        # tick labels' overhang, so panels can never overlap or overrun the
        # fixed-width preview crop.
        geo = []
        for (ckey, col_title, type_disp, groups) in flat_panels:
            span = _aaai_panel_span(groups)
            # One uniform slot density everywhere (user request); only a
            # panel too dense for the text width compresses, as the old
            # full-width layout already did.
            slot = _AAAI_SLOT_PT
            w = span * slot
            max_w = _AAAI_TEXTWIDTH_PT - _AAAI_PANEL_DECO_PT
            if w > max_w:
                slot, w = max_w / span, max_w
            geo.append((ckey, col_title, type_disp, groups, span, slot, w))
        # FIRST-FIT packing: each panel joins the FIRST existing row with
        # room, else opens a new row. (Plain greedy-adjacent packing missed
        # easy shares: a wide panel between two narrow ones kept the narrow
        # pair apart.) A row's first panel pays the full 2-line-ylabel
        # reserve; every later panel pays the 1-line reserve plus the
        # inter-panel air.
        budget = _AAAI_TEXTWIDTH_PT - 6.0  # margin for edge-label overhang
        rows, row_w = [], []
        for g in geo:
            join = _AAAI_PANEL_HSEP_PT + _AAAI_PANEL_DECO_NEXT_PT + g[6]
            for i in range(len(rows)):
                if row_w[i] + join <= budget:
                    rows[i].append(g)
                    row_w[i] += join
                    break
            else:
                rows.append([g])
                row_w.append(_AAAI_PANEL_DECO_PT + g[6])
        # \small = 9pt everywhere: every piece of figure text is as big as
        # the AAAI caption (user request; also the 9pt label floor).
        # height=2.2cm (down from 2.5): with the two-line arch+dataset titles
        # a 4-row float at 2.5cm lands ~580pt, a hair over the ~572pt import
        # cap, and would be scaled to ~8.9pt text; 2.2cm keeps 4 rows at 1:1.
        axis_common = (
            r"scale only axis, height=2.2cm, clip=false, ymin=0," "\n"
            r"  ylabel style={font=\small, align=center},"
            r" y tick label style={font=\small}," "\n"
            # HORIZONTAL x labels (user request), two lines per cluster
            # (target class over perturbation size); align=center makes the
            # \\ line break work inside each label.
            r"  x tick label style={font=\small, align=center}," "\n"
            r"  title style={font=\small, align=center}," "\n"
            r"  ymajorgrids, major grid style={gray!25}")
        row_blocks = []
        for ri, row in enumerate(rows):
            out = [r"\begin{tikzpicture}"]
            for ci, (ckey, col_title, type_disp, groups,
                     span, slot, w) in enumerate(row):
                # label_pt=0: no value labels are drawn, so the y-axis only
                # needs to clear the bars/I-beams themselves.
                pymax, pytick = (_panel_axis(groups, 0.0)
                                 if groups else (1.0, ""))
                # CLUSTERED drawing (user request): clusters sharing a source
                # class sit together, each group labelled "source = <num>"
                # once below the horizontal per-cluster target/size labels
                # (group_order="given" keeps the caller's cs-ascending order;
                # yshift -25 tucks that label right under the two-line \small
                # tick labels). bar width: _AAAI_BAR_PT physical over this
                # panel's own slot width.
                # SIZE-RUN layout (user request): tight pitch, one size
                # label per run of adjacent same-size clusters; positions
                # and labels are computed in _aaai_size_run_layout.
                content, _n, xtick, xticklabels = _aaai_bar_content(
                    groups, power, cluster=True, size_runs=True,
                    series=series, bar_w=_AAAI_BAR_PT / slot,
                    group_order="given")
                # Dashed timeout level, only when a bar runs close to the cap.
                tline, tymin = _panel_timeout_line(groups, ckey, span)
                content += tline
                pymax = max(pymax, tymin)
                name = f"pnl{ri}x{ci}"
                opt = (f"name={name}, width={w:.2f}pt,\n  " + axis_common
                       + f",\n  xmin=0.5, xmax={span + 0.5:.4g}, "
                       f"ymax={pymax:.4g}")
                if ci > 0:
                    # Anchor on the previous panel's plot box, shifted right
                    # by the inter-panel AIR plus this panel's (1-line) left
                    # decorations, so panels can never touch. The shift is a
                    # plain \path ++ coordinate: the terser
                    # ([xshift=..]node.anchor) modifier needs the tikz `calc`
                    # library, and WITHOUT it the shift is silently dropped,
                    # gluing the panels together (the bug this replaces).
                    shift = _AAAI_PANEL_HSEP_PT + _AAAI_PANEL_DECO_NEXT_PT
                    out.append(
                        f"\\path (pnl{ri}x{ci - 1}.south east) "
                        f"++({shift:.2f}pt,0pt) coordinate (anc{ri}x{ci});")
                    opt += (f", at={{(anc{ri}x{ci})}}, "
                            f"anchor=south west")
                if pytick:
                    opt += ", " + pytick.rstrip(", ")
                if xtick and xticklabels:
                    opt += f", xtick={{{xtick}}}, xticklabels={{{xticklabels}}}"
                elif xtick:
                    # Staggered labels are drawn as nodes inside the axis;
                    # keep the tick marks, suppress pgfplots' own labels.
                    opt += f", xtick={{{xtick}}}, xticklabels=\\empty"
                else:
                    opt += r", xtick=\empty"
                # Every panel carries the architecture in its title and the
                # perturbation name rotated on its left; the gray
                # "time (minutes)" unit line (inheriting the \small
                # caption-size ylabel font) is written ONCE per row, on the
                # row's first panel only (user request).
                opt += r", title={" + (col_title or "") + r"}"
                if ci == 0:
                    opt += (", ylabel={" + type_disp
                            + r"\\{\color{gray!55!black} "
                            + unit_label + r"}}")
                else:
                    opt += ", ylabel={" + type_disp + "}"
                out.append(f"\\begin{{axis}}[{opt}]")
                out += content
                out.append(r"\end{axis}")
            out.append(r"\end{tikzpicture}")
            row_blocks.append(out)
        if return_rows:
            # The caller (the combined time figure) chunks the rows across
            # MULTIPLE figure* floats when one would exceed the import height
            # cap and be scaled below 9pt.
            return row_blocks
        joined = []
        for ri, blk in enumerate(row_blocks):
            if ri:
                joined.append(r"\par\smallskip")
            joined += blk
        return joined
    ncol, nrow = len(c_cols), len(types_order)
    # Every panel shares ONE x-range so all bars render at the same physical
    # width: the range spans the figure-wide maximum cluster count `nmax`
    # (sparser panels simply leave empty space on the right rather than widening
    # their bars). `scale only axis` makes `width` the plot-box width itself, so
    # the y-labelled left column gets the SAME box (and the same data->cm scale)
    # as the others -- otherwise its bars would be ~15% narrower. Panel width is
    # derived from \textwidth so the whole figure* fills the page without ever
    # overflowing it (AAAI-safe): an allowance covers the left column's y-label /
    # ticks plus the 4pt inter-panel separations.
    slots_global = max(nmax, 1)
    # Panel width is derived from the figure-wide MAXIMUM column count
    # (`ncol_max`), not this figure's own `ncol`, so every architecture's bars
    # are the same physical width: the densest figure exactly fills \textwidth,
    # and an architecture with fewer columns yields a narrower (centred) figure
    # rather than wider bars. Sizing for ncol_max guarantees no figure* overruns
    # the text width (AAAI-safe). Because every panel now prints its OWN y-tick
    # numbers (no `yticklabels at=edge left`), the inter-panel gap is widened so
    # an inner panel's y-numbers clear the neighbour to its left, and that extra
    # `horizontal sep` is folded into the width overhead so the figure still fits
    # \textwidth.
    overhead_cm = 1.5 + 0.75 * max(ncol_max - 1, 0)
    panel_w = (r"\dimexpr(\textwidth-" + f"{overhead_cm:.2g}" +
               f"cm)/{ncol_max}" + r"\relax")
    # Mirror the \dimexpr in Python (exact under `scale only axis`) to convert
    # the hard-coded physical bar width into this figure's data units, so bars
    # here match every other evaluation figure.
    panel_w_pt = max(_AAAI_TEXTWIDTH_PT - overhead_cm * _AAAI_PT_PER_CM,
                     1.0) / ncol_max
    bar_w_data = _aaai_bar_data_w(slots_global, panel_w_pt)
    out = [r"\begin{tikzpicture}"]
    out.append(
        r"\begin{groupplot}[" "\n"
        f"  group style={{group size={ncol} by {nrow}, horizontal sep=20pt,"
        r" vertical sep=24pt}," "\n"
        r"  scale only axis, width=" + panel_w + r", height=2.5cm, clip=false,"
        "\n"
        r"  ymin=0," "\n"
        # \small (9pt) throughout: AAAI's floor for rendered figure text.
        r"  ylabel style={font=\small, align=center},"
        r" y tick label style={font=\small}," "\n"
        r"  x tick label style={font=\small, align=center}," "\n"
        # align=center lets a column title carry a \\ line break (the merged
        # solve-time grid puts the architecture on line 1, the dataset on line 2).
        r"  title style={font=\small, align=center}," "\n"
        r"  ymajorgrids, major grid style={gray!25}]")
    for ri, type_disp in enumerate(types_order):
        for ci, c in enumerate(c_cols):
            groups = cell_map.get((type_disp, c), [])
            # Shared figure-wide x-range (`slots_global`) so every panel uses the
            # same data->cm scale and therefore the same physical bar width; a
            # panel with fewer perturbation sizes leaves empty space on the right
            # rather than widening its bars. Clusters sit at x=1..n_local.
            slots = slots_global
            # This graph's OWN y-axis, fitted to its own tallest bar
            # (label_pt=0: no value labels are drawn above the bars).
            pymax, pytick = _panel_axis(groups, 0.0) if groups else (1.0, "")
            if groups:
                # Centre this panel's clusters within the shared x-range: a panel
                # with fewer perturbation sizes than the figure-wide maximum is
                # shifted right by half the empty slots so its bars sit in the
                # middle of the graph rather than flush left.
                x_offset = (slots_global - len(groups)) / 2.0
                # No per-bar value labels (user request); `series` fixes the
                # per-method palette (solve-time vs bounds-difference).
                content, _n, xtick, xticklabels = _aaai_bar_content(
                    groups, power, cluster=False, x_offset=x_offset,
                    series=series, bar_w=bar_w_data)
                # Dashed timeout level, only when a bar runs close to the cap.
                tline, tymin = _panel_timeout_line(groups, c, slots)
                content += tline
                pymax = max(pymax, tymin)
            else:
                content, xtick, xticklabels = [], "", ""
            opt = f"xmin=0.5, xmax={slots + 0.5:g}, ymax={pymax:.4g}"
            if pytick:
                opt += ", " + pytick.rstrip(", ")
            if xtick:
                opt += f", xtick={{{xtick}}}, xticklabels={{{xticklabels}}}"
            else:
                opt += r", xtick=\empty"
            if ri == 0:
                # Column title (top row only): an explicit `col_titles` entry
                # (the merged solve-time grid -> arch + dataset) wins; else the
                # merged c_s column gets "mean over c_s" and a genuine per-Cs
                # column keeps its "$c_s{=}k$" title.
                if col_titles is not None:
                    opt += r", title={" + col_titles.get(c, "") + r"}"
                elif c == _MERGED_COL:
                    opt += r", title={mean over $c_s$}"
                else:
                    opt += r", title={$c_s{=}" + str(c) + r"$}"
            if ci == 0:
                # Left-column axis label, written ONCE per row: the perturbation
                # name plus, on a second (inner) vertical line to its right, the
                # y-axis unit (`unit_label` -- "time (minutes)" for the solve-time
                # figures, the bound-difference unit for the bounds-difference
                # ones). The rotated 2-line label stacks the first line on the
                # OUTER (left) side and the second on the INNER (right,
                # axis-facing) side (see `align=center` in the ylabel style).
                # Gray (inheriting the \small ylabel font, 9pt AAAI floor) so
                # the name stays dominant.
                opt += (", ylabel={" + type_disp
                        + r"\\{\color{gray!55!black} "
                        + unit_label + r"}}")
            out.append(f"\\nextgroupplot[{opt}]")
            out += content
    out.append(r"\end{groupplot}")
    out.append(r"\end{tikzpicture}")
    return out


def _aaai_group_grid_figure(arch_rows, ylabel, dataset_disp, label_base,
                            force_timeout=None, series=None,
                            unit_label="time (minutes)", bounddiff=False):
    """Emit one two-column `figure*` PER architecture (user: separate each
    architecture into its own figure). Each figure has its own shared legend,
    that architecture's 2-D groupplot grid (columns = c_s, rows = perturbation
    type, each row with its own linear y-axis fitted to that perturbation), and
    a caption naming the architecture;
    its label is `<label_base>-<arch>` (the first figure also carries the bare
    `<label_base>` so any older `\\ref` still resolves). `arch_rows` is a list
    of (arch, arch_disp, c_cols, types_order, cell_map). Every bar across every
    figure renders at the SAME physical width: panels share one figure-wide
    x-range (`nmax`) and are sized from the figure-wide maximum column count
    (`ncol_max`) under `scale only axis` (see _aaai_arch_typegrid), so a figure
    with fewer columns is simply narrower (centred) rather than carrying wider
    bars.

    `series`/`unit_label`/`bounddiff` select the SOLVE-TIME rendering (default)
    or the BOUNDS-DIFFERENCE one (distinct palette, gap unit, gap-specific
    caption) for the clusters where all three methods reached the timeout."""
    # Resolve against the LIVE global (set_aaai_chart_taus may have
    # rebuilt it); a default arg would have frozen it at import time.
    if series is None:
        series = _AAAI_CHART_SERIES
    nmax = max((len(groups)
                for (_ar, _ad, _c, _t, cm) in arch_rows
                for groups in cm.values()), default=1)
    ncol_max = max((len(c) for (_ar, _ad, c, _t, _cm) in arch_rows), default=1)
    out = []
    for ri, (arch, arch_disp, c_cols, types_order, cell_map) in enumerate(
            arch_rows):
        out.append(r"\begin{figure*}[tp]")
        out.append(r"\centering")
        out += _aaai_standalone_legend(series=series)
        out.append(r"\par\smallskip")
        # Solve-time figures mark the solver timeout (dashed level) on panels
        # whose tallest bar runs close to it; the cap is per-architecture and
        # this figure is one architecture, so a scalar covers every panel. The
        # bounds-difference figures chart a gap, not a time -- no line there.
        _tm = _ft_for(force_timeout, arch)
        out += _aaai_arch_typegrid(c_cols, types_order, cell_map, ylabel, nmax,
                                   ncol_max, series=series,
                                   unit_label=unit_label,
                                   timeout_min_of=(None if bounddiff
                                                   or _tm is None
                                                   else _tm / 60.0))
        if bounddiff:
            # Bounds-difference figure: only the all-three-timeout clusters,
            # where the bar height is the remaining bound gap delta_u-delta_l
            # (shorter = tighter) rather than a solve time.
            cap = (r"Remaining bound difference $\delta_u-\delta_l$ (as a "
                   r"percentage of $\delta_{\max}$; shorter is tighter) of "
                   r"\tool compared to \baseline (baseline) on "
                   + dataset_disp + r" (" + arch_disp + r"), for the "
                   r"perturbation clusters where all three methods reach the "
                   r"timeout. Each bar is the mean over the source classes "
                   r"$c_s$; a cluster is only shown for a $c_s$ that has all "
                   r"three methods, and any $c_s$ absent from a cluster is "
                   r"marked on its label.")
        else:
            cap = (r"\tool compared to \baseline (baseline) on "
                   + dataset_disp + r" (" + arch_disp + r") across different perturbation "
                   r"types and sizes. Each bar is the mean solve time over the "
                   r"source classes $c_s$; a cluster is only shown for a $c_s$ that "
                   r"has all three methods, and any $c_s$ absent from a cluster is "
                   r"marked on its label. Clusters where all three methods reach "
                   r"the timeout are moved to the bounds-difference figure.")
        cap += _timeout_caption_clause(force_timeout, arch)
        cap += _benchmark_provenance_clause(dataset_disp)
        out.append(f"\\caption{{{cap}}}")
        if ri == 0:
            out.append(f"\\label{{{label_base}}}")
        out.append(f"\\label{{{label_base}-{arch.replace('_', '')}}}")
        out.append(r"\end{figure*}")
        out.append("")
    return out


def _aaai_combined_time_figure(col_order, col_titles, types_order, cell_map,
                               caption, labels, tex_dir=None, basename=None,
                               force_timeout=None):
    """Emit ONE solve-time `figure*`: a 2-D groupplot grid whose COLUMNS are
    `col_order` (architectures, titled via `col_titles`) and whose ROWS are the
    perturbation types `types_order`, so every little graph sits at
    (row=perturbation type, col=architecture). The bars/values are unchanged
    from the old per-arch figures -- only their layout moves. Bars auto-narrow
    because the panel width divides `\\textwidth` by the number of columns.

    `cell_map[(type_disp, col_key)]` is the list of per-size bar-groups for that
    panel. `caption` is the caption TEXT and `labels` the list of `\\label`
    names to emit (the caller assigns per-database labels + arch aliases)."""
    # Panel x-extent now includes the inter-group gaps of the source-class
    # clustering (see _aaai_panel_span), so the shared figure-wide x-range --
    # and hence the physical bar width conversion -- stays exact.
    nmax = max((_aaai_panel_span(groups) for groups in cell_map.values()),
               default=1)
    ncol_max = max(len(col_order), 1)
    # FLAT layout: only the NON-EMPTY (arch, perturbation) plots, ordered
    # architecture-major (all of one architecture's perturbations together),
    # each its own plot with the architecture in the title and the perturbation
    # rotated on the left -- so the empty grid cells disappear.
    #
    # Panel title: ONE line, "arch, dataset", one style/colour (user request
    # -- the earlier arch-over-gray-dataset stack and the size-in-title
    # factoring are gone; the perturbation size lives on the x-axis, as the
    # second line of every cluster label).
    flat_panels = []
    for c in col_order:
        for type_disp in types_order:
            groups = cell_map.get((type_disp, c))
            if not groups:
                continue
            arch_disp, ds = (col_titles.get(c, ("", ""))
                             if isinstance(col_titles.get(c), tuple)
                             else (col_titles.get(c, ""), ""))
            title = arch_disp + (", " + ds if ds else "")
            flat_panels.append((c, title, type_disp, groups))
    # Per-column solver cap in MINUTES, for the dashed timeout level on
    # panels whose tallest bar runs close to it. A column key is either a
    # bare arch or, in the pooled cross-dataset layout, a (dataset, arch)
    # pair -- the cap is per-arch in both cases. Columns with no cap map to
    # None (no line ever).
    timeout_min_of = {}
    for c in col_order:
        arch_of_c = c[1] if isinstance(c, tuple) else c
        ft = _ft_for(force_timeout, arch_of_c)
        timeout_min_of[c] = ft / 60.0 if ft is not None else None
    # # A caption clause is only warranted when some panel actually shows the
    # # line: a panel qualifies when its tallest bar is within the slack of its
    # # column's cap (the same test _panel_timeout_line applies).
    # has_timeout_line = any(
    #     tm is not None
    #     and tm - max((pair[0] for _t, _l, bars in groups
    #                   for pair in bars.values()), default=0.0)
    #     <= _AAAI_TIMEOUT_LINE_SLACK_MIN
    #     for (c, _ct, _td, groups) in flat_panels
    #     for tm in [timeout_min_of.get(c)])
    # if has_timeout_line:
    #     caption += (r" A dashed horizontal line marks the solver timeout, "
    #                 r"shown when a bar runs close to it.")
    # The graphic (legend + packed rows) is pgfplots, which the AAAI style bans
    # in the paper source, so it is rendered to a PDF here and imported. The
    # caption and labels stay in the paper: they are typeset by the AAAI style
    # at 10pt roman and must not be baked into the image.
    row_blocks = _aaai_arch_typegrid(
        col_order, types_order, cell_map, r"time (min)", nmax, ncol_max,
        series=_AAAI_CHART_SERIES, unit_label="time (minutes)",
        col_titles=col_titles, flat_panels=flat_panels,
        timeout_min_of=timeout_min_of, return_rows=True)
    # A figure taller than the import cap (height=0.88\textheight,
    # keepaspectratio) would be scaled DOWN and its \small text would land
    # under AAAI's 9pt floor. Chunk the rows across as many figure* floats as
    # needed (balanced, so no float gets a lone straggler row); every chunk
    # then renders 1:1.
    n_figs = max(1, math.ceil(len(row_blocks) / _AAAI_MAX_ROWS_PER_FIG))
    per_fig = math.ceil(len(row_blocks) / n_figs) if row_blocks else 1
    out = []
    for fi in range(n_figs):
        chunk = row_blocks[fi * per_fig:(fi + 1) * per_fig]
        graphic = []
        graphic += _aaai_standalone_legend(series=_AAAI_CHART_SERIES)
        graphic.append(r"\par\smallskip")
        for ri, blk in enumerate(chunk):
            if ri:
                graphic.append(r"\par\smallskip")
            graphic += blk
        bname = basename if fi == 0 else f"{basename}{'bcdef'[fi - 1]}"
        cap = caption
        if fi > 0:
            cap = (cap[:-1] if cap.endswith(".") else cap) + " (continued)."
        out.append(r"\begin{figure*}[tp]")
        out.append(r"\centering")
        out.append(_aaai_render_chart_pdf(graphic, bname, tex_dir))
        out.append(f"\\caption{{{cap}}}")
        if fi == 0:
            for lb in labels:
                out.append(f"\\label{{{lb}}}")
        out.append(r"\end{figure*}")
        out.append("")
    return out


# Sentinel column key for the MERGED-over-source-classes chart: the old grid
# had one column per source class c_s; now the c_s columns are collapsed into a
# single column whose bars are the mean over c_s (see _render_aaai_n2_charts).
_MERGED_COL = "__merged_cs__"


def _render_aaai_n2_charts(rows, archs, dataset, delta_max_by_key=None,
                           force_timeout=None, rerun_timeout_eps=30.0,
                           requested_c_targets=None,
                           requested_c_sources=None):
    """Emit, per architecture, up to TWO merged-over-c_s figures for the three
    methods: a SOLVE-TIME figure (bar height = mean minutes) for clusters where
    at least one method finishes, and a BOUNDS-DIFFERENCE figure (bar height =
    mean delta_u - delta_l as a percentage of delta_max, shorter is tighter)
    for clusters where all three methods reach the timeout -- there a time bar
    carries no signal, so the remaining bound gap is charted instead. The two
    figures use distinct palettes but the SAME bar textures. Bucketing mirrors
    the N2 per-cell table so the bar heights equal the table's cells (geom side
    where present, else base)."""
    if not rows:
        return [], []
    if force_timeout is not None:
        rows = [r for r in rows
                if not _aaai_is_timeout_mismatch(r, force_timeout,
                                                 rerun_timeout_eps)]
    from collections import defaultdict
    buckets = defaultdict(lambda: defaultdict(lambda: {
        "t_base": [], "t_geom": [],
        "lb_base": [], "lb_geom": [], "ub_base": [], "ub_geom": [],
        "c_targets": set(), "c_targets_geom": set(),
        # Per-target solve times (keyed by c_t) so the solve-time figure can draw
        # ONE cluster per (c_s, c_t) pair instead of a single averaged bar.
        "t_base_by_ct": defaultdict(list), "t_geom_by_ct": defaultdict(list),
        # Per-target bounds (keyed by c_t) so the bounds-difference figure can
        # draw ONE cluster per all-timeout (c_s, c_t) pair (no averaging).
        "lb_base_by_ct": defaultdict(list), "lb_geom_by_ct": defaultdict(list),
        "ub_base_by_ct": defaultdict(list),
        "ub_geom_by_ct": defaultdict(list)}))
    for r in rows:
        if r.get("role") != "N2":
            continue
        key = (r["arch"], r["perturbation"], r["perturbation_size"],
               r["c_source"])
        cell = buckets[key][r["combo"]]
        is_geom = bool(r.get("geom"))
        t = r.get("t_total")
        if t is not None:
            (cell["t_geom"] if is_geom else cell["t_base"]).append(t)
        lb = r.get("lb_total")
        if lb is not None and math.isfinite(lb):
            (cell["lb_geom"] if is_geom else cell["lb_base"]).append(lb)
        ub = r.get("ub_total")
        if ub is not None and math.isfinite(ub):
            (cell["ub_geom"] if is_geom else cell["ub_base"]).append(ub)
        # Track which target classes contributed, per side, so a bar whose
        # mean covers only some of the expected c_targets can be flagged
        # with a red asterisk (same partial-coverage notion as the tables).
        ct = r.get("c_target")
        if ct is not None:
            try:
                cti = int(ct)
                (cell["c_targets_geom"] if is_geom
                 else cell["c_targets"]).add(cti)
                # Keep this run's time and bounds under their own target so
                # the figures can split cells into per-(c_s, c_t) clusters.
                if t is not None:
                    (cell["t_geom_by_ct"] if is_geom
                     else cell["t_base_by_ct"])[cti].append(t)
                if lb is not None and math.isfinite(lb):
                    (cell["lb_geom_by_ct"] if is_geom
                     else cell["lb_base_by_ct"])[cti].append(lb)
                if ub is not None and math.isfinite(ub):
                    (cell["ub_geom_by_ct"] if is_geom
                     else cell["ub_base_by_ct"])[cti].append(ub)
            except (TypeError, ValueError):
                pass

    dataset_disp = _dataset_display_name(dataset)
    series_keys = [k for k, _lbl, _sty, _pat in _AAAI_CHART_SERIES]
    # Two independent figures now: the SOLVE-TIME grid (clusters where at least
    # one method finished) and the BOUNDS-DIFFERENCE figure (clusters where all
    # three methods reached the timeout, so a time bar carries no signal and the
    # remaining delta_u-delta_l gap is charted instead).
    #
    # The solve-time charts are merged into ONE grid whose COLUMNS are the
    # networks -- an (architecture, dataset) pair -- and whose ROWS are the
    # perturbation types. Since that grid pools EVERY dataset, this function only
    # COLLECTS its dataset's time cells (each tagged with dataset + arch) and
    # RETURNS them; the caller pools across datasets and emits the one figure.
    time_cells = []   # (dataset_disp, arch, arch_disp, type_disp, c_src,
                      #  item_label, bars)
    # The bounds-difference figure is a SINGLE flat axis (not a per-arch grid):
    # every all-three-timeout cluster across all architectures is collected here
    # as ("", full_x_label, bars), the x label naming type+size, arch, dataset.
    bd_groups = []
    for arch in archs:
        arch_disp = _AAAI_ARCH_DISPLAY.get(arch, arch.replace("_", r"\_"))
        c_sources = sorted({k[3] for k in buckets if k[0] == arch},
                           key=lambda c: int(c))
        # The per-c_s columns are MERGED into one: for each (perturbation type,
        # size) we average every method's solve time (and I-beam gap) over the
        # source classes whose cell passed the "all three methods present +
        # complete" filter, then note any universe Cs that did NOT pass for that
        # cluster on the x label. merge_acc keeps, per (type_disp, size_disp),
        # the passing per-Cs bar dicts so they can be averaged below.
        merge_acc = {}  # (type_disp, size_disp) -> {c_src: bars}
        for c_src in c_sources:
            dmax_ub = None
            if delta_max_by_key is not None:
                dm = delta_max_by_key.get((arch, "N2", c_src))
                if dm is not None:
                    u = dm.get("upper")
                    if u is not None and math.isfinite(u) and abs(u) > 1e-9:
                        dmax_ub = u
            # Expected target classes for every cell in this block: every
            # class index except the source class. A bar whose mean covers
            # a strict subset of these is "partial" and gets a red asterisk.
            # When restricted via --ct (requested_c_targets, 0-indexed), the
            # expected set is exactly that request minus the self-pair, so the
            # "*" flags a requested c_target with no data.
            n_classes = _num_classes_for_dataset(dataset)
            _req_cts = _ct_for(requested_c_targets, arch)
            if _req_cts is not None:
                expected_cts = {ct for ct in _req_cts
                                if ct != int(c_src)}
            else:
                expected_cts = {ct for ct in range(n_classes)
                                if ct != int(c_src)}
            pert_keys = sorted(
                (k for k in buckets if k[0] == arch and k[3] == c_src),
                key=lambda k: (k[1], k[2]))  # (pert, p_size)
            for k in pert_keys:
                _, pert, p_size, _ = k
                cells = buckets[k]
                _ft = _ft_for(force_timeout, arch)
                times = {sk: _aaai_chart_time(cells.get(sk), _ft,
                                              prefer_geom=(sk != "vaghar"))
                         for sk in series_keys}  # sk -> (value, side)
                present = [v for v, _s in times.values() if v is not None]
                if not present:
                    continue
                type_disp = (pert.replace("_", r"\_")
                             .replace("linf", r"$\ell_\infty$"))
                size_disp = ("(" + p_size + ")").replace("_", r"\_")
                # Every perturbation is a regular time bar (clamped to the
                # wall-clock cap): even a cell where all three methods time
                # out is now kept here -- its bars sit flat at the cap and are
                # told apart by their I-beam (the remaining delta_u-delta_l
                # gap), so no separate bound-gap figure is needed.
                bars = {}
                for sk in series_keys:
                    v, side = times[sk]
                    if v is not None:
                        partial = _aaai_partial(cells.get(sk), side,
                                                expected_cts)
                        # Remaining bound gap for this bar's I-beam (same
                        # delta_u-delta_l, in pp of delta_max, the appendix
                        # N2 table reports); None when it cannot be computed
                        # -> the bar simply carries no marker.
                        gap, _gside = _aaai_chart_gap(
                            cells.get(sk), dmax_ub,
                            prefer_geom=(sk != "vaghar"))
                        # Individual per-target times (minutes) for this cell:
                        # the raw class-pair samples the cluster pools for the
                        # confidence-interval I-beam.
                        tlist, _tside = _aaai_chart_times_list(
                            cells.get(sk), _ft, prefer_geom=(sk != "vaghar"))
                        # Individual per-target bound gaps for this cell: the raw
                        # class-pair samples the bounds-difference I-beam pools,
                        # mirroring tlist for the time bars.
                        glist, _glside = _aaai_chart_gaps_list(
                            cells.get(sk), dmax_ub,
                            prefer_geom=(sk != "vaghar"))
                        # Per-(c_t) times for this cell so the solve-time figure
                        # can draw one cluster per (c_s, c_t) pair (no averaging).
                        tbyct, _tbcside = _aaai_chart_time_by_ct(
                            cells.get(sk), _ft, prefer_geom=(sk != "vaghar"))
                        # Per-(c_t) bound gaps: the bounds-difference analogue,
                        # so all-timeout pairs get their own cluster too.
                        gbyct, _gbcside = _aaai_chart_gap_by_ct(
                            cells.get(sk), dmax_ub,
                            prefer_geom=(sk != "vaghar"))
                        bars[sk] = (v, partial, gap, tlist, glist, tbyct,
                                    gbyct)
                # Keep the per-Cs filter (user request): a source class Cs
                # contributes to the merged cluster ONLY if it has ALL THREE
                # methods (ours with transfer, ours, vaghar) and each covers
                # every EXPECTED target class. A Cs failing this is simply not
                # averaged in (and shows up as a "missing Cs=k" note instead).
                if (len(bars) == len(series_keys)
                        and not any(pair[1] for pair in bars.values())):
                    merge_acc.setdefault((type_disp, size_disp), {})[c_src] = bars

        # Collapse the passing per-Cs bars into ONE merged column, then ROUTE
        # each cluster to one of the two figures: if all three merged bars hit
        # the timeout the cluster goes to the BOUNDS-DIFFERENCE grid (bar height
        # = mean remaining delta_u-delta_l gap); otherwise it stays in the
        # SOLVE-TIME grid (bar height = mean solve time). Any universe Cs absent
        # from a cluster is appended to its x label as "missing Cs=k" in both.
        _ft = _ft_for(force_timeout, arch)
        # A merged bar "reached the timeout" when its mean solve time is pinned
        # at the wall-clock cap (all contributing Cs runs were clamped there);
        # rerun_timeout_eps (seconds) is the same slack the sweep uses.
        cap_min = (_ft / 60.0) if _ft is not None else None
        eps_min = rerun_timeout_eps / 60.0

        def _timed_out(v):
            return cap_min is not None and v >= cap_min - eps_min

        for (type_disp, size_disp), bars_by_cs in merge_acc.items():
            # PAIR-LEVEL routing (user request): every (c_s, c_t) pair is its
            # own cluster. A pair where ALL THREE methods reach the timeout
            # goes to the bounds-difference figures (its own remaining gap,
            # no averaging); a pair with solve-time signal goes to the
            # solve-time grid unless the baseline already solves it within
            # the 15-minute cut. This also closes the old coverage gap where
            # all-timeout pairs inside a mixed cell appeared in NEITHER
            # figure.
            for c_src in sorted(bars_by_cs, key=lambda c: int(c)):
                cbars = bars_by_cs[c_src]
                ct_sets = [set(cbars[sk][5].keys()) for sk in series_keys
                           if sk in cbars and len(cbars[sk]) > 5]
                if len(ct_sets) != len(series_keys):
                    continue
                common_cts = set.intersection(*ct_sets)
                for ct in sorted(common_cts, key=int):
                    pair_times = {sk: cbars[sk][5][ct]
                                  for sk in series_keys}
                    if all(_timed_out(pair_times[sk])
                           for sk in series_keys):
                        # All three at the cap: no time signal. Chart the
                        # pair's own remaining delta_u-delta_l gap instead
                        # (skipped only if some method's per-target bounds
                        # are unavailable).
                        pair_gaps = {sk: cbars[sk][6].get(ct)
                                     for sk in series_keys
                                     if len(cbars[sk]) > 6}
                        if (len(pair_gaps) != len(series_keys)
                                or any(g is None
                                       for g in pair_gaps.values())):
                            continue
                        bd_bars = {sk: (pair_gaps[sk], False)
                                   for sk in series_keys}
                        bd_groups.append((dataset_disp, type_disp,
                                          size_disp, arch_disp,
                                          int(c_src), bd_bars))
                        continue
                    # Pairs EVERY method already solves quickly say nothing
                    # about acceleration -- drop them from the chart (they
                    # remain in the appendix tables; the caption states the
                    # cut). Cutting on the baseline alone also hid the pairs
                    # where \baseline is quick but a method is not, which is
                    # exactly where the comparison has something to say, so
                    # the whole cluster has to clear the cut. A missing time
                    # keeps the pair, since it cannot be shown to be quick.
                    if all(pair_times.get(sk) is not None
                           and pair_times[sk] <= _AAAI_TIME_CHART_MIN_SOLVE_MIN
                           for sk in series_keys):
                        continue
                    # Every bar also carries its remaining bound gap
                    # (delta_u - delta_l, in pp of delta_max, the number the
                    # appendix tables report) as a 4th slot. It is not drawn on
                    # the time bar: it becomes the companion GAP ROW under this
                    # panel, so both quantities are bar lengths on their own
                    # axis and any two clusters compare directly. Carried for
                    # solved bars too, where it is ~0 and draws a flat bar --
                    # which is the honest reading, nothing left open.
                    pair_bars = {}
                    for sk in series_keys:
                        _g = (cbars[sk][6].get(ct)
                              if len(cbars[sk]) > 6 else None)
                        pair_bars[sk] = (pair_times[sk], False, None, _g)
                    # Cluster x label, HORIZONTAL, two lines: the target
                    # class ($c_t{=}N$, matching the paper's notation) on
                    # top, the perturbation size below. The source class
                    # is NOT repeated here -- clusters sharing a source
                    # sit together as a group whose "$c{=}<num>$" label
                    # is printed once under the group (see the cluster
                    # branch of _aaai_bar_content).
                    pair_label = (r"$c_t{=}" + str(ct) + r"$\\"
                                  + size_disp)
                    # Collected (not emitted): the caller pools these across
                    # every dataset into ONE grid at (row=type, col=arch).
                    time_cells.append(
                        (dataset_disp, arch, arch_disp, type_disp,
                         int(c_src), pair_label, pair_bars))

    # Return the solve-time cells and the bounds-difference clusters; BOTH are
    # pooled across datasets by the caller and drawn as ONE combined figure each
    # (the solve-time grid with network columns, the bounds-difference flat
    # figure). Nothing is emitted per dataset here.
    return time_cells, bd_groups


def regenerate_aaai_n2_charts_section(tex_path, cwd, dataset, arch_runs,
                                      parse_result_file,
                                      seeds_filter=None,
                                      force_timeout=None,
                                      rerun_timeout_eps=30.0,
                                      begin_mark=AAAI_N2_CHARTS_BEGIN_MARK,
                                      end_mark=AAAI_N2_CHARTS_END_MARK,
                                      ds_label_suffix="",
                                      advstd_meta_fn=None,
                                      perts=None,
                                      combination_filter=None,
                                      requested_c_targets=None,
                                      requested_c_sources=None,
                                      stale_fn=None):
    """Collect the same per-cell rows as the N2 table and emit the
    per-perturbation solve-time and bound-gap charts into the evaluation
    body between the chart marks.

    Like regenerate_aaai_wide_perarch_section, the transfer (advstd N2) rows
    come from the advStd .txt files directly when `advstd_meta_fn`/`perts` are
    supplied, else from the CSVs. `stale_fn` (_is_pre_fix_dropped) drops
    pre-fix files that relaxed >=1 binary, on both the vaghar/ours and
    transfer rows.

    Returns the dataset's BOUNDS-DIFFERENCE clusters (`bd_groups`, each already
    labelled with this dataset) so the caller can pool them across every dataset
    into a single combined bounds-difference figure. Returns [] on any error."""
    try:
        rows = _collect_wide_perarch_cells(arch_runs, cwd, dataset,
                                           parse_result_file,
                                           seeds_filter=seeds_filter,
                                           stale_fn=stale_fn)
        # Emit archs in Table-1 order so the per-arch charts (and the combined
        # cross-dataset grids, which follow encounter order) match the paper's
        # Networks table rather than the --arch_timeouts command order.
        archs = sorted((a for a, _ in arch_runs), key=_tab1_arch_sort_key)
        if advstd_meta_fn is not None and perts is not None:
            rows += _load_advstd_rows_for_wide_from_txt(
                cwd, dataset, archs, perts, parse_result_file, advstd_meta_fn,
                seeds_filter=seeds_filter,
                combination_filter=combination_filter,
                force_timeout=force_timeout,
                rerun_timeout_eps=rerun_timeout_eps,
                stale_fn=stale_fn)
        else:
            rows += _load_advstd_rows_for_wide(cwd, dataset, archs,
                                               seeds_filter=seeds_filter)
        rows = _filter_rows_by_c_targets(rows, requested_c_targets)
        rows = _filter_rows_by_c_sources(rows, requested_c_sources)
        delta_max_by_key = _load_delta_max_values(cwd, dataset, archs)
        time_cells, bd_groups = _render_aaai_n2_charts(
            rows, archs, dataset,
            delta_max_by_key=delta_max_by_key,
            force_timeout=force_timeout,
            rerun_timeout_eps=rerun_timeout_eps,
            requested_c_targets=requested_c_targets,
            requested_c_sources=requested_c_sources)
        # The per-dataset time block is retired: the solve-time charts are now
        # ONE cross-dataset grid written by regenerate_aaai_time_combined_section.
        # Clear this dataset's old block so no stale per-dataset figure lingers.
        update_aaai_wide_perarch_tex(
            tex_path,
            r"% (solve-time charts moved to the combined cross-dataset figure)",
            begin_mark=begin_mark, end_mark=end_mark,
            label_suffix=ds_label_suffix)
        return time_cells, bd_groups
    except SystemExit as exc:
        print(f"[update_advstd_tex_tables] aaai_n2_charts block skipped: "
              f"{exc}")
    except Exception as exc:
        print(f"[update_advstd_tex_tables] aaai_n2_charts block error: "
              f"{exc}")
    return [], []


AAAI_N2_TIME_COMBINED_BEGIN_MARK = "% BEGIN AUTO: aaai_n2_time_combined"
AAAI_N2_TIME_COMBINED_END_MARK   = "% END AUTO: aaai_n2_time_combined"

AAAI_N2_TIME_APPENDIX_BEGIN_MARK = "% BEGIN AUTO: aaai_n2_time_appendix"
AAAI_N2_TIME_APPENDIX_END_MARK   = "% END AUTO: aaai_n2_time_appendix"

# The perturbations whose solve-time figures go in the EVALUATION body, one
# pooled figure each; every other perturbation goes to the appendix. Matched
# against the normalised perturbation key, so the display form ($\ell_\infty$)
# and the raw name (linf) both resolve.
_AAAI_BODY_PERT_KEYS = ("patch", "rotation", "linf", "translation")


def _time_in_body(cell):
    """True for the solve-time cells the EVALUATION charts: patch, rotation and
    linf in full, plus translation at size (3,1) only (user request). The rest
    go to the appendix, so one perturbation can be split by SIZE between the
    two -- which is why this is a per-cell predicate rather than a set of
    perturbation names."""
    key = _aaai_pert_key(cell[3])
    if key in ("patch", "rotation", "linf"):
        return True
    size = (cell[5].split(r"\\")[-1].strip().strip("()").replace(" ", "")
            if len(cell) > 5 else "")
    return key == "translation" and size == "3,1"


def _aaai_pert_key(type_disp):
    """Normalise a perturbation's DISPLAY string back to its plain name, so a
    figure can be routed by perturbation whatever the label looks like.
    '$\\ell_\\infty$' -> 'linf'; 'occ' -> 'occ'."""
    t = str(type_disp)
    if "ell_" in t or "infty" in t:
        return "linf"
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def _aaai_time_gap_ymax(time_cells):
    """The single right-hand (bound-difference) scale shared by EVERY solve-time
    row of the paper (user request). Computed over ALL cells, so the body and
    the appendix figures agree and a line's height means the same thing
    wherever it is read."""
    gmax = 0.0
    for cell in time_cells or []:
        for pair in cell[6].values():
            if len(pair) > 3 and pair[3] is not None:
                gmax = max(gmax, float(pair[3]))
    return max(gmax * 1.15, 1.0)


def _aaai_time_pooled_rows(entries, force_timeout=None, gap_ymax=None):
    """The solve-time ROWS for ONE perturbation, pooling EVERY model onto a
    single axis (user request: "all patch results on the same slot, just
    create bigger gaps between groups of clusters of different models").

    Returns (panels, one_model): `panels` is a list of PANEL DESCRIPTORS, one
    per chunk (content lines plus the geometry the row packer needs), NOT
    finished pictures -- the caller stacks the rows
    of every perturbation into ONE figure per section (user request: one
    figure in the Evaluation, one in the appendix) and adds the legend once. A
    perturbation too wide for \\textwidth yields several rows, split only at a
    (model, size, c_s) boundary so no shared label line is torn in half.

    Same layout machinery as the bounds-difference figure: clusters at the
    hard-coded pitch, `_AAAI_MODEL_GAP_PT` of air where the model changes, the
    perturbation and size above each block, the source class under it, and the
    model under that.

    `entries` are the (dataset_disp, arch, arch_disp, type_disp, c_src,
    item_label, bars) tuples collected per dataset, already filtered to this
    one perturbation.
    """
    power = _AAAI_YAXIS_POWER
    # Bars carry a 3rd (CI) and 4th (bound gap) slot that this figure does not
    # draw; strip to (value, partial) so the axis top clears the bars alone.
    # A perturbation evaluated on ONE model repeats that model's name over
    # every block for no information; name it in the caption instead and leave
    # the line off the plot.
    models = {(e[2], e[0]) for e in entries}
    one_model = models.pop() if len(models) == 1 else None

    def _groups3(chunk):
        # Three-level structure for the size-run layout, ordered MODEL-major
        # (user request: every cluster of one model sits together whatever its
        # c_s, and within a model the clusters of one perturbation size sit
        # together):
        #   group key    -- the model, "arch over dataset" on TWO lines. It is
        #                   the layout's OUTER key, so a model's clusters are
        #                   contiguous and only a model boundary earns the wide
        #                   gap. Printed once under each block (empty, hence
        #                   skipped, on a single-model figure).
        #   label line 2 -- the perturbation as "<type>(<size>)", the RUN
        #                   inside a model, so same-size clusters are
        #                   adjacent. Printed above; the type rides on the
        #                   label rather than being named in the caption
        #                   (user request), and size_disp already carries its
        #                   own parentheses.
        #   label line 1 -- "$c_s{=}N$", once per adjacent stretch, below.
        out = []
        for (ds, _arch, arch_disp, type_disp, c_src, item_label,
             bars) in chunk:
            size_disp = (item_label.split(r"\\", 1)[1]
                         if r"\\" in item_label else "")
            # EVERY block names its model, even when the whole figure has
            # only one (user request). The old shortcut dropped the line for a
            # single-model row and left the caption to name it, which stopped
            # working once the caption was shortened: rotation, linf and
            # translation are each evaluated on one network, so their rows
            # ended up naming the model nowhere.
            model_key = arch_disp + r"\\" + ds
            out.append((
                model_key,
                r"$c_s{=}" + str(c_src) + r"$\\" + type_disp + size_disp,
                # slot 3 stays None (no CI marker); slot 4 is the bound gap,
                # which feeds the overlaid line graph.
                {sk: (pair[0], pair[1], None,
                      pair[3] if len(pair) > 3 else None)
                 for sk, pair in bars.items()}))
        return out

    # MODEL-major: one model's clusters are contiguous, then its clusters of
    # one perturbation size, then its source classes. The layout re-sorts
    # inside a model group on the same key, so the two agree.
    pooled = sorted(entries, key=lambda g: (g[2], g[0], g[3],
                                            g[5].split("\\\\")[-1],
                                            g[4], g[5]))
    plotbox = max(_AAAI_TEXTWIDTH_PT - 1.5 * _AAAI_PT_PER_CM, 1.0)
    model_extra = ((_AAAI_MODEL_GAP_PT - _AAAI_CLUSTER_GAP_PT)
                   / _AAAI_SLOT_PT)
    # Indivisible packing units: a (size, model, dataset, c_s) block never
    # straddles two figures, or its shared label lines would be torn in half.
    blocks, prev_key = [], None
    for g in pooled:
        key = (g[2], g[0], g[3], g[5].split("\\\\")[-1], g[4])
        if key != prev_key:
            blocks.append([])
            prev_key = key
        blocks[-1].append(g)
    chunks, cur = [], []
    for blk in blocks:
        cand = cur + blk
        w = _aaai_panel_span(_groups3(cand), sort_by_label=True,
                             run_gap_extra=0.0,
                             group_gap_extra=model_extra) * _AAAI_SLOT_PT
        if cur and w > plotbox:
            chunks.append(cur)
            cur = list(blk)
        else:
            cur = cand
    if cur:
        chunks.append(cur)

    panels = []
    for fi, chunk in enumerate(chunks):
        groups = _groups3(chunk)
        vmax = max((pair[0] for _t, _l, bars in groups
                    for pair in bars.values()), default=1.0)
        ymax, ytick_clause = _aaai_yaxis(vmax)
        # Dashed timeout level, drawn once this row's tallest bar passes
        # _AAAI_TIME_LINE_MIN. ONE line per distinct cap among the row's
        # models: a pooled row may hold models given different caps (3x100
        # runs five hours, the rest three), so a single level would be wrong
        # for some of its bars.
        caps = []
        if vmax > _AAAI_TIME_LINE_MIN:
            caps = sorted({_ft_for(force_timeout, e[1]) for e in chunk
                           if _ft_for(force_timeout, e[1]) is not None})
            for ft_secs in caps:
                ymax = max(ymax, (ft_secs / 60.0) ** power * 1.05)
        span = _aaai_panel_span(groups, sort_by_label=True,
                                run_gap_extra=0.0,
                                group_gap_extra=model_extra)
        slot = _AAAI_SLOT_PT
        w = span * slot
        if w > plotbox:
            slot, w = plotbox / span, plotbox
        line_pts = []
        content, n, xtick, xticklabels = _aaai_bar_content(
            groups, power, cluster=True, size_runs=True,
            runs_above=True, run_sort_by_label=True,
            run_gap_extra=0.0, group_gap_extra=model_extra,
            cs_subrun_labels=True, group_order="given",
            series=_AAAI_CHART_SERIES, bar_w=_AAAI_BAR_PT / slot,
            line_out=line_pts)
        for ft_secs in caps:
            y = (ft_secs / 60.0) ** power
            content.append(
                f"\\draw[dashed, darkgray, line width=0.6pt] "
                f"(axis cs:0.5,{y:.4g}) -- "
                f"(axis cs:{n + 0.5:.4g},{y:.4g});")

        panels.append({
            "content": content, "w": w, "n": n, "ymax": ymax,
            "ytick": ytick_clause, "xtick": xtick,
            "xticklabels": xticklabels, "line_pts": line_pts,
        })
    return panels, one_model


def _aaai_time_single_figure(by_pert, order, force_timeout=None,
                             tex_dir=None, basename="n2_time",
                             label_base="fig:n2-time", extra_labels=(),
                             gap_ymax=None, float_spec="tp"):
    """ONE figure holding the solve-time rows of EVERY perturbation given
    (user request: a single figure in the Evaluation and a single one in the
    appendix, rather than one per perturbation).

    Each perturbation contributes one or more rows from _aaai_time_pooled_rows;
    the rows are stacked in `order`, the legend is drawn once at the top, and
    the whole stack renders to a single PDF that one figure* imports. The
    perturbation is named on each block's label (\"<type>(<size>)\"), so the
    caption does not have to enumerate them.
    """
    graphic = []
    graphic += _aaai_standalone_legend(series=_AAAI_CHART_SERIES)
    # ONE packing pass PER PERTURBATION (user request): a panel never mixes
    # perturbation types. Several such panels then SHARE a row when they fit
    # \textwidth together, which is what keeps the figure short without ever
    # putting two perturbations on one slot.
    panels = []
    for key in order:
        if key not in by_pert:
            continue
        _disp, cells = by_pert[key]
        pans, _one_model = _aaai_time_pooled_rows(
            cells, force_timeout=force_timeout, gap_ymax=gap_ymax)
        panels += pans
    # First-fit into rows. A row pays the left decoration once (the rotated
    # "time (minutes)" label and its tick numbers), the right decoration once
    # (the shared bound-difference axis), and the separation between panels.
    rows, row_w = [], []
    for pan in panels:
        # A later panel in a row keeps its OWN time scale, so it must also
        # carry its own tick numbers: reserve their width in the budget.
        joined = _AAAI_PANEL_HSEP_PT + _AAAI_PANEL_TICKS_PT + pan["w"]
        for ri in range(len(rows)):
            if (row_w[ri] + joined + _AAAI_RIGHT_DECO_PT
                    <= _AAAI_TEXTWIDTH_PT):
                rows[ri].append(pan)
                row_w[ri] += joined
                break
        else:
            rows.append([pan])
            row_w.append(_AAAI_PANEL_DECO_PT + pan["w"])
    n_rows = len(rows)
    for ri, row in enumerate(rows):
        graphic.append(r"\par\smallskip")
        graphic.append(r"\begin{tikzpicture}")
        for ci, pan in enumerate(row):
            first, last = (ci == 0), (ci == len(row) - 1)
            opt = (f"name=bax{ri}x{ci}, scale only axis, "
                   f"width={pan['w']:.2f}pt, height=2.2cm, clip=false,\n"
                   f"  ymin=0, ymax={pan['ymax']:.4g}, xmin=0.5, "
                   f"xmax={pan['n'] + 0.5:.4g}, {pan['ytick']}\n"
                   r"  x tick label style={align=center, font=\small},"  "\n"
                   r"  y tick label style={font=\small}," "\n"
                   r"  axis background/.style={fill=black!4}," "\n"
                   r"  ymajorgrids, major grid style={gray!25}")
            if pan["xtick"] and pan["xticklabels"]:
                opt += (f", xtick={{{pan['xtick']}}}, "
                        f"xticklabels={{{pan['xticklabels']}}}")
            elif pan["xtick"]:
                opt += f", xtick={{{pan['xtick']}}}, xticklabels=\\empty"
            else:
                opt += r", xtick=\empty"
            if first:
                opt += r", ylabel={time (minutes)}, ylabel style={font=\small}"
            else:
                # Every panel keeps its OWN time scale (user request), so it
                # prints its own tick numbers. Only the unit NAME is written
                # once, on the row's first panel.
                anc = f"anc{ri}x{ci}"
                graphic.append(
                    f"\\path (bax{ri}x{ci - 1}.south east) "
                    f"++({_AAAI_PANEL_HSEP_PT:.2f}pt,0pt) coordinate ({anc});")
                opt += f", at={{({anc})}}, anchor=south west"
            graphic.append(f"\\begin{{axis}}[{opt}]")
            graphic += pan["content"]
            graphic.append(r"\end{axis}")
            # The overlaid line axis. Its scale is figure-wide, so only the
            # row's LAST panel draws the axis line, its ticks and its label;
            # the others place their lines against the same scale silently.
            if pan["line_pts"]:
                gopt = (f"name=lax{ri}x{ci}, "
                        f"at={{(bax{ri}x{ci}.south west)}}, "
                        r"anchor=south west," "\n"
                        f"  scale only axis, width={pan['w']:.2f}pt, "
                        r"height=2.2cm, clip=false," "\n"
                        f"  ymin=0, ymax={gap_ymax or 100.0:.4g}, xmin=0.5, "
                        f"xmax={pan['n'] + 0.5:.4g}," "\n"
                        r"  axis x line=none, ")
                if last:
                    gopt += (r"axis y line*=right," "\n"
                             r"  ylabel={$\delta_u-\delta_l$ (\%$\delta_{\max}$)},"
                "\n"
                             r"  ylabel style={font=\small}," "\n"
                             r"  y tick label style={font=\small}]")
                else:
                    gopt += (r"axis y line=none, yticklabels=\empty]")
                graphic.append(f"\\begin{{axis}}[{gopt}")
                for pts in pan["line_pts"]:
                    pts = sorted(pts)
                    path = " -- ".join(f"(axis cs:{px:.4g},{pg:.4g})"
                                       for px, pg in pts)
                    graphic.append(
                        f"\\draw[black, line width=0.6pt] {path};")
                    for px, pg in pts:
                        graphic.append(
                            f"\\node[circle, fill=black, inner sep=0pt,"
                            f" minimum size=2.2pt] at "
                            f"(axis cs:{px:.4g},{pg:.4g}) {{}};")
                graphic.append(r"\end{axis}")
        graphic.append(r"\end{tikzpicture}")
    if not n_rows:
        return []
    out = [r"\begin{figure*}[" + float_spec + "]", r"\centering",
           _aaai_render_chart_pdf(graphic, basename, tex_dir)]
    cap = (r"Solve time of \tool compared to \baseline (baseline)")
    out.append(f"\\caption{{{cap}}}")
    out.append(f"\\label{{{label_base}}}")
    for lb in extra_labels:
        out.append(f"\\label{{{lb}}}")
    out.append(r"\end{figure*}")
    out.append("")
    return out


def _aaai_ct_of_label(item_label):
    """The target class out of a cluster's '$c_t{=}N$\\\\<size>' label."""
    m = re.search(r"c_t\{=\}(\d+)", str(item_label))
    return m.group(1) if m else "?"


def _aaai_time_cells_by_pert(time_cells):
    """Split the solve-time cells into the BODY perturbations (one pooled
    figure each, in _AAAI_BODY_PERT_KEYS order) and everything else, which goes
    to the appendix. Returns (body, rest): both {pert_key: (display, cells)}."""
    def _group(cells):
        by = {}
        for cell in cells:
            by.setdefault(_aaai_pert_key(cell[3]), (cell[3], []))[1].append(cell)
        return by
    cells = time_cells or []
    return (_group([c for c in cells if _time_in_body(c)]),
            _group([c for c in cells if not _time_in_body(c)]))


def regenerate_aaai_time_appendix_section(
        tex_path, time_cells, force_timeout=None,
        begin_mark=AAAI_N2_TIME_APPENDIX_BEGIN_MARK,
        end_mark=AAAI_N2_TIME_APPENDIX_END_MARK):
    """The solve-time figures for every perturbation the body does NOT show
    (user request: patch, rotation and linf in the Evaluation, anything else
    here), one pooled figure per perturbation, same layout as the body's."""
    try:
        _body, rest = _aaai_time_cells_by_pert(time_cells)
        if rest:
            tex_dir = os.path.dirname(os.path.dirname(os.path.abspath(tex_path)))
            figs = _aaai_time_single_figure(
                rest, sorted(rest), force_timeout=force_timeout,
                tex_dir=tex_dir, basename="n2_time_app",
                label_base="fig:n2-time-app",
                gap_ymax=_aaai_time_gap_ymax(time_cells))
            body_tex = "\n".join(figs)
        else:
            body_tex = r"% (every perturbation is shown in the body)"
        update_aaai_wide_perarch_tex(tex_path, body_tex,
                                     begin_mark=begin_mark,
                                     end_mark=end_mark, label_suffix="")
    except SystemExit as exc:
        print(f"[update_advstd_tex_tables] aaai_n2_time_appendix block "
              f"skipped: {exc}")
    except Exception as exc:
        print(f"[update_advstd_tex_tables] aaai_n2_time_appendix block error: "
              f"{exc}")


def regenerate_aaai_time_combined_section(
        tex_path, time_cells, force_timeout=None,
        begin_mark=AAAI_N2_TIME_COMBINED_BEGIN_MARK,
        end_mark=AAAI_N2_TIME_COMBINED_END_MARK):
    """Emit the body's solve-time figures: ONE pooled figure per perturbation
    in _AAAI_BODY_PERT_KEYS (patch, rotation, linf), each carrying EVERY model
    on one axis with a wider gap between models. Every other perturbation is
    emitted by regenerate_aaai_time_appendix_section instead. `time_cells` is a
    list of (dataset_disp, arch, arch_disp, type_disp, c_src, item_label,
    bars)."""
    try:
        body, _rest = _aaai_time_cells_by_pert(time_cells)
        if body:
            # Charts are pre-rendered to PDFs next to the paper (AAAI bans
            # pgfplots in the source); tex_path is <paper>/sections/<file>.tex.
            tex_dir = os.path.dirname(os.path.dirname(os.path.abspath(tex_path)))
            # ONE figure for the body (user request), stacking the rows of
            # every body perturbation. The per-perturbation aliases stay as
            # extra \labels on it so the section's existing \refs resolve.
            figs = _aaai_time_single_figure(
                body, [k for k in _AAAI_BODY_PERT_KEYS if k in body],
                force_timeout=force_timeout, tex_dir=tex_dir,
                basename="n2_time", label_base="fig:n2-time",
                extra_labels=[f"fig:n2-time-{k}"
                              for k in _AAAI_BODY_PERT_KEYS if k in body],
                gap_ymax=_aaai_time_gap_ymax(time_cells),
                float_spec="t")
            body_out = "\n".join(figs)
        else:
            body_out = r"% (no solve-time clusters)"
        update_aaai_wide_perarch_tex(tex_path, body_out,
                                     begin_mark=begin_mark,
                                     end_mark=end_mark,
                                     label_suffix="")
    except SystemExit as exc:
        print(f"[update_advstd_tex_tables] aaai_n2_time_combined block skipped: "
              f"{exc}")
    except Exception as exc:
        print(f"[update_advstd_tex_tables] aaai_n2_time_combined block error: "
              f"{exc}")


HAR_CONFIDENCE_BEGIN_MARK = "% BEGIN AUTO: har_confidence_figures"
HAR_CONFIDENCE_END_MARK   = "% END AUTO: har_confidence_figures"


def _parse_har_bound_rows(path):
    """[(c_source, lower_bound, upper_bound), ...] from a HAR result .txt.
    Each line is 'key=val,key=val,...'; rows without finite bounds are dropped."""
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or "lower_bound" not in line:
                    continue
                kv = {}
                for tok in line.split(","):
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        kv[k.strip()] = v.strip()
                try:
                    cs = int(float(kv["c_source"]))
                    lo = float(kv["lower_bound"])
                    up = float(kv["upper_bound"])
                except (KeyError, ValueError):
                    continue
                if math.isfinite(lo) and math.isfinite(up):
                    rows.append((cs, lo, up))
    except OSError:
        return []
    return rows


def _har_method_bounds(cwd):
    """{method: {c_src: (delta_l, delta_u)}} for HAR, method in
    'vaghar' / 'ours' / 'transfer'. delta_l / delta_u are the MAX over the
    target classes of the verified lower / upper bounds (the source class's
    global-robustness delta). Method buckets mirror the paper tables:
      * 'vaghar'   -- the NO-perturbed-intervals baseline: vagharNoPerturbed_*
                      dirs, or an all-off N2stdBoost cell (no zono / no SibGate
                      / no PerturbedIntervals). HAR currently has neither.
      * 'ours'     -- any other N2stdBoost cell (zono / SibGate / PI on).
      * 'transfer' -- advStd_* dirs. HAR currently has none.
    vagharWithPerturbed_* is the separate 'PI' variant, NOT \\baseline, so it is
    deliberately excluded (see _wide_combo_of_dir). Returns {} if no HAR data."""
    import glob  # module convention: glob is imported locally
    root = os.path.join(cwd, "paper_experiments", "har", "har_exp")
    if not os.path.isdir(root):
        return {}
    acc = {}   # method -> c_src -> ([lowers], [uppers])
    for eps_dir in glob.glob(os.path.join(root, "*", "eps_*")):
        for d in sorted(glob.glob(os.path.join(eps_dir, "*"))):
            if not os.path.isdir(d):
                continue
            base = os.path.basename(d)
            if base.startswith("advStd_"):
                fixed_method = "transfer"
            elif base.startswith("vagharNoPerturbed_"):
                fixed_method = "vaghar"
            elif base.startswith("N2stdBoost_"):
                fixed_method = None   # classified per file below
            else:
                continue               # vagharWithPerturbed (PI), n1_state, model_*
            for tf in glob.glob(os.path.join(d, "*.txt")):
                method = fixed_method
                if method is None:
                    fname = os.path.basename(tf)
                    is_all_off = ("zono" not in fname
                                  and "SibGate" not in fname
                                  and "PertruebedIntervals" not in fname
                                  and "PerturbedIntervals" not in fname)
                    method = "vaghar" if is_all_off else "ours"
                for cs, lo, up in _parse_har_bound_rows(tf):
                    los, ups = acc.setdefault(method, {}).setdefault(
                        cs, ([], []))
                    los.append(lo)
                    ups.append(up)
    out = {}
    for method, per in acc.items():
        for cs, (los, ups) in per.items():
            if los and ups:
                out.setdefault(method, {})[cs] = (max(los), max(ups))
    return out


def _har_forward_margins(cwd, c_srcs, n_samples=1000, seed=0):
    """{c_src: numpy array (n_samples,)} of the HAR target net N2's clean-input
    class margin  f(x)[c_src] - max_{k != c_src} f(x)[k]  for x drawn uniformly
    from the verification box [-1,1]^561. Returns None if torch / the model is
    unavailable. The int8 N2 is 'Linear(561->500) -> ReLU -> Linear(500->6)', so
    the forward pass is reconstructed straight from the state_dict."""
    try:
        import numpy as np
        import torch
    except ImportError:
        return None
    cand = [
        os.path.join(cwd, "paper_experiments", "har", "har_exp",
                     "model_har_int8", "model.pth"),
        os.path.join(cwd, "models", "har", "model.pth"),
    ]
    sd_path = next((p for p in cand if os.path.isfile(p)), None)
    if sd_path is None:
        return None
    try:
        sd = torch.load(sd_path, map_location="cpu")
        w1 = sd["fc1.weight"].cpu().numpy()
        b1 = sd["fc1.bias"].cpu().numpy()
        w2 = sd["fc2.weight"].cpu().numpy()
        b2 = sd["fc2.bias"].cpu().numpy()
    except Exception:
        return None
    n_in = w1.shape[1]
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, n_in))
    h = np.maximum(0.0, x @ w1.T + b1)
    out = h @ w2.T + b2               # (n_samples, n_classes)
    n_cls = out.shape[1]
    margins = {}
    for c in c_srcs:
        if 0 <= c < n_cls:
            mask = np.ones(n_cls, dtype=bool)
            mask[c] = False
            margins[c] = out[:, c] - out[:, mask].max(axis=1)
    return margins


def render_har_confidence_figures(cwd, n_samples=50000, seed=0):
    """Auto-generate, for HAR only, one figure per source class c_tag: the
    network's clean-input class margin over `n_samples` inputs sampled uniformly
    from the verification box [-1,1]^561 (a sampling stand-in for the
    dataset-empirical delta_d, which HAR has no dataset for), with horizontal
    delta_l / delta_u lines per method (ours / vaghar / transfer) that has HAR
    results. Renders the PDFs under neta-s-paper/figures/ and returns the LaTeX
    figure floats (empty string if HAR data / torch / matplotlib is missing)."""
    try:
        import numpy as np  # noqa: F401  (used indirectly via margins arrays)
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
    except ImportError:
        return ""
    bounds = _har_method_bounds(cwd)
    # Source classes = whatever the HAR results cover; fall back to {0, 1}.
    c_srcs = sorted({cs for per in bounds.values() for cs in per} or {0, 1})
    margins = _har_forward_margins(cwd, c_srcs, n_samples=n_samples, seed=seed)
    if margins is None:
        print("[full-results] HAR confidence figures skipped "
              "(torch / model unavailable)")
        return ""
    fig_dir = os.path.join(cwd, "neta-s-paper", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    # Canonical three methods, drawn in this order. 'ours' is red so the
    # single method HAR has data for reads as the requested red bound lines.
    method_order = ["ours", "vaghar", "transfer"]
    method_color = {"ours": "#d62728", "vaghar": "#1f77b4",
                    "transfer": "#2ca02c"}
    parts = [HAR_CONFIDENCE_BEGIN_MARK,
             r"\section*{HAR --- random-input confidence "
             r"($\delta_d$ stand-in)}"]
    # The figure is a full-width figure* (\includegraphics[width=\textwidth]),
    # so a 7in-wide canvas is displayed ~1:1 and 10pt matplotlib text prints at
    # ~10pt -- the paper's body size. Fonts are set here (not shrunk by an
    # includegraphics downscale) so the in-figure text matches the running text.
    rc = {"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
          "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10}
    for c in c_srcs:
        y = margins.get(c)
        if y is None:
            continue
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(7.0, 3.9))
            # Small, semi-transparent dots so 50k points read as a density
            # band rather than one solid block.
            ax.scatter(range(len(y)), y, s=2, color="0.4", alpha=0.35,
                       linewidths=0, zorder=2,
                       label="class margin (random inputs)")
            drawn, absent = [], []
            for m in method_order:
                b = bounds.get(m, {}).get(c)
                if b is None:
                    # Keep the label in the legend with no line drawn.
                    ax.plot([], [], ls="none", marker="",
                            label=f"{m}: no HAR data")
                    absent.append(m)
                    continue
                dl, du = b
                col = method_color[m]
                ax.axhline(dl, ls="--", lw=1.8, color=col, zorder=3,
                           label=rf"{m} $\delta_l$")
                ax.axhline(du, ls="-", lw=1.8, color=col, zorder=3,
                           label=rf"{m} $\delta_u$")
                drawn.append(m)
            ax.set_xlabel("random input index")
            ax.set_ylabel(
                r"$f(x)_{c_\mathrm{tag}}-\max_{k\neq c_\mathrm{tag}}f(x)_k$")
            ax.set_title(rf"HAR, $c_\mathrm{{tag}}={c + 1}$")
            ax.legend(loc="best", framealpha=0.9, ncol=2)
            fig.tight_layout()
            pdf = os.path.join(fig_dir, f"har_confidence_ctag{c + 1}.pdf")
            fig.savefig(pdf)
            plt.close(fig)
        drawn_txt = (", ".join(drawn) if drawn else "no method")
        absent_txt = (r" \emph{" + "}, \\emph{".join(absent) + "} have no "
                      "HAR runs, so they carry no line." if absent else "")
        parts.append("\n".join([
            r"\begin{figure*}[t]",
            r"\centering",
            rf"\includegraphics[width=\textwidth]"
            rf"{{figures/har_confidence_ctag{c + 1}}}",
            r"\caption{HAR, source class $c_\mathrm{tag}=" + str(c + 1)
            + r"$: the target network $N$'s clean-input class margin "
            r"$f(x)_{c_\mathrm{tag}}-\max_{k\neq c_\mathrm{tag}}f(x)_k$ over "
            + f"{n_samples:,}" + r" inputs drawn uniformly from the verification "
            r"box $[-1,1]^{561}$. This samples the box as a stand-in for the "
            r"dataset-empirical $\delta_d$ (HAR ships with no public dataset), "
            r"so its peak under-estimates the true $\delta_d$. Horizontal lines "
            r"mark the verified $\delta_l$ (dashed) and $\delta_u$ (solid), "
            r"taken as the max over target classes, for " + drawn_txt + "."
            + absent_txt + r"}",
            rf"\label{{fig:har-confidence-ctag{c + 1}}}",
            r"\end{figure*}",
        ]))
    parts.append(HAR_CONFIDENCE_END_MARK)
    if len(parts) <= 3:   # markers + section header only, no figure emitted
        return ""
    return "\n".join(parts)


AAAI_N2_BOUNDDIFF_BEGIN_MARK = "% BEGIN AUTO: aaai_n2_bounddiff"
AAAI_N2_BOUNDDIFF_END_MARK   = "% END AUTO: aaai_n2_bounddiff"

AAAI_N2_BOUNDDIFF_APP_BEGIN_MARK = "% BEGIN AUTO: aaai_n2_bounddiff_appendix"
AAAI_N2_BOUNDDIFF_APP_END_MARK   = "% END AUTO: aaai_n2_bounddiff_appendix"


def _bd_in_body(entry):
    """True for the bounds-difference clusters the EVALUATION shows (user
    request): every HAR cluster, every conv4 cluster, contrast at 1.2, occ at
    (3,3,5) outside c_s 0 and 4, and brightness 0.25 on conv1/MNIST outside
    c_s 2. Everything else -- translation, contrast 1.5, and the excluded
    source classes -- goes to the appendix.

    `entry` is (dataset_disp, type_disp, size_disp, arch_disp, c_src, bars);
    size_disp arrives parenthesised, e.g. "(3,3,5)"."""
    ds, type_disp, size_disp = entry[0], entry[1], entry[2]
    arch_disp = entry[3] if len(entry) > 3 else ""
    try:
        c_src = int(entry[4])
    except (IndexError, TypeError, ValueError):
        c_src = None
    if str(ds).strip().lower() == "har":
        return True
    if "conv4" in str(arch_disp):
        return True
    key = _aaai_pert_key(type_disp)
    size = str(size_disp).strip().strip("()").replace(" ", "")
    if key == "contrast" and size == "1.2":
        return True
    # occ (3,3,5) except the c_s the appendix takes.
    if key == "occ" and size == "3,3,5" and c_src not in (0, 4):
        return True
    # brightness 0.25 on conv1/MNIST, except c_s=2.
    if (key == "brightness" and size == "0.25"
            and "conv1" in str(arch_disp)
            and str(ds).strip() == "MNIST" and c_src != 2):
        return True
    return False


def _bd_split(bd_groups):
    """(body, appendix) partition of the bounds-difference clusters."""
    body = [g for g in (bd_groups or []) if _bd_in_body(g)]
    rest = [g for g in (bd_groups or []) if not _bd_in_body(g)]
    return body, rest


def regenerate_aaai_bounddiff_appendix_section(
        tex_path, bd_groups, force_timeout=None,
        begin_mark=AAAI_N2_BOUNDDIFF_APP_BEGIN_MARK,
        end_mark=AAAI_N2_BOUNDDIFF_APP_END_MARK):
    """The bounds-difference figures for every cluster the body does NOT show
    (see _bd_in_body). Same renderer as the body's, different AUTO block."""
    _body, rest = _bd_split(bd_groups)
    return regenerate_aaai_bounddiff_section(
        tex_path, rest, force_timeout=force_timeout,
        begin_mark=begin_mark, end_mark=end_mark,
        label_stem="fig:n2-bounddiff-app", basename_stem="n2_bounddiff_app",
        preselected=True)


def regenerate_aaai_bounddiff_section(tex_path, bd_groups, force_timeout=None,
                                      begin_mark=AAAI_N2_BOUNDDIFF_BEGIN_MARK,
                                      end_mark=AAAI_N2_BOUNDDIFF_END_MARK,
                                      label_stem="fig:n2-bounddiff",
                                      basename_stem="n2_bounddiff",
                                      preselected=False, float_spec="tp"):
    """Emit EXACTLY TWO bounds-difference figures (user request: no
    per-dataset split anymore). `bd_groups` is the pooled list of
    (dataset_disp, type_disp, size_disp, x_label, bars) clusters from every
    dataset; here they are sorted so clusters sharing a perturbation type and
    size sit TOGETHER (stable, so within a type+size the dataset/arch/c_s
    order is preserved), then split in half across the two `figure*`s. The
    x labels carry the dataset ("arch, dataset" line), so no figure belongs
    to a dataset. Both figures live in the one AUTO block."""
    if not preselected:
        # The EVALUATION shows only the selected clusters; the rest are
        # emitted by regenerate_aaai_bounddiff_appendix_section.
        bd_groups, _rest = _bd_split(bd_groups)
    try:
        if bd_groups:
            # Same perturbation type+size adjacent (then arch, then dataset,
            # then c_s, so identical arch+dataset clusters sit adjacent too
            # -- their shared label lines are factored out per block
            # downstream).
            pooled = sorted(bd_groups,
                            key=lambda g: (g[1], g[2], g[3], g[0], g[4]))

            def _groups3(entries):
                # Three-level structure for the size-run layout:
                # group key = "type (size)" (printed once per block, below),
                # label line 1 = the per-cluster "$c_s{=}N$" tick label,
                # label line 2 = "arch, dataset" (printed once per adjacent
                # sub-run, ABOVE the plot box).
                return [(g[1] + r"\ " + g[2],
                         r"$c_s{=}" + str(g[4]) + r"$\\"
                         + g[3] + ", " + g[0],
                         g[5]) for g in entries]

            # As many figures as the labels need (user request: no overlap;
            # splitting is fine, and clusters may move between figures --
            # but a (type, size, model, c_s) combo must never be torn apart,
            # so THOSE are the indivisible packing units).
            plotbox = max(_AAAI_TEXTWIDTH_PT - 1.5 * _AAAI_PT_PER_CM, 1.0)
            model_extra = ((_AAAI_MODEL_GAP_PT - _AAAI_CLUSTER_GAP_PT)
                           / _AAAI_SLOT_PT)
            blocks, prev_key = [], None
            for g in pooled:
                key = (g[1], g[2], g[3], g[0], g[4])
                if key != prev_key:
                    blocks.append([])
                    prev_key = key
                blocks[-1].append(g)
            chunks, cur = [], []
            for blk in blocks:
                cand = cur + blk
                w = _aaai_panel_span(
                    _groups3(cand), sort_by_label=True,
                    run_gap_extra=model_extra) * _AAAI_SLOT_PT
                if cur and w > plotbox:
                    chunks.append(cur)
                    cur = list(blk)
                else:
                    cur = cand
            if cur:
                chunks.append(cur)
            # ONE figure holding EVERY chunk (user request): the chunks
            # become stacked rows of a single graphic, the legend is drawn
            # once, and the lot renders to one PDF under one caption.
            tex_dir = os.path.dirname(os.path.dirname(os.path.abspath(tex_path)))
            graphic = []
            graphic += _aaai_standalone_legend(series=_AAAI_CHART_SERIES_GAP)
            ds_all = []
            for chunk in chunks:
                for g in chunk:
                    if g[0] not in ds_all:
                        ds_all.append(g[0])
                graphic.append(r"\par\smallskip")
                graphic += _aaai_bd_single_figure(
                    _groups3(chunk), force_timeout=force_timeout,
                    tex_dir=tex_dir)
            figs = [r"\begin{figure*}[" + float_spec + "]", r"\centering",
                    _aaai_render_chart_pdf(graphic, f"{basename_stem}_1",
                                           tex_dir)]
            cap = (r"Bound difference $\delta_u-\delta_l$ of \tool compared "
                   r"to \baseline (baseline) when reached timeout")
            figs.append(f"\\caption{{{cap}}}")
            figs.append(f"\\label{{{label_stem}}}")
            if label_stem == "fig:n2-bounddiff":
                for d in ds_all:
                    figs.append(
                        "\\label{" + label_stem + "-"
                        + re.sub(r"[^a-z0-9]+", "-", d.lower()).strip("-")
                        + "}")
            figs.append(r"\end{figure*}")
            body = "\n".join(figs)
        else:
            body = (r"% (no cluster has all three methods at the timeout -- "
                    r"no bounds-difference figure)")
        update_aaai_wide_perarch_tex(tex_path, body,
                                     begin_mark=begin_mark,
                                     end_mark=end_mark,
                                     label_suffix="")
    except SystemExit as exc:
        print(f"[update_advstd_tex_tables] aaai_n2_bounddiff block skipped: "
              f"{exc}")
    except Exception as exc:
        print(f"[update_advstd_tex_tables] aaai_n2_bounddiff block error: "
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
    # CIFAR-10 convolutional nets: internal/data keys cnn4/cnn5 render as the
    # paper's conv3/conv4 (matching the Networks table in sec_evaluation.tex).
    "cnn4": r"\emph{conv3}",
    "cnn5": r"\emph{conv4}",
    "3x50": r"3$\times$50",
    "3x10": r"3$\times$10",
    "3x100": r"3$\times$100",
    # HAR ships one pretrained net, so its arch key IS the dataset name. The
    # rendering must match the Network cell of Table 1 verbatim, since
    # _filter_tab_networks pairs the two to decide which rows Table 1 keeps.
    "har": r"1$\times$500",
}


# Canonical architecture order matching Table 1 (tab:networks) in
# sec_evaluation.tex: fully-connected nets by width, then the convolutional
# nets conv1..conv4 (internal keys cnn1, cnn2, cnn4, cnn5 -- see
# _AAAI_ARCH_DISPLAY), then the ACAS/HAR nets. Every per-cell table and
# per-arch chart iterates archs through _tab1_arch_sort_key so their sequence
# matches the paper's Networks table. Archs not listed here sort last, in
# alphabetical order, so a new arch never silently jumps to the front.
_TAB1_ARCH_ORDER = [
    "3x10", "3x50", "3x100", "6x100", "9x200",
    "cnn0", "cnn1", "cnn2", "cnn3", "cnn4", "cnn5",
    "6x50", "fc",
    # The benchmark nets close Table 1, in the order the evaluation text
    # introduces them (ACAS Xu, then HAR).
    "acas", "har",
]


def _tab1_arch_sort_key(arch):
    """Sort key placing `arch` in Table-1 (tab:networks) order; unknown archs
    sort after all known ones, alphabetically."""
    try:
        return (0, _TAB1_ARCH_ORDER.index(arch))
    except ValueError:
        return (1, arch)


_AAAI_SUMMARY_NUM_CLASSES = dict(_DATASET_NUM_CLASSES)


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


def _load_advstd_rows_for_wide_from_txt(cwd, dataset, archs, perts,
                                        parse_result_file, advstd_meta_fn,
                                        seeds_filter=None,
                                        combination_filter=None,
                                        force_timeout=None,
                                        rerun_timeout_eps=30.0,
                                        stale_fn=None):
    """Read advStd per-cell rows DIRECTLY from the advStd .txt result files.

    Discovery is FILESYSTEM-driven: for each arch, scan every real perturbation
    dir (`_WIDE_PERT_SUBDIRS`) and its eps_* subdirs, and read every
    "_N2_advStd_" file under advStd_*/. This mirrors _collect_wide_perarch_cells
    (which finds the vaghar/ours columns the same way), so the transfer column
    covers exactly the same on-disk cells. It does NOT consult
    run_relaxation_sweep.PERTURBATIONS: that list only schedules which jobs to
    launch, so commenting a perturbation out of it (to run fewer jobs at once)
    must never drop already-computed results from the tables/charts. The `perts`
    parameter is kept only as the truthy sentinel the callers use to select this
    txt-direct path. There is no standard-baseline pairing: an advStd cell is
    emitted as soon as its .txt line exists, so advStd results are never dropped
    for want of a vaghar baseline (and no CSV is involved).

    role is always 'N2' (advstd is transfer mode N1 -> N2). Only the two
    advstd N2 combos in _ADVSTD_WIDE_COMBOS are emitted. Row shape matches
    _collect_wide_perarch_cells / the CSV-based _load_advstd_rows_for_wide so
    transfer rows bucket against vaghar/ours rows on identical keys
    (perturbation=pert_type, perturbation_size=eps_str -- exactly as the old
    CSV rows were keyed).

    Filters:
      - seeds_filter: keep only the requested Gurobi seed(s).
      - combination_filter: when set (--combination_table), keep only combos
        whose (bt, vh, tau[+sg]) _combo_label is listed.
      - solve status: keep only OPTIMAL or a genuine timeout
        (_AAAI_TIMEOUT_STATUSES); partial/killed runs are dropped.
      - force_timeout: honored here exactly as the renderer does it
        (_aaai_is_timeout_mismatch) so cross-cap timeout re-runs (e.g. an old
        1800s run vs a new 10800s run) are deduplicated -- the same dedup the
        old --find_advstd ... --force_timeout 10800 command performed.

    `advstd_meta_fn` is run_relaxation_sweep._extract_advstd_file_metadata,
    passed in so this module needs no new filename-parsing logic.
    """
    import glob
    seeds_filter = set(str(s) for s in seeds_filter) if seeds_filter else None
    rows = []
    for arch in (archs or []):
        exp_base = os.path.join(cwd, "paper_experiments", dataset,
                                f"{arch}_exp")
        if not os.path.isdir(exp_base):
            continue
        # Discover perturbation/eps dirs straight from the filesystem -- the
        # SAME way _collect_wide_perarch_cells finds the vaghar/ours columns --
        # so the transfer column covers exactly the same cells. The `perts`
        # argument (run_relaxation_sweep.PERTURBATIONS) is deliberately NOT used
        # for discovery: that list only schedules which jobs to launch, and
        # commenting a perturbation out of it must not drop already-computed
        # results from the tables/charts. It survives only as the truthy
        # sentinel the callers use to pick this txt-direct path.
        for pert_type in _WIDE_PERT_SUBDIRS:
            pert_base = os.path.join(exp_base, pert_type)
            if not os.path.isdir(pert_base):
                continue
            for eps_dir in sorted(glob.glob(os.path.join(pert_base, "eps_*"))):
                eps_str = os.path.basename(eps_dir).replace("eps_", "")
                for ad in sorted(glob.glob(os.path.join(eps_dir, "advStd_*"))):
                    for tf in sorted(glob.glob(os.path.join(ad, "*.txt"))):
                        fname = os.path.basename(tf)
                        if "_N2_advStd" not in fname:
                            continue
                        # Leave-one-out ablation files never feed the paper
                        # columns. _ablation marks all component-removed combos
                        # from the sweep's --advstd_ablations; _noPI is the
                        # defensive net for any pi=false run (its
                        # (zb, vh, rt, sg) tags are identical to the paper
                        # run's and would pool into "ours with transfer").
                        if "_ablation" in fname or "_noPI" in fname:
                            continue
                        # Drop only files made stale by the perturbation-dependency
                        # soundness fix: a pre-fix file is unsound ONLY if it relaxed
                        # (dropped) >=1 ReLU binary, because the missing
                        # has_a_o && has_a_p guard mis-coupled relaxed neurons. A
                        # pre-fix file that dropped nothing is byte-identical under
                        # the fix and is kept. `stale_fn` is
                        # run_relaxation_sweep._is_pre_fix_dropped (the same predicate
                        # the sweep skip-check uses); None disables the gate.
                        if stale_fn is not None and stale_fn(fname):
                            continue
                        meta = advstd_meta_fn(fname)
                        # Parity with the CSV loader's pre-filters.
                        if (meta.get("bound_tightening") != "yes"
                                or meta.get("branch_priorities") != "off"
                                or meta.get("n1_probe", "off") != "off"):
                            continue
                        if seeds_filter and meta.get("seed") not in seeds_filter:
                            continue
                        # Restrict to the wide-table advstd combos (one per
                        # rendered tau; see _advstd_wide_combos).
                        combo_label = None
                        for label, zb, vh, rt, sg in _advstd_wide_combos():
                            if (meta.get("zono_bounds") == zb
                                    and meta.get("var_hint") == vh
                                    and meta.get("relax_threshold") == rt
                                    and meta.get("sibling_gate", "no") == sg):
                                combo_label = label
                                break
                        if combo_label is None:
                            continue
                        # --combination_table filter: reuse _combo_label so the
                        # spec form ('zono:prev_pgd:0.5+sg') matches verbatim.
                        if combination_filter is not None:
                            grp_key = (meta["mip_start"], meta["branch_priorities"],
                                       meta["lp_basis"], meta["bound_tightening"],
                                       meta["var_hint"], meta["zono_bounds"],
                                       meta["n1_probe"], meta["relax_threshold"],
                                       meta["sibling_gate"])
                            if _combo_label(grp_key) not in combination_filter:
                                continue
                        seed_val = meta.get("seed", "0")
                        is_geom = "_geomInt" in fname
                        for (cs, ct), val in parse_result_file(tf).items():
                            t = val.get("total_time", 0.0) or 0.0
                            if t <= 0:
                                continue
                            # Only completed cells belong in the table: a proven
                            # optimum or a genuine timeout. Anything else (e.g.
                            # INTERRUPTED / killed mid-solve) is a partial result
                            # and is dropped.
                            status_norm = (val.get("solve_status", "") or "") \
                                .upper().replace(" ", "_")
                            is_opt = status_norm == "OPTIMAL"
                            is_to = any(tag.replace(" ", "_") in status_norm
                                        for tag in _AAAI_TIMEOUT_STATUSES)
                            if not (is_opt or is_to):
                                continue
                            lb = val.get("lower_bound")
                            ub = val.get("upper_bound")
                            lb = lb if (isinstance(lb, (int, float))
                                        and math.isfinite(lb)) else None
                            ub = ub if (isinstance(ub, (int, float))
                                        and math.isfinite(ub)) else None
                            row = {
                                "arch": arch,
                                "role": "N2",
                                "perturbation": pert_type,
                                "perturbation_size": eps_str,
                                "c_source": cs, "c_target": ct,
                                "seed": seed_val,
                                "combo": combo_label,
                                "t_total": t,
                                # Gurobi solver wall-clock only (no hyper-attack
                                # overhead); used for the timeout-cap dedup.
                                "t_opt": val.get("optimization_time"),
                                "lb_total": lb,
                                "ub_total": ub,
                                "solve_status": val.get("solve_status", ""),
                                # Dropped binaries per copy; see the same fields
                                # in _collect_wide_perarch_cells.
                                "relaxed_org": _as_int(
                                    val.get("n2_org_relaxed_binaries")),
                                "relaxed_pert": _as_int(
                                    val.get("n2_pert_relaxed_binaries")),
                                "geom": is_geom,
                                # Source file for _dedupe_wide_rows; advStd
                                # rows never come from a baseline dir.
                                "src_file": fname,
                                "src_baseline_dir": False,
                            }
                            # Honor --force_timeout: drop a timeout cell whose
                            # wall-clock does not match the requested cap (the
                            # cross-cap re-run dedup the old path did at render).
                            if _aaai_is_timeout_mismatch(row, force_timeout,
                                                         rerun_timeout_eps):
                                continue
                            rows.append(row)
    return _dedupe_wide_rows(rows)


def regenerate_wide_perarch_section(tex_path, cwd, dataset, arch_runs,
                                     parse_result_file, seeds_filter=None,
                                     begin_mark=WIDE_BEGIN_MARK,
                                     end_mark=WIDE_END_MARK,
                                     ds_label_suffix=""):
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
        update_wide_perarch_tex(tex_path, body, begin_mark=begin_mark,
                                end_mark=end_mark,
                                label_suffix=ds_label_suffix)
    except SystemExit as exc:
        print(f"[update_advstd_tex_tables] wide_perarch block skipped: "
              f"{exc}")
    except Exception as exc:
        print(f"[update_advstd_tex_tables] wide_perarch block error: {exc}")


# ── Ablation appendix tables (leave-one-out component study) ────────────────
# Rendered from the `_ablation`-tagged result files the sweep's
# --advstd_ablations / --stdboost_ablations runs produce. One table per
# (dataset, arch) that has ablation data; rows are the two paper methods
# ("ours" = N2stdBoost, "ours with transfer" = advstd-N2), columns are the
# removed component, cells are the mean solve time over the (c_s, c_t) cells
# completed by EVERY variant of that row (so times are comparable).

ABLATION_BEGIN_MARK = "% BEGIN AUTO: ablation_tables"
ABLATION_END_MARK = "% END AUTO: ablation_tables"

_ABLATION_COLUMNS = ("full", "zono", "triangle", "zono_triangle",
                     "pert_intervals", "var_hint")
_ABLATION_COL_HEADERS = {
    "full": r"\tool",
    "zono": r"w/o zonotope",
    "triangle": r"w/o relaxation",
    # Both bound-tightening techniques removed together: zono and triangle
    # tighten the same ReLU bounds, so either alone can be masked by the
    # other still doing the work.
    "zono_triangle": r"w/o zonotope \& relaxation",
    "pert_intervals": r"w/o perturbation difference",
    # var_hint drops only the N1-derived variable hints (the PGD warm start
    # still runs, so those files keep the _HyperAttackHints tag). The
    # warm_start variant (drops both; no _HyperAttackHints tag) is still
    # classified from disk but has no column here (removed per user).
    "var_hint": r"w/o warm start",
}
#: A technique whose ablation means something different per mode. "ours" is
#: single-network and has no N_pre, so its zonotope row can only be the whole
#: technique off; "transfer" instead ablates only the zonotope's N_pre input
#: (user request), which is the claim the paper actually makes about transfer.
_ABLATION_MODE_COMPONENT = {("transfer", "zono"): "zono_npre"}

_ABLATION_ROW_LABELS = {"ours": r"\emph{ours}",
                        "transfer": r"\emph{ours with transfer}"}

_ABL_SEED_RE = re.compile(r"_seed(\d+)(?=_)(?!_itr)")
_ABL_BTPR_RE = re.compile(r"_(?:BTPR|BoundTightPertRelax)(\d+(?:\.\d+)?)")


#: Cache of {result dir -> {short name: full name}} read from the
#: _filename_legend.txt that safe_filepath writes beside truncated results.
_LEGEND_CACHE = {}


def _resolve_truncated_name(dirpath, fname):
    """The FULL result filename for `fname`, which may be hash-truncated.

    Julia's safe_filepath caps a name at the 255-byte Linux limit by cutting it
    and appending a 16-hex hash, writing `<short> => <full>` into
    _filename_legend.txt beside it. The cut lands on the TAIL, which is exactly
    where the semantically load-bearing tags live (_depGuardFix, _cTagN,
    _PerturbedIntervals), so parsing the on-disk name silently misreads such a
    file -- most damagingly as pre-fix/stale, which drops it entirely. Resolve
    through the legend before any tag is inspected; names that were never
    truncated pass straight through."""
    if not re.search(r"_[0-9a-f]{16}\.txt$", fname):
        return fname
    leg = _LEGEND_CACHE.get(dirpath)
    if leg is None:
        leg = {}
        lp = os.path.join(dirpath, "_filename_legend.txt")
        if os.path.exists(lp):
            try:
                with open(lp) as fh:
                    for line in fh:
                        if " => " in line:
                            short, full = line.strip().split(" => ", 1)
                            leg[short] = full
            except OSError:
                pass
        _LEGEND_CACHE[dirpath] = leg
    return leg.get(fname, fname)


def _classify_ablation_filename(fname):
    """Map a result filename to ('ours'|'transfer', variant) or None.

    variant is 'full' for the un-ablated control combo and the removed
    component ('zono' | 'zono_npre' | 'triangle' | 'zono_triangle' |
    'pert_intervals' |
    'var_hint') for the `_ablation`-tagged runs -- leave-one-out except
    'zono_triangle', which removes both bound-tightening techniques at once.
    Non-control non-ablation files (grid combos, τ=0 rows, ...) return None.
    """
    is_abl = "_ablation" in fname
    bm = _ABL_BTPR_RE.search(fname)
    tau = float(bm.group(1)) if bm else -1.0
    if "_N2_advStd" in fname:
        if is_abl:
            if "_noPI" in fname:
                return ("transfer", "pert_intervals")
            if "_varHint" not in fname:
                # Both warm-start ablations set var_hint=off; the PGD tag is
                # the only on-disk difference. Present => hints removed but
                # the warm start kept (var_hint); absent => no warm start at
                # all (warm_start).
                if ("_HyperAttackHints" in fname
                        or "_HyperAttackCutoff" in fname):
                    return ("transfer", "var_hint")
                return ("transfer", "warm_start")
            # Combined removal first: it is the only variant missing BOTH
            # tags, so testing it after the single-component checks would
            # let it be swallowed by the 'zono' branch.
            # zono_npre keeps the _zonoBounds tag and adds _noNpreZono, so it
            # must be tested BEFORE the tag-absence checks below, which would
            # otherwise class it as the untouched control.
            if "_noNpreZono" in fname:
                return ("transfer", "zono_npre")
            if "_zonoBounds" not in fname and "_SibGate" not in fname:
                return ("transfer", "zono_triangle")
            if "_zonoBounds" not in fname:
                return ("transfer", "zono")
            if "_SibGate" not in fname:
                return ("transfer", "triangle")
            return None
        # control: the full paper combo at τ > 0 (τ=0 wide-grid rows and
        # partial combos are not the ablation reference).
        if ("_noPI" not in fname and "_noNpreZono" not in fname
                and "_varHintPrevPGD" in fname
                and "_zonoBounds" in fname and "_SibGate" in fname
                and tau > 0.0):
            return ("transfer", "full")
        return None
    if "_stdBoost_" in fname:
        if is_abl:
            if "_PertruebedIntervals" not in fname:
                return ("ours", "pert_intervals")
            if "_stdBoost_zono" not in fname and "_SibGate" not in fname:
                return ("ours", "zono_triangle")
            if "_stdBoost_zono" not in fname:
                return ("ours", "zono")
            if "_SibGate" not in fname:
                return ("ours", "triangle")
            return None
        if ("_PertruebedIntervals" in fname and "_stdBoost_zono" in fname
                and "_SibGate" in fname and tau > 0.0):
            return ("ours", "full")
        return None
    return None


def _collect_ablation_cells(cwd, dataset, arch, parse_result_file,
                            seeds_filter=None, stale_fn=None,
                            unify_taus=None):
    """({(row, variant): {(pert, eps, cs, ct): (total_time, solve_status,
    lower_bound, upper_bound, relaxed_binaries),
                                                lower_bound, upper_bound)}},
        chosen_tau_or_None).

    'ours' cells come from N2stdBoost_* dirs (target network only), 'transfer'
    cells from advStd_* dirs. Later files (higher timestamp token) win on key
    collisions, matching the sweep's newest-run-wins convention.

    Every <pert>/eps_* directory on disk is scanned -- deliberately NOT the
    sweep's active PERTURBATIONS list, so flipping that list to launch a
    different sweep does not silently change (or empty) the ablation table.
    The render-side abl_pe filter keeps the table scoped to perturbations
    with actual _ablation data.

    `unify_taus` (candidate relaxation thresholds, from --paper_taus) makes
    the table tau-consistent: cells are collected separately per candidate
    tau -- a file counts toward candidate t only when its BTPR tag equals t,
    except the 'triangle'/'zono_triangle' variants, which remove the
    relaxation and always run at BTPR 0.0, so they are tau-agnostic and
    shared by every candidate -- and the candidate with the most cells wins
    (tie: earlier in the list). Without it, any tau > 0 is accepted per cell,
    newest file wins (the prior, possibly tau-mixing behavior).
    """
    import glob  # module convention: glob is imported locally
    seeds = ({str(s) for s in seeds_filter} if seeds_filter else None)
    exp_base = os.path.join(cwd, "paper_experiments", dataset, f"{arch}_exp")
    candidates = ([float(t) for t in unify_taus] if unify_taus else None)
    outs = ({t: {} for t in candidates} if candidates else {None: {}})
    for eps_dir in sorted(glob.glob(os.path.join(exp_base, "*", "eps_*"))):
            pert_type = os.path.basename(os.path.dirname(eps_dir))
            eps_str = os.path.basename(eps_dir)[len("eps_"):]
            dir_globs = (glob.glob(os.path.join(eps_dir, "advStd_*"))
                         + glob.glob(os.path.join(eps_dir, "N2stdBoost_*")))
            for d in sorted(dir_globs):
                for tf in sorted(glob.glob(os.path.join(d, "*.txt"))):
                    fname = _resolve_truncated_name(os.path.dirname(tf),
                                                    os.path.basename(tf))
                    cls = _classify_ablation_filename(fname)
                    if cls is None:
                        continue
                    if stale_fn is not None and stale_fn(fname):
                        continue
                    sm = _ABL_SEED_RE.search(fname)
                    if seeds is not None and (sm.group(1) if sm else "0") not in seeds:
                        continue
                    bm = _ABL_BTPR_RE.search(fname)
                    ftau = float(bm.group(1)) if bm else None
                    parsed = parse_result_file(tf)
                    for cand, out in outs.items():
                        # Tau gate: only for candidates, only for variants
                        # that keep the relaxation (see docstring).
                        if (cand is not None and ftau is not None
                                and cls[1] not in ("triangle", "zono_triangle")
                                and abs(ftau - cand) > 1e-9):
                            continue
                        cells = out.setdefault(cls, {})
                        for (cs, ct), info in parsed.items():
                            t = info.get("total_time")
                            if t is None or t <= 0:
                                continue
                            status = (info.get("solve_status", "") or "") \
                                .upper().replace(" ", "_")
                            if status == "INTERRUPTED":
                                continue
                            # 5th slot: binaries the relaxation dropped,
                            # summed over BOTH MIP copies (clean N(x) and
                            # perturbed N(x')), the same convention Table 2
                            # uses. None when the run recorded neither count.
                            def _int_or_none(x):
                                try:
                                    return int(str(x).strip())
                                except (TypeError, ValueError):
                                    return None
                            _ro = _int_or_none(info.get("n2_org_relaxed_binaries"))
                            _rp = _int_or_none(info.get("n2_pert_relaxed_binaries"))
                            _rel = None if (_ro is None and _rp is None) else \
                                ((_ro or 0) + (_rp or 0))
                            cells[(pert_type, eps_str, cs, ct)] = (
                                t, status,
                                info.get("lower_bound"),
                                info.get("upper_bound"),
                                _rel)
    if candidates is None:
        return outs[None], None
    chosen = max(candidates,
                 key=lambda t: (sum(len(c) for c in outs[t].values()),
                                -candidates.index(t)))
    return outs[chosen], chosen


def _collect_ablation_baseline(cwd, dataset, arch, perts, parse_result_file,
                               seeds_filter=None):
    """{(pert, eps, cs, ct): (t, status, lower_bound, upper_bound)} for the
    \\baseline (vaghar) runs on the ABLATION's target network (N2).

    The \\baseline column of the per-cell tables is the combo key 'vaghar'
    (see _aaai_wide_columns_for_tau) -- the NO-perturbed-intervals,
    no-relaxation baseline: dependencies + hyper-attack alone. This mirrors
    the two sources _collect_wide_perarch_cells pools into that bucket, so
    the ablation tables compare against exactly the same runs the paper
    tables call \\baseline:

      * `vagharNoPerturbed_*` dirs whose role is N2, and
      * all-off N2stdBoost cells (zono=no, SibGate=no, tau=0, PI=no) -- that
        MIP is identical to the baseline, only the Gurobi seed differs, so
        _collect_wide_perarch_cells relabels them 'vaghar' too.

    `vagharWithPerturbed_*` is the SEPARATE 'PI' column, not \\baseline, and
    is deliberately excluded. Later files (higher timestamp token) win on key
    collisions, matching the sweep's newest-run-wins convention.
    """
    import glob  # module convention: glob is imported locally
    seeds = ({str(s) for s in seeds_filter} if seeds_filter else None)
    exp_base = os.path.join(cwd, "paper_experiments", dataset, f"{arch}_exp")
    out = {}
    for _pname, pert_spec in perts:
        pert_type = pert_spec.split(":", 1)[0]
        eps_glob = os.path.join(exp_base, pert_type, "eps_*")
        for eps_dir in sorted(glob.glob(eps_glob)):
            # Key on the eps of the directory being read, NOT on the one in
            # the spec: this loop walks every eps_* dir of the perturbation,
            # so keying on the spec collapses all of them onto one key and
            # the keys stop matching the ablation cells (which carry the real
            # eps).
            eps_str = os.path.basename(eps_dir).replace("eps_", "")
            dir_globs = sorted(set(
                glob.glob(os.path.join(eps_dir, "vagharNoPerturbed_*"))
                + glob.glob(os.path.join(eps_dir, "N2stdBoost_*"))))
            for d in dir_globs:
                if _role_of_stdboost_dir(d) != "N2":
                    continue
                is_boost_dir = _wide_combo_of_dir(d) == "boost"
                for tf in sorted(glob.glob(os.path.join(d, "*.txt"))):
                    fname = os.path.basename(tf)
                    if is_boost_dir:
                        combo = _classify_stdboost_filename(fname)
                        if combo is None:
                            continue
                        if seeds and combo["seed"] not in seeds:
                            continue
                        z, sg, pi, tau = _wide_boost_key(combo)
                        # Only the all-off, PI-off cell IS the baseline MIP.
                        if not (z == "0" and sg == "0" and pi == "0"
                                and tau in ("0", "0.0")):
                            continue
                    elif "_stdBoost_" in fname:
                        # A boost file that landed in a baseline dir is not
                        # the baseline (same guard _collect_wide_perarch_cells
                        # applies). Plain baseline files carry no seed tag, so
                        # they intentionally bypass `seeds`.
                        continue
                    for (cs, ct), info in parse_result_file(tf).items():
                        lb = info.get("lower_bound")
                        ub = info.get("upper_bound")
                        if lb is None or ub is None:
                            continue
                        if lb != lb or ub != ub:   # NaN from a missing field
                            continue
                        t = info.get("total_time")
                        status = (info.get("solve_status", "") or "") \
                            .upper().replace(" ", "_")
                        out[(pert_type, eps_str, cs, ct)] = (t, status, lb, ub)
    return out


def _ablation_mean_gap(cells, keys, delta_max_by_key=None, arch=None):
    """Mean bound gap over `keys`, in percentage points of delta_max when the
    Phase-0.5 delta_max is known for that (arch, N2, c_s) and raw units
    otherwise. Returns None when any key is missing or has no usable bounds.

    `cells` maps key -> (lb, ub) or (t, status, lb, ub)."""
    gaps = []
    for k in keys:
        val = cells.get(k)
        if not val:
            return None
        lb, ub = (val[2], val[3]) if len(val) >= 4 else (val[0], val[1])
        if lb is None or ub is None or lb != lb or ub != ub:
            return None
        gap = ub - lb
        if gap < 0:
            gap = 0.0
        dm = None
        if delta_max_by_key is not None and arch is not None:
            entry = delta_max_by_key.get((arch, "N2", k[2]))
            if entry:
                dm = entry.get("upper")
        if dm:
            gap = gap / dm * 100.0
        gaps.append(gap)
    if not gaps:
        return None
    return sum(gaps) / len(gaps)


def _ablation_mean_loss(cells, keys, baseline, delta_max_by_key=None,
                        arch=None):
    """Mean PRECISION LOSS of an ablation variant over `keys`: how far its
    bound interval provably sits from the exact delta, as a percentage of
    delta_max. Higher means further from the true bound; 0 means the variant
    is consistent with the same delta as \baseline.

    Per cell this is the distance between the variant's [dl, du] and
    \baseline's, the same measure Table 2 uses (_relax_gap_loss):

        100 * max(0, max(dl_m, dl_b) - min(du_m, du_b)) / delta_max

    \baseline relaxes no binary, so its interval soundly contains the exact
    delta whether or not it proved optimality; a variant interval disjoint
    from it therefore differs from the exact delta by AT LEAST this much.
    Cells with no baseline run, no bounds, or no delta_max are skipped rather
    than counted as 0, so the mean never credits a variant for a cell it
    cannot be judged on. Returns None when no cell can be judged."""
    losses = []
    for k in keys:
        m = (cells or {}).get(k)
        b = (baseline or {}).get(k)
        if not m or not b or len(m) < 4 or len(b) < 4:
            continue
        ml, mu, bl, bu = m[2], m[3], b[2], b[3]
        if any(v is None or v != v for v in (ml, mu, bl, bu)):
            continue
        dm = None
        if delta_max_by_key is not None and arch is not None:
            entry = delta_max_by_key.get((arch, "N2", k[2]))
            if entry:
                dm = entry.get("upper")
        if not dm:
            continue
        losses.append(100.0 * max(0.0, max(ml, bl) - min(mu, bu)) / dm)
    if not losses:
        return None
    return sum(losses) / len(losses)


def _fmt_ablation_time(sec):
    return f"{sec:.0f}" if sec >= 100 else f"{sec:.1f}"


def _ablation_pert_phrase(cells):
    """The perturbations this ablation table covers, as "type (size)" joined
    for a caption. Read off the cell keys of the ABLATED variants only: the
    "full" row is the ordinary run and carries every perturbation on disk, so
    including it would list perturbations the ablation never touched."""
    seen = []
    for (_row, variant), grid in (cells or {}).items():
        if variant == "full":
            continue
        for key in (grid or {}):
            if len(key) < 2:
                continue
            pert, eps = key[0], key[1]
            disp = (str(pert).replace("_", r"\_")
                    .replace("linf", r"$\ell_\infty$")) + r" (" + str(eps) + r")"
            if disp not in seen:
                seen.append(disp)
    if not seen:
        return ""
    if len(seen) == 1:
        return seen[0]
    return ", ".join(seen[:-1]) + " and " + seen[-1]


def _render_ablation_table(dataset, arch, cells, label_suffix="",
                           expected=None, unified_tau=None,
                           delta_max_by_key=None, baseline=None):
    """One booktabs table for (dataset, arch), or None when no ablation
    variant has data. Cells: each variant's mean total solve time in seconds
    over the grid cells it completed.

    `expected` is the planned grid from --ablation_expected as
    (ct0s, css_or_None): a variant is starred when it is missing any
    (pert, eps, cs, ct) cell of that grid (perts restricted to the ones the
    row has any data for -- an untested perturbation does not flag anything).
    Without `expected`, the star falls back to the relative rule: missing a
    cell that another variant of the same row already has."""
    have_abl = any(v != "full" and cells.get(("ours", v)) or
                   v != "full" and cells.get(("transfer", v))
                   for v in _ABLATION_COLUMNS)
    if not have_abl:
        return None
    # TRANSPOSED (techniques as ROWS, the two modes as the only two data
    # columns) so the table fits ONE text column at full size. The untransposed
    # form needs six technique columns, which only fit via \resizebox-style
    # shrinking -- banned by AAAI, and it would drop the in-table text under
    # the 9pt floor. \small is GROUPED around the body so the caption stays
    # 10pt.
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"{\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        # Each mode carries a time column and a precision column (user
        # request); precision is the mean remaining bound difference
        # delta_u - delta_l, the same quantity the figures and the per-cell
        # tables report, in percentage points of delta_max.
        " & " + " & ".join(r"\multicolumn{2}{c}{" + _ABLATION_ROW_LABELS[r_]
                           + "}" for r_ in ("ours", "transfer")) + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & $t$ (s) & loss & $t$ (s) & loss \\",
        r"\midrule",
    ]
    STAR = r"\textcolor{red}{$^*$}"  # same partial-coverage mark as the
    #                                   per-cell appendix tables
    any_star = False
    grid = {}   # (mode, technique) -> solve-time cell, emitted transposed below
    prec = {}   # (mode, technique) -> mean precision loss
    for row in ("ours", "transfer"):
        # Fetch the MODE-MAPPED component, not the column name: for
        # ("transfer", "zono") the data lives under "zono_npre", which is not
        # in _ABLATION_COLUMNS. Keying this dict on the column name alone left
        # that cell permanently "---".
        variants = {_ABLATION_MODE_COMPONENT.get((row, v), v):
                    cells.get((row, _ABLATION_MODE_COMPONENT.get((row, v), v)))
                    for v in _ABLATION_COLUMNS}
        if row == "ours":
            # The warm-start ablation is not produced for standard mode:
            # var_hint needs N1, which standard mode has no notion of.
            variants["var_hint"] = None
        # Perturbation scope: only perturbations the ABLATION variants have
        # data for belong to this experiment. Control ("full") results exist
        # for every paper perturbation (e.g. occ), and without this filter
        # they would drag those perturbations into the grid and star every
        # ablation variant for cells the ablation never targeted.
        abl_pe = {(k[0], k[1]) for v, c in variants.items()
                  if v != "full" and c for k in c}
        if abl_pe:
            variants = {v: ({k: val for k, val in c.items()
                             if (k[0], k[1]) in abl_pe} if c else None)
                        for v, c in variants.items()}
        # With --ablation_expected, the table is COMPUTED over the provided
        # (c_s, c_t) grid only: cells outside it (e.g. leftovers from an
        # earlier, wider sweep) are dropped before any mean/star/coverage.
        exp_keys = None
        if expected is not None:
            union_raw = set()
            for c in variants.values():
                if c:
                    union_raw |= set(c.keys())
            if union_raw:
                ct0s, css = expected
                cs_ref = (css if css is not None
                          else {cs for (_p, _e, cs, _ct) in union_raw})
                variants = {
                    v: ({k: val for k, val in c.items()
                         if k[2] in cs_ref and k[3] in ct0s} if c else None)
                    for v, c in variants.items()}
                pe_pairs = {(p, e) for (p, e, _cs, _ct) in union_raw}
                exp_keys = {(p, e, cs, ct) for (p, e) in pe_pairs
                            for cs in cs_ref for ct in ct0s if ct != cs}
        present = {v: c for v, c in variants.items() if c}
        union = set()
        for c in present.values():
            union |= set(c.keys())
        # Completeness reference for the red "*": the planned grid when
        # given, otherwise the union of observed cells (relative rule).
        if exp_keys is None:
            exp_keys = union
        # Values are collected per (mode, technique) here and emitted BELOW,
        # one line per technique, since the table is transposed.
        for v in _ABLATION_COLUMNS:
            c = variants.get(_ABLATION_MODE_COMPONENT.get((row, v), v))
            if not c:
                grid[(row, v)] = "---"
                prec[(row, v)] = "---"
                continue
            # Each variant averages over ALL the grid cells it completed
            # (its own set, not the row-wide intersection): the command line
            # defines the population, and the red "*" is the only signal
            # that a variant's mean covers less than the full grid.
            keys_v = set(c.keys())
            times = [c[k][0] for k in keys_v]
            cell = _fmt_ablation_time(sum(times) / len(times))
            # Red "*" (same convention as the per-cell tables): this
            # variant has not yet completed every cell of the completeness
            # reference (planned grid, or the row's best-covered variant).
            if exp_keys - keys_v:
                cell += STAR
                any_star = True
            grid[(row, v)] = cell
            loss = _ablation_mean_loss(c, keys_v, baseline,
                                       delta_max_by_key=delta_max_by_key,
                                       arch=arch)
            prec[(row, v)] = ("---" if loss is None
                              else (r"%.2f\%%" % loss))
    for v in _ABLATION_COLUMNS:
        cells_out = []
        for r_ in ("ours", "transfer"):
            cells_out.append(grid.get((r_, v), "---"))
            cells_out.append(prec.get((r_, v), "---"))
        lines.append(_ABLATION_COL_HEADERS[v] + " & "
                     + " & ".join(cells_out) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}%",
        r"\caption{Ablation experiments over "
        + _AAAI_ARCH_DISPLAY.get(arch, arch.replace("_", r"\_"))
        + r" (" + _dataset_display_name(dataset) + r")"
        + (r" for " + _ablation_pert_phrase(cells) if _ablation_pert_phrase(cells)
           else "")
        + r": the mean solve time and the mean precision loss (how far the "
        r"bound provably sits from the exact one, as a percentage of "
        r"$\delta_{\max}$) with one component removed at a time"
        + ((r", for $\tau{=}%g$" % unified_tau)
           if unified_tau is not None else "")
        + (r"; a red $^*$ marks a variant with incomplete cells"
           if any_star else "")
        + r".}",
        r"\label{tab:ablation-" + dataset + label_suffix + "-" + arch + r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


ABLATION_FULL_BEGIN_MARK = "% BEGIN AUTO: ablation_full_tables"
ABLATION_FULL_END_MARK = "% END AUTO: ablation_full_tables"


def _ablation_full_tables(dataset, arch, cells, baseline=None,
                          delta_max_by_key=None, label_suffix=""):
    """PER-EXPERIMENT ablation tables (user request): one table per
    (perturbation, size, c_s), rows = technique, columns = c_t. Every cell is
    exactly ONE run -- no averaging anywhere -- and carries both modes, each
    with its solve time and its precision loss.

    This is the un-aggregated companion to _render_ablation_table, which means
    the same numbers over the whole grid. It lives in table_full_results.tex,
    where a wide table is fine.
    """
    # Which (pert, eps, c_s) blocks exist, from the ABLATED variants only:
    # the 'full' row is an ordinary run covering the whole sweep and would
    # invent blocks the ablation never touched.
    blocks = {}
    for (row, variant), grid in (cells or {}).items():
        if variant == "full":
            continue
        for (pert, eps, cs, ct) in (grid or {}):
            blocks.setdefault((pert, eps, cs), set()).add(ct)
    out = []
    for (pert, eps, cs) in sorted(blocks, key=lambda b: (b[0], b[1], b[2])):
        cts = sorted(blocks[(pert, eps, cs)])
        if not cts:
            continue
        head1 = " & ".join(r"\multicolumn{4}{c}{$c_t{=}%d$}" % ct for ct in cts)
        head2 = " & ".join(r"\multicolumn{2}{c}{\emph{ours}} & "
                           r"\multicolumn{2}{c}{\emph{ours with transfer}}"
                           for _ in cts)
        head3 = " & ".join(r"$t$ (s) & loss & $t$ (s) & loss" for _ in cts)
        rules = " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 4 * i, 5 + 4 * i)
                         for i in range(len(cts)))
        lines = [r"\begin{table*}[!tbp]", r"\centering", r"\small",
                 r"\setlength{\tabcolsep}{4pt}",
                 r"\begin{adjustbox}{max width=\textwidth,center}%",
                 r"\begin{tabular}{@{}l" + "rr rr" * len(cts) + "@{}}",
                 r"\toprule",
                 " & " + head1 + r" \\", rules,
                 " & " + head2 + r" \\",
                 " & " + head3 + r" \\", r"\midrule"]
        for v in _ABLATION_COLUMNS:
            cells_out = []
            for ct in cts:
                for row in ("ours", "transfer"):
                    _v = _ABLATION_MODE_COMPONENT.get((row, v), v)
                    got = ((cells or {}).get((row, _v)) or {}).get(
                        (pert, eps, cs, ct))
                    if not got:
                        cells_out += ["---", "---"]
                        continue
                    cells_out.append(_fmt_ablation_time(got[0]))
                    loss = _ablation_mean_loss(
                        {(pert, eps, cs, ct): got},
                        [(pert, eps, cs, ct)], baseline,
                        delta_max_by_key=delta_max_by_key, arch=arch)
                    cells_out.append("---" if loss is None
                                     else (r"%.2f\%%" % loss))
            lines.append(_ABLATION_COL_HEADERS[v] + " & "
                         + " & ".join(cells_out) + r" \\")
        pert_disp = (str(pert).replace("_", r"\_")
                     .replace("linf", r"$\ell_\infty$"))
        lines += [
            r"\bottomrule", r"\end{tabular}%", r"\end{adjustbox}",
            r"\caption{Ablation experiments over "
            + _AAAI_ARCH_DISPLAY.get(arch, arch.replace("_", r"\_"))
            + r" (" + _dataset_display_name(dataset) + r") for "
            + pert_disp + r" (" + str(eps) + r"), source class $c_s{=}"
            + str(cs) + r"$. Every cell is a single run: its solve time and "
            r"its precision loss against \baseline.}",
            r"\label{tab:ablation-full-" + dataset + label_suffix + "-" + arch
            + "-" + re.sub(r"[^a-z0-9]+", "", str(pert).lower())
            + "-" + re.sub(r"[^0-9]+", "", str(eps)) + "-cs" + str(cs) + r"}",
            r"\end{table*}", ""]
        out.append("\n".join(lines))
    return out


def regenerate_ablation_full_section(tex_path, cwd, dataset, arch_runs,
                                     parse_result_file, seeds_filter=None,
                                     stale_fn=None,
                                     begin_mark=ABLATION_FULL_BEGIN_MARK,
                                     end_mark=ABLATION_FULL_END_MARK,
                                     ds_label_suffix="", expected_map=None,
                                     taus=None):
    """Write the PER-EXPERIMENT ablation tables into `tex_path`'s AUTO block.

    Same collection path as regenerate_ablation_appendix_section (so both
    describe the same runs), but rendered un-aggregated by
    _ablation_full_tables. `expected_map` selects which (dataset, arch) pairs
    render, exactly as it does for the aggregated table."""
    bodies = []
    for arch, _mp in arch_runs:
        pinned_tau = None
        if expected_map:
            hit = (expected_map.get((dataset, arch))
                   or expected_map.get((None, arch)))
            if not hit:
                continue
            if len(hit) > 2 and hit[2] is not None:
                pinned_tau = hit[2]
        try:
            cells, _chosen = _collect_ablation_cells(
                cwd, dataset, arch, parse_result_file,
                seeds_filter=seeds_filter, stale_fn=stale_fn,
                unify_taus=([pinned_tau] if pinned_tau is not None else taus))
            bodies += _ablation_full_tables(
                dataset, arch, cells,
                baseline=_collect_ablation_baseline(
                    cwd, dataset, arch,
                    [(p_, p_ + ":") for p_ in _WIDE_PERT_SUBDIRS],
                    parse_result_file, seeds_filter=seeds_filter),
                delta_max_by_key=_load_delta_max_values(cwd, dataset, [arch]),
                label_suffix=ds_label_suffix)
        except Exception as exc:
            print(f"[update_advstd_tex_tables] ablation full tables "
                  f"{dataset}/{arch} error: {exc}")
            continue
    body = ("\n\n".join(bodies) if bodies
            else "% (no _ablation result files found for this dataset)")
    with open(tex_path) as f:
        text = f.read()
    if begin_mark not in text or end_mark not in text:
        raise SystemExit(f"ablation_full_tables markers not found in {tex_path}")
    pre, rest = text.split(begin_mark, 1)
    _old, post = rest.split(end_mark, 1)
    updated = f"{pre}{begin_mark}\n{body}\n{end_mark}{post}"
    if updated == text:
        print(f"[update_advstd_tex_tables] no changes to ablation_full_tables "
              f"block ({dataset})")
        return
    with open(tex_path, "w") as f:
        f.write(updated)
    print(f"[update_advstd_tex_tables] wrote ablation_full_tables block "
          f"({dataset}, {len(bodies)} table(s)) in {tex_path}")


def regenerate_ablation_appendix_section(tex_path, cwd, dataset, arch_runs,
                                         parse_result_file,
                                         seeds_filter=None, stale_fn=None,
                                         begin_mark=ABLATION_BEGIN_MARK,
                                         end_mark=ABLATION_END_MARK,
                                         ds_label_suffix="",
                                         expected_map=None,
                                         taus=None):
    """Rewrite the ablation_tables AUTO block in `tex_path` with one table
    per arch of `arch_runs` that has `_ablation` result files.

    `expected_map` ({(dataset_or_None, arch): (ct0s, css)}, from
    --ablation_expected) supplies the planned grid the red '*' is judged
    against; dataset-scoped entries win over unscoped ones. When given, it is
    also the table FILTER: only the (dataset, arch) pairs it lists are
    rendered, so leftover _ablation files from earlier experiments (other
    archs/datasets) do not spawn extra tables. Without it, every arch with
    _ablation files renders (the prior behavior)."""
    bodies = []
    for arch, _mp in arch_runs:
        expected = None
        pinned_tau = None
        if expected_map:
            expected = (expected_map.get((dataset, arch))
                        or expected_map.get((None, arch)))
            # --ablation_expected doubles as the render filter: a
            # (dataset, arch) it does not list gets no table at all.
            if expected is None:
                continue
            # '~TAU' entries carry a third element pinning this table's
            # unified threshold; downstream (star grid) keeps the 2-tuple.
            if len(expected) == 3:
                ct0s_, css_, pinned_tau = expected
                expected = (ct0s_, css_)
        try:
            cells, chosen_tau = _collect_ablation_cells(
                cwd, dataset, arch, parse_result_file,
                seeds_filter=seeds_filter, stale_fn=stale_fn,
                unify_taus=([pinned_tau] if pinned_tau is not None else taus))
            if chosen_tau is not None and cells:
                print(f"[update_advstd_tex_tables] ablation table "
                      f"{dataset}/{arch}: unified tau = {chosen_tau:g}")
            # The precision column measures each variant against \baseline's
            # sound interval on the SAME cell, normalised by delta_max --
            # the same construction Table 2 uses.
            tab = _render_ablation_table(
                dataset, arch, cells,
                label_suffix=ds_label_suffix,
                expected=expected, unified_tau=chosen_tau,
                delta_max_by_key=_load_delta_max_values(cwd, dataset, [arch]),
                baseline=_collect_ablation_baseline(
                    # (name, "type:eps") pairs, the shape this collector
                    # parses; the eps half is unused (it globs eps_*), so
                    # every perturbation dir on disk is covered.
                    cwd, dataset, arch,
                    [(p_, p_ + ":") for p_ in _WIDE_PERT_SUBDIRS],
                    parse_result_file, seeds_filter=seeds_filter))
        except Exception as exc:
            print(f"[update_advstd_tex_tables] ablation table {dataset}/{arch} "
                  f"error: {exc}")
            continue
        if tab:
            bodies.append(tab)
    body = ("\n\n".join(bodies) if bodies
            else "% (no _ablation result files found for this dataset)")
    with open(tex_path) as f:
        text = f.read()
    if begin_mark not in text or end_mark not in text:
        raise SystemExit(f"ablation_tables markers not found in {tex_path}")
    pre, rest = text.split(begin_mark, 1)
    _old, post = rest.split(end_mark, 1)
    updated = f"{pre}{begin_mark}\n{body}\n{end_mark}{post}"
    if updated == text:
        print(f"[update_advstd_tex_tables] no changes to ablation_tables "
              f"block ({dataset})")
        return
    with open(tex_path, "w") as f:
        f.write(updated)
    print(f"[update_advstd_tex_tables] wrote ablation_tables block "
          f"({dataset}, {len(bodies)} table(s)) in {tex_path}")


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
