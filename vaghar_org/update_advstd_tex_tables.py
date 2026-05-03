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
        k = (r["mip_start"], r["branch_priorities"], r["lp_basis"],
             r["bound_tightening"], r["var_hint"], r["zono_bounds"],
             r["n1_probe"], r["relax_threshold"])
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
    """Parse a 'bt:varHint:tau' string for --combination_table.

    Format, e.g. 'zono:prev_pgd:0.5':
      - bt   ∈ {none, interval, zono, interval+lp, zono+lp} — matches the
               bound_tight column rendered in the per-arch tables.
      - vh   ∈ {no, off, prev, direct, direct_pgd, prev_pgd} — matches the
               CSV var_hint column verbatim ('off' is accepted as an alias
               for 'no').
      - tau  ∈ {off, 0.0, 0.01, 0.05, 0.1, 0.5, 1.0} — matches the
               relax_threshold column verbatim.
    Returns None if spec is empty / None.
    """
    if not spec:
        return None
    parts = [p.strip() for p in str(spec).split(":")]
    if len(parts) != 3:
        raise SystemExit(
            f"--combination_table: expected 'bt:varHint:tau', got {spec!r}")
    bt, vh, tau = parts
    if vh.lower() == "off":
        vh = "no"
    return (bt, vh, tau)


def _combo_label(grp_key):
    """(bt_label, vh, rt) for a combo's grp[0] tuple, matching the format
    rendered into the per-arch tables. Used to filter combos against a
    --combination_table spec."""
    _ms, _bp, _lb, bt, vh, zb, np_, rt = grp_key
    if bt != "yes":
        bt_label = "none"
    else:
        base = "zono" if zb == "yes" else "interval"
        bt_label = base + ("+lp" if np_ == "lp" else "")
    return (bt_label, vh, str(rt).strip())


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
        _ms, bp, _lb, bt, vh, zb, np_, rt = grp[0]
        if bt != "yes":
            bt_label = "none"
        else:
            base = "zono" if zb == "yes" else "interval"
            bt_label = base + ("+lp" if np_ == "lp" else "")
        prefix = (f"\\rowcolor{{{row_color}}} " if np_ != "lp" else "")
        pert, psize = ps
        combo_head = True
        # Per-c_src mean sp (mean of per-cell t_std/t_adv), shown as an
        # extra column on the first row of each c_src group within a combo.
        mean_sp_by_csrc = {}
        mean_dd_by_csrc = {}
        seen_csrc_in_combo = set()
        by_csrc = defaultdict(list)
        for cell in cells:
            by_csrc[cell[0]].append(cell)
        for c_src_key, group in by_csrc.items():
            # Exclude mismatched-timeout cells from the per-c_src means.
            # Their `sp` is meaningless (different `--timout` caps), so
            # including them would skew the average and the user couldn't
            # tell whether a low average reflects real slowdowns or just
            # an unfair budget.
            sps = [g[2] / g[3] for g in group
                   if g[3] > 0 and not g[10]]
            mean_sp_by_csrc[c_src_key] = (
                (sum(sps) / len(sps)) if sps else float("nan"))
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
            rt_c = rt if combo_head else ""
            nr_c = (f"{n_relax:5d}"
                    if isinstance(n_relax, int) else "  ---")
            if _tom:
                # Mismatched `--timout` caps on this cell — replace the
                # last 3 columns (sp, mean-sp, mean-delta-diff) with a
                # single \multicolumn warning that records both wall-clocks
                # so the reader knows the comparison is unsound and which
                # leg needs a 2nd attempt under the matching cap. We do
                # NOT add c_src to `seen_csrc_in_combo`, so the per-c_src
                # avg display defers to the next non-mismatched cell.
                warn_inner = (
                    r"\itshape\bfseries timeouts differ; 2nd attempt "
                    rf"required ($T_{{\mathrm{{std}}}}={ts:.0f}\,$s, "
                    rf"$T_{{\mathrm{{adv}}}}={ta:.0f}\,$s)")
                warning_cell = (
                    r"\multicolumn{3}{l}{" + warn_inner + r"}")
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
                msp = mean_sp_by_csrc[c_src]
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
                avg_sp_c = "       "
                dd_c = "       "
            out.append(
                f"{prefix}{pert_c:11s} & {size_c:20s} & {bt_c:11s} & "
                f"{vh_c:3s} & {rt_c:4s} & "
                f"{c_src:>5s} & {c_tgt:>5s} & {nr_c} & "
                f"{ts:7.2f} & {ta:7.2f} & {su_c} & {avg_sp_c} & "
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
    out.append(r"\begin{tabular}{@{}l l l l l | r r r r r r r r@{}}")
    out.append(r"\hline")
    out.append(r"p\_type & p\_size & \tech{bound\_tight} & "
               r"\tech{varHint} & $\tau$ & "
               r"c\_src & c\_tgt & n\_relax & "
               r"t\_std & t\_adv & $\text{sp}$ & "
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


def _render_merged_combo_tables(arch, arch_label, by_ps, combination_filter,
                                seed, tau, has_timeout_table):
    """Emit one \\begin{table} per (pert, psize) for the filtered combo.

    All c_src groups for that (pert, psize) are stacked into a single
    table; each c_src group is rendered as its own block via
    `_emit_block_rows` with `row_color=_csrc_color(c_src)`, so c_src=0
    rows stay green, c_src=1 yellow, c_src=2 magenta, etc. — the same
    palette used in the unfiltered tables. Block ordering inside a
    table is by ascending c_src.
    """
    by_pp = defaultdict(lambda: defaultdict(list))
    for ps, sub_combos in by_ps.items():
        pert, psize, c_src, _bucket = ps
        # Multiple tau_buckets may collapse here (with a filter, in
        # practice only one survives); merge them into the same c_src
        # block so the output is one table per (pert, psize).
        by_pp[(pert, psize)][c_src].extend(sub_combos)
    ordered_pp = sorted(
        by_pp.keys(),
        key=lambda pp: (PERT_ORDER.index(pp[0]) if pp[0] in PERT_ORDER
                        else 99, pp[1]))

    bt_tex, vh_tex, tau_tex = combination_filter
    vh_tex_safe = vh_tex.replace("_", r"\_")
    combo_tex = f"\\texttt{{{bt_tex}:{vh_tex_safe}:{tau_tex}}}"

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
                f"merged per-perturbation table for combo {combo_tex} "
                f"at {_seed_tau_phrase(seed, tau)}; perturbation "
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
                f"combo {combo_tex} ({block_cells} cell rows); "
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
    # tables (the c_src-tinted blocks) to only the matching combo. The
    # overall summary and the timeout-gap table are NOT filtered — they
    # remain useful as global ranking / context.
    if combination_filter is not None:
        all_combos = [e for e in all_combos
                      if _combo_label(e[0][0]) == combination_filter]
    if not all_combos:
        return (f"% no rows for arch={arch} at seed={seed}, "
                f"tau={tau}"
                + (f", combo={combination_filter}"
                   if combination_filter else "")
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
        rt = grp[0][-1]
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
    # timeout-gap table to the matching combo too. The combo key on
    # each timeout entry is `e[0][0]` (same shape as the per-arch
    # combo grp), so we can apply `_combo_label` exactly like in
    # `render_table`.
    if combination_filter is not None:
        timeout_all = [e for e in timeout_all
                       if _combo_label(e[0][0]) == combination_filter]
    if not timeout_all:
        return (f"% no timeout cells for arch={arch} at seed={seed}, "
                f"tau={tau}"
                + (f", combo={combination_filter}"
                   if combination_filter else "")
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
            _ms, bp, _lb, bt, vh, zb, np_, rt = grp[0]
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
            out.append(
                f"{prefix}{pert_c:11s} & {size_c:20s} & {bt_label:11s} & "
                f"{vh_tex:3s} & {rt:4s} & {c_src:>5s} & "
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
            # when a single combo was requested, drop every other combo
            # from this aggregation so the row count, win-rate, and
            # loser/winner classifications all reflect only that combo.
            if (combination_filter is not None
                    and _combo_label(key) != combination_filter):
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
                    agg[key]["sub_sps"][
                        (arch, pert, psize, c_src)].append(ratio)
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
                by_triple[(a, p, ps)].append(
                    (cs, sum(sub_sps) / len(sub_sps)))
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
                by_pair[(a, cs)].append(
                    (p, ps, sum(sub_sps) / len(sub_sps)))
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
            _ms, bp, _lb, bt, vh, zb, np_, rt = key
            if bt != "yes":
                bt_label = "none"
            else:
                base = "zono" if zb == "yes" else "interval"
                bt_label = base + ("+lp" if np_ == "lp" else "")
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
                f"{rt:4s} & {sp_c} & {n_wins:5d} & {n_cells:6d} & "
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
        bt_tex, vh_tex, tau_tex = combination_filter
        vh_tex = vh_tex.replace("_", r"\_")
        intro += (
            "\n\n"
            r"\noindent\textbf{Note:} every table below --- the overall "
            r"combination ranking, the per-arch perturbation tables, "
            r"and the per-arch \texttt{TIME\_LIMIT} gap-comparison "
            r"tables --- is restricted to the single combination "
            r"\texttt{" + f"{bt_tex}:{vh_tex}:{tau_tex}" + r"} "
            r"(via \texttt{--combination\_table}).")
    intro += (
        "\n\n"
        + f"% auto-generated: archs=[{', '.join(summary)}], seed={seed}, "
          f"tau={tau}"
        + (f", combo_filter={':'.join(combination_filter)}"
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
    return (intro + overall_table + "\n".join(sections_main)
            + mid_section)


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
                    metavar="BT:VH:TAU",
                    help="Restrict the per-arch perturbation tables (the "
                         "c_src-tinted blocks) to a single combo. "
                         "Format '<bound_tight>:<varHint>:<tau>', e.g. "
                         "'zono:prev_pgd:0.5'. The overall ranking and "
                         "timeout-gap tables remain unfiltered.")
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
