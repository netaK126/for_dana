#!/usr/bin/env python3
"""Regenerate the per-architecture safe-combo tables in advstd_techniques.tex.

Reads the per-cell CSVs produced by --find_advstd_faster_than_standard
(advstd_faster_than_standard{_vs_withPerturbed}.csv and
standard_faster_than_advstd{_vs_withPerturbed}.csv) and rewrites the
tables between the AUTO markers in advstd_techniques.tex.

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
    paths = [
        os.path.join(csv_dir, f"advstd_faster_than_standard{suffix}.csv"),
        os.path.join(csv_dir, f"standard_faster_than_advstd{suffix}.csv"),
    ]
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[update_advstd_tex_tables] missing: {p}", file=sys.stderr)
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get("arch"):
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


def collect_arch(rows, arch, seed, tau):
    filt = [r for r in rows
            if r["arch"] == arch
            and r["seed"] == seed
            and r["relax_threshold"] == tau
            and r["bound_tightening"] == "yes"]
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
        seen[(grp, cell)] = (float(r["time_standard"]),
                             float(r["time_advstd"]),
                             t_adv_lp,
                             r_total)
    by_grp = defaultdict(list)
    for (grp, cell), tup in seen.items():
        by_grp[grp].append(tup)

    safe = []
    for grp, tuples in by_grp.items():
        t_std = sum(t for t, _, _, _ in tuples) / len(tuples)
        t_adv = sum(a for _, a, _, _ in tuples) / len(tuples)
        lp_vals = [lp for _, _, lp, _ in tuples if lp is not None]
        t_adv_lp = (sum(lp_vals) / len(lp_vals)) if lp_vals else None
        r_vals = [r for _, _, _, r in tuples if r is not None]
        n_relax = (sum(r_vals) / len(r_vals)) if r_vals else None
        hmfs = [a / t for t, a, _, _ in tuples if t > 0]
        if not hmfs:
            continue
        gm = math.exp(sum(math.log(h) for h in hmfs) / len(hmfs))
        mn = max(hmfs)
        tier = classify(mn)
        if tier in ("dominant", "avg-win") and gm < 1.0 / 1.05:
            su_lp = (t_adv_lp / t_std) if (t_adv_lp is not None
                                            and t_std > 0) else None
            safe.append((grp, t_std, t_adv, t_adv / t_std,
                         t_adv_lp, su_lp, gm, mn, tier, n_relax))
    return safe


def render_table(arch, safe, seed, tau):
    if not safe:
        return (f"% no safe-tier rows for arch={arch} at seed={seed}, "
                f"tau={tau}\n")

    by_ps = defaultdict(list)
    for e in safe:
        by_ps[(e[0][1], e[0][2])].append(e)
    ordered_ps = sorted(
        by_ps.keys(),
        key=lambda ps: (PERT_ORDER.index(ps[0]) if ps[0] in PERT_ORDER
                        else 99, ps[1]))

    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append(r"\scriptsize")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\begin{tabular}{@{}l l l l l l r r r r r r r@{}}")
    out.append(r"\toprule")
    out.append(r"p\_type & p\_size & \tech{bound\_tight} & \tech{bp} & "
               r"\tech{varHint} & $\tau$ & "
               r"t\_std & t\_adv & $\text{sp}$ & "
               r"t\_adv\_lp & $\text{sp}_\mathrm{lp}$ & min\_speed\_up & "
               r"n\_relax \\")
    out.append(r"\midrule")

    first = True
    for ps in ordered_ps:
        if not first:
            out.append(r"\midrule")
        first = False
        block = sorted(by_ps[ps], key=lambda r: r[3])  # speed_up ascending
        head = True
        for (grp, ts, ta, su, ta_lp, su_lp, gm, mn, tier, n_relax) in block:
            _ms, bp, _lb, bt, vh, zb, np_, rt = grp[0]
            if bt != "yes":
                bt_label = "none"
            else:
                base = "zono" if zb == "yes" else "interval"
                bt_label = base + ("+lp" if np_ == "lp" else "")
            pert, psize = ps
            pert_c = pert if head else ""
            size_c = f"\\texttt{{{psize}}}" if head else ""
            ms = f"{mn:.3f}$\\times$"
            ta_lp_c = f"{ta_lp:7.2f}" if ta_lp is not None else "    ---"
            su_lp_c = f"{su_lp:.3f}" if su_lp is not None else "  ---"
            nr_c = f"{n_relax:6.1f}" if n_relax is not None else "   ---"
            prefix = r"\rowcolor{green!15} " if (bp == "off"
                                                  and np_ != "lp") else ""
            out.append(
                f"{prefix}{pert_c:11s} & {size_c:20s} & {bt_label:11s} & {bp:6s} & "
                f"{vh:3s} & {rt:4s} & {ts:7.2f} & {ta:7.2f} & "
                f"{su:.3f} & {ta_lp_c} & {su_lp_c} & {ms} & {nr_c} \\\\")
            head = False
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(
        f"\\caption{{Architecture \\textbf{{{arch}}} --- safe advstd "
        f"combinations at \\textsf{{seed}}$={seed}$, $\\tau={tau}$, broken "
        f"out by perturbation type and size. \\textsf{{t\\_std}} and "
        f"\\textsf{{t\\_adv}} are mean $N_2$ solve times (seconds) averaged "
        f"across $(c_\\mathrm{{tag}}, c_\\mathrm{{target}})$ cells in each "
        f"block; \\textsf{{sp}}$=\\textsf{{t\\_adv}}/\\textsf{{t\\_std}}$ "
        f"($<\\!1$ means advstd was faster). \\textsf{{t\\_adv\\_lp}} adds "
        f"the N1-probe LP construction time (i.e.\\ "
        f"$\\textsf{{t\\_adv}}+t_\\mathrm{{LP}}$) and "
        f"$\\textsf{{sp}}_\\mathrm{{lp}}=\\textsf{{t\\_adv\\_lp}}/"
        f"\\textsf{{t\\_std}}$; both are \\textsf{{---}} when the "
        f"\\tech{{bound\\_tight}} label has no \\texttt{{+lp}} suffix. "
        f"\\tech{{bound\\_tight}} collapses Technique~2 and its optional "
        f"Source~C into a single label: "
        f"\\texttt{{interval}} / \\texttt{{interval+lp}} / "
        f"\\texttt{{zono}} / \\texttt{{zono+lp}} / \\texttt{{none}} "
        f"(maps to \\tech{{boundTight}}$\\in\\{{$yes, no$\\}}$, "
        f"\\tech{{zono}}$\\in\\{{$yes, no$\\}}$, "
        f"\\tech{{n1Probe}}$\\in\\{{$lp, off$\\}}$). All "
        f"{len(safe)} rows pass the safe-set criterion "
        f"(\\textsf{{perf\\_tier}}$\\in\\{{\\textsf{{dominant}},"
        f"\\textsf{{avg-win}}\\}}$), based on "
        f"$t_\\mathrm{{adv}}/t_\\mathrm{{std}}$; "
        f"\\textsf{{min\\_speed\\_up}} is the worst-cell ratio "
        f"$\\max(t_\\mathrm{{adv}}/t_\\mathrm{{std}})$ per group. "
        f"\\textsf{{n\\_relax}} is the mean number of N2 binary variables "
        f"removed by Technique~4 (sum of N2 original + perturbed copies), "
        f"averaged across $(c_\\mathrm{{tag}}, c_\\mathrm{{target}})$ cells; "
        f"\\textsf{{---}} when $\\tau=\\textsf{{off}}$. "
        f"\\tech{{mipStart}} and \\tech{{lpBasis}} are always off "
        f"(omitted for brevity). Rows shaded green satisfy both "
        f"\\tech{{bp}}=\\textsf{{off}} and \\tech{{n1Probe}}=\\textsf{{off}} "
        f"(no \\texttt{{+lp}} suffix), the simplified configuration "
        f"recommended for explainability. Auto-generated by "
        f"\\texttt{{update\\_advstd\\_tex\\_tables.py}}.}}")
    label = arch.replace("=", "_")
    out.append(f"\\label{{tab:safe_{label}}}")
    out.append(r"\end{table}")
    out.append("")
    return "\n".join(out)


def render_all(archs, rows, seed, tau):
    sections = []
    summary = []
    for arch in archs:
        safe = collect_arch(rows, arch, seed, tau)
        sections.append(render_table(arch, safe, seed, tau))
        summary.append(f"{arch}={len(safe)}")

    intro = (
        r"Tables below list the safe-to-use combinations at "
        rf"\textsf{{seed}}$={seed}$ and relax threshold $\tau={tau}$, "
        r"evaluated per-architecture and broken out by perturbation type "
        r"and size. For each (combo, perturbation, size) row we report "
        r"\textsf{t\_std} and \textsf{t\_adv} (mean wall-clock $N_2$ "
        r"solve times in seconds, averaged across all "
        r"$(c_\mathrm{tag}, c_\mathrm{target})$ cells in the block) and "
        r"\textsf{sp}$=\textsf{t\_adv}/\textsf{t\_std}$ (lower is better; "
        r"$<\!1$ means advstd finished faster than the "
        r"with-perturbed-intervals baseline). We also report "
        r"\textsf{t\_adv\_lp} and "
        r"$\textsf{sp}_\mathrm{lp}=\textsf{t\_adv\_lp}/\textsf{t\_std}$, "
        r"which include the N1-probe LP construction time on top of "
        r"\textsf{t\_adv}; both are \textsf{---} when the "
        r"\tech{bound\_tight} label has no \texttt{+lp} suffix. The "
        r"\tech{bound\_tight} column labels which Technique~2 variant is "
        r"used (\texttt{interval} / \texttt{interval+lp} / \texttt{zono} / "
        r"\texttt{zono+lp} / \texttt{none}), and the $\tau$ column is the "
        r"Technique~4 threshold. Rows are kept only if "
        r"\textsf{perf\_tier}$\in\{\textsf{dominant},\textsf{avg-win}\}$ "
        r"(geometric-mean per-cell time ratio $<0.952\times$ and worst-cell "
        r"$t_\mathrm{adv}/t_\mathrm{std}\le 1.333\times$); within each "
        r"(perturbation, size) block rows are sorted by ascending "
        r"\textsf{sp}. \tech{mipStart} and \tech{lpBasis} are always off "
        r"and omitted for brevity."
        "\n\n"
        f"% auto-generated: archs=[{', '.join(summary)}], seed={seed}, "
        f"tau={tau}\n")
    return intro + "\n".join(sections)


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
    ap.add_argument("--seed", default="4")
    ap.add_argument("--tau", default="0.1")
    ap.add_argument("--archs", nargs="+", default=None,
                    help="archs to emit tables for (default: infer from CSV)")
    ap.add_argument("--vs_with_perturbed", action="store_true", default=True)
    ap.add_argument("--no_vs_with_perturbed", dest="vs_with_perturbed",
                    action="store_false")
    args = ap.parse_args()

    suffix = "_vs_withPerturbed" if args.vs_with_perturbed else ""
    rows = load_rows(args.csv_dir, suffix)
    if not rows:
        raise SystemExit(
            f"no rows loaded from {args.csv_dir} (suffix='{suffix}')")

    archs = args.archs
    if archs is None:
        archs = sorted({r["arch"] for r in rows if r.get("arch")})
    body = render_all(archs, rows, args.seed, args.tau)
    update_tex(args.tex, body)


if __name__ == "__main__":
    main()
