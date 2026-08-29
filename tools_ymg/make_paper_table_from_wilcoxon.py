\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def parse_labels(methods: List[str], labels_csv: Optional[str]) -> Dict[str, str]:
    if not labels_csv:
        return {m: m for m in methods}
    labels = [x.strip() for x in labels_csv.split(",")]
    if len(labels) != len(methods):
        raise ValueError("--method-labels must have the same length as --methods")
    return dict(zip(methods, labels))

def fmt_num(x: float, digits: int = 2, strip_zeros: bool = True) -> str:
    if pd.isna(x):
        return ""
    s = f"{float(x):.{digits}f}"
    if strip_zeros:
        s = s.rstrip("0").rstrip(".")
    return s

def sig_symbol(
    method_value: float,
    baseline_value: float,
    pvalue: float,
    direction: str,
    alpha: float,
) -> str:

    if pd.isna(method_value) or pd.isna(baseline_value) or pd.isna(pvalue):
        return ""
    if float(pvalue) >= alpha:
        return r"\approx"
    if direction == "max":
        return "+" if method_value > baseline_value else "-"
    return "+" if method_value < baseline_value else "-"

def symbol_for_csv(symbol: str) -> str:
    return {"\\approx": "≈"}.get(symbol, symbol)

def escape_latex(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(s))


def load_table_data(
    df: pd.DataFrame,
    eval_point: int,
    methods: List[str],
    baseline: str,
    alpha: float,
) -> Tuple[pd.DataFrame, str]:
    d = df[(df["eval"].astype(int) == int(eval_point)) & (df["instance"] != "__ALL__")].copy()
    if d.empty:
        raise ValueError(f"No rows found for eval={eval_point}")


    direction = str(d["direction"].dropna().iloc[0])
    if direction == "auto":
        direction = "min"

    rows = []
    for ins, g in d.groupby("instance", sort=False):
        row = {"instance": ins}


        bg = g[g["baseline"] == baseline]
        if not bg.empty:
            row[baseline] = float(bg["baseline_mean"].dropna().iloc[0])
            row[f"{baseline}__pvalue"] = np.nan
            row[f"{baseline}__symbol"] = ""
        else:
            row[baseline] = np.nan
            row[f"{baseline}__pvalue"] = np.nan
            row[f"{baseline}__symbol"] = ""

        for m in methods:
            if m == baseline:
                continue
            mg = g[g["method"] == m]
            if mg.empty:
                row[m] = np.nan
                row[f"{m}__pvalue"] = np.nan
                row[f"{m}__symbol"] = ""
                continue
            rec = mg.iloc[0]
            mv = float(rec["method_mean"])
            bv = float(rec["baseline_mean"])
            pv = float(rec["pvalue"])
            row[m] = mv
            row[f"{m}__pvalue"] = pv
            row[f"{m}__symbol"] = sig_symbol(mv, bv, pv, direction, alpha)
        rows.append(row)

    out = pd.DataFrame(rows)
    return out, direction


def add_ranks(table: pd.DataFrame, methods: List[str], direction: str) -> Dict[str, float]:
    ranks = {m: [] for m in methods}
    ascending = direction != "max"
    for _, row in table.iterrows():
        vals = pd.Series({m: row.get(m, np.nan) for m in methods}, dtype="float64").dropna()
        if vals.empty:
            continue
        rr = vals.rank(method="average", ascending=ascending)
        for m, r in rr.items():
            ranks[m].append(float(r))
    return {m: (float(np.mean(v)) if v else np.nan) for m, v in ranks.items()}


def plus_approx_minus_counts(table: pd.DataFrame, methods: List[str], baseline: str) -> Dict[str, Tuple[int, int, int]]:
    out = {}
    for m in methods:
        if m == baseline:
            out[m] = (0, 0, 0)
            continue
        syms = table.get(f"{m}__symbol", pd.Series([], dtype=str)).fillna("").tolist()
        plus = sum(s == "+" for s in syms)
        approx = sum(s == r"\approx" for s in syms)
        minus = sum(s == "-" for s in syms)
        out[m] = (plus, approx, minus)
    return out


def build_csv_table(
    table: pd.DataFrame,
    methods: List[str],
    labels: Dict[str, str],
    baseline: str,
    direction: str,
    digits: int,
) -> pd.DataFrame:
    rows = []
    for _, r in table.iterrows():
        row = {"instance": r["instance"]}
        for m in methods:
            val = fmt_num(r.get(m, np.nan), digits)
            sym = symbol_for_csv(r.get(f"{m}__symbol", ""))
            row[labels[m]] = (val if m == baseline or not sym else f"{val} {sym}").strip()
        rows.append(row)

    counts = plus_approx_minus_counts(table, methods, baseline)
    row = {"instance": "+ / ≈ / −"}
    for m in methods:
        row[labels[m]] = "-" if m == baseline else f"{counts[m][0]}/{counts[m][1]}/{counts[m][2]}"
    rows.append(row)

    avg = add_ranks(table, methods, direction)
    row = {"instance": "Average rank"}
    for m in methods:
        row[labels[m]] = fmt_num(avg[m], 2)
    rows.append(row)
    return pd.DataFrame(rows)


def latex_cell(
    value: float,
    symbol: str,
    is_best: bool,
    is_second: bool,
    method: str,
    baseline: str,
    digits: int,
    color_best: str,
    color_second: str,
    color_worse_sig: str,
) -> str:
    if pd.isna(value):
        return ""
    body = fmt_num(value, digits)
    if is_best:
        body = r"\textbf{" + body + "}"
    if symbol:
        body += " " + symbol

    if is_best:
        return rf"\cellcolor{{{color_best}}}{body}"
    if is_second:
        return rf"\cellcolor{{{color_second}}}{body}"
    if method != baseline and symbol == "+":
        return rf"\cellcolor{{{color_second}}}{body}"
    if method != baseline and symbol == "-":
        return rf"\cellcolor{{{color_worse_sig}}}{body}"
    return body


def build_latex(
    table: pd.DataFrame,
    methods: List[str],
    labels: Dict[str, str],
    baseline: str,
    direction: str,
    eval_point: int,
    problem_label: str,
    caption: str,
    label: str,
    digits: int,
    model_free: List[str],
    saea: List[str],
    color_best: str,
    color_second: str,
    color_worse_sig: str,
) -> str:
    lines = []
    colspec = "ll" + "r" * len(methods)
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(rf"\caption{{{escape_latex(caption)}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")


    if model_free or saea:
        group = ["", ""]
        i = 0
        while i < len(methods):
            m = methods[i]
            if m in model_free:
                j = i
                while j < len(methods) and methods[j] in model_free:
                    j += 1
                group.append(rf"\multicolumn{{{j-i}}}{{c}}{{Model-free}}")
                i = j
            elif m in saea:
                j = i
                while j < len(methods) and methods[j] in saea:
                    j += 1
                group.append(rf"\multicolumn{{{j-i}}}{{c}}{{SAEA}}")
                i = j
            else:
                group.append("")
                i += 1
        lines.append(" & ".join(group) + r" \\")
        lines.append(r"\cmidrule(lr){3-" + str(2 + len(methods)) + "}")

    header = ["Problem", "Prob. size"] + [escape_latex(labels[m]) for m in methods]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for ridx, (_, r) in enumerate(table.iterrows()):
        vals = pd.Series({m: r.get(m, np.nan) for m in methods}, dtype="float64").dropna()
        if direction == "max":
            sorted_methods = vals.sort_values(ascending=False).index.tolist()
        else:
            sorted_methods = vals.sort_values(ascending=True).index.tolist()
        best = sorted_methods[0] if sorted_methods else None
        second = sorted_methods[1] if len(sorted_methods) > 1 else None

        problem_cell = escape_latex(problem_label) if ridx == 0 else ""
        inst = str(r["instance"])

        import re
        nums = re.findall(r"\d+", inst)
        size = nums[-1] if nums else ""

        cells = [problem_cell + " " + escape_latex(inst) if problem_cell else escape_latex(inst), size]
        for m in methods:
            cells.append(
                latex_cell(
                    r.get(m, np.nan),
                    r.get(f"{m}__symbol", ""),
                    m == best,
                    m == second,
                    m,
                    baseline,
                    digits,
                    color_best,
                    color_second,
                    color_worse_sig,
                )
            )
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    counts = plus_approx_minus_counts(table, methods, baseline)
    cells = [r"$+ / \approx / -$", ""]
    for m in methods:
        cells.append("-" if m == baseline else f"{counts[m][0]}/{counts[m][1]}/{counts[m][2]}")
    lines.append(" & ".join(cells) + r" \\")
    avg = add_ranks(table, methods, direction)
    rank_vals = pd.Series(avg, dtype="float64")
    best_rank = rank_vals.idxmin() if not rank_vals.dropna().empty else None
    second_rank = rank_vals.sort_values().index[1] if rank_vals.dropna().shape[0] > 1 else None
    cells = ["Average rank", ""]
    for m in methods:
        v = fmt_num(avg[m], 2)
        if m == best_rank:
            v = rf"\cellcolor{{{color_best}}}\textbf{{{v}}}"
        elif m == second_rank:
            v = rf"\cellcolor{{{color_second}}}{v}"
        cells.append(v)
    lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="", help="result root")
    ap.add_argument("--eval", type=int, default= 300, help="FE/eval point to make the table for")
    ap.add_argument("--methods", nargs="+", default=["RLEA", "gbdtma", "fatrls", "umm", "rflos"], help="Column order. Include baseline here.")
    ap.add_argument("--baseline", default="RLEA", help="Baseline method. Usually the proposed method.")
    ap.add_argument("--method-labels", default="RLEA,GBDTMA,FAT-RLS,UMM,RFLoS", help="Comma-separated labels")
    ap.add_argument("--instance-order", nargs="*", default=None)
    ap.add_argument("--problem-label", default="ALL")
    ap.add_argument("--caption", default=None)
    ap.add_argument("--label", default="tab:comparison")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--digits", type=int, default=2)
    ap.add_argument("--model-free", nargs="*", default=[])
    ap.add_argument("--saea", nargs="*", default=[])

    ap.add_argument("--color-best", default="green!18")
    ap.add_argument("--color-second", default="green!8")
    ap.add_argument("--color-worse-sig", default="red!8")
    args = ap.parse_args()

    if args.baseline not in args.methods:
        raise ValueError("--baseline must be included in --methods")

    summary_csv = Path(args.root) / f"wilcoxon_fe_summary_all.csv"

    df = pd.read_csv(summary_csv)
    labels = parse_labels(args.methods, args.method_labels)
    table, direction = load_table_data(df, args.eval, args.methods, args.baseline, args.alpha)

    if args.instance_order:
        order = {name: i for i, name in enumerate(args.instance_order)}
        table["_ord"] = table["instance"].map(order).fillna(10**9).astype(int)
        table = table.sort_values(["_ord", "instance"], kind="stable").drop(columns=["_ord"])

    csv_table = build_csv_table(table, args.methods, labels, args.baseline, direction, args.digits)
    out_prefix = Path(args.root) / f"table_{args.problem_label}_fe{args.eval}_man"
    csv_path = out_prefix.with_suffix(".csv")
    tex_path = out_prefix.with_suffix(".tex")
    csv_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    caption = args.caption or f"Performance comparison and statistical results under {args.eval} FEs"
    tex = build_latex(
        table=table,
        methods=args.methods,
        labels=labels,
        baseline=args.baseline,
        direction=direction,
        eval_point=args.eval,
        problem_label=args.problem_label,
        caption=caption,
        label=args.label,
        digits=args.digits,
        model_free=args.model_free,
        saea=args.saea,
        color_best=args.color_best,
        color_second=args.color_second,
        color_worse_sig=args.color_worse_sig,
    )
    tex_path.write_text(tex, encoding="utf-8")

    print(f"[OK] CSV  -> {csv_path}")
    print(f"[OK] LaTeX -> {tex_path}")


if __name__ == "__main__":
    main()
