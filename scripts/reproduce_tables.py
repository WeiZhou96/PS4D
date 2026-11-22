"""Recompute the paper's result tables from the per-clip results files.

Tables (LaTeX rows as in the paper, plus a plain-text version on stdout):
  main_rows.tex      Table 1: depth motion on the three test sets (median r, eps x 100, beta; paired PS4D - Pi3X)
  ablation_rows.tex  Table 2: ablations (median r and eps x 100)
  x4d_rows.tex       supplement: PS4D with each 4D model (track lifting, 4D branch, PS4D, paired difference)
  compl_rows.tex     supplement: averaging the 4D branch with another 4D branch, and Pi3X with a global gain
Summaries: medians over the clips on which every compared method has a value; paired differences are medians of
per-clip differences with 95% bootstrap intervals (2000 resamples, seed 0, clustered by recording sequence on HOT3D).

Usage: python scripts/reproduce_tables.py [--results results] [--out tables_out] [--check paper_tables_dir]
--check compares every cell with existing LaTeX row files of the same names and reports differences.
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ps4d.io import read_results  # noqa: E402
from ps4d.metrics import complete, median, median_ci, paired  # noqa: E402

TEST = {"DEX": "test/dexycb.json.gz", "Q3": "test/hot3d_quest3.json.gz", "DROID": "test/droid.json.gz"}
DEV = {"A": "dev/droid_dev.json.gz", "B12": "dev/dexycb_01_02.json.gz", "B35": "dev/dexycb_03_05.json.gz",
       "H3": "dev/hot3d_aria.json.gz"}
SETS = list(TEST)
# a clip enters the tables when all of these methods have a value
NAMES = ["ps4d_pi3x", "lift_pi3x", "ps", "branch4d_pi3x", "spatracker2", "lift_vda", "lift_moge2", "joint_graph",
         "disagreement_weights"]
OTHER_LIFTS = ("lift_pi3", "lift_vggt", "lift_cut3r")


def ranked(vals):
    v = sorted(set(round(x, 3) for x in vals if x is not None))
    return {"best": v[0] if v else None, "second": v[1] if len(v) > 1 else None}


def fmt(x, rk, nd):
    if x is None:
        return "--"
    s = f"{x:.{nd}f}"
    if rk["best"] is not None and round(x, 3) == rk["best"]:
        return r"\textbf{" + s + "}"
    if rk["second"] is not None and round(x, 3) == rk["second"]:
        return r"\underline{" + s + "}"
    return s


def neg(s):
    return s.replace("-", "$-$")


def main_tables(R, out):
    P = {s: complete(R[s], NAMES) for s in SETS}

    def stats(s, n):
        Q = [c for c in P[s] if n in c["methods"]] if n in OTHER_LIFTS else P[s]
        if not Q:
            return None
        return (median(Q, n), 100 * median(Q, n, "eps"), median(Q, n, "beta"), len(Q))

    rows = [("lift_moge2", "CoTracker3 + MoGe-2"), ("lift_vda", "CoTracker3 + VDA-L"),
            ("spatracker2", "SpatialTrackerV2"), ("lift_pi3", "$\\pi^3$"), ("lift_vggt", "VGGT"),
            ("lift_cut3r", "CUT3R"), ("lift_pi3x", "Pi3X"), None,
            ("branch4d_pi3x", "4D branch: Pi3X depth in our masks"), ("ps", "Projective-scale branch"),
            ("ps4d_pi3x", "\\method{} (average of both branches)")]
    S = {s: {n: stats(s, n) for n, _ in [r for r in rows if r]} for s in SETS}
    rk = {(s, j): ranked([S[s][n][j] if S[s][n] else None for n, _ in [r for r in rows if r]])
          for s in SETS for j in (0, 1)}
    lines = []
    for r in rows:
        if r is None:
            lines.append("\\midrule")
            continue
        n, lab = r
        cells = []
        for s in SETS:
            v = S[s][n]
            cells += ["--"] * 3 if v is None else [fmt(v[0], rk[(s, 0)], 3), fmt(v[1], rk[(s, 1)], 2), neg(f"{v[2]:.2f}")]
        lines.append(("\\rowcolor{ourstint}" if n == "ps4d_pi3x" else "") + lab + " & " + " & ".join(cells) + " \\\\")
    d = []
    for s in SETS:
        (m, lo, hi), _, _ = paired(P[s], "ps4d_pi3x", "lift_pi3x")
        d.append("\\multicolumn{3}{c}{" + f"${m:+.3f}$ $[{lo:+.3f},{hi:+.3f}]$" + "}")
    lines += ["\\midrule", "\\method{} $-$ Pi3X, paired $\\Delta r$ & " + " & ".join(d) + " \\\\"]
    write(out, "main_rows.tex", lines)

    ab = [("size_unweighted", "Size cue, re-anchored, unweighted"), ("ps_equal", "+ rigid measurement, robust graph"),
          ("ps", "+ learned reliability"), ("ps4d_pi3x", "+ average with 4D branch (\\method{})"), None,
          ("joint_graph", "Joint graph, learned 4D variances"), ("disagreement_weights", "Disagreement-based weights")]
    A = {s: {n: stats(s, n) for n, _ in [r for r in ab if r]} for s in SETS}
    rka = {(s, j): ranked([A[s][n][j] for n, _ in [r for r in ab if r]]) for s in SETS for j in (0, 1)}
    lines = []
    for r in ab:
        if r is None:
            lines.append("\\midrule")
            continue
        n, lab = r
        cells = []
        for s in SETS:
            v = A[s][n]
            cells += [fmt(v[0], rka[(s, 0)], 3), fmt(v[1], rka[(s, 1)], 2)]
        lines.append(("\\rowcolor{ourstint}" if n == "ps4d_pi3x" else "") + lab + " & " + " & ".join(cells) + " \\\\")
    write(out, "ablation_rows.tex", lines)

    keys = {"pi3x": "Pi3X", "pi3": "$\\pi^3$", "vggt": "VGGT", "cut3r": "CUT3R", "any4d": "Any4D", "da3": "DA3"}

    def cell3(Q, n):
        return [f"{median(Q, n):.3f}", f"{100 * median(Q, n, 'eps'):.2f}", neg(f"{median(Q, n, 'beta'):.2f}")]

    G = {s: {m: [c for c in P[s] if all(k in c["methods"] for k in ("lift_" + m, "branch4d_" + m, "ps4d_" + m))]
             for m in keys} for s in SETS}
    lines = ["Projective-scale branch (shared) & " + " & ".join(c for s in SETS for c in cell3(G[s]["pi3x"], "ps")) + " \\\\"]
    for m, lab in keys.items():
        if any(not G[s][m] for s in SETS):
            continue
        lines.append("\\midrule")
        for k, txt in (("lift_" + m, ", tracks lifted"), ("branch4d_" + m, ", 4D branch")):
            lines.append(lab + txt + " & " + " & ".join(c for s in SETS for c in cell3(G[s][m], k)) + " \\\\")
        lines.append("\\rowcolor{ourstint}\\method{} with " + lab + " & "
                     + " & ".join(c for s in SETS for c in cell3(G[s][m], "ps4d_" + m)) + " \\\\")
        d = []
        for s in SETS:
            (mm, lo, hi), _, _ = paired(G[s][m], "ps4d_" + m, "lift_" + m)
            d.append("\\multicolumn{3}{c}{" + f"${mm:+.3f}$ $[{lo:+.3f},{hi:+.3f}]$" + "}")
        lines.append("\\quad paired $\\Delta r$, \\method{} $-$ " + lab + " & " + " & ".join(d) + " \\\\")
    write(out, "x4d_rows.tex", lines)
    return P


# ------------------------------------------------------------------------------------------- complementarity table
FOUR = ["branch4d_pi3x", "branch4d_pi3", "branch4d_vggt", "branch4d_cut3r"]
ALT = [("ps", "projective scale (\\method{})"), ("branch4d_pi3", "$\\pi^3$, 4D branch"),
       ("branch4d_vggt", "VGGT, 4D branch"), ("branch4d_cut3r", "CUT3R, 4D branch"),
       ("avg4", "mean of all four 4D branches")]
DEV_OF = {"DEX": ["B12", "B35"], "Q3": ["H3"], "DROID": ["A"]}
GRID = np.round(np.arange(0.50, 3.0001, 0.01), 2)


def _centred(x):
    x = np.asarray(x, float)
    return x - x.mean()


def _r_beta(x, ref):
    x, ref = _centred(x), _centred(ref)
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref)), float(x @ ref / (ref @ ref))


def _series(R, k):
    if k == "avg4":
        return [None if any(R[f][i] is None for f in FOUR) else sum(R[f][i] for f in FOUR) / 4 for i in range(len(R["k"]))]
    return R[k]


def _pair_stats(c, k):
    """PS4D and the average of the 4D branch with series k on their common frames: (r_ps4d, r_alt, error corr)."""
    R = c["frames"]
    S = _series(R, k)
    idx = [i for i in range(len(R["k"])) if R["ref"][i] is not None and R["branch4d_pi3x"][i] is not None
           and R["ps"][i] is not None and S[i] is not None]
    if len(idx) < 5:
        return None
    ref = np.array([R["ref"][i] for i in idx])
    m4 = np.array([R["branch4d_pi3x"][i] for i in idx])
    a = np.array([R["ps"][i] for i in idx])
    x = np.array([S[i] for i in idx])
    r0, _ = _r_beta(0.5 * (m4 + a), ref)
    if k == "avg4":
        return r0, _r_beta(x, ref)[0], None
    r1, _ = _r_beta(0.5 * (m4 + x), ref)
    e_m4, e_x = _centred(m4) - _centred(ref), _centred(x) - _centred(ref)
    return r0, r1, float(e_m4 @ e_x / (np.linalg.norm(e_m4) * np.linalg.norm(e_x)))


def _r_gain(r, b, g):
    return np.sqrt(np.maximum(g * g * (r * r + 2 * b - 1) - 2 * g * b + 1, 0.0))


def _best_gain(P, n="lift_pi3x"):
    r = np.array([c["methods"][n]["r"] for c in P])
    b = np.array([c["methods"][n]["beta"] for c in P])
    return float(GRID[int(np.argmin([np.median(_r_gain(r, b, g)) for g in GRID]))])


def _fmt_d(t):
    m, lo, hi = t
    return f"\\mbox{{${m:.3f}$ {{\\scriptsize$[{lo:.2f},{hi:.2f}]$}}}}"


def compl_table(P, D, out):
    res = {s: {} for s in SETS}
    for s in SETS:
        for k, _ in ALT:
            Q = []
            for c in P[s]:
                if "frames" not in c:
                    continue
                v = _pair_stats(c, k)
                if v is not None:
                    Q.append({"sequence": c.get("sequence"), "r0": v[0], "r1": v[1], "c": v[2]})
            cc = np.array([q["c"] for q in Q if q["c"] is not None])
            res[s][k] = {"r": float(np.median([q["r1"] for q in Q])), "c": float(np.median(cc)) if len(cc) else None,
                         "d": None if k == "ps" else median_ci(Q, lambda q: q["r0"] - q["r1"])}
        dev = [c for d in DEV_OF[s] for c in complete(D[d], ["lift_pi3x"])]
        for lab, g in (("dev", _best_gain(dev)), ("test", _best_gain(P[s]))):
            rg = {c["id"]: float(_r_gain(c["methods"]["lift_pi3x"]["r"], c["methods"]["lift_pi3x"]["beta"], g)) for c in P[s]}
            res[s]["gain_" + lab] = {"g": g, "r": float(np.median(list(rg.values()))),
                                     "d": median_ci(P[s], lambda c: c["methods"]["ps4d_pi3x"]["r"] - rg[c["id"]])}
    lines = []
    for k, lab in ALT:
        cells = []
        for s in SETS:
            v = res[s][k]
            cells += ["--" if v["c"] is None else neg(f"{v['c']:.2f}"), f"{v['r']:.3f}",
                      "--" if v["d"] is None else _fmt_d(v["d"])]
        lines.append(("\\rowcolor{ourstint}" if k == "ps" else "") + lab + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\midrule", "\\emph{Pi3X with a global gain} & " + " & ".join(["$g$", "$r\\downarrow$", "$\\Delta r$"] * 3)
              + " \\\\", "\\midrule"]
    for lab, name in (("dev", "$g$ from development data"), ("test", "$g$ from the test set (bound)")):
        cells = []
        for s in SETS:
            v = res[s]["gain_" + lab]
            cells += [f"{v['g']:.2f}", f"{v['r']:.3f}", _fmt_d(v["d"])]
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
    write(out, "compl_rows.tex", lines)


def write(out, name, lines):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"== {name}")
    for l in lines:
        print(l)


def cells(line):
    """Row label and cells of a LaTeX table row, with citations, row colours and \\marg{...} wrappers removed."""
    line = re.sub(r"~?\\cite\{[^}]*\}", "", line)
    line = line.replace("\\rowcolor{ourstint}", "").replace("\\csname @@input\\endcsname", "")
    line = re.sub(r"\\marg\{(.*)\}(?=\s*\\\\)", r"\1", line)
    line = re.sub(r"\\marg\{([^{}]*(\{[^{}]*\})*[^{}]*)\}", r"\1", line)
    line = re.sub(r"\\pending\{[^}]*\}", "--", line)
    line = line.strip().rstrip("\\").strip()
    return [c.strip() for c in line.split("&")]


def check(out, ref_dir):
    bad = 0
    for name in ("main_rows.tex", "ablation_rows.tex", "x4d_rows.tex", "compl_rows.tex"):
        p = os.path.join(ref_dir, name)
        if not os.path.exists(p):
            print(f"{name}: no reference file")
            continue
        mine = [cells(l) for l in open(os.path.join(out, name), encoding="utf-8") if l.strip()]
        ref = [cells(l) for l in open(p, encoding="utf-8") if l.strip()]
        n_cells = 0
        if len(mine) != len(ref):
            print(f"{name}: {len(mine)} rows vs {len(ref)} in the reference")
            bad += 1
        for i, (a, b) in enumerate(zip(mine, ref)):
            if len(a) != len(b):
                print(f"{name} row {i}: {len(a)} cells vs {len(b)}")
                bad += 1
                continue
            for j, (x, y) in enumerate(zip(a, b)):
                n_cells += 1
                if j == 0 and i < len(ref):      # labels: compare without LaTeX spacing differences
                    x, y = x.replace(" ", ""), y.replace(" ", "")
                if x != y:
                    print(f"{name} row {i} cell {j}: {x!r} vs reference {y!r}")
                    bad += 1
        print(f"{name}: {len(mine)} rows, {n_cells} cells compared")
    print("all cells identical" if bad == 0 else f"{bad} differences")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--results", default=os.path.join(root, "results"))
    ap.add_argument("--out", default="tables_out")
    ap.add_argument("--check", default=None, help="directory with reference LaTeX row files")
    a = ap.parse_args()
    R = {s: read_results(os.path.join(a.results, f))["clips"] for s, f in TEST.items()}
    D = {s: read_results(os.path.join(a.results, f))["clips"] for s, f in DEV.items()}
    P = main_tables(R, a.out)
    compl_table(P, D, a.out)
    print({s: len(P[s]) for s in SETS}, "clips per test set")
    if a.check:
        sys.exit(1 if check(a.out, a.check) else 0)


if __name__ == "__main__":
    main()
