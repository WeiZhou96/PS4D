"""Summary of a results file: median r / eps / beta per method and paired comparisons of PS4D with the others.

Clips: those on which every listed method has a value.  Paired differences: median of per-clip differences with a 95%
bootstrap interval (2000 resamples, seed 0; clustered by recording sequence when every clip has one, e.g. HOT3D;
--by-group clusters by group, e.g. DROID buildings).
Usage: python scripts/summarize.py results/test/droid.json.gz [--methods ...] [--by-group]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ps4d.io import read_results  # noqa: E402
from ps4d.metrics import complete, median, paired  # noqa: E402

DEFAULT = ["ps4d_pi3x", "lift_pi3x", "ps", "branch4d_pi3x", "spatracker2", "lift_vda", "lift_moge2", "joint_graph",
           "disagreement_weights"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--methods", nargs="+", default=DEFAULT, help="the first is compared with each of the others")
    ap.add_argument("--by-group", action="store_true", help="bootstrap clusters = groups (e.g. DROID buildings)")
    a = ap.parse_args()
    clips = read_results(a.results)["clips"]
    P = complete(clips, a.methods)
    print(f"{len(P)} of {len(clips)} clips have every method; {len({c['group'] for c in P})} groups")
    print(f"{'method':24s} {'r':>7s} {'eps x100':>9s} {'beta':>6s}")
    for m in a.methods:
        print(f"{m:24s} {median(P, m):7.3f} {100 * median(P, m, 'eps'):9.2f} {median(P, m, 'beta'):6.2f}")
    x = a.methods[0]
    by = [c["group"] for c in P] if a.by_group else None
    print(f"\npaired differences, {x} minus each method (r; eps x100; share of clips where {x} is better)")
    for y in a.methods[1:]:
        (m, lo, hi), (e, elo, ehi), win = paired(P, x, y, by)
        print(f"  - {y:22s} r {m:+.3f} [{lo:+.3f}, {hi:+.3f}]   eps {100 * e:+.2f} [{100 * elo:+.2f}, {100 * ehi:+.2f}]"
              f"   win {win:.2f}")


if __name__ == "__main__":
    main()
