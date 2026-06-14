"""Spawn-safe test for main_parallel's gap-safe resume and sorted output.

Run with:  python3 test_resume_gapfix.py
The `if __name__ == "__main__"` guard is REQUIRED: macOS multiprocessing uses
the 'spawn' start method, so each worker re-imports this module; the guard
prevents the test body from re-running (and fork-bombing) inside workers.
"""
import math
import shutil

import pandas as pd

import apollo11


def _close(a, b):
    # NaN-vs-NaN counts as equal; NaN-vs-value does not. Check this BEFORE
    # float()/isclose, because float('nan') succeeds but isclose(nan,nan) is
    # False — which would spuriously flag empty string/categorical cells.
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return a == b


def main():
    d = "outputs/_gaptest"
    shutil.rmtree(d, ignore_errors=True)

    # Full clean run of 4 trials.
    apollo11.main_parallel(n=4, outdir=d, seed=11, resume=False, workers=4)
    df = pd.read_csv(f"{d}/results.csv")
    print("after full run: trials =", df["trial"].tolist())
    ref_trial1 = df[df["trial"] == 1].iloc[0].to_dict()

    # Punch an interior hole: drop trial 1, leaving 0,2,3 (max=3).
    # The old max+1 resume logic would start at 4 and skip trial 1 forever.
    df[df["trial"] != 1].to_csv(f"{d}/results.csv", index=False)
    print("punched hole -> remaining trials =",
          sorted(pd.read_csv(f'{d}/results.csv')['trial'].tolist()))

    # Resume: must detect trial 1 missing and re-run ONLY it.
    apollo11.main_parallel(n=4, outdir=d, seed=11, resume=True, workers=4)
    df2 = pd.read_csv(f"{d}/results.csv")
    order = df2["trial"].tolist()
    print("after resume: row order =", order)

    got_trial1 = df2[df2["trial"] == 1].iloc[0].to_dict()
    mismatch = [k for k in ref_trial1 if not _close(ref_trial1[k], got_trial1.get(k))]

    all_present = sorted(order) == [0, 1, 2, 3]
    sorted_rows = order == sorted(order)
    refilled = not mismatch

    print("trial-1 refilled identically:", refilled, "| mismatched cols:", mismatch[:5])
    print("rows written sorted by trial:", sorted_rows)
    print("all 0..3 present, no dupes:", all_present)

    shutil.rmtree(d, ignore_errors=True)
    print("RESULT:", "PASS" if (refilled and all_present and sorted_rows) else "FAIL")


if __name__ == "__main__":
    main()
