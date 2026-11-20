"""Compare an endurance replay (replay_endurance.py) with its fleet. Read-only.

Usage, from the repository root:
  python analysis/lifeboat_replay/replay_compare.py <replay_out_dir> <cfg> <fleet_dir> <out_csv> [--no-return-control] [--trials <dir>]
<cfg> is the file prefix (asflown, directabort); files <cfg>_c_*.jsonl flew on the cluster's compute
nodes, <cfg>_g_*.jsonl on its gpu nodes. With --no-return-control the fleet was not flown in this
configuration (directabort), so only the failure label and strike time are required to match the
fleet. --trials names the fleet's per-trial JSONs (the release asset), which keep the exact
floating-point values; without it the fleet's CSV is used and both sides are rounded to 16 digits.
Writes <out_csv>, one row per replayed trial, and prints the report.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

KEYS = ("noloi_return", "noloi_return_h", "noloi_dv_ms")


def nz(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return x


ROUND16 = False


def same(a, b):
    a, b = nz(a), nz(b)
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    if ROUND16:
        return float("%.16g" % float(a)) == float("%.16g" % float(b))
    return float(a) == float(b)


def load(rdir, cfg, typ):
    rows, err = {}, 0
    files = sorted(glob.glob(os.path.join(rdir, f"{cfg}_{typ}_*.jsonl")))
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if "error" in r:
                err += 1
                continue
            rows[(int(r["trial"]), float(r["hours"]))] = r
    return rows, err, files


def main():
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--trials"]
    rdir, cfg, fleet, out_csv = args[:4]
    no_ctl = "--no-return-control" in sys.argv
    global ROUND16
    ROUND16 = "--trials" not in sys.argv
    src = "results_with_survival.csv" if os.path.exists(os.path.join(fleet, "results_with_survival.csv")) else "results.csv"
    d = pd.read_csv(os.path.join(fleet, src), low_memory=False, float_precision="round_trip").sort_values("trial").reset_index(drop=True)
    tl = d[d.mission_failure.fillna("") == "sm_failure_translunar"].copy()
    # The fleet CSV keeps 16 significant digits; the per-trial JSONs keep the exact floats. Take the
    # exact values from the JSONs when a directory of them is given (--trials <dir>).
    tdir = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--trials"), None)
    if tdir:
        cols = ("sm_failure_get_h",) + KEYS
        nj = 0
        for i, t in zip(tl.index, tl.trial):
            fn = os.path.join(tdir, f"trial_{int(t)}.json")
            if os.path.exists(fn):
                j = json.load(open(fn))
                for k in cols:
                    if k in j:
                        tl.at[i, k] = j[k] if j[k] is not None else np.nan
                nj += 1
        print(f"exact fleet values from {nj} trial JSONs in {tdir}")
    else:
        print("fleet values from the CSV (16 significant digits); the replay is rounded to match")
    print(f"fleet {fleet} ({src}): {len(d)} trials; translunar service-module trials {len(tl)}; configuration {cfg}"
          + (" (not the fleet's own configuration: label and strike time are the only controls)" if no_ctl else ""))
    R = {}
    for typ, name in (("c", "compute"), ("g", "gpu")):
        rows, err, files = load(rdir, cfg, typ)
        R[typ] = rows
        hosts = sorted({r["host"].split(".")[0] for r in rows.values()})
        lim = sorted({(r["hours"], r["limit_in_worker"]) for r in rows.values()})
        hrs = sorted({h for _, h in rows})
        print(f"\n[{name} nodes] files {len(files)}; runs {len(rows)}; errors {err}; hosts {hosts}; (asked, in worker) limits {lim}"
              f"; replay time median {np.median([r['replay_time_s'] for r in rows.values()]):.0f} s"
              f", sum {sum(r['replay_time_s'] for r in rows.values()) / 3600:.1f} core-h" if rows else f"\n[{name} nodes] no runs")
        if not rows:
            continue
        assert all(a == b for a, b in lim), "a worker flew at a limit other than the one asked"
        base, alt = max(hrs), min(hrs)
        miss = [(int(t), h) for t in tl.trial for h in (base, alt) if (int(t), h) not in rows]
        print(f"  missing runs: {len(miss)} {miss[:8]}")
        lab = ok = 0
        bad = []
        for t in tl.trial:
            c = rows.get((int(t), base))
            if c is None:
                continue
            f = tl[tl.trial == t].iloc[0]
            l_ok = c.get("mission_failure") == f.mission_failure and same(c.get("sm_failure_get_h"), f.sm_failure_get_h)
            lab += l_ok
            r_ok = l_ok and all(same(c.get(k), f.get(k)) for k in KEYS)
            ok += r_ok
            if not (l_ok if no_ctl else r_ok):
                bad.append(int(t))
        n_c = sum((int(t), base) in rows for t in tl.trial)
        if bad and not no_ctl:  # size of the differences that remain: same return class?
            cls = dr = ddv = 0.0
            ncls = 0
            for t in bad:
                c = rows[(t, base)]
                f = tl[tl.trial == t].iloc[0]
                fr = f.noloi_return if isinstance(f.noloi_return, str) else None
                if nz(c.get("noloi_return")) == fr and c.get("mission_failure") == f.mission_failure:
                    ncls += 1
                    if nz(c.get("noloi_return_h")) is not None and not pd.isna(f.noloi_return_h):
                        dr = max(dr, abs(float(c["noloi_return_h"]) - float(f.noloi_return_h)))
                    if nz(c.get("noloi_dv_ms")) is not None and not pd.isna(f.noloi_dv_ms):
                        ddv = max(ddv, abs(float(c["noloi_dv_ms"]) - float(f.noloi_dv_ms)))
            print(f"  of the {len(bad)} that differ: same failure label and return class {ncls}; largest difference "
                  f"in return time {dr:.3g} h, in burn {ddv:.3g} m/s")
        print(f"  controls at {base} h: {n_c}; same failure label and strike time as the fleet: {lab}"
              + ("" if no_ctl else f"; return class, return time and burn equal to the fleet: {ok}") + f"; differ: {len(bad)} {bad[:12]}")
    # node types against each other
    common = sorted(set(R["c"]) & set(R["g"]))
    diff = [k for k in common if not all(same(R["c"][k].get(x), R["g"][k].get(x)) for x in KEYS + ("mission_failure", "sm_failure_get_h"))]
    print(f"\ncompute against gpu nodes: {len(common)} runs on both; identical label, strike, return class, return time and burn: "
          f"{len(common) - len(diff)}; differ: {len(diff)} {diff[:10]}")
    for k in diff[:10]:
        print("   ", k, {x: (R["c"][k].get(x), R["g"][k].get(x)) for x in KEYS})
    # per-trial table: take the compute run, fall back to gpu
    hrs = sorted({h for rows in R.values() for _, h in rows})
    if not hrs:
        return
    base, alt = max(hrs), min(hrs)
    out = []
    for _, f in tl.iterrows():
        t = int(f.trial)
        c = R["c"].get((t, base)) or R["g"].get((t, base))
        v = R["c"].get((t, alt)) or R["g"].get((t, alt))
        if c is None or v is None:
            continue
        g = lambda r, k: (nz(r.get(k)) if k != "noloi_return" else (nz(r.get(k)) or "none"))  # noqa: E731
        out.append(dict(trial=t, strike_get_h=f.sm_failure_get_h,
                        fleet_return=f.noloi_return if isinstance(f.noloi_return, str) else "none",
                        fleet_return_h=f.noloi_return_h, fleet_dv_ms=f.noloi_dv_ms, fleet_reason=f.get("noloi_reason"),
                        fleet_family=f.get("noloi_family"), fleet_crew_outcome=f.get("crew_outcome"),
                        fleet_crew_prob=f.get("crew_survival_prob"),
                        base_h=base, base_return=g(c, "noloi_return"), base_return_h=g(c, "noloi_return_h"),
                        base_dv_ms=g(c, "noloi_dv_ms"), base_reason=g(c, "noloi_reason"), base_family=g(c, "noloi_family"),
                        alt_h=alt, alt_return=g(v, "noloi_return"), alt_return_h=g(v, "noloi_return_h"),
                        alt_dv_ms=g(v, "noloi_dv_ms"), alt_reason=g(v, "noloi_reason"), alt_family=g(v, "noloi_family")))
    T = pd.DataFrame(out)
    T.to_csv(out_csv, index=False)
    print(f"\nper-trial table: {out_csv} ({len(T)} trials)")
    print(f"\nreturn class at {base} h in this configuration (rows) against {alt} h (columns):")
    print(pd.crosstab(T.base_return, T.alt_return).to_string())
    if no_ctl:
        print(f"\nfleet return class as flown (rows) against this configuration at {base} h (columns):")
        print(pd.crosstab(T.fleet_return, T.base_return).to_string())
        gained = T[(T.fleet_return == "none") & (T.base_return != "none")]
        print(f"returns this configuration adds at {base} h: {len(gained)}; strikes {gained.strike_get_h.min():.2f}-{gained.strike_get_h.max():.2f} h GET;"
              f" burn {gained.base_dv_ms.min():.1f}-{gained.base_dv_ms.max():.1f} m/s; family {gained.base_family.value_counts().to_dict()}")
        still = T[(T.base_return == "none")]
        print(f"no return at {base} h: {len(still)}; strikes {sorted(round(x, 2) for x in still.strike_get_h)}")
    lost = T[(T.base_return != "none") & (T.alt_return == "none")]
    gain = T[(T.base_return == "none") & (T.alt_return != "none")]
    sw = T[(T.base_return != "none") & (T.alt_return != "none") & (T.base_return != T.alt_return)]
    print(f"\nlose their return at {alt} h: {len(lost)}" + (f" | class at {base} h {lost.base_return.value_counts().to_dict()}"
          f" | family {lost.base_family.value_counts(dropna=False).to_dict()} | strikes {lost.strike_get_h.min():.2f}-{lost.strike_get_h.max():.2f} h GET"
          f" | fleet crew value {sorted(set(lost.fleet_crew_prob.dropna()))}" if len(lost) else ""))
    print("  trials:", sorted(lost.trial))
    print(f"gain a return at {alt} h: {len(gain)} {sorted(gain.trial)}")
    print(f"change class, still return: {len(sw)}" + (f" | {sw.base_return.value_counts().to_dict()} -> {sw.alt_return.value_counts().to_dict()}"
          f" | returns at {base} h {sw.base_return_h.min():.2f}-{sw.base_return_h.max():.2f} h | strikes {sw.strike_get_h.min():.2f}-{sw.strike_get_h.max():.2f} h GET" if len(sw) else ""))
    both = T[(T.base_return == "dps_abort") & (T.alt_return == "dps_abort")]
    chg = both[(both.base_return_h - both.alt_return_h).abs() > 1e-9]
    print(f"descent-engine abort at both limits: {len(both)} | earlier and dearer at {alt} h: {len(chg)}"
          + (f" (mean burn {chg.base_dv_ms.mean():.1f} -> {chg.alt_dv_ms.mean():.1f} m/s, largest {chg.alt_dv_ms.max():.1f})" if len(chg) else ""))
    if "noloi_return_h" in d:
        other = d[d.mission_failure.fillna("").str.startswith("sm_failure") & (d.mission_failure != "sm_failure_translunar")]
        late = other[other.noloi_return_h > alt]
        print(f"other service-module trials: {len(other)}; with a recorded return longer than {alt} h: {len(late)} {list(late.trial)[:10]}")


if __name__ == "__main__":
    main()
