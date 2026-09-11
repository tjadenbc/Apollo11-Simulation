#!/usr/bin/env python3
"""Fleet-wide Sutton-Graves entry heating for the definitive run, sharded for SLURM.

Re-flies EVERY trial's entry through the model's own ``phase_entry`` from an EI
state rebuilt out of the recorded interface conditions, and integrates
Sutton-Graves stagnation-point convective heating along each result. The physics
helpers are imported from ``entry_heating_validate`` so this is the same code
path that passed the 25-trial gate (all 25 reproduced their recorded max_g to
0.00%), not a reimplementation.

The point of the exercise: the paper's heatshield breach is a flat per-entry
probability independent of the trajectory, and Section 7.3 currently has to
concede that the integrated heat load -- the quantity the SHALLOW entries set --
is not bounded anywhere in the work. This bounds it.

Sharding is by stride (trial index % n_shards), matching cluster_run.py's
convention, so shards are disjoint and the union is the whole fleet.

    python3 entry_heating_fleet.py <shard> <n_shards> <workers>

Writes heat_shard_<shard>.json into $APOLLO_HEAT_OUTDIR (default outputs/final).
Merge with:  python3 entry_heating_fleet.py merge <n_shards>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import entry_heating_validate as V          # env + apollo11 + run_config flags, then helpers

A = V.A
OUTDIR = V.OUTDIR
_PERTURBS = None                            # per-process cache: 10,000 draws generated once


def _init(n_total):
    """Build the whole perturbation stream once per worker process.

    Regenerating per trial would be O(n^2) across the fleet; the stream is
    sequential, so one pass per process and index into it.
    """
    global _PERTURBS
    rng = np.random.default_rng(V.SEED)
    _PERTURBS = [A.sample_perturbation(rng) for _ in range(n_total)]


def _one(job):
    i, lat, lon, t, v, fpa, brg, tgt_lat, tgt_lon, rec_g = job
    try:
        st, az, az_fb = V._ei_state(lat, lon, t, v, fpa, brg)
        res = A.phase_entry(st, t, _PERTURBS[int(i)],
                            target_lat_deg=tgt_lat, target_lon_deg=tgt_lon)
        out = dict(trial=int(i), fpa=float(fpa), azimuth_deg=az, az_fallback=az_fb,
                   recorded_max_g=float(rec_g),
                   reflown_max_g=float(res.get("max_g", np.nan)),
                   success=bool(res.get("success", False)), reason=res.get("reason"))
        tt, yy = res.get("trajectory_t"), res.get("trajectory_y")
        if tt is not None and yy is not None and len(tt) > 3:
            out.update(V._heating(np.asarray(tt), np.asarray(yy)))
        return out
    except Exception as e:
        return dict(trial=int(i), error=f"{type(e).__name__}: {e}")


def _jobs():
    d = pd.read_csv(os.path.join(OUTDIR, "results_with_survival.csv"), low_memory=False)
    e = d[(d.reached_entry == True) & d.fpa_at_entry_deg.notna()              # noqa: E712
          & d.recovery_zone_lat.notna() & d.max_g.notna()]
    jobs = [(r.trial, r.ei_lat, r.ei_lon, r.ei_t_s, r.entry_speed_ms, r.fpa_at_entry_deg,
             V._bearing_deg(r.ei_lat, r.ei_lon, r.recovery_zone_lat, r.recovery_zone_lon),
             r.recovery_zone_lat, r.recovery_zone_lon, r.max_g) for r in e.itertuples()]
    return jobs, int(d.trial.max()) + 1


def merge(n_shards):
    rows, missing = [], []
    for k in range(n_shards):
        f = os.path.join(OUTDIR, f"heat_shard_{k}.json")
        if os.path.exists(f):
            rows += json.load(open(f))
        else:
            missing.append(k)
    rows.sort(key=lambda r: r["trial"])
    out = os.path.join(OUTDIR, "entry_heating_fleet.json")
    json.dump(rows, open(out, "w"), default=float)
    ok = [r for r in rows if "reflown_max_g" in r and np.isfinite(r["reflown_max_g"])]
    print(f"merged {len(rows)} rows from {n_shards - len(missing)}/{n_shards} shards -> {out}")
    if missing:
        print(f"  MISSING shards: {missing}")
    if ok:
        rel = np.array([r["reflown_max_g"] / r["recorded_max_g"] - 1
                        for r in ok if r["recorded_max_g"]])
        q = np.array([r["q_peak_W_cm2"] for r in ok if "q_peak_W_cm2" in r])
        Q = np.array([r["Q_kJ_cm2"] for r in ok if "Q_kJ_cm2" in r])
        print(f"  max_g agreement: median {100*np.median(rel):+.3f}%  worst {100*np.abs(rel).max():.3f}%")
        print(f"  peak flux  W/cm2 : min {q.min():.1f}  median {np.median(q):.1f}  max {q.max():.1f}")
        print(f"  heat load kJ/cm2 : min {Q.min():.2f}  median {np.median(Q):.2f}  max {Q.max():.2f}")


def main():
    if sys.argv[1] == "merge":
        return merge(int(sys.argv[2]))
    shard, n_shards, workers = (int(sys.argv[1]), int(sys.argv[2]),
                                int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    jobs, n_total = _jobs()
    mine = [j for j in jobs if int(j[0]) % n_shards == shard]
    print(f"shard {shard}/{n_shards}: {len(mine)} of {len(jobs)} entries on {workers} workers",
          flush=True)
    from multiprocessing import Pool
    with Pool(workers, initializer=_init, initargs=(n_total,)) as p:
        rows = p.map(_one, mine)
    f = os.path.join(OUTDIR, f"heat_shard_{shard}.json")
    json.dump(rows, open(f, "w"), default=float)
    bad = [r for r in rows if "error" in r]
    print(f"shard {shard}: wrote {len(rows)} rows ({len(bad)} errors) -> {f}", flush=True)
    for r in bad[:5]:
        print(f"   trial {r['trial']}: {r['error'][:100]}", flush=True)


if __name__ == "__main__":
    main()
