#!/usr/bin/env python3
"""Find the integrated-heat-load ceiling at the shallow edge of the entry corridor.

The fleet's maximum convective load (23.96 kJ/cm2, trial 8689 at -5.914 deg) is an
ORDER STATISTIC of a very thin tail: only two captured entries in 8,208 lie shallower
than -6.0 deg, and those two are the maximum and the runner-up, while median load is
still climbing steeply at the last populated bin. Quoting it as a bound on what the
corridor admits would be wrong.

This sweeps entry flight-path angle across the capture boundary (the shallowest
captured entry is -5.912 deg; the shallowest skip-out is -5.859 deg), holding one
trial's other interface conditions fixed, so the load can be reported as a corridor
ceiling rather than as whatever the fleet happened to draw.

    python3 entry_heating_boundary.py [base_trial] [workers]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import entry_heating_validate as V

A = V.A
OUT = os.path.join(V.HERE, "entry_heating_boundary.json")
FPAS = [-6.20, -6.10, -6.00, -5.95, -5.92, -5.90, -5.88, -5.86, -5.84, -5.80, -5.75]


def _one(job):
    fpa, lat, lon, t, v, brg, tlat, tlon, perturb = job
    try:
        st, az, az_fb = V._ei_state(lat, lon, t, v, fpa, brg)
        res = A.phase_entry(st, t, perturb, target_lat_deg=tlat, target_lon_deg=tlon)
        out = dict(fpa=float(fpa), az_fallback=az_fb, max_g=float(res.get("max_g", np.nan)),
                   captured=bool(res.get("success", False)), reason=res.get("reason"))
        tt, yy = res.get("trajectory_t"), res.get("trajectory_y")
        if tt is not None and yy is not None and len(tt) > 3:
            tt = np.asarray(tt); yy = np.asarray(yy)
            out.update(V._heating(tt, yy))
            out["arc_s"] = float(tt[-1] - tt[0])
            out["min_alt_km"] = float((np.linalg.norm(yy[:3], axis=0) - A.R_EARTH).min() / 1000.0)
        return out
    except Exception as e:
        return dict(fpa=float(fpa), error=f"{type(e).__name__}: {e}")


def main():
    base = int(sys.argv[1]) if len(sys.argv) > 1 else 8689
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    d = pd.read_csv(os.path.join(V.OUTDIR, "results_with_survival.csv"), low_memory=False)
    r = d[d.trial == base].iloc[0]
    rng = np.random.default_rng(V.SEED)
    for _ in range(base + 1):
        perturb = A.sample_perturbation(rng)
    brg = V._bearing_deg(r.ei_lat, r.ei_lon, r.recovery_zone_lat, r.recovery_zone_lon)
    jobs = [(f, r.ei_lat, r.ei_lon, r.ei_t_s, r.entry_speed_ms, brg,
             r.recovery_zone_lat, r.recovery_zone_lon, perturb) for f in FPAS]
    print(f"boundary sweep off trial {base} (its own FPA {r.fpa_at_entry_deg:.4f}, "
          f"load 23.96): {len(jobs)} angles on {workers} workers", flush=True)
    from multiprocessing import Pool
    with Pool(workers) as p:
        rows = p.map(_one, jobs)
    json.dump(rows, open(OUT, "w"), indent=1, default=float)
    print(f"\n{'FPA':>8} {'captured':>9} {'max_g':>7} {'q_peak':>8} {'load':>7} {'arc_s':>8} {'min_alt':>8}  reason")
    for x in sorted(rows, key=lambda z: z["fpa"]):
        if "error" in x:
            print(f"{x['fpa']:>8.3f}  ERROR {x['error'][:60]}"); continue
        print(f"{x['fpa']:>8.3f} {str(x['captured']):>9} {x['max_g']:>7.3f} "
              f"{x.get('q_peak_W_cm2', float('nan')):>8.1f} {x.get('Q_kJ_cm2', float('nan')):>7.2f} "
              f"{x.get('arc_s', float('nan')):>8.1f} {x.get('min_alt_km', float('nan')):>8.1f}  {x.get('reason')}")
    capd = [x for x in rows if x.get("captured") and "Q_kJ_cm2" in x]
    if capd:
        top = max(capd, key=lambda z: z["Q_kJ_cm2"])
        print(f"\nCEILING: highest load among CAPTURED trajectories in the sweep = "
              f"{top['Q_kJ_cm2']:.2f} kJ/cm2 at FPA {top['fpa']:.3f} "
              f"({top['Q_kJ_cm2']/18.27:.3f}x the nominal's 18.27)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
