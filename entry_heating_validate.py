#!/usr/bin/env python3
"""Validation gate for a fleet-wide Sutton-Graves entry-heating analysis.

The definitive run records each trial's entry-interface conditions but not its
entry trajectory, so heating has to be recovered by re-flying ONLY the entry
through the model's own ``phase_entry`` from a reconstructed EI state. Two
assumptions have to hold for that to be legitimate, and this script tests both
before any cluster time is spent:

  1. the EI state can be rebuilt from what is recorded -- position from
     (ei_lat, ei_lon) at the 121.92 km interface, speed and flight-path angle
     from the recorded INERTIAL values, and the missing azimuth from the bearing
     to the trial's own recovery zone, which sits 2,784 km down its entry ground
     track (so the bearing IS the entry azimuth, by construction);
  2. a standalone ``phase_entry`` call reproduces what the full mission produced.

The gate on both is the recorded ``max_g``: if the re-flown value matches the
archived one across the corridor, the reconstruction is sound and the heating
computed along the same trajectory inherits that agreement. If it does not, the
fleet-wide run is not worth submitting.

Heating is Sutton-Graves stagnation-point CONVECTIVE flux,
q = k*sqrt(rho/Rn)*V_rel^3 with k = 1.7415e-4 (SI), evaluated on the model's own
_entry_density and Earth-relative velocity. The nose-radius factor is stored
separately (as the Rn-free quantity sqrt(rho)*V^3) so any Rn can be applied
afterwards without re-flying anything.

    python3 entry_heating_validate.py [n_trials] [n_workers]      # default 25 10

Writes entry_heating_validation.json next to the run directory.
"""
import json
import os
import sys

# Must precede the apollo11 import, and must run in every spawned worker too
# (spawn re-imports __main__), so these sit at module scope deliberately.
os.environ.setdefault("APOLLO_JPL_EPHEMERIS", "1")
os.environ.setdefault("APOLLO_SKIP_NOMINAL_CHECK", "1")

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("APOLLO_HEAT_OUTDIR", os.path.join(HERE, "outputs", "final"))
SEED = 37
EI_ALT_M = 121_920.0          # entry interface, apollo11.py:5940
SG_K = 1.7415e-4              # Sutton-Graves, SI (NASA TR R-376 air value)
RN_M = 4.694                  # Apollo CM heatshield radius of curvature

sys.path.insert(0, HERE)
import apollo11 as A                                    # noqa: E402

_cfg = json.load(open(os.path.join(OUTDIR, "run_config.json")))
for _k, _v in _cfg.items():
    if _k.startswith("ENABLE_") and hasattr(A, _k):
        setattr(A, _k, _v)


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from 1 to 2, degrees east of north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    return np.degrees(np.arctan2(np.sin(dl) * np.cos(p2),
                                 np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)))


def _ei_state(lat, lon, t, v_inert, fpa_inert_deg, bearing_deg):
    """Rebuild the ECI entry state. Speed/FPA are the recorded INERTIAL values;
    the azimuth is solved so the EARTH-RELATIVE ground track takes the given
    bearing, which is the frame the recovery-zone geometry lives in."""
    r = np.asarray(A.latlon_alt_to_eci(lat, lon, EI_ALT_M, t), dtype=float)
    up = r / np.linalg.norm(r)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    fpa = np.radians(fpa_inert_deg)
    omega = np.array([0.0, 0.0, A.OMEGA_E])

    def v_of(alpha):
        horiz = np.sin(alpha) * east + np.cos(alpha) * north
        return v_inert * (np.sin(fpa) * up + np.cos(fpa) * horiz)

    def resid(alpha):
        vr = v_of(alpha) - np.cross(omega, r)
        got = np.degrees(np.arctan2(np.dot(vr, east), np.dot(vr, north)))
        return (got - bearing_deg + 180.0) % 360.0 - 180.0

    lo, hi = np.radians(bearing_deg) - 1.2, np.radians(bearing_deg) + 1.2
    try:
        alpha = brentq(resid, lo, hi, xtol=1e-10)
    except ValueError:
        # No sign change in the bracket. Falling back to the ground bearing is an
        # inertial-vs-Earth-relative error of order 2 deg, and a row produced that way is
        # indistinguishable from a good one -- so record it rather than swallow it.
        alpha = np.radians(bearing_deg)
        return (np.concatenate([r, v_of(alpha), [A.CSM_CM_MASS]]),
                float(np.degrees(alpha)), True)
    return (np.concatenate([r, v_of(alpha), [A.CSM_CM_MASS]]),
            float(np.degrees(alpha)), False)


def _heating(tt, yy):
    """Sutton-Graves peak flux and integrated load along a returned entry arc."""
    r, v = yy[:3], yy[3:6]
    omega = np.array([0.0, 0.0, A.OMEGA_E])
    v_rel = v - np.cross(omega, r.T).T
    alt = np.linalg.norm(r, axis=0) - A.R_EARTH
    rho = np.array([A._entry_density(float(a)) for a in alt])
    vr = np.linalg.norm(v_rel, axis=0)
    base = np.sqrt(np.maximum(rho, 0.0)) * vr ** 3          # Rn-free part
    q = SG_K * base / np.sqrt(RN_M)                          # W/m^2
    return dict(q_peak_W_cm2=float(q.max()) / 1e4,
                Q_kJ_cm2=float(np.trapezoid(q, tt)) / 1e7,
                q_peak_alt_km=float(alt[int(np.argmax(q))]) / 1000.0,
                base_peak=float(base.max()), base_int=float(np.trapezoid(base, tt)))


def _one(job):
    i, lat, lon, t, v, fpa, brg, tgt_lat, tgt_lon, rec_g = job
    try:
        rng = np.random.default_rng(SEED)
        for _ in range(i + 1):
            perturb = A.sample_perturbation(rng)          # trial i's own dispersions
        st, az, az_fallback = _ei_state(lat, lon, t, v, fpa, brg)
        res = A.phase_entry(st, t, perturb, target_lat_deg=tgt_lat, target_lon_deg=tgt_lon)
        out = dict(trial=int(i), fpa=float(fpa), azimuth_deg=az, az_fallback=az_fallback,
                   recorded_max_g=float(rec_g), reflown_max_g=float(res.get("max_g", np.nan)),
                   success=bool(res.get("success", False)), reason=res.get("reason"))
        tt, yy = res.get("trajectory_t"), res.get("trajectory_y")
        if tt is not None and yy is not None and len(tt) > 3:
            out.update(_heating(np.asarray(tt), np.asarray(yy)))
        return out
    except Exception as e:                                  # never lose the whole pool to one trial
        return dict(trial=int(i), error=f"{type(e).__name__}: {e}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    d = pd.read_csv(os.path.join(OUTDIR, "results_with_survival.csv"), low_memory=False)
    e = d[(d.reached_entry == True) & d.fpa_at_entry_deg.notna()          # noqa: E712
          & d.recovery_zone_lat.notna() & d.max_g.notna()].sort_values("fpa_at_entry_deg")
    idx = np.unique(np.linspace(0, len(e) - 1, n).round().astype(int))    # span the corridor
    sel = e.iloc[idx]
    jobs = [(r.trial, r.ei_lat, r.ei_lon, r.ei_t_s, r.entry_speed_ms, r.fpa_at_entry_deg,
             _bearing_deg(r.ei_lat, r.ei_lon, r.recovery_zone_lat, r.recovery_zone_lon),
             r.recovery_zone_lat, r.recovery_zone_lon, r.max_g) for r in sel.itertuples()]
    print(f"re-flying {len(jobs)} entries on {workers} workers; "
          f"FPA {sel.fpa_at_entry_deg.min():.2f} to {sel.fpa_at_entry_deg.max():.2f} deg", flush=True)
    from multiprocessing import Pool
    with Pool(workers) as p:
        rows = p.map(_one, jobs)
    ok = [r for r in rows if "reflown_max_g" in r and np.isfinite(r["reflown_max_g"])]
    print(f"\n{'trial':>6} {'FPA':>7} {'rec g':>7} {'reflown':>8} {'err%':>7} "
          f"{'q_pk W/cm2':>11} {'Q kJ/cm2':>9}  reason")
    for r in sorted(rows, key=lambda x: x.get("fpa", 0)):
        if "error" in r:
            print(f"{r['trial']:>6}  ERROR {r['error'][:70]}"); continue
        err = 100 * (r["reflown_max_g"] / r["recorded_max_g"] - 1) if r["recorded_max_g"] else float("nan")
        print(f"{r['trial']:>6} {r['fpa']:>7.3f} {r['recorded_max_g']:>7.3f} {r['reflown_max_g']:>8.3f} "
              f"{err:>+7.2f} {r.get('q_peak_W_cm2', float('nan')):>11.1f} "
              f"{r.get('Q_kJ_cm2', float('nan')):>9.2f}  {r.get('reason')}")
    if ok:
        rel = np.array([r["reflown_max_g"] / r["recorded_max_g"] - 1 for r in ok])
        print(f"\nGATE: {len(ok)}/{len(rows)} re-flew. max_g agreement: median {100*np.median(rel):+.2f}%, "
              f"mean {100*rel.mean():+.2f}%, worst {100*np.abs(rel).max():.2f}%")
        print("  -> reconstruction is sound if the worst case is within a few percent and unbiased.")
    json.dump(rows, open(os.path.join(HERE, "entry_heating_validation.json"), "w"), indent=1, default=float)
    print(f"\nwrote entry_heating_validation.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
