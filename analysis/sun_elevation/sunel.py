"""Sun elevation at the lunar touchdown point, per trial, for one fleet.

    APOLLO_LAUNCH_DELAY_H=<slot delay> venv/bin/python sunel.py <run_dir> <out.csv>

Touchdown time = pdi_get_s + 760 s (Apollo 11 PDI->touchdown 12.6 min; the sun moves ~0.5 deg/h, so the error is ~0.1 deg).
Site direction from land_lat_deg / land_lon_deg (IAU selenographic, as phase_descent reports them).
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repo root
import numpy as np, pandas as pd
import apollo11 as A

run, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"{run}/run_config.json"))
assert abs(cfg["_LAUNCH_DELAY_H"] - A._LAUNCH_DELAY_H) < 1e-9, (cfg["_LAUNCH_DELAY_H"], A._LAUNCH_DELAY_H)
df = pd.read_csv(f"{run}/results.csv", low_memory=False)
ok = df.land_lat_deg.notna() & df.pdi_get_s.notna()

def sun_el(t, lat, lon):
    xb, yb, zb = A._moon_fixed_axes_ofdate(t)
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    u = np.cos(la) * np.cos(lo) * xb + np.cos(la) * np.sin(lo) * yb + np.sin(la) * zb
    rm = np.asarray(A.moon_state(t)[0])
    s = np.asarray(A.sun_state(t))[:3] - rm
    return float(np.rad2deg(np.arcsin(np.dot(u, s / np.linalg.norm(s)))))

rows = []
for _, r in df[ok].iterrows():
    t = float(r.pdi_get_s) + 760.0
    rows.append((int(r.trial), t, r.land_lat_deg, r.land_lon_deg, sun_el(t, r.land_lat_deg, r.land_lon_deg),
                 bool(r.full_success) if pd.notna(r.full_success) else False, r.mission_failure))
o = pd.DataFrame(rows, columns=["trial", "t_td_s", "lat", "lon", "sun_el_deg", "full_success", "mission_failure"])
o.to_csv(out, index=False)
e = o.sun_el_deg
print(f"{run}: delay {A._LAUNCH_DELAY_H:g} h, {len(o)} touchdowns | sun el min {e.min():.2f} p1 {e.quantile(.01):.2f} "
      f"median {e.median():.2f} p99 {e.quantile(.99):.2f} max {e.max():.2f} | landing GET median {o.t_td_s.median()/3600:.2f} h "
      f"| site median {o.lat.median():.3f}, {o.lon.median():.3f}")
for lo_, hi_ in ((5, 14), (6.7, 16.8)):
    print(f"   outside [{lo_}, {hi_}] deg: {int(((e < lo_) | (e > hi_)).sum())}")
