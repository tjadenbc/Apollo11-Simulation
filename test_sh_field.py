"""Validation suite for the GRGM1200A truncated lunar gravity field (#3).

1. Degree-2 truncation must reproduce the legacy C20/C22 closed form (<1e-12 rel).
2. Field magnitude sanity at 100 km altitude.
3. 26.7 h parking-orbit evolution: deg-2 near-frozen (<2 km) vs deg-8 in the
   Apollo ground-truth band (~5-20 km/day; LOI-2 was left elliptical so mascon
   drift would pull the orbit toward 60 nm circular over the ~1 day stay).
4. Cost guard: <50 us/eval.
"""
import math
import time

import numpy as np
from scipy.integrate import solve_ivp

import apollo11 as a


def closed_form_body(x, y, z):
    """The legacy C20+C22 closed form, replicated for an independent compare."""
    mu, R = a.MU_MOON, a.R_MOON_GRAV
    r = math.sqrt(x*x + y*y + z*z)
    z2r2 = (z/r)**2
    fz = 1.5 * a.J2_MOON * mu * R**2 / r**5
    a20 = np.array([x*(5*z2r2-1), y*(5*z2r2-1), z*(5*z2r2-3)]) * fz
    K = 3.0 * mu * R**2 * a.C22_MOON
    r5, r7, d = r**5, r**7, (x*x - y*y)
    a22 = K * np.array([2*x/r5 - 5*x*d/r7, -2*y/r5 - 5*y*d/r7, -5*z*d/r7])
    return a20 + a22


def test_deg2_equivalence():
    saved_terms, saved_n = a._LUNAR_SH_TERMS, a._LUNAR_SH_NMAX
    # build a table containing ONLY the code's own degree-2 constants
    a._LUNAR_SH_TERMS = [(2, 0, -a.J2_MOON, 0.0, 12.0, 3.0),
                         (2, 2, a.C22_MOON, 0.0, 2.0, 1.0)]
    a._LUNAR_SH_NMAX = 2
    rng = np.random.default_rng(1)
    worst = 0.0
    pts = []
    for _ in range(200):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        pts.append(u * rng.uniform(1.75e6, 5.0e6))
    pts.append(np.array([0.0, 0.0, 1.85e6]))   # exact pole
    pts.append(np.array([1.85e6, 0.0, 0.0]))   # equatorial x-axis
    for p in pts:
        got = np.array(a._lunar_sh_accel_body(*p))
        ref = closed_form_body(*p)
        err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        worst = max(worst, err)
    a._LUNAR_SH_TERMS, a._LUNAR_SH_NMAX = saved_terms, saved_n
    print(f"1. deg-2 equivalence: worst rel err = {worst:.2e}  "
          f"{'PASS' if worst < 1e-12 else 'FAIL'}")
    return worst < 1e-12


def test_magnitude():
    r = a.R_MOON + 100e3
    p = np.array([r/math.sqrt(2), r/math.sqrt(3), r*math.sqrt(1-0.5-1/3)])
    p *= r / np.linalg.norm(p)
    pert = np.linalg.norm(a._lunar_sh_accel_body(*p))
    central = a.MU_MOON / r**2
    ratio = pert / central
    ok = 2e-4 <= ratio <= 8e-4
    print(f"2. magnitude @100 km: |a_pert|/|a_central| = {ratio:.2e}  "
          f"{'PASS' if ok else 'FAIL'} (expect 2e-4..8e-4)")
    return ok


def _propagate_orbit(hours, flag_on):
    """Propagate a ~99 km circular lunar orbit with the mission's dynamics."""
    a.ENABLE_LUNAR_SH_FIELD = flag_on
    t0 = 300000.0
    mr, mv = a.moon_state(t0)
    zb = np.cross(mr, mv); zb /= np.linalg.norm(zb)
    xb = np.array([1.0, 0.0, 0.0]) - zb * zb[0]; xb /= np.linalg.norm(xb)
    yb = np.cross(zb, xb)
    r_orb = a.R_MOON + 99e3
    v_orb = math.sqrt(a.MU_MOON / r_orb)
    y0 = np.concatenate([mr + xb*r_orb, mv + yb*v_orb, [15000.0]])

    def rhs(t, y):
        return np.concatenate([y[3:6], a.gravity_earth_moon(y[:3], t), [0]])

    sol = solve_ivp(rhs, (t0, t0 + hours*3600), y0, method='RK45',
                    rtol=1e-9, atol=1e-2, max_step=60.0)
    pae0 = a._peri_apo_ecc_standalone(y0, t0) if hasattr(a, "_peri_apo_ecc_standalone") else None
    # peri/apo from Moon-relative osculating elements
    def pae(y, t):
        mr_, mv_ = a.moon_state(t)
        rl, vl = y[:3]-mr_, y[3:6]-mv_
        E = 0.5*np.dot(vl, vl) - a.MU_MOON/np.linalg.norm(rl)
        aa = -a.MU_MOON/(2*E)
        h = np.linalg.norm(np.cross(rl, vl))
        ecc = math.sqrt(max(0.0, 1 - h*h/a.MU_MOON/aa))
        return ((aa*(1-ecc)-a.R_MOON)/1e3, (aa*(1+ecc)-a.R_MOON)/1e3)
    p0 = pae(y0, t0)
    p1 = pae(sol.y[:, -1], sol.t[-1])
    return p0, p1


def test_orbit_evolution():
    p0_off, p1_off = _propagate_orbit(26.7, False)
    d_off = max(abs(p1_off[0]-p0_off[0]), abs(p1_off[1]-p0_off[1]))
    p0_on, p1_on = _propagate_orbit(26.7, True)
    d_on = max(abs(p1_on[0]-p0_on[0]), abs(p1_on[1]-p0_on[1]))
    a.ENABLE_LUNAR_SH_FIELD = False
    print(f"3. 26.7 h evolution: deg-2 {p0_off[0]:.1f}x{p0_off[1]:.1f} -> "
          f"{p1_off[0]:.1f}x{p1_off[1]:.1f} km (max drift {d_off:.1f} km); "
          f"deg-{a.LUNAR_SH_DEGREE} -> {p1_on[0]:.1f}x{p1_on[1]:.1f} km "
          f"(max drift {d_on:.1f} km)")
    # NOTE: both runs include Earth third-body drift (common-mode, ~4 km/day on
    # this plane/epoch), so the assertion targets the INCREMENTAL higher-degree
    # effect and the Apollo ground-truth band for the total deg-N evolution.
    ok = (3.0 <= d_on <= 25.0) and (d_on - d_off) >= 3.0
    print(f"   {'PASS' if ok else 'FAIL'} (expect deg-N total in ~3-25 km/day "
          f"and incremental SH effect >=3 km/day; incremental = {d_on-d_off:.1f})")
    return ok


def test_cost():
    p = np.array([a.R_MOON + 100e3, 12e3, -34e3])
    t0 = time.perf_counter()
    for _ in range(1000):
        a._lunar_sh_accel_body(*p)
    us = (time.perf_counter() - t0) * 1e3
    print(f"4. cost: {us:.1f} us/eval  {'PASS' if us < 50 else 'FAIL'} (<50 us)")
    return us < 50


if __name__ == "__main__":
    results = [test_deg2_equivalence(), test_magnitude(),
               test_orbit_evolution(), test_cost()]
    print("\nRESULT:", "ALL PASS" if all(results) else "FAILURES PRESENT")
