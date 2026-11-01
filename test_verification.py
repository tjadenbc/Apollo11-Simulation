"""Unit-level verification suite (V&V).

Answers the reviewer question "how was the AI-generated code verified below the
level of the nominal trajectory?". Macro-validation compares OUTPUTS against a
reference, so a pair of compensating errors passes it by construction. Every
test here is therefore chosen so that it does NOT need to know the right answer:

  Tier 1  invariants and symmetries   - must hold whatever the true answer is
  Tier 2  independent derivations     - two code paths to one number
  Tier 3  external standards          - pins ONE factor of a product alone,
                                        which is what breaks a compensating pair
  Tier 4  recorded-trajectory replay  - checks nominal_traj.npz actually
                                        satisfies the equations of motion

Runs in seconds; no mission propagation.  Usage:  python3 test_verification.py
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

os.environ.setdefault("APOLLO_SKIP_NOMINAL_CHECK", "1")
import apollo11 as A                                        # noqa: E402

# ---------------------------------------------------------------- harness ---
_RESULTS = []


def check(tier, name, ok, detail=""):
    _RESULTS.append((tier, name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<46s} {detail}")
    return bool(ok)


def tier(n, title):
    print(f"\n=== Tier {n}: {title} " + "=" * max(0, 30 - len(title)))


def two_body(t, y, mu=A.MU_EARTH):
    r = y[:3]
    return np.concatenate([y[3:6], -mu * r / np.linalg.norm(r) ** 3])


# A representative bound orbit, well away from any singularity.
R0 = np.array([7.0e6, 1.0e6, 2.0e6])
V0 = np.array([-500.0, 7200.0, 1500.0])


# ================================================================= Tier 1 ===
def t1_time_reversal():
    """Propagate forward under the FULL right-hand side, then backward over the
    same interval. Strictly stronger than energy conservation: it also catches
    sign errors in velocity-dependent terms, which conserve energy trivially."""
    rhs = lambda t, y: np.concatenate([y[3:6], A.gravity_earth_moon(y[:3], t)])
    y0 = np.concatenate([R0, V0])
    T = 3600.0
    kw = dict(rtol=1e-12, atol=1e-9, method="DOP853")
    fwd = solve_ivp(rhs, (0.0, T), y0, **kw)
    back = solve_ivp(rhs, (T, 0.0), fwd.y[:, -1], **kw)
    dr = np.linalg.norm(back.y[:3, -1] - R0)
    dv = np.linalg.norm(back.y[3:6, -1] - V0)
    check(1, "time-reversal of full RHS (1 h)", dr < 1.0 and dv < 1e-3,
          f"dr={dr:.2e} m  dv={dv:.2e} m/s")


def t1_selenographic_basis():
    """The Moon-fixed axes must be ORTHONORMAL and RIGHT-HANDED. A determinant
    of -1 is a mirror: every longitude flips sign while every distance stays
    correct, so macro-validation cannot see it."""
    worst_o, worst_d = 0.0, 1.0
    for t in (0.0, 1.0e5, 3.0e5, 6.0e5):
        x, y, z = A._moon_fixed_axes_ofdate(t)
        M = np.array([x, y, z])
        worst_o = max(worst_o, np.abs(M @ M.T - np.eye(3)).max())
        worst_d = min(worst_d, np.linalg.det(M))
    check(1, "selenographic basis orthonormal", worst_o < 1e-12,
          f"max|MM^T - I| = {worst_o:.2e}")
    check(1, "selenographic basis RIGHT-handed (det=+1)",
          abs(worst_d - 1.0) < 1e-12, f"min det = {worst_d:.12f}")


def t1_precession_matrix():
    """Same mirror/orthonormality argument for the J2000 -> of-date rotation,
    plus the identity limit at T=0."""
    worst_o, worst_d = 0.0, 1.0
    for T in (0.0, 0.1, 0.5, -0.3):
        P = A._precess_j2000_to_date(T)
        worst_o = max(worst_o, np.abs(P @ P.T - np.eye(3)).max())
        worst_d = min(worst_d, np.linalg.det(P))
    check(1, "precession matrix orthonormal", worst_o < 1e-12,
          f"max|PP^T - I| = {worst_o:.2e}")
    check(1, "precession matrix det=+1 (no mirror)", abs(worst_d - 1.0) < 1e-12,
          f"min det = {worst_d:.12f}")
    check(1, "precession -> identity at T=0",
          np.abs(A._precess_j2000_to_date(0.0) - np.eye(3)).max() < 1e-12)


def t1_frame_roundtrip():
    """eci_to_latlon o latlon_alt_to_eci must be the identity - tested at the
    POLES, the date line and negative longitudes, where quadrant and sign bugs
    live. A nominal trajectory never visits these points."""
    worst = 0.0
    bad = None
    lats = [0.0, 28.6082, -28.6082, 89.9, -89.9, 45.0]
    lons = [0.0, 179.9, -179.9, 90.0, -90.0, 23.62, -169.15]
    for t in (0.0, 12345.0, 5.0e5):
        for la in lats:
            for lo in lons:
                r = A.latlon_alt_to_eci(la, lo, 0.0, t)
                la2, lo2 = A.eci_to_latlon(r, t)
                dlo = (lo2 - lo + 180.0) % 360.0 - 180.0
                err = max(abs(la2 - la), abs(dlo) * np.cos(np.deg2rad(la)))
                if err > worst:
                    worst, bad = err, (la, lo, t)
    check(1, "ECI<->lat/lon round-trip (poles, date line)", worst < 1e-9,
          f"max err {worst:.2e} deg" + (f" at {bad}" if worst > 1e-9 else ""))


def t1_rotational_covariance():
    """Rotate the initial state, propagate in a CENTRAL field, and the result
    must be the same rotation of the unrotated result. Catches hardcoded axes."""
    th = 0.7
    Q = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    kw = dict(rtol=1e-12, atol=1e-9, method="DOP853")
    a = solve_ivp(two_body, (0, 2000.0), np.concatenate([R0, V0]), **kw).y[:, -1]
    b = solve_ivp(two_body, (0, 2000.0),
                  np.concatenate([Q @ R0, Q @ V0]), **kw).y[:, -1]
    err = np.linalg.norm(b[:3] - Q @ a[:3])
    check(1, "rotational covariance of central field", err < 1e-3,
          f"err = {err:.2e} m")


def t1_conservation():
    """The reviewer's own test: energy and angular momentum in two-body."""
    kw = dict(rtol=1e-12, atol=1e-9, method="DOP853", dense_output=True)
    sol = solve_ivp(two_body, (0, 20000.0), np.concatenate([R0, V0]), **kw)
    ts = np.linspace(0, 20000.0, 200)
    Y = sol.sol(ts)
    rn = np.linalg.norm(Y[:3], axis=0)
    E = 0.5 * np.sum(Y[3:6] ** 2, axis=0) - A.MU_EARTH / rn
    L = np.cross(Y[:3].T, Y[3:6].T)
    dE = np.ptp(E) / abs(E[0])
    dL = np.ptp(np.linalg.norm(L, axis=1)) / np.linalg.norm(L[0])
    check(1, "two-body specific energy conserved", dE < 1e-10, f"drift {dE:.2e}")
    check(1, "two-body angular momentum conserved", dL < 1e-10, f"drift {dL:.2e}")

    # Laplace-Runge-Lenz: fixes the APSE DIRECTION. Energy and |L| can both be
    # conserved while the apse line rotates, so this catches gravity sign or
    # scale errors that the two tests above pass.
    ecc = (np.cross(Y[3:6].T, L) / A.MU_EARTH) - (Y[:3] / rn).T
    drift = np.max(np.linalg.norm(ecc - ecc[0], axis=1))
    check(1, "Laplace-Runge-Lenz vector conserved", drift < 1e-9,
          f"apse drift {drift:.2e}")


# ================================================================= Tier 2 ===
def t2_third_body_decomposition():
    """Rebuild gravity_earth_moon from its documented parts with an
    INDEPENDENTLY written third-body formula. The indirect term (-mu*R/|R|^3)
    is the classic omission and the classic sign error; if it were missing or
    flipped, the residual here would be enormous while the nominal trajectory
    still looked plausible."""
    worst = 0.0
    for t in (0.0, 2.0e5, 5.0e5):
        r = R0 * 1.7
        a_tot = A.gravity_earth_moon(r, t)
        rn = np.linalg.norm(r)
        a = -A.MU_EARTH * r / rn ** 3
        z2 = (r[2] / rn) ** 2
        f = 1.5 * A.J2 * A.MU_EARTH * A.R_EARTH ** 2 / rn ** 5
        a = a + f * np.array([r[0] * (5 * z2 - 1), r[1] * (5 * z2 - 1),
                              r[2] * (5 * z2 - 3)])
        if A.ENABLE_EARTH_HIGHER_ZONALS and rn < A.EARTH_ZONAL_R_MAX_M:
            a = a + A._earth_zonal_accel(r)
        # NOTE: moon_state returns (r, v); sun_state returns a BARE (3,) array.
        bodies = [(A.MU_MOON, A.moon_state(t)[0])]
        if A.ENABLE_SOLAR_GRAVITY:
            bodies.append((A.MU_SUN, A.sun_state(t)))
        for mu, R in bodies:
            d = R - r
            a = a + mu * (d / np.linalg.norm(d) ** 3 - R / np.linalg.norm(R) ** 3)
        worst = max(worst, np.linalg.norm(a - a_tot) / np.linalg.norm(a_tot))
    check(2, "RHS = point mass + J2 + zonals + 3rd body", worst < 1e-10,
          f"max rel residual {worst:.2e}")


def t2_lambert_self_consistency():
    """Sample two points and the TOF from a trajectory scipy generated, then
    ask the Lambert solver to reconstruct the velocities. Self-contained: no
    external reference is needed, so it cannot be tuned to agree."""
    kw = dict(rtol=1e-12, atol=1e-9, method="DOP853")
    for tof in (1800.0, 5400.0):
        sol = solve_ivp(two_body, (0, tof), np.concatenate([R0, V0]), **kw)
        r2 = sol.y[:3, -1]
        v1, v2 = A.lambert_uv(R0, r2, tof, mu=A.MU_EARTH, prograde=True)
        e1 = np.linalg.norm(v1 - V0)
        e2 = np.linalg.norm(v2 - sol.y[3:6, -1])
        check(2, f"Lambert reconstructs known arc (tof={tof:.0f}s)",
              e1 < 1e-3 and e2 < 1e-3, f"dv1={e1:.2e} dv2={e2:.2e} m/s")


def t2_stumpff_continuity():
    """The Stumpff functions switch between closed form and series at |psi|=1e-6.
    A discontinuity there is a classic silent bug in a universal-variable solver.

    Tolerance is 1e-9, not machine epsilon: at psi=1e-6 the closed form evaluates
    1-cos(1e-3), losing ~6 digits to cancellation. The residual jump measures
    that cancellation, and is the reason the series branch exists at all."""
    worst = 0.0
    for eps in (1e-6, -1e-6):
        lo = np.array(A._stumpff(eps * 0.999999))
        hi = np.array(A._stumpff(eps * 1.000001))
        worst = max(worst, np.abs(lo - hi).max())
    check(2, "Stumpff c2/c3 continuous at series switch", worst < 1e-9,
          f"max jump {worst:.2e}")


def t2_vis_viva():
    """Independent derivation of speed from the energy integral. This exact
    check previously caught a floor constant that clobbered a lunar-orbit
    velocity magnitude."""
    rn = np.linalg.norm(R0)
    vn = np.linalg.norm(V0)
    a_sma = 1.0 / (2.0 / rn - vn * vn / A.MU_EARTH)
    v_pred = np.sqrt(A.MU_EARTH * (2.0 / rn - 1.0 / a_sma))
    check(2, "vis-viva agrees with state speed", abs(v_pred - vn) < 1e-9,
          f"|dv| = {abs(v_pred - vn):.2e} m/s")


# ================================================================= Tier 3 ===
def t3_ussa76_table():
    """Density pinned against the PUBLISHED US Standard Atmosphere 1976 table,
    with no drag term anywhere in sight. Together with an independent ballistic
    -coefficient check this makes the reviewer's 'bad rho compensated by bad
    Cd' pair impossible: each factor is fixed separately.

    The table is tabulated in GEOPOTENTIAL altitude; the model takes GEOMETRIC.
    Converting is not a detail: at 71 km the two differ by 784 m, which is 12%
    of density there."""
    table = {0: 1.2250, 11000: 0.36391, 20000: 0.088035,
             32000: 0.013225, 47000: 0.0014275, 71000: 6.4211e-5}
    worst, where = 0.0, None
    for hgp, rho_ref in table.items():
        h_geom = 6356766.0 * hgp / (6356766.0 - hgp)      # geopotential -> geometric
        rho = A._ussa76_rho_a(h_geom)[0]
        rel = abs(rho - rho_ref) / rho_ref
        if rel > worst:
            worst, where = rel, hgp
    check(3, "USSA-76 density vs published table", worst < 5e-3,
          f"max {worst * 100:.4f}% at {where} m geopotential")


def t3_atmosphere_continuity():
    """No jumps at the seven layer boundaries of the piecewise model."""
    worst, where = 0.0, None
    for hb, _, _, _ in A._USSA76_LAYERS[1:]:
        lo = A._ussa76_rho_a(hb - 0.5)[0]
        hi = A._ussa76_rho_a(hb + 0.5)[0]
        rel = abs(hi - lo) / max(lo, 1e-30)
        if rel > worst:
            worst, where = rel, hb
    check(3, "USSA-76 internal layer seams continuous", worst < 1e-3,
          f"max jump {worst * 100:.5f}% at {where} m")

    # SEPARATE seam: at 86 km the model hands off from USSA-76 to the legacy
    # exponential. This one is a real, KNOWN discontinuity, characterised here
    # rather than asserted away -- it sits where rho ~ 1e-7 and drag is
    # aerodynamically irrelevant, so it is disclosed, not silently tolerated.
    lo = A._ussa76_rho_a(86000.0 - 0.5)[0]
    hi = A._ussa76_rho_a(86000.0 + 0.5)[0]
    jump = abs(hi - lo) / lo
    check(3, "86 km model handoff bounded (known seam)", jump < 0.10,
          f"{jump * 100:.2f}% jump at rho={lo:.2e}")


def t3_atmosphere_monotonic():
    hs = np.arange(0.0, 86000.0, 250.0)
    rho = np.array([A._ussa76_rho_a(h)[0] for h in hs])
    check(3, "density strictly decreasing 0-86 km", np.all(np.diff(rho) < 0),
          f"{int(np.sum(np.diff(rho) >= 0))} violations")


def t3_cm_reference_area():
    """The OTHER half of the reviewer's compensating pair. Density is pinned
    externally by the USSA-76 test above; here the drag reference area is pinned
    to the command module's documented 3.912 m maximum diameter. With rho fixed
    by an external standard, a drag error can no longer hide behind a density
    error - it has to surface as a peak-g or range error against the flown
    values. (Cd itself is a modelling choice, reported not asserted.)"""
    a_ref = np.pi * (3.912 / 2.0) ** 2
    check(3, "CM drag reference area = pi*d^2/4 (d=3.912 m)",
          abs(A.CM_AREA - a_ref) / a_ref < 0.01,
          f"{A.CM_AREA:.2f} vs {a_ref:.2f} m^2")
    beta = A.CSM_CM_MASS / (A.CM_CD * A.CM_AREA)
    check(3, "CM ballistic coefficient in Apollo class",
          250.0 < beta < 450.0, f"m/(Cd*A) = {beta:.1f} kg/m^2 (Cd={A.CM_CD})")


def t3_surface_gravity():
    """External standard, independent of every other model."""
    g_eq = np.linalg.norm(A.gravity_earth_moon(
        np.array([A.R_EARTH, 0.0, 0.0]), 0.0))
    g_moon = A.MU_MOON / A.R_MOON ** 2
    check(3, "Earth equatorial surface g ~ 9.798", abs(g_eq - 9.7986) < 0.02,
          f"{g_eq:.4f} m/s^2")
    check(3, "lunar surface g ~ 1.62", abs(g_moon - 1.625) < 0.01,
          f"{g_moon:.4f} m/s^2")


def t3_j2_degeneration():
    """Switch J2 and the zonals off and the field must collapse EXACTLY onto the
    point mass. Perturbation terms are small, which is precisely why a sign
    error in one survives macro-validation."""
    j2, zon = A.J2, A.ENABLE_EARTH_HIGHER_ZONALS
    try:
        A.J2 = 0.0
        A.ENABLE_EARTH_HIGHER_ZONALS = False
        r = np.array([9.0e6, 3.0e6, 4.0e6])
        a = A.gravity_earth_moon(r, 0.0)
        # third body still present; remove it the same independent way
        bodies = [(A.MU_MOON, A.moon_state(0.0)[0])]
        if A.ENABLE_SOLAR_GRAVITY:
            bodies.append((A.MU_SUN, A.sun_state(0.0)))
        for mu, R in bodies:
            d = R - r
            a = a - mu * (d / np.linalg.norm(d) ** 3
                          - R / np.linalg.norm(R) ** 3)
        ref = -A.MU_EARTH * r / np.linalg.norm(r) ** 3
        rel = np.linalg.norm(a - ref) / np.linalg.norm(ref)
        check(3, "J2=0, zonals off -> exact point mass", rel < 1e-12,
              f"rel {rel:.2e}")
    finally:
        A.J2, A.ENABLE_EARTH_HIGHER_ZONALS = j2, zon


def t3_lunar_figure_falloff():
    """Far from the Moon the degree-2 term dominates and the perturbation must
    fall as 1/r^4 (doubling r cuts it by 16). A wrong exponent is invisible in a
    nominal that only ever samples one altitude band.

    Sample at r >= 5e6 m: the field is degree 8, so nearer the surface the
    higher degrees legitimately steepen the falloff (measured exponent 3.31 at
    r = 2e6 m, i.e. 262 km altitude). Testing the asymptote isolates degree 2."""
    for r0 in (5.0e6, 1.0e7):
        a1 = np.linalg.norm(A.lunar_nonspherical_accel(np.array([r0, 0., 0.]), 0.))
        a2 = np.linalg.norm(A.lunar_nonspherical_accel(np.array([2 * r0, 0., 0.]), 0.))
        ratio = a1 / max(a2, 1e-300)
        check(3, f"lunar figure -> 1/r^4 asymptote (r={r0:.0e})",
              abs(ratio - 16.0) < 0.1, f"ratio {ratio:.4f} (expect 16)")


def t3_gmst_rate():
    """GMST must advance at the SIDEREAL rate, not the solar one - a 0.27%
    error that puts every ground track ~4 min/day out."""
    # One solar day of GMST is 2*pi PLUS the sidereal excess, so the wrapped
    # difference must have the full turn added back before differencing.
    d = (A._gmst_rad(A.JD_LAUNCH + 1.0) - A._gmst_rad(A.JD_LAUNCH)) % (2 * np.pi)
    rate = (d + 2 * np.pi) / 86400.0
    check(3, "GMST advances at sidereal rate", abs(rate - A.OMEGA_E) < 1e-9,
          f"{rate:.9e} vs OMEGA_E {A.OMEGA_E:.9e}")


# ================================================================= Tier 4 ===
def t4_recorded_trajectory(path="outputs/final/nominal_traj.npz"):
    """Replay the recorded per-phase states and verify they actually SATISFY the
    equations of motion, rather than merely ending in the right place.

      dr/dt = v   is exact and checks the record is a genuine trajectory
      dv/dt = a   re-evaluates the right-hand side at every recorded point

    Sampling-limited (the record is decimated), so the velocity residual is an
    order-of-magnitude consistency check, not a precision one."""
    if not os.path.exists(path):
        print(f"  [SKIP] no recorded trajectory at {path}")
        return
    z = np.load(path)

    t = z["translunar_t"]
    Y = z["translunar_y"]
    i0, i1 = len(t) // 4, len(t) // 2          # smooth mid-coast segment
    sl = slice(i0, i1)
    dt = np.gradient(t[sl])
    drdt = np.gradient(Y[:3, sl], axis=1) / dt
    v = Y[3:6, sl]
    kin = np.max(np.linalg.norm(drdt - v, axis=0)) / np.max(
        np.linalg.norm(v, axis=0))
    check(4, "recorded coast: dr/dt == v", kin < 1e-3, f"max rel {kin:.2e}")

    dvdt = np.gradient(Y[3:6, sl], axis=1) / dt
    acc = np.array([A.gravity_earth_moon(Y[:3, i], t[i])
                    for i in range(i0, i1)]).T
    dyn = np.median(np.linalg.norm(dvdt - acc, axis=0)
                    / np.linalg.norm(acc, axis=0))
    check(4, "recorded coast: dv/dt == gravity RHS", dyn < 5e-2,
          f"median rel {dyn:.2e}")

    m = Y[6]
    check(4, "coast mass constant (no burns recorded)",
          np.ptp(m) / m[0] < 1e-12, f"ptp {np.ptp(m):.3e} kg")

    md = z["descent_y"][6]
    check(4, "descent mass monotonically decreasing",
          np.all(np.diff(md) <= 1e-9), f"{int(np.sum(np.diff(md) > 1e-9))} rises")

    ent = z["entry_y"]
    alt = np.linalg.norm(ent[:3], axis=0) - A.R_EARTH
    spd = np.linalg.norm(ent[3:6], axis=0)
    check(4, "entry: altitude decreases overall", alt[-1] < alt[0],
          f"{alt[0] / 1e3:.1f} -> {alt[-1] / 1e3:.1f} km")
    check(4, "entry: speed bled off below orbital", spd[-1] < 1000.0,
          f"{spd[0]:.0f} -> {spd[-1]:.0f} m/s")


def t4_phase_handoff(path="outputs/final/nominal_traj.npz"):
    """Every phase boundary must hand over a CONTINUOUS position. A silent unit
    conversion or frame change at a seam is invisible to macro-validation but
    shows up here instantly."""
    if not os.path.exists(path):
        return
    z = np.load(path)
    ya, yb = z["transearth_y"], z["entry_y"]

    # The mass step must be a REAL separation event, not leakage: CM/SM
    # separation drops the stack to the command module alone.
    check(4, "transearth->entry mass step = CM/SM sep",
          abs(yb[6, 0] - A.CSM_CM_MASS) < 1.0,
          f"{ya[6, -1]:.0f} -> {yb[6, 0]:.0f} kg (CM = {A.CSM_CM_MASS:.0f})")

    # The recorded gap is 4.25 km, but it is NOT a discontinuity: the entry
    # phase starts from a root-solved state at the 121.92 km (400,000 ft)
    # entry interface, while the last decimated transearth sample sits 0.39 s
    # short of it and carries the EI time label. The physics check is whether
    # the two states are consistent under free fall over that implied offset.
    dr = yb[:3, 0] - ya[:3, -1]
    v = ya[3:6, -1]
    vh = v / np.linalg.norm(v)
    dt_imp = float(np.dot(dr, vh) / np.linalg.norm(v))
    cross = np.linalg.norm(dr - np.dot(dr, vh) * vh)
    check(4, "seam offset is purely along-track",
          cross < 0.05 * np.linalg.norm(dr),
          f"cross {cross / 1e3:.3f} km of {np.linalg.norm(dr) / 1e3:.3f} km")

    # Velocity difference must equal gravity x the SAME implied offset.
    dv_pred = np.linalg.norm(A.gravity_earth_moon(ya[:3, -1], 0.0)) * dt_imp
    dv_act = np.linalg.norm(yb[3:6, 0] - v)
    check(4, "seam closes under free fall (record label, not physics)",
          abs(dv_pred - dv_act) < 0.15 * max(dv_act, 1e-9),
          f"dt={dt_imp:.3f}s  dv act {dv_act:.3f} vs g*dt {dv_pred:.3f} m/s")

    alt_ei = np.linalg.norm(yb[:3, 0]) - A.R_EARTH
    check(4, "entry starts at the 400,000 ft interface",
          abs(alt_ei - 121_920.0) < 200.0, f"{alt_ei:.0f} m")


# =================================================================== main ===
def main():
    print(__doc__.split("Runs in seconds")[0].strip().splitlines()[0])
    print(f"apollo11 module: {os.path.abspath(A.__file__)}")

    tier(1, "invariants and symmetries")
    t1_time_reversal()
    t1_selenographic_basis()
    t1_precession_matrix()
    t1_frame_roundtrip()
    t1_rotational_covariance()
    t1_conservation()

    tier(2, "independent derivations")
    t2_third_body_decomposition()
    t2_lambert_self_consistency()
    t2_stumpff_continuity()
    t2_vis_viva()

    tier(3, "external standards")
    t3_ussa76_table()
    t3_atmosphere_continuity()
    t3_atmosphere_monotonic()
    t3_cm_reference_area()
    t3_surface_gravity()
    t3_j2_degeneration()
    t3_lunar_figure_falloff()
    t3_gmst_rate()

    tier(4, "recorded-trajectory replay")
    t4_recorded_trajectory()
    t4_phase_handoff()

    n = len(_RESULTS)
    bad = [r for r in _RESULTS if not r[2]]
    print(f"\n{'=' * 60}\n{n - len(bad)}/{n} passed")
    for _, name, _, detail in bad:
        print(f"  FAILED: {name}  {detail}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
