"""P63-style PDI site targeting (ENABLE_PDI_SITE_TARGETING): from the consumer's own nominal descent-orbit perilune
(tests/fixtures/pdi_perilune_jul16.json), PDI at perilune lands ~43 km long of Site 2; _target_pdi_tig moves the TIG
along the descent orbit until the predicted touchdown is at Site 2's downrange. ~15 s (0.5 s per predicted descent)."""
import json
import os

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import delivered as D

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "pdi_perilune_jul16.json")


@pytest.fixture(scope="module")
def pdi(package_path, consumer):
    A = consumer
    D.bind_package(package_path, A)                 # the epoch and site the fixture was captured with
    fx = json.load(open(FIXTURE))
    return A, np.asarray(fx["state"], float), float(fx["t_get_s"]), float(fx["descent_m0_kg"])


def _n_hat(A, s, t):
    mr, mv = A.moon_state(t)
    n = np.cross(s[:3] - mr, s[3:6] - mv)
    return n / np.linalg.norm(n)


def _coast(A, s, t0, t1):
    def rhs(t, y):
        return np.concatenate([y[3:6], A.gravity_earth_moon(y[:3], t), [0]])
    return solve_ivp(rhs, (t0, t1), s, method="DOP853", rtol=1e-10, atol=1e-3, max_step=30.0).y[:, -1].copy()


def test_perilune_pdi_lands_long(pdi):
    """Why the targeting exists: PDI at the perilune of the as-planned DOI lands ~43 km past Site 2."""
    A, s, t, m0 = pdi
    d = A.phase_powered_descent(s, t, None, m0_override=m0)
    assert d["success"]
    dr, xr = A._land_site_error_km(d, _n_hat(A, s, t))
    assert 35.0 < dr < 50.0                                          # measured +43.4 km (long, west)
    assert abs(xr) < 3.0                                             # LOI-1 holds the plane: +0.7 km


def test_tig_targeting_lands_at_site_downrange(pdi):
    A, s, t, m0 = pdi
    r = A._target_pdi_tig(s, t, m0)
    assert r is not None
    assert abs(r["downrange_pred_km"]) < A.PDI_SITE_DOWNRANGE_TOL_KM
    assert -30.0 < r["dt_s"] < -20.0                                 # measured -25.8 s (1.68 km/s of ground speed)
    assert r["n_eval"] <= 4                                          # measured 2: downrange is linear in TIG
    d = A.phase_powered_descent(r["state"], r["t"], None, m0_override=m0)   # the nominal flies the prediction
    dr, xr = A._land_site_error_km(d, _n_hat(A, s, t))
    assert abs(dr) < A.PDI_SITE_DOWNRANGE_TOL_KM and abs(xr) < 3.0
    assert d["fuel_margin_s"] > 90.0                                 # measured 102 s (100 s at perilune)


@pytest.mark.parametrize("phase_s", [-60.0, 60.0])
def test_tig_targeting_absorbs_parking_orbit_phase(pdi, phase_s):
    """A dispersed parking-orbit phase moves the perilune along track (here +/-60 s, ~100 km): the targeting re-solves
    from that perilune and still lands at Site 2's downrange."""
    A, s, t, m0 = pdi
    s2 = _coast(A, s, t, t + phase_s)                                # a perilune displaced along the same orbit
    r = A._target_pdi_tig(s2, t + phase_s, m0)
    assert r is not None and abs(r["downrange_pred_km"]) < A.PDI_SITE_DOWNRANGE_TOL_KM
    assert abs((r["t"] - t) - (-25.8)) < 3.0                         # same TIG on the orbit, whatever the seed point
