"""Consumer 3-DOF LOI-1 site-plane corrector (dispersed trials with a bound package). From the delivered LOI-1
ignition state with SPS and arrival dispersions, _solve_loi_3dof re-solves (dv, tilt, yaw) to the planned
60.0 x 169.7 nmi ellipse AND the Site 2 plane at the PDI epoch (the solver's LOI-1 residual), so the plane change
the free-return package moved into LOI-1 survives the re-solve. ~2 min."""
import numpy as np
import pytest

import delivered as D


@pytest.fixture(scope="module")
def loi1(package_path, consumer):
    A = consumer
    t, _ = D.bind_package(package_path, A)
    d = A._delivered_burn("loi1")
    if d is None or d["yaw_deg"] == 0.0:
        pytest.skip("package has no LOI-1 plane change")
    m = A.CSM_CM_MASS + A.CSM_SM_DRY + A.SPS_PROP_INIT + A.LM_DESC_TOTAL + A.LM_ASCT_TOTAL   # the consumer's stack
    ig = np.concatenate([d["state_at_tig"], [m]])
    return A, t, d, ig


def _h_hat(A, ig, t):
    mr, mv = A.moon_state(t)
    h = np.cross(ig[:3] - mr, ig[3:6] - mv)
    return h / np.linalg.norm(h), (ig[3:6] - mv) / np.linalg.norm(ig[3:6] - mv)


def test_yaw_zero_is_bit_identical(loi1):
    A, _, d, ig = loi1
    a = A.phase_loi(ig.copy(), d["t_ign"], d["dv_mps"], None, tilt_deg=d["tilt_deg"])
    b = A.phase_loi(ig.copy(), d["t_ign"], d["dv_mps"], None, tilt_deg=d["tilt_deg"], yaw_deg=0.0)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_tilt_yaw_reproduce_the_delivered_direction(loi1):
    """phase_loi's (tilt, yaw) construction is the solver's sps_burn_direction: at the delivered ignition state it
    flies the delivered frozen direction (measured 2.2 arcsec, ~0.01 m/s over the 883 m/s burn)."""
    A, _, d, ig = loi1
    s_ty, t_ty, _ = A.phase_loi(ig.copy(), d["t_ign"], d["dv_mps"], None, tilt_deg=d["tilt_deg"], yaw_deg=d["yaw_deg"])
    s_dir, t_dir, _ = A.phase_loi(ig.copy(), d["t_ign"], d["dv_mps"], None, burn_dir_eci=d["dir_eci"])
    assert abs(t_ty - t_dir) < 1e-9
    assert np.linalg.norm(s_ty[3:6] - s_dir[3:6]) < 0.05
    assert np.linalg.norm(s_ty[:3] - s_dir[:3]) < 20.0


def test_delivered_burn_meets_the_site_plane(loi1):
    """The consumer's own residual on the delivered burn agrees with the package's LOI-1 site goal."""
    A, t, d, ig = loi1
    s, tt, _ = A.phase_loi(ig.copy(), d["t_ign"], d["dv_mps"], None, burn_dir_eci=d["dir_eci"])
    assert abs(A._loi_site_xtrack_km(s, tt)) < A.LOI1_SITE_XTRACK_TOL_KM
    assert abs(A._loi_plane_change_deg(ig, d["t_ign"], s, tt) - t["loi1_plane_change_deg"]) < 0.05


CASES = {
    "sps_isp-0.5%_thrust+1%": ({"sps_isp_factor": 0.995, "sps_thrust_factor": 1.01}, 0.0, 0.0),
    "arrival_+1.5mps_normal_-2mps_along": ({}, 1.5, -2.0),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_3dof_corrector_holds_the_site_plane(loi1, case):
    A, t, d, ig = loi1
    perturb, dv_n, dv_a = CASES[case]
    h, v_hat = _h_hat(A, ig, d["t_ign"])
    s0 = ig.copy()
    s0[3:6] += dv_n * h + dv_a * v_hat
    out = A._solve_loi_3dof(s0, d["t_ign"], A.LOI1_PERI_TARGET_KM, A.LOI1_APO_TARGET_KM, perturb,
                            d["dv_mps"], d["tilt_deg"], d["yaw_deg"])
    assert out is not None
    dv, tilt, conv, s, tt, bt, info = out
    assert conv and info["ellipse_ok"]
    peri, apo = A._loi_peri_apo(s, tt)
    assert abs(peri - A.LOI1_PERI_TARGET_KM) < 1.0 and abs(apo - A.LOI1_APO_TARGET_KM) < 1.0
    assert abs(info["site_xtrack_km"]) < A.LOI1_SITE_XTRACK_TOL_KM
    assert abs(A._loi_site_xtrack_km(s, tt) - info["site_xtrack_km"]) < 1e-6      # the reported residual is the coasted one
    assert abs(info["yaw_deg"] - d["yaw_deg"]) < 1.0 and abs(dv - d["dv_mps"]) < 15.0
    assert 2.3 < A._loi_plane_change_deg(s0, d["t_ign"], s, tt) < 3.3          # delivered 2.76 deg
    assert info["n_coast"] <= 3                                                  # measured 2 (proxy + drift bias)


@pytest.mark.parametrize("drop_m", [5_000.0, 9_000.0, 25_000.0])
def test_3dof_accepts_least_squares_best_when_the_ellipse_is_infeasible(loi1, drop_m):
    """A low arrival (ignition state 5 / 9 km radially in: approach perilune 109.0 / 105.0 km) caps the capture
    perilune below the 111.12 km target's 1 km gate. The 2-DOF misses (the legacy in-plane single-dv burn would fly);
    the 3-DOF's Levenberg-Marquardt step ends at the least-squares best (apolune met, perilune as high as the arrival
    allows) inside LOI1_ELLIPSE_ACCEPT_KM with the site plane still met (seed 42 trials 3 / 5 arrive at 109.5 /
    105.6 km). A 25 km drop (~89 km approach perilune, like hazards trial 24) leaves a ~21 km perilune shortfall:
    accepted because the apolune is inside the band and the capture perilune is above LOI1_CAPTURE_PERI_FLOOR_KM."""
    A, _, d, ig = loi1
    mr, _ = A.moon_state(d["t_ign"])
    r_hat = (ig[:3] - mr) / np.linalg.norm(ig[:3] - mr)
    s0 = ig.copy()
    s0[:3] -= drop_m * r_hat
    two = A._solve_loi_2dof(s0.copy(), d["t_ign"], A.LOI1_PERI_TARGET_KM, A.LOI1_APO_TARGET_KM, {},
                            dv_seed=d["dv_mps"], tilt_seed=d["tilt_deg"], yaw_deg=d["yaw_deg"])
    assert two is None or not two[2]
    out = A._solve_loi_3dof(s0, d["t_ign"], A.LOI1_PERI_TARGET_KM, A.LOI1_APO_TARGET_KM, {},
                            d["dv_mps"], d["tilt_deg"], d["yaw_deg"])
    dv, tilt, conv, s, tt, bt, info = out
    assert conv and not info["ellipse_ok"] and info["ellipse_accepted"]
    peri, apo = A._loi_peri_apo(s, tt)
    assert A.LOI1_PERI_TARGET_KM - peri > 1.0                                   # the perilune carries the shortfall
    assert abs(apo - A.LOI1_APO_TARGET_KM) < (1.0 if drop_m < 10_000 else A.LOI1_ELLIPSE_ACCEPT_KM)
    assert info["ellipse_err_km"] < A.LOI1_ELLIPSE_ACCEPT_KM or peri >= A.LOI1_CAPTURE_PERI_FLOOR_KM
    assert abs(info["site_xtrack_km"]) < A.LOI1_SITE_XTRACK_TOL_KM


def test_in_plane_2dof_drops_the_plane_change(loi1):
    """Why the corrector exists: the in-plane re-solve meets the ellipse but leaves Site 2 ~56 km off the plane."""
    A, _, d, ig = loi1
    out = A._solve_loi_2dof(ig.copy(), d["t_ign"], A.LOI1_PERI_TARGET_KM, A.LOI1_APO_TARGET_KM, {},
                            dv_seed=d["dv_mps"], tilt_seed=d["tilt_deg"])
    assert out is not None and out[2]
    assert abs(A._loi_site_xtrack_km(out[3], out[4])) > 30.0
