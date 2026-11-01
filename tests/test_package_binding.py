import math

import numpy as np

import check_artifact as CA
import delivered as D


def test_adapter_keys(package_path):
    t = D.adapt_delivered_package(CA.load(package_path))
    assert set(t) == D.DELIVERED_TARGET_KEYS
    assert t["tli_cutoff_rule"] == "inertial_speed"
    assert len(t["insertion_state"]) == 7 and len(t["tei_dv_ric_mps"]) == 3
    env = D.env_for_import(t)
    assert env["APOLLO_JPL_EPHEMERIS"] == "1" and float(env["APOLLO_LAUNCH_AZ_DEG"]) == t["launch_azimuth_deg"]
    assert abs(float(env["APOLLO_LAUNCH_DELAY_H"])) < 1e-9        # the Jul-16 package is at the consumer's literal epoch


def test_bind_sets_live_globals(package_path, consumer):
    t, manifest = D.bind_package(package_path, consumer)
    assert consumer.LAUNCH_AZIMUTH_DEG == t["launch_azimuth_deg"]
    assert consumer._CACHED_LAUNCH_TLI == (t["tli_ign_angle_deg"], t["tli_vcut_mps"], t["tli_pitch_deg"], t["tli_yaw_deg"])
    assert consumer._solve_launch_tli() == consumer._CACHED_LAUNCH_TLI     # the consumer's TLI solve is bypassed
    assert consumer.JD_LAUNCH == t["liftoff_jd"] and consumer.LUNAR_DOI_GET_S == t["doi_get_s"]
    assert consumer.SPLASH_TARGET_LAT_DEG == t["splash_lat_deg"] and consumer.TEI_PLAN_GET_S == t["tei_plan_get_s"]
    assert consumer._EI_TARGET["t_ei"] == t["ei_get_s"] and consumer._PRESET_REF["t_ins"] == t["insertion_get_s"]
    assert manifest["package_sha256"] == t["package_sha256"]
    assert not any("loi1_*" in s or "tei_*" in s for s in manifest["not_consumed_yet"])
    assert "tei_dir_eci" in manifest["consumed_by_phases"]
    assert consumer._delivered_burn("loi1")["dv_mps"] == t["loi1_dv_mps"]
    assert consumer._delivered_burn("tei")["t_ign"] == t["tei_t_ignition_get_s"]


def test_consumer_launch_and_tli_refly_the_package(package_path, consumer):
    """Deterministic re-fly of the first two phases: the consumer's own IGM ascent at the bound azimuth, then its
    own steered S-IVB burn with the bound preset, reproduce the delivered insertion and TLI cutoff states."""
    t, _ = D.bind_package(package_path, consumer)
    launch = consumer.phase_saturn_v_launch({})
    assert launch["success"]
    ins = np.asarray(launch["state"], float)
    # The solver's insertion is a cubic spline through 52 captured consumer ascents: the residual is a ~2 ms insertion
    # timing shift along the same orbit (16 m along-track, 0.019 m/s), not a different orbit.
    dt_ins = launch["t_insertion"] - t["insertion_get_s"]
    dpos = np.linalg.norm(ins[:3] - np.asarray(t["insertion_state"][:3]))
    assert abs(dt_ins) < 0.01
    assert dpos < 50.0
    assert abs(dpos - abs(dt_ins) * np.linalg.norm(ins[3:6])) < 2.0      # offset explained by the time shift
    assert np.linalg.norm(ins[3:6] - np.asarray(t["insertion_state"][3:6])) < 0.1
    out = consumer._fly_launched_tli(launch["state"], launch["t_insertion"], {}, t["tli_ign_angle_deg"], t["tli_vcut_mps"],
                                     steer_pitch_deg=t["tli_pitch_deg"], steer_yaw_deg=t["tli_yaw_deg"])
    assert out is not None
    state, t_cut, dv = out
    post = np.asarray(t["tli_state_post"])
    assert abs(t_cut - t["tli_t_cutoff_get_s"]) < 0.1
    assert np.linalg.norm(state[:3] - post[:3]) < 100.0          # measured 0.29 m (consumer RK45 vs solver DOP853)
    assert np.linalg.norm(state[3:6] - post[3:6]) < 0.1          # measured 0.4 mm/s
    assert math.isclose(np.linalg.norm(state[3:6]), t["tli_vcut_mps"], abs_tol=0.1)
