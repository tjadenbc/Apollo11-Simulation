"""Fleet execution (sim plan S5): APOLLO_DELIVERED_PACKAGE binds the package at import, so a spawned multiprocessing
worker (macOS start method; spawn re-imports apollo11) and a cluster job fly the same package as the parent. ~30 s."""
import json
import multiprocessing as mp
import os
import subprocess
import sys

import check_artifact as CA
import delivered as D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _probe(_):
    import apollo11 as A
    d = A._DELIVERED or {}
    return {"label": d.get("label"), "sha": d.get("package_sha256"), "tli": A._CACHED_LAUNCH_TLI,
            "jd": A.JD_LAUNCH, "az": A.LAUNCH_AZIMUTH_DEG, "doi": A.LUNAR_DOI_GET_S,
            "cfg_sha": A.run_config()["DELIVERED_PACKAGE_SHA256"], "pkg_env": os.environ.get("APOLLO_DELIVERED_PACKAGE")}


def test_spawned_worker_binds_the_package(package_path, monkeypatch):
    t = D.adapt_delivered_package(CA.load(package_path))
    monkeypatch.setenv("APOLLO_DELIVERED_PACKAGE", package_path)
    with mp.get_context("spawn").Pool(1) as pool:
        w = pool.map(_probe, [0])[0]
    assert w["sha"] == t["package_sha256"] == w["cfg_sha"] and w["label"] == t["label"]
    assert w["tli"] == (t["tli_ign_angle_deg"], t["tli_vcut_mps"], t["tli_pitch_deg"], t["tli_yaw_deg"])  # no own TLI solve
    assert w["jd"] == t["liftoff_jd"] and w["az"] == t["launch_azimuth_deg"] and w["doi"] == t["doi_get_s"]
    assert os.path.isabs(w["pkg_env"])                                           # children get an absolute path


def _import_rc(env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("APOLLO_")}
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", "import apollo11"], cwd=HERE, env=env, capture_output=True, text=True)


def test_conflicting_environment_is_refused(package_path):
    r = _import_rc({"APOLLO_DELIVERED_PACKAGE": package_path, "APOLLO_LAUNCH_AZ_DEG": "72.0"})
    assert r.returncode != 0 and "conflicts with the delivered package" in r.stderr


def test_pinned_sha_is_enforced(package_path):
    sha = json.load(open(package_path))["manifest"]["package_sha256"]
    assert _import_rc({"APOLLO_DELIVERED_PACKAGE": package_path, "APOLLO_DELIVERED_PACKAGE_SHA256": sha[:12]}).returncode == 0
    r = _import_rc({"APOLLO_DELIVERED_PACKAGE": package_path, "APOLLO_DELIVERED_PACKAGE_SHA256": "deadbeef"})
    assert r.returncode != 0 and "not the pinned deadbeef" in r.stderr


def test_cluster_stages_refuse_unbound(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("APOLLO_")}
    r = subprocess.run([sys.executable, "cluster_run.py", "setup", str(tmp_path / "o"), "1", "42"], cwd=HERE, env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "APOLLO_DELIVERED_PACKAGE is not set" in (r.stderr + r.stdout)


def test_fleet_drivers_start_from_the_package_ei_target(package_path, consumer):
    """main() / main_parallel() start from _delivered_ei_target(), not None: the fleet nominal flies the delivered TEI
    corrector to the package's EI and pins THAT target (a None start re-solved the TEI: 1,024 vs 1,018 m/s)."""
    t, _ = D.bind_package(package_path, consumer)
    ei = consumer._delivered_ei_target()
    assert ei == {"lat": t["ei_lat_gc_deg"], "lon": t["ei_lon_deg"], "fpa": t["ei_fpa_deg"], "t_ei": t["ei_get_s"]}
    assert ei == consumer._EI_TARGET
    saved = consumer._DELIVERED
    try:
        consumer._DELIVERED = None
        assert consumer._delivered_ei_target() is None
    finally:
        consumer._DELIVERED = saved
