"""S4: the deterministic re-fly of the delivered nominal in the consumer (slow: one full mission, ~10 min).

Run with `make refly-test` or `pytest -m refly`; `make test-fast` deselects it.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import refly_delivered_nominal as RF  # noqa: E402

pytestmark = pytest.mark.refly


@pytest.fixture(scope="module")
def report(package_path, tmp_path_factory):
    rep, _ = RF.refly(package_path, verbose=False)
    out = tmp_path_factory.mktemp("refly") / "refly_report.json"
    out.write_text(json.dumps(rep, default=float))
    return rep


def test_refly_reaches_splashdown_from_delivered_burns(report):
    assert report["reached_splashdown"], report["mission_failure"]
    assert report["sources"]["loi1"] == "delivered_open_loop"
    assert report["sources"]["loi2"] == "delivered_open_loop"
    assert report["sources"]["tei_ignition"] == "delivered"


def test_refly_hard_tolerances(report):
    hard = [r for r in report["rows"] if r["hard"]]
    assert hard, "no hard rows"
    failed = [r for r in hard if not r["ok"]]
    assert not failed, failed
    assert report["verdict"] != "FAIL"


def test_refly_milestones_present(report):
    m = report["milestones"]
    for k in ("insertion", "tli_cutoff", "perilune", "loi1_cutoff", "loi1_exec", "loi2_cutoff", "loi2_exec",
              "tei_ignition", "tei_cutoff_open_loop", "tei_exec", "ei_open_loop", "tei_corrector", "ei", "splash"):
        assert k in m and not m[k].get("missing"), k


def test_refly_tight_coast_holds_tei_phase(report):
    """The truth coasts run DOP853 1e-10 (ENABLE_TIGHT_COAST): the delivered TEI epoch finds the CSM within a few km
    of the package's state after 55 h / 28 lunar revs (92 km / 57 s of phase at the original RK45 1e-9)."""
    assert report["tight_coast"]
    m = report["milestones"]
    assert m["loi2_ignition"]["dr_m"] < 1_000.0
    assert m["tei_ignition"]["dr_m"] < 10_000.0
    assert m["tei_ignition"]["dv_mps"] < 10.0
    assert abs(m["tei_corrector"]["flown_dv_mps"] - m["tei_corrector"]["delivered_dv_mps"]) < 2.0


def test_refly_loi1_holds_the_site_plane(report):
    """The open-loop delivered LOI-1 carries the Site 2 plane change: the consumer's own residual (flown plane coasted
    to PDI) meets the package goal, and the flown plane change matches the delivered one."""
    rows = {r["id"]: r for r in report["rows"]}
    assert abs(rows["loi1_site.xtrack_km"]["value"]) < 1.0
    assert abs(rows["loi1_site.dplane_change_deg"]["value"]) < 0.05
    assert report["flown"]["loi1_yaw_deg"] == report["delivered"]["loi1_yaw_deg"]


def test_refly_lands_at_site2(report):
    """PDI site targeting: the nominal touches down at Site 2's downrange (was 43 km long with PDI at perilune)."""
    rows = {r["id"]: r for r in report["rows"]}
    assert abs(rows["landing.site_downrange_km"]["value"]) < 1.0
    assert abs(rows["landing.site_crossrange_km"]["value"]) < 3.0
    assert -30.0 < report["flown"]["pdi_tig_offset_s"] < -20.0
