"""ENABLE_WATER_ZONE_REAIM: the recovery-zone search along the entry ground track and its predicted-entry check.

Geometry from the package 10Ks (seed 37, trial 8036): a scan-fallback TEI ignited 4.96 h late, entry at
9.42 S 96.58 E, and the zone 2,784 km along its ground track fell at 15.11 N 101.63 E (Thailand); the guidance
landed 4.1 km from it, on land. The track crosses water (Gulf of Thailand) between ~2,100 and ~2,480 km. The
entry predictions are stubbed here (a real one is a ~2 min entry simulation); the cluster re-fly of trial 8036
exercises the real predictor.
"""
import numpy as np
import pytest

EI_8036 = (-9.423196061010923, 96.58013825118032)
ZONE_8036 = (15.109473014959352, 101.62593428545063)
ZONE_NOMINAL_JUL16 = (13.437216830595025, -169.91344163809475)
STATE = np.zeros(7)            # the stub predictors ignore the state


def _bearing(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.deg2rad, (la1, lo1, la2, lo2))
    return float(np.arctan2(np.sin(lo2 - lo1) * np.cos(la2),
                            np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1)))


def _range_km(lat, lon):
    """Great-circle range from the 8036 entry interface (spherical R 6,371 km, as the zone placement)."""
    la1, lo1, la2, lo2 = map(np.deg2rad, (*EI_8036, lat, lon))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(min(1.0, a))))


def _stub(rule, calls=None):
    """predict(state, t, lat, lon) from rule(range_km) -> (success, miss_km, max_g)."""
    def predict(state, t, lat, lon):
        d = _range_km(lat, lon)
        if calls is not None:
            calls.append(round(d))
        ok, miss, g = rule(d)
        return {"success": ok, "miss_km": miss, "max_g": g}
    return predict


@pytest.fixture(scope="module")
def track_8036(consumer):
    consumer._is_land_splash(*ZONE_8036)          # initialises the land mask
    if consumer._LANDMASK is None:
        pytest.skip("global-land-mask not installed")
    return np.deg2rad(EI_8036[0]), np.deg2rad(EI_8036[1]), _bearing(*EI_8036, *ZONE_8036)


def _reaim(consumer, track, predict):
    return consumer._water_zone_reaim(STATE, 0.0, *track, predict=predict)


def test_flag_default_off_and_in_run_config(consumer):
    assert "ENABLE_WATER_ZONE_REAIM" in consumer.run_config()
    assert consumer.RECOVERY_ZONE_RANGE_KM == 2784.0


def test_track_reproduces_recorded_zone(consumer, track_8036):
    la, lo = consumer._gc_dest_deg(*track_8036, consumer.RECOVERY_ZONE_RANGE_KM)
    assert abs(la - ZONE_8036[0]) < 1e-6 and abs(lo - ZONE_8036[1]) < 1e-6


def test_land_zone_is_near_land_and_ocean_zone_is_not(consumer, track_8036):
    assert consumer._is_land_splash(*ZONE_8036)
    assert consumer._zone_near_land(*ZONE_8036, consumer.WATER_ZONE_MARGIN_KM)
    assert not consumer._zone_near_land(*ZONE_NOMINAL_JUL16, consumer.WATER_ZONE_MARGIN_KM)


def test_accepts_the_nearest_flyable_water(consumer, track_8036):
    calls = []
    out = _reaim(consumer, track_8036, _stub(lambda d: (True, 3.0, 6.0), calls))
    assert out["status"] == "water" and out["n_pred"] == 1
    assert 2100.0 <= out["d_km"] <= 2500.0, out          # Gulf of Thailand, short of 2,784 km
    assert calls[0] == 2784                               # the design-range reference comes first
    assert not consumer._zone_near_land(out["lat"], out["lon"], consumer.WATER_ZONE_MARGIN_KM)
    la2, lo2 = consumer._gc_dest_deg(*track_8036, out["d_km"] + consumer.WATER_ZONE_STEP_KM)
    assert consumer._zone_near_land(la2, lo2, consumer.WATER_ZONE_MARGIN_KM)   # nearest to the design range


def test_miss_rejection_skips_retry_distance(consumer, track_8036):
    first = _reaim(consumer, track_8036, _stub(lambda d: (True, 3.0, 6.0)))["d_km"]
    limit = first - 1.0                                   # reject the nearest candidate only
    out = _reaim(consumer, track_8036, _stub(lambda d: (True, 50.0 if d > limit else 3.0, 6.0)))
    assert out["status"] == "water" and out["n_pred"] == 2
    assert out["d_km"] <= first - consumer.WATER_ZONE_RETRY_KM


def test_g_margin_is_relative_to_the_design_range(consumer, track_8036):
    g_design = 5.9

    def rule_high(d):
        return (True, 3.0, g_design) if abs(d - 2784.0) < 1 else (True, 3.0, g_design + 1.3)
    out = _reaim(consumer, track_8036, _stub(rule_high))
    assert out["status"] == "rejected" and out["g_design"] == pytest.approx(g_design)

    def rule_ok(d):
        return (True, 3.0, g_design) if abs(d - 2784.0) < 1 else (True, 3.0, g_design + 0.75)
    assert _reaim(consumer, track_8036, _stub(rule_ok))["status"] == "water"


def test_failed_design_prediction_uses_the_fallback_reference(consumer, track_8036):
    def rule(d):
        return (False, float("inf"), 9.9) if abs(d - 2784.0) < 1 else (True, 3.0, 7.0)
    out = _reaim(consumer, track_8036, _stub(rule))
    assert out["g_design"] == consumer.WATER_ZONE_G_REF_FALLBACK
    assert out["status"] == "water"                      # 7.0 <= 6.5 + 1.0


def test_prediction_cap(consumer, track_8036):
    calls = []
    out = _reaim(consumer, track_8036, _stub(lambda d: (False, float("inf"), 9.0), calls))
    assert out["status"] == "rejected"
    assert out["n_pred"] <= consumer.WATER_ZONE_MAX_PRED
    assert len(calls) == out["n_pred"] + 1                # + the design-range reference


def test_none_found_when_the_span_is_all_land(consumer, track_8036, monkeypatch):
    monkeypatch.setattr(consumer, "WATER_ZONE_SEARCH_KM", (2600.0, 3400.0))
    out = _reaim(consumer, track_8036, _stub(lambda d: (True, 3.0, 6.0)))
    assert out["status"] == "none_found" and out["n_pred"] == 0
