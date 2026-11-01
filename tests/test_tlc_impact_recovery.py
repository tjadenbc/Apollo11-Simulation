"""ENABLE_TLC_IMPACT_RECOVERY: the pre-MCC SOI-miss pre-screen (_tlc_gross_miss).

Uncorrected periapsis altitudes are the four Jul-18 seed-37 trials the screen ended as missed_lunar_soi (72, 157,
162, 194: -1,371 / -1,018 / -1,054 / -1,040 km); with the screen's impact side off all four were full successes.
"""
import pytest

JUL18_SCREENED_KM = (-1370.787960, -1018.132758, -1053.560083, -1039.901729)


def test_flag_default_off_and_in_run_config(consumer):
    assert "ENABLE_TLC_IMPACT_RECOVERY" in consumer.run_config()


def test_flag_off_keeps_the_original_screen(consumer, monkeypatch):
    monkeypatch.setattr(consumer, "ENABLE_TRANS_LUNAR_MCC", True)
    monkeypatch.setattr(consumer, "ENABLE_TLC_IMPACT_RECOVERY", False)
    for alt in JUL18_SCREENED_KM:
        assert consumer._tlc_gross_miss(alt)
    assert not consumer._tlc_gross_miss(-999.0)
    assert not consumer._tlc_gross_miss(5136.0)            # a Jul-16 end-of-window value
    assert consumer._tlc_gross_miss(60001.0)


def test_flag_on_drops_only_the_impact_side(consumer, monkeypatch):
    monkeypatch.setattr(consumer, "ENABLE_TRANS_LUNAR_MCC", True)
    monkeypatch.setattr(consumer, "ENABLE_TLC_IMPACT_RECOVERY", True)
    for alt in JUL18_SCREENED_KM + (-1.0e6,):
        assert not consumer._tlc_gross_miss(alt)
    assert consumer._tlc_gross_miss(60001.0)               # far miss still screened


@pytest.mark.parametrize("flag", [False, True])
def test_without_mcc_chain_the_tight_gate_is_unchanged(consumer, monkeypatch, flag):
    monkeypatch.setattr(consumer, "ENABLE_TRANS_LUNAR_MCC", False)
    monkeypatch.setattr(consumer, "ENABLE_TLC_IMPACT_RECOVERY", flag)
    assert consumer._tlc_gross_miss(-201.0)
    assert consumer._tlc_gross_miss(1501.0)
    assert not consumer._tlc_gross_miss(100.0)
