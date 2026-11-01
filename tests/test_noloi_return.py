"""ENABLE_NOLOI_RETURN: the no-LOI return flown instead of assumed (apollo11._noloi_*, crew_survival.py).

The coast and burn solve are checked on a pure Earth-return conic placed well inside the Moon's influence-free
region, where the answer is known in closed form; the per-slot behaviour on the real arrival geometry is a fleet
result, not a unit test. The crew-survival mapping is checked on hand-built rows.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("flag", ["ENABLE_NOLOI_RETURN", "ENABLE_LIFEBOAT_LIMIT"])
def test_flags_default_off_and_in_run_config(consumer, flag):
    assert getattr(consumer, flag) is False
    assert consumer.run_config()[flag] is False


def _inbound_state(consumer, r0_km, fpa_ei_deg, v_ei=11_000.0):
    """Earth-centred state at r0 on the inbound leg of the conic that crosses the entry interface at (v_ei, fpa)."""
    mu = consumer.MU_EARTH
    r_ei = consumer.R_EARTH + consumer.NOLOI_EI_ALT_M
    eps = 0.5 * v_ei ** 2 - mu / r_ei
    h = r_ei * v_ei * np.cos(np.deg2rad(fpa_ei_deg))
    r0 = r0_km * 1e3
    v0 = np.sqrt(2.0 * (eps + mu / r0))
    vt = h / r0
    vr = -np.sqrt(v0 ** 2 - vt ** 2)                      # inbound
    return np.array([r0, 0.0, 0.0, vr, vt, 0.0, 30_000.0])


@pytest.mark.parametrize("fpa", [-6.5, -5.0])
def test_coast_matches_the_entry_coast(consumer, fpa):
    # Earth's zonals bend the arc off the two-body conic on the way in (~0.1 deg here), so the reference is the
    # sim's own entry coast under the same force model, and h_err follows the FPA it reads at the shell.
    t0 = 520_000.0                                         # trans-Earth coast GET: Moon ~380,000 km away
    y = _inbound_state(consumer, 30_000.0, fpa)
    c = consumer._noloi_coast(y, t0)
    ref = consumer.phase_transearth_coast(y, t0)
    assert c is not None and c["reached_ei"] and ref["reached_entry"]
    assert c["fpa_ei_deg"] == pytest.approx(ref["fpa_at_entry_deg"], abs=0.01)
    assert c["t_end"] == pytest.approx(ref["final_t"], abs=5.0)
    g, g0 = np.deg2rad(c["fpa_ei_deg"]), np.deg2rad(consumer.MCC_TARGET_FPA_DEG)
    assert c["h_err"] == pytest.approx(np.cos(g) / np.cos(g0) - 1.0, abs=1e-6)


def test_solve_puts_a_shallow_return_on_the_corridor(consumer):
    t0 = 520_000.0
    y = _inbound_state(consumer, 60_000.0, -4.0)            # skips out as flown
    dv, c = consumer._noloi_solve(y, t0)
    assert dv is not None
    assert abs(c["h_err"]) < 5e-5
    assert c["fpa_ei_deg"] == pytest.approx(consumer.MCC_TARGET_FPA_DEG, abs=0.05)
    assert 1.0 < np.linalg.norm(dv) < 200.0


def test_solve_meets_a_deadline_when_asked(consumer):
    t0 = 520_000.0
    y = _inbound_state(consumer, 200_000.0, -6.5)
    c0 = consumer._noloi_coast(y, t0)
    deadline = c0["t_end"] - 3600.0                        # an hour earlier than the ballistic arrival
    dv, c = consumer._noloi_solve(y, t0, deadline_t=deadline)
    assert dv is not None
    assert c["t_end"] <= deadline
    assert abs(c["h_err"]) < 5e-5


@pytest.mark.parametrize("limit", [False, True])
def test_lifeboat_limit_acts_only_with_its_flag(consumer, monkeypatch, limit):
    # an on-corridor return whose ballistic arrival is one hour past the (shortened) lifeboat limit: without the
    # flag it flies as a free return and records its length; with it the burn buys the hour
    t0 = 520_000.0
    y = _inbound_state(consumer, 200_000.0, -6.5)
    t_end0 = consumer._noloi_coast(y, t0)["t_end"]
    t_fail = t0 - consumer.NOLOI_PC_DELAY_S - 1.0          # too late for the early burn: PC+2h (= t0) only
    monkeypatch.setattr(consumer, "NOLOI_LIFEBOAT_MAX_H", (t_end0 - 3600.0 - t_fail) / 3600.0)
    monkeypatch.setattr(consumer, "ENABLE_LIFEBOAT_LIMIT", limit)
    tlc = {"log_t": np.array([t0]), "log_states": y.reshape(7, 1), "t_closest": t0 - consumer.NOLOI_PC_DELAY_S}
    out = consumer._noloi_return("sm_failure_translunar", tlc, t_fail, 18_000.0)
    if limit:
        assert out["noloi_return"] == "dps_abort"
        assert out["noloi_ei_get_h"] * 3600.0 <= t_end0 - 3600.0
    else:
        assert out["noloi_return"] == "free_return"
        assert out["noloi_return_h"] * 3600.0 == pytest.approx(t_end0 - t_fail, abs=60.0)


def test_sm_failure_before_tli_is_left_unscored(consumer):
    out = consumer._noloi_return("sm_failure_translunar", {"t_closest": 273_000.0}, 2_000.0, 18_000.0, t_tli=9_000.0)
    assert out == {"noloi_return": None, "noloi_reason": "before_tli"}


def _rows(**extra):
    base = {"trial": [0, 1, 2, 3], "full_success": [False] * 4,
            "mission_failure": ["sm_failure_translunar"] * 3 + ["sps_ignition_failure_loi"]}
    base.update(extra)
    return pd.DataFrame(base)


def test_crew_survival_reads_the_flown_return():
    import crew_survival as CS
    out = CS.apply_survival_model(_rows(noloi_return=["none", "dps_abort", "free_return", "none"]), seed=1)
    assert list(out["psurv_Armstrong"]) == [0.0, 0.90, 0.90, 0.0]
    assert list(out["crew_outcome"]) == ["no_return_without_loi", "dps_abort", "lm_lifeboat_abort",
                                         "no_return_without_loi"]
    assert not out.loc[[0, 3], "crew_survived"].any()


def test_crew_survival_unchanged_without_the_column():
    import crew_survival as CS
    out = CS.apply_survival_model(_rows(), seed=1)
    assert list(out["psurv_Armstrong"]) == [0.90, 0.90, 0.90, 0.97]
