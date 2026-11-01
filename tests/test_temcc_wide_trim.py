"""ENABLE_TEMCC_WIDE_TRIM: the trans-Earth corridor FPA trim's sample range (_temcc_trim_dv).

Stub fpa_err(dv) functions stand in for the entry projection (a real one is a coast integration). The no-entry
stub mirrors Jul-18 trials 1448 / 2001: every dv in -8..+8 m/s leaves the return above the atmosphere (+90),
and entry needs about -20 m/s retrograde.
"""
import numpy as np
import pytest


def _old_trim(fpa_err):
    """The trim's search as it was inline in solve_correction (bd719c47), verbatim."""
    grid = np.linspace(-8.0, 8.0, 9)
    samples = [(dv, fpa_err(dv)) for dv in grid]
    best_dv = 0.0
    bracket = None
    for i in range(len(samples) - 1):
        a, fa = samples[i]; b, fb = samples[i+1]
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            bracket = (a, b); break
    if bracket is not None:
        try:
            from scipy.optimize import brentq
            best_dv = brentq(fpa_err, bracket[0], bracket[1], xtol=0.05, maxiter=12)
        except Exception:
            best_dv = min(samples, key=lambda s: abs(s[1]))[0]
    else:
        best_dv = min(samples, key=lambda s: abs(s[1]))[0]
    return best_dv


def _counted(f):
    calls = []

    def g(dv):
        calls.append(float(dv))
        return f(dv)
    return g, calls


def _no_entry_until(root_dv, graze_dv):
    """Entry FPA error vs dv: steeper with more retrograde dv, no entry (+90) above the graze."""
    def f(dv):
        return 90.0 if dv > graze_dv else 0.15 * (dv - root_dv)
    return f


STUBS = {
    "root inside": lambda dv: 0.2 * (dv - 3.3),
    "root at a sample": lambda dv: 0.2 * (dv + 4.0),
    "all steep": lambda dv: -0.5 - 0.01 * dv * dv,
    "no entry anywhere": lambda dv: 90.0,
    "no entry until -20": _no_entry_until(-20.0, -17.0),
    "nonlinear": lambda dv: np.tanh(0.3 * (dv - 1.7)) + 0.05,
}


def test_flag_default_off_and_in_run_config(consumer):
    cfg = consumer.run_config()
    assert "ENABLE_TEMCC_WIDE_TRIM" in cfg
    assert consumer.TEMCC_TRIM_WIDE_MS == 30.0


@pytest.mark.parametrize("name", sorted(STUBS))
def test_flag_off_matches_the_old_search(consumer, name):
    f = STUBS[name]
    g_old, c_old = _counted(f)
    g_new, c_new = _counted(f)
    assert consumer._temcc_trim_dv(g_new, wide=False) == _old_trim(g_old)
    assert c_new == c_old


@pytest.mark.parametrize("name", ["root inside", "nonlinear"])
def test_bracketed_within_8_is_unchanged_by_the_flag(consumer, name):
    f = STUBS[name]
    g_off, c_off = _counted(f)
    g_on, c_on = _counted(f)
    assert consumer._temcc_trim_dv(g_on, wide=True) == consumer._temcc_trim_dv(g_off, wide=False)
    assert c_on == c_off                                   # no extra projections


def test_exact_zero_at_a_sample_keeps_that_sample(consumer):
    # fa * fb < 0 sees no sign change across an exact zero, so the wide search runs, but the zero sample still wins
    f = STUBS["root at a sample"]
    assert consumer._temcc_trim_dv(f, wide=True) == consumer._temcc_trim_dv(f, wide=False) == -4.0


def test_no_entry_within_8_reaches_the_root_when_wide(consumer):
    f = STUBS["no entry until -20"]
    assert consumer._temcc_trim_dv(f, wide=False) == -8.0  # the old ceiling
    dv = consumer._temcc_trim_dv(f, wide=True)
    assert dv == pytest.approx(-20.0, abs=0.06)            # brentq xtol 0.05
    assert abs(f(dv)) < 0.01


def test_wide_samples_extend_in_2_ms_steps_to_30(consumer):
    g, calls = _counted(STUBS["no entry anywhere"])
    dv = consumer._temcc_trim_dv(g, wide=True)
    assert sorted(calls) == [float(x) for x in range(-30, 31, 2)]
    assert dv == -30.0                                     # every sample ties at +90: the first is kept


def test_root_beyond_30_is_not_chased(consumer):
    f = _no_entry_until(-40.0, -35.0)
    assert consumer._temcc_trim_dv(f, wide=True) == -30.0
