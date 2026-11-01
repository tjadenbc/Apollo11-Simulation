"""APOLLO_DISPERSION_SCALE: one scalar on the guidance, navigation and execution dispersions.

The scale multiplies each dispersion after its draw, so the RNG streams are unchanged: at 1.0 every value is what
the unscaled code produced, and at k the scaled terms are exactly k times their 1.0 values while every other
perturbation is untouched. The nested _estimate / _exec_residual draws cannot be called in isolation, so their
sites are checked in the source of the phases that hold them.
"""
import inspect

import numpy as np
import pytest

SCALED = ("insertion_v_err", "tli_pointing_rad", "tli_dv_bias_ms", "tei_dv_bias_ms")


def _sample(consumer, scale, seed=37):
    old = consumer.DISPERSION_SCALE
    consumer.DISPERSION_SCALE = scale
    try:
        return consumer.sample_perturbation(np.random.default_rng(seed))
    finally:
        consumer.DISPERSION_SCALE = old


def test_default_is_one_and_recorded(consumer):
    assert consumer.DISPERSION_SCALE == 1.0
    assert consumer.run_config()["DISPERSION_SCALE"] == 1.0


@pytest.mark.parametrize("k", [2.0, 4.0])
def test_only_the_listed_terms_scale(consumer, k):
    base, scaled = _sample(consumer, 1.0), _sample(consumer, k)
    assert set(base) == set(scaled)
    for key in base:
        a, b = np.asarray(base[key]), np.asarray(scaled[key])
        if key in SCALED:
            np.testing.assert_array_equal(b, a * k)
        else:
            np.testing.assert_array_equal(b, a)


def test_nav_and_exec_sites_carry_the_scale(consumer):
    tl = inspect.getsource(consumer.phase_translunar_mcc)
    te = inspect.getsource(consumer.phase_transearth_mcc)
    rm = inspect.getsource(consumer.run_mission)
    assert tl.count("MSFN_TL_POS_SIGMA_M, 3) * DISPERSION_SCALE") == 1
    assert tl.count("MSFN_TL_VEL_SIGMA_MS, 3) * DISPERSION_SCALE") == 1
    assert te.count("MSFN_TE_POS_SIGMA_M, 3) * DISPERSION_SCALE") == 1
    assert te.count("MSFN_TE_VEL_SIGMA_MS, 3) * DISPERSION_SCALE") == 1
    for src in (tl, te):                                   # every execution-residual draw in both chains
        draws = [l.strip() for l in src.splitlines() if "rng_exec.normal(" in l]
        assert draws and all("DISPERSION_SCALE" in l for l in draws), draws
    assert "MSFN_TEI_EXEC_SIGMA_MS * DISPERSION_SCALE" in rm
