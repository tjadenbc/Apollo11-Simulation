"""DOI site retargeting (ENABLE_DOI_SITE_RETARGET): from the consumer's nominal post-LOI-2 state
(tests/fixtures/loi2_cutoff_jul16.json), the as-planned DOI at the Apollo-timed GET puts PDI-at-perilune 43 km long of
Site 2, inside what PDI targeting absorbs (26 s), so the DOI is not moved; a DOI plan displaced by a parking-orbit
phase error (+600 s / -1500 s, like the low-arrival trials 18 / 24 of the local 25) is shifted back along the orbit
to the same DOI time. ~30 s."""
import json
import os

import numpy as np
import pytest

import delivered as D

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "loi2_cutoff_jul16.json")


@pytest.fixture(scope="module")
def park(package_path, consumer):
    D.bind_package(package_path, consumer)
    fx = json.load(open(FIXTURE))
    return consumer, np.asarray(fx["state"], float), float(fx["t_get_s"])


def test_apollo_timed_doi_is_kept_when_pdi_can_absorb_it(park):
    A, s, t = park
    r = A._retarget_doi_time(s, t, A.LUNAR_DOI_GET_S)
    assert r is not None and r["shift_s"] == 0.0 and r["n_eval"] == 1
    assert 35.0 < r["downrange0_km"] < 50.0                      # measured +43.3 km = 25.8 s of PDI offset
    assert abs(r["downrange0_km"]) / 1.68 < A.PDI_TIG_OFFSET_SOFT_S


@pytest.mark.parametrize("plan_offset_s", [600.0, -1500.0])
def test_phase_error_is_absorbed_by_moving_doi(park, plan_offset_s):
    A, s, t = park
    r = A._retarget_doi_time(s, t, A.LUNAR_DOI_GET_S + plan_offset_s)
    assert r is not None
    assert abs(r["downrange0_km"]) > 1000.0                      # PDI targeting alone would need 600-1500 s
    assert abs(r["downrange_km"]) < A.DOI_RETARGET_TOL_KM
    assert abs(r["t_doi"] - (A.LUNAR_DOI_GET_S - 25.0)) < 15.0   # both converge to DOI ~26 s before the plan
    assert r["n_eval"] <= 5
