"""The compiled kernels (fast_kernels.c) against the Python code they replace: every
result must agree in every bit. Skipped when the library is not built for this machine
(`python fast_kernels.py build`)."""
import os

import numpy as np
import pytest

fk = pytest.importorskip("fast_kernels")
if not os.path.exists(fk.lib_path()):
    pytest.skip("fast_kernels library not built on this machine", allow_module_level=True)


@pytest.fixture(scope="module")
def segs(consumer):
    seg = consumer._jpl()
    if seg is None:
        pytest.skip("DE440 kernel not available")
    _, moon, earth, sun, emb = seg
    fk.ephem_load([moon, earth, sun, emb], consumer.JD_LAUNCH - 2.0, consumer.JD_LAUNCH + 60.0)
    return moon, earth, sun, emb


def _epochs(consumer, segs, n=4000):
    jd0 = consumer.JD_LAUNCH
    rng = np.random.default_rng(7)
    jds = list(jd0 + rng.uniform(-1.0, 55.0, n))
    for s in (segs[0], segs[2]):                 # record boundaries and their neighbours
        init, intlen, _ = s._data
        k0 = int(((jd0 - 2451545.0) * 86400.0 - init) // intlen)
        for k in range(k0, k0 + 12):
            b = 2451545.0 + (init + k * intlen) / 86400.0
            if jd0 - 1.0 < b < jd0 + 55.0:
                jds += [b, np.nextafter(b, 0.0), np.nextafter(b, 1e9)]
    return [float(j) for j in jds]


def test_moon_state_bit_identical(consumer, segs):
    moon, earth, _, _ = segs
    for jde in _epochs(consumer, segs):
        rm, vm = moon.compute_and_differentiate(jde)
        re, ve = earth.compute_and_differentiate(jde)
        r, v = fk.moon_rv(jde)
        assert np.array_equal(r, (rm - re) * 1000.0) and np.array_equal(v, (vm - ve) * 1000.0 / 86400.0), jde
        assert np.array_equal(fk.moon_r(jde), (moon.compute(jde) - earth.compute(jde)) * 1000.0), jde


def test_sun_bit_identical(consumer, segs):
    _, earth, sun, emb = segs
    for jde in _epochs(consumer, segs):
        assert np.array_equal(fk.sun_r(jde), (sun.compute(jde) - (emb.compute(jde) + earth.compute(jde))) * 1000.0), jde


def test_outside_slice_falls_back(consumer, segs):
    assert fk.moon_rv(consumer.JD_LAUNCH + 400.0) is None
    assert fk.sun_r(consumer.JD_LAUNCH - 30.0) is None


def test_lunar_field_bit_identical(consumer):
    fk.sh_load(consumer._LUNAR_SH_NMAX, consumer.R_MOON_GRAV, consumer.MU_MOON, consumer._LUNAR_SH_TERMS)
    rng = np.random.default_rng(11)
    R = consumer.R_MOON_GRAV
    pts = [(0.0, 0.0, 1.8e6), (0.0, 0.0, -1.8e6), (1.8e6, 0.0, 0.0), (1e-9, 1e-9, 1.75e6)]
    for _ in range(20000):
        u = rng.normal(size=3)
        u *= rng.uniform(0.99 * R, 3.5e6) / np.linalg.norm(u)
        pts.append(tuple(float(c) for c in u))
    for x, y, z in pts:
        assert fk.sh_accel(x, y, z) == consumer._lunar_sh_accel_body(x, y, z), (x, y, z)


def test_no_fused_multiply_add_in_build():
    """-ffp-contract=off must have held: a fused multiply-add rounds once where Python rounds twice."""
    import subprocess
    import sys
    tool = ["otool", "-tv"] if sys.platform == "darwin" else ["objdump", "-d"]
    try:
        out = subprocess.run(tool + [fk.lib_path()], capture_output=True, text=True, check=True).stdout.lower()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("no disassembler")
    for op in ("fmadd", "fmsub", "fnmadd", "fnmsub", "vfmadd", "vfmsub", "vfnmadd"):
        assert op not in out, op


# ------------------------------------------------------------- APOLLO_VECTOR_KERNELS
def _gravity_points(consumer, n, seed):
    """Positions from LEO to beyond the Moon, a share of them inside the zonal gate
    (< 3e7 m) and a share within 5e7 m of the Moon (the lunar-figure gate)."""
    rng = np.random.default_rng(seed)
    pts = []
    for k in range(n):
        t = float(rng.uniform(0.0, 8.0 * 86400.0))
        kind = k % 4
        if kind == 0:
            r = rng.standard_normal(3); r *= rng.uniform(6.5e6, 3.0e7) / np.linalg.norm(r)
        elif kind == 1:
            r = rng.standard_normal(3); r *= rng.uniform(3.0e7, 4.5e8) / np.linalg.norm(r)
        else:
            mr, _ = consumer.moon_state(t)
            d = rng.standard_normal(3); d *= rng.uniform(1.75e6, 6.0e7) / np.linalg.norm(d)
            r = mr + d
        pts.append((np.ascontiguousarray(r, dtype=np.float64), t))
    return pts


@pytest.mark.parametrize("solar,zonals", [(True, True), (False, True), (True, False)])
def test_gravity_tail_bit_identical(consumer, monkeypatch, solar, zonals):
    monkeypatch.setattr(consumer, "FAST_VECTOR_KERNELS", False)     # gravity_earth_moon = numpy path
    monkeypatch.setattr(consumer, "ENABLE_SOLAR_GRAVITY", solar)
    monkeypatch.setattr(consumer, "ENABLE_EARTH_HIGHER_ZONALS", zonals)
    bad = 0
    for r, t in _gravity_points(consumer, 6000, 11):
        a_np = consumer.gravity_earth_moon(r, t)
        a_fk = consumer._gravity_earth_moon_fk(r, t)
        assert type(a_fk) is np.ndarray and a_fk.dtype == np.float64 and a_fk.shape == (3,)
        bad += int(not np.array_equal(a_np, a_fk))
    assert bad == 0, f"{bad} gravity evaluations differ"


def test_zonal_accel_bit_identical(consumer):
    fk.grav_load(consumer.MU_EARTH, consumer.R_EARTH, consumer.J2, consumer.MU_MOON,
                 consumer.MU_SUN, consumer.EARTH_ZONAL_R_MAX_M, consumer.EARTH_ZONALS)
    consumer._FK_GV[0] = None                   # make the next fast gravity call reload
    rng = np.random.default_rng(12)
    for _ in range(20000):
        r = rng.standard_normal(3); r *= rng.uniform(6.3e6, 3.0e7) / np.linalg.norm(r)
        assert np.array_equal(consumer._earth_zonal_accel(r), fk.zonal_accel(r))


def test_norm3_cross3_bit_identical(consumer):
    rng = np.random.default_rng(13)
    for scale in (6.4e6, 4.0e8, 1.1e4, 1.0, 1e-3):
        V = rng.standard_normal((20000, 3)) * scale
        W = rng.standard_normal((20000, 3)) * scale
        for v, w in zip(V, W):
            n3 = consumer._norm3(v)
            assert type(n3) is np.float64 and n3 == np.linalg.norm(v)
            c3 = consumer._cross3(v, w)
            assert c3.dtype == np.float64 and np.array_equal(c3, np.cross(v, w))
    # slices of a solver state vector, as the call sites pass them
    y = rng.standard_normal(7) * 1e6
    assert consumer._norm3(y[:3]) == np.linalg.norm(y[:3])
    assert np.array_equal(consumer._cross3(y[:3], y[3:6]), np.cross(y[:3], y[3:6]))
    # anything that is not a float64 3-vector goes to numpy unchanged
    f32 = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    assert consumer._norm3(f32) == np.linalg.norm(f32)
    assert consumer._norm3([3.0, 4.0, 12.0]) == 13.0
    assert consumer._norm3(np.ones((2, 3))) == np.linalg.norm(np.ones((2, 3)))
    assert np.array_equal(consumer._cross3([1.0, 0, 0], [0, 1.0, 0]), [0, 0, 1.0])
    assert np.array_equal(consumer._cross3(f32, f32[::-1]), np.cross(f32, f32[::-1]))
