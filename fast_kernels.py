"""Compiled kernels for apollo11.py: a cached DE440 coefficient slice, the degree-N
lunar field, and the arithmetic of gravity_earth_moon and the Earth zonals (see fast_kernels.c).

Each reproduces the Python code it replaces bit for bit. They are used only when their
switches are set (APOLLO_EPHEM_CACHE=1 -> FAST_EPHEM_CACHE, APOLLO_COMPILED_LUNAR_FIELD=1 -> FAST_LUNAR_FIELD,
APOLLO_VECTOR_KERNELS=1 -> FAST_VECTOR_KERNELS); with the switches off apollo11.py never
imports this module.

Build once per machine type (the library name carries the platform, so one shared
directory serves a Mac and a Linux cluster side by side):
    python fast_kernels.py build
The build uses the system C compiler with -O2 -ffp-contract=off and no -ffast-math;
a fused multiply-add would round once where the Python code rounds twice.
"""
import ctypes
import os
import platform
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "fast_kernels.c")
CFLAGS = ["-O2", "-ffp-contract=off", "-fno-fast-math", "-fPIC", "-shared"]


def lib_path():
    return os.path.join(HERE, f"_fast_kernels.{sys.platform}-{platform.machine()}.so")


def build(cc=None, verbose=True):
    """Compile fast_kernels.c into this machine type's shared library."""
    cc = cc or os.environ.get("CC", "cc")
    out = lib_path()
    tmp = out + f".tmp{os.getpid()}"
    cmd = [cc] + CFLAGS + ["-o", tmp, SRC, "-lm"]
    if verbose:
        print(" ".join(cmd))
    subprocess.check_call(cmd)
    os.replace(tmp, out)            # atomic, so a concurrent loader never sees half a file
    return out


_LIB = None
_D3 = ctypes.c_double * 3
_DP = ctypes.POINTER(ctypes.c_double)


def lib():
    """The loaded library. Fails loudly if it was never built for this machine type."""
    global _LIB
    if _LIB is None:
        path = lib_path()
        if not os.path.exists(path):
            raise RuntimeError(
                f"compiled kernels requested but {os.path.basename(path)} is not built for this "
                f"machine; run `python fast_kernels.py build` in {HERE} (or unset "
                "APOLLO_EPHEM_CACHE / APOLLO_COMPILED_LUNAR_FIELD / APOLLO_VECTOR_KERNELS)")
        L = ctypes.CDLL(path)
        L.fk_seg_load.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_long,
                                  ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int, _DP]
        L.fk_seg_load.restype = ctypes.c_int
        for name, nargs in (("fk_moon_rv", 2), ("fk_moon_r", 1), ("fk_sun_r", 1)):
            fn = getattr(L, name)
            fn.argtypes = [ctypes.c_double] + [_DP] * nargs
            fn.restype = ctypes.c_int
        L.fk_seg_eval.argtypes = [ctypes.c_int, ctypes.c_double, _DP, _DP]
        L.fk_seg_eval.restype = ctypes.c_int
        _IP = ctypes.POINTER(ctypes.c_int)
        L.fk_sh_load.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                                 _IP, _IP, _DP, _DP, _DP, _DP]
        L.fk_sh_load.restype = ctypes.c_int
        L.fk_sh_accel.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, _DP]
        L.fk_sh_accel.restype = ctypes.c_int
        L.fk_grav_load.argtypes = [ctypes.c_double] * 6 + [_DP, _IP]
        L.fk_grav_load.restype = ctypes.c_int
        L.fk_zonal_accel.argtypes = [ctypes.c_double] * 3 + [_DP]
        L.fk_zonal_accel.restype = ctypes.c_int
        L.fk_grav_tail.argtypes = [ctypes.c_double] * 9 + [ctypes.c_int, _DP]
        L.fk_grav_tail.restype = ctypes.c_int
        _LIB = L
    return _LIB


# ---------------------------------------------------------------- ephemeris cache
_EPH = {"window": None}


def ephem_load(segments, jd_lo, jd_hi):
    """Copy the Chebyshev records covering [jd_lo, jd_hi] (TDB Julian dates) of the four
    jplephem segments (Moon/EMB, Earth/EMB, Sun/SSB, EMB/SSB) into the library."""
    L = lib()
    for which, seg in enumerate(segments):
        init, intlen, coef = seg._data                 # (ncoef, ncomp, n), jplephem's own arrays
        ncoef, ncomp, n = coef.shape
        r0 = max(0, int(((jd_lo - 2451545.0) * 86400.0 - init) // intlen) - 1)
        r1 = min(n - 1, int(((jd_hi - 2451545.0) * 86400.0 - init) // intlen) + 1)
        sl = np.ascontiguousarray(np.transpose(coef[:, :, r0:r1 + 1], (2, 0, 1)), dtype=np.float64)
        rc = L.fk_seg_load(which, float(init), float(intlen), int(n), r0, r1 - r0 + 1,
                           int(ncoef), int(ncomp), sl.ctypes.data_as(_DP))
        if rc:
            raise RuntimeError(f"fk_seg_load failed ({rc}) for segment {which}")
    _EPH["window"] = (jd_lo, jd_hi)


def ephem_window():
    return _EPH["window"]


_BR, _BV = _D3(), _D3()


def moon_rv(jde):
    """(r, v) J2000 [m, m/s] before the precession, or None outside the cached slice."""
    if _LIB.fk_moon_rv(jde, _BR, _BV):
        return None
    return np.array(_BR), np.array(_BV)


def moon_r(jde):
    if _LIB.fk_moon_r(jde, _BR):
        return None
    return np.array(_BR)


def sun_r(jde):
    if _LIB.fk_sun_r(jde, _BR):
        return None
    return np.array(_BR)


def seg_eval(which, tdb):
    """Raw (position km, rate km/day) of one cached segment, for the tests."""
    p, v = _D3(), _D3()
    if lib().fk_seg_eval(which, tdb, p, v):
        return None
    return np.array(p), np.array(v)


# ---------------------------------------------------------------- lunar field
def sh_load(nmax, R, mu, terms):
    """terms: apollo11._LUNAR_SH_TERMS, a list of (n, m, Cn, Sn, f, g)."""
    L = lib()
    nt = len(terms)
    n = (ctypes.c_int * nt)(*[int(t[0]) for t in terms])
    m = (ctypes.c_int * nt)(*[int(t[1]) for t in terms])
    cols = [(ctypes.c_double * nt)(*[float(t[k]) for t in terms]) for k in (2, 3, 4, 5)]
    for k in (4, 5):        # f and g must be exact as doubles (they are small integers or floats)
        for t in terms:
            if float(t[k]) != t[k]:
                raise ValueError(f"lunar-field term {t[:2]}: factor {t[k]!r} is not exact as a double")
    rc = L.fk_sh_load(int(nmax), float(R), float(mu), nt, n, m, *cols)
    if rc:
        raise RuntimeError(f"fk_sh_load failed ({rc})")


_BA = _D3()


def sh_accel(x, y, z):
    """The body-frame perturbing acceleration (ax, ay, az), as _lunar_sh_accel_body returns it."""
    _LIB.fk_sh_accel(x, y, z, _BA)
    return _BA[0], _BA[1], _BA[2]


# ---------------------------------------------------------------- gravity tail
_GV = {"key": None}
_BG = (ctypes.c_double * 4)()


def grav_load(mu_e, re, j2, mu_m, mu_s, zr_max, zonals):
    """Hand the gravity constants to the library. kj2 is formed here in the order
    gravity_earth_moon writes it: 1.5 * J2 * MU_EARTH * R_EARTH**2."""
    J = (ctypes.c_double * 7)(*[float(zonals.get(n, 0.0)) for n in range(7)])
    has = (ctypes.c_int * 7)(*[int(n in zonals) for n in range(7)])
    kj2 = 1.5 * j2 * mu_e * re ** 2
    rc = lib().fk_grav_load(mu_e, re, kj2, mu_m, mu_s, zr_max, J, has)
    if rc:
        raise RuntimeError(f"fk_grav_load failed ({rc})")


def grav_tail(r, mr, sr, zonals_on):
    """gravity_earth_moon without the lunar-figure term: (a as a new float64 array, |mr - r|).
    sr is None when solar gravity is off."""
    rx, ry, rz = r.tolist()
    mx, my, mz = mr.tolist()
    if sr is None:
        flags, sx, sy, sz = (1 if zonals_on else 0), 0.0, 0.0, 0.0
    else:
        flags = (1 if zonals_on else 0) | 2
        sx, sy, sz = sr.tolist()
    _LIB.fk_grav_tail(rx, ry, rz, mx, my, mz, sx, sy, sz, flags, _BG)
    return np.array(_BG[0:3]), _BG[3]


def zonal_accel(r):
    """_earth_zonal_accel(r) as a new float64 array (for the tests)."""
    x, y, z = r.tolist()
    if lib().fk_zonal_accel(x, y, z, _BG):
        raise RuntimeError("gravity constants not loaded")
    return np.array(_BG[0:3])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        print("built", build())
    else:
        print(__doc__)
