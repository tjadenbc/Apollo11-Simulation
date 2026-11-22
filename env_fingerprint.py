"""Record the software environment a run computed in, plus a numerical fingerprint of it.

Why: after a routine system-library update on a compute cluster, 38% of a 10K fleet's
trials came out different from an earlier run of the same code, seed and pinned nominal,
starting in the TEI solve. Nothing recorded what the earlier run had computed with, so the
cause could not be proven. Each run now records it, and the fingerprint shows whether two
environments compute the same bits WITHOUT re-flying trials.

What is recorded:
  * host, CPU model, a hash of the CPU feature flags (SIMD dispatch depends on them),
    OS / kernel, glibc, installed system packages (rpm, when present)
  * Python, numpy / scipy / pandas / jplephem versions, numpy's BLAS / LAPACK build
  * sha256 of the shared libraries actually mapped into the process (Linux: libm, libc,
    OpenBLAS, libgfortran, ...), of apollo11.py, fast_kernels and cluster_run.py
  * threading / FP-relevant environment variables and the SLURM job ids
  * numerical fingerprint: sha256 of the exact bits of a fixed battery of computations,
    per category -- libm (Python math), numpy ufuncs (SIMD, long and short arrays),
    BLAS (dot, norm, 3x3 / 6x6 / 64x64 matmul), LAPACK (solve, inv, SVD, QR, eigh,
    lstsq), scipy (least_squares trf and lm, solve_ivp DOP853 two-body orbit). Equal
    hashes = identical bits; a differing category localizes the change.

Read-only with respect to the simulation: it touches no apollo11 state, uses its own
seeded Generator, and never raises (collect() returns errors as strings).

Usage:
  python env_fingerprint.py [out.json]        print (and optionally write) the record
  python env_fingerprint.py --compare a.json b.json   say which categories differ
"""
import hashlib
import json
import os
import platform
import socket
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = 1

_ENV_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "GLIBC_TUNABLES", "LD_LIBRARY_PATH",
             "LD_PRELOAD", "PYTHONHASHSEED", "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID",
             "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME", "SLURM_CPUS_PER_TASK")
_LIB_HINTS = ("libm.", "libm-", "libc.", "libc-", "libmvec", "openblas", "gfortran", "quadmath",
              "libgcc_s", "libstdc++", "lapack", "libblas", "libcblas", "accelerate", "veclib")


def _sha256_file(path, n=16):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:n]
    except Exception as e:
        return f"error: {e}"


def _run(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        # rpm -q exits nonzero when ANY named package is absent but still lists the
        # installed ones (and "package X is not installed") on stdout: keep stdout.
        if p.stdout.strip():
            return p.stdout.strip()
        return f"rc={p.returncode}: {p.stderr.strip()[:200]}"
    except Exception as e:
        return f"unavailable: {e}"


def _cpu():
    out = {"machine": platform.machine(), "processor": platform.processor()}
    if os.path.exists("/proc/cpuinfo"):
        model, flags = None, None
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and model is None:
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("flags") and flags is None:
                    flags = line.split(":", 1)[1].split()
        out["model"] = model
        if flags:
            out["flags_sha256"] = hashlib.sha256(" ".join(sorted(flags)).encode()).hexdigest()[:16]
            out["simd"] = sorted(f for f in flags if f.startswith(("avx", "fma", "sse4", "f16c")))
    elif sys.platform == "darwin":
        out["model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return out


def _system():
    out = {"hostname": socket.gethostname(), "platform": platform.platform(),
           "kernel": platform.release(), "python": sys.version.replace("\n", " "),
           "python_executable": sys.executable}
    try:
        out["libc"] = os.confstr("CS_GNU_LIBC_VERSION")
    except (ValueError, OSError, AttributeError):
        out["libc"] = " ".join(platform.libc_ver()) or "n/a"
    if sys.platform == "darwin":
        out["macos"] = platform.mac_ver()[0]
    if os.path.exists("/usr/bin/rpm") or os.path.exists("/bin/rpm"):
        out["rpm"] = _run(["rpm", "-q", "glibc", "libgcc", "libgfortran", "libstdc++",
                           "libquadmath", "kernel-core", "openblas"]).splitlines()
    return out


def _packages():
    out = {}
    for mod in ("numpy", "scipy", "pandas", "jplephem"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception as e:
            out[mod] = f"unavailable: {e}"
    try:
        import numpy as np
        deps = np.show_config(mode="dicts").get("Build Dependencies", {})
        out["numpy_blas"] = {k: {kk: v for kk, v in deps[k].items()
                                 if kk in ("name", "version", "detection method", "openblas configuration")}
                             for k in ("blas", "lapack") if k in deps}
    except Exception as e:
        out["numpy_blas"] = f"unavailable: {e}"
    return out


def _mapped_libraries():
    """sha256 of the math-relevant shared libraries mapped into this process (Linux only)."""
    if not os.path.exists("/proc/self/maps"):
        return "n/a (no /proc/self/maps)"
    paths = set()
    with open("/proc/self/maps") as f:
        for line in f:
            p = line.split()[-1] if len(line.split()) >= 6 else ""
            base = os.path.basename(p).lower()
            if p.startswith("/") and ".so" in base and any(h in base for h in _LIB_HINTS):
                paths.add(p)
    return {p: _sha256_file(p) for p in sorted(paths)}


def _code():
    out = {}
    for name in ("apollo11.py", "cluster_run.py", "fast_kernels.py", "fast_kernels.c"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            out[name] = _sha256_file(p)
    for name in sorted(os.listdir(HERE)):
        if name.startswith("_fast_kernels") or (name.startswith("fast_kernels") and name.endswith((".so", ".dylib"))):
            out[name] = _sha256_file(os.path.join(HERE, name))
    return out


def _numeric():
    """Hash the exact bits of a fixed battery of computations, per category."""
    import math
    import numpy as np
    rng = np.random.default_rng(20261121)
    res = {}

    def h(chunks):
        m = hashlib.sha256()
        for c in chunks:
            m.update(np.ascontiguousarray(c, dtype=np.float64).tobytes())
        return m.hexdigest()[:16]

    def guard(name, fn):
        try:
            res[name] = fn()
        except Exception as e:
            res[name] = f"error: {e}"

    small = rng.uniform(-10.0, 10.0, 4000)
    pos = rng.uniform(1e-6, 50.0, 4000)
    big = rng.uniform(-1e6, 1e6, 4000)            # large trig arguments (range reduction)
    unit = rng.uniform(-0.999, 0.999, 4000)
    xy = rng.normal(0.0, 1e4, (4000, 2))

    def libm():
        f = [[math.exp(x) for x in small], [math.log(x) for x in pos], [math.log10(x) for x in pos],
             [math.sin(x) for x in big], [math.cos(x) for x in big], [math.tan(x) for x in small],
             [math.sin(x) for x in small], [math.cos(x) for x in small],
             [math.asin(x) for x in unit], [math.acos(x) for x in unit], [math.atan(x) for x in small],
             [math.atan2(a, b) for a, b in xy], [math.hypot(a, b) for a, b in xy],
             [math.sinh(x) for x in small], [math.cosh(x) for x in small], [math.tanh(x) for x in small],
             [math.expm1(x) for x in small], [math.log1p(x) for x in pos],
             [math.pow(a, b) for a, b in zip(pos, small)], [math.pow(a, 1.5) for a in pos],
             [math.sqrt(x) for x in pos]]
        return h(f)

    def ufunc():
        f = []
        for arr in (small, small[:3], small[:7]):                 # SIMD loop and short/remainder paths
            f += [np.exp(arr), np.sin(arr), np.cos(arr), np.tan(arr), np.arctan(arr), np.tanh(arr)]
        for arr in (big, big[:3]):
            f += [np.sin(arr), np.cos(arr)]
        for arr in (pos, pos[:3]):
            f += [np.log(arr), np.sqrt(arr), np.power(arr, 1.5), np.power(arr, small[:len(arr)]), np.cbrt(arr)]
        f += [np.arctan2(xy[:, 0], xy[:, 1]), np.hypot(xy[:, 0], xy[:, 1]),
              np.arcsin(unit), np.arccos(unit), np.sum(big), np.cumsum(small)]
        return h(f)

    def blas():
        v = rng.normal(0.0, 1e7, (500, 3))
        w = rng.normal(0.0, 1e3, (500, 3))
        A3 = rng.normal(size=(50, 3, 3))
        A6 = rng.normal(size=(20, 6, 6))
        A64 = rng.normal(size=(64, 64))
        f = [[np.dot(a, b) for a, b in zip(v, w)], [np.linalg.norm(a) for a in v],
             [np.cross(a, b) for a, b in zip(v, w)],
             [A3[i] @ A3[i + 1] for i in range(49)], [A3[i] @ v[i] for i in range(50)],
             [A6[i] @ A6[i + 1] for i in range(19)], A64 @ A64, A64 @ A64[:, 0]]
        return h(f)

    def lapack():
        import scipy.linalg as sl
        A6 = rng.normal(size=(6, 6)) + 6 * np.eye(6)
        S = A6 @ A6.T
        b = rng.normal(size=6)
        M = rng.normal(size=(12, 4))
        y = rng.normal(size=12)
        f = [np.linalg.solve(A6, b), np.linalg.inv(A6), np.linalg.svd(A6, compute_uv=False),
             np.abs(np.linalg.qr(M)[1]), np.linalg.eigh(S)[0], np.linalg.cholesky(S),
             np.linalg.lstsq(M, y, rcond=None)[0], sl.lstsq(M, y)[0], sl.solve(S, b, assume_a="pos")]
        return h(f)

    def scipy_opt():
        from scipy.optimize import least_squares
        from scipy.integrate import solve_ivp
        t = np.linspace(0.0, 4.0, 40)
        data = 2.5 * np.exp(-1.3 * t) * np.cos(3.1 * t + 0.4)

        def resid(p):
            return p[0] * np.exp(-p[1] * t) * np.cos(p[2] * t + p[3]) - data
        f = [least_squares(resid, [1.0, 1.0, 3.0, 0.0], method=m).x for m in ("trf", "lm")]
        mu = 4.9028e12
        r0 = 1.85e6
        y0 = [r0, 0.0, 0.0, 0.0, math.sqrt(mu / r0) * 1.02, 30.0]

        def rhs(_t, y):
            r3 = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]) ** 1.5
            return [y[3], y[4], y[5], -mu * y[0] / r3, -mu * y[1] / r3, -mu * y[2] / r3]
        sol = solve_ivp(rhs, (0.0, 7200.0), y0, method="DOP853", rtol=1e-12, atol=1e-6)
        f += [sol.y[:, -1], sol.t]
        return h(f)

    for name, fn in (("libm", libm), ("numpy_ufunc", ufunc), ("blas", blas),
                     ("lapack", lapack), ("scipy", scipy_opt)):
        guard(name, fn)
    res["all"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()[:16]
    return res


def collect():
    """Return the environment record as a dict (never raises)."""
    t0 = time.time()
    rec = {"fingerprint_version": VERSION,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for key, fn in (("system", _system), ("cpu", _cpu), ("packages", _packages),
                    ("libraries", _mapped_libraries), ("code", _code), ("numeric", _numeric)):
        try:
            rec[key] = fn()
        except Exception as e:
            rec[key] = f"error: {e}"
    rec["env"] = {k: os.environ[k] for k in _ENV_KEYS if k in os.environ}
    rec["collect_s"] = round(time.time() - t0, 2)
    return rec


def write(path):
    """Write collect() to path as JSON; return the record (never raises)."""
    rec = collect()
    try:
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
    except Exception as e:
        print(f"env_fingerprint: could not write {path}: {e}", flush=True)
    return rec


def compare(a, b):
    """List the numeric categories (and key versions) that differ between two records."""
    diffs = []
    na, nb = a.get("numeric", {}), b.get("numeric", {})
    for k in sorted(set(na) | set(nb)):
        if na.get(k) != nb.get(k):
            diffs.append(f"numeric.{k}: {na.get(k)} vs {nb.get(k)}")
    for sect, keys in (("system", ("libc", "kernel", "python")), ("packages", ("numpy", "scipy", "pandas"))):
        for k in keys:
            va, vb = a.get(sect, {}).get(k), b.get(sect, {}).get(k)
            if va != vb:
                diffs.append(f"{sect}.{k}: {va} vs {vb}")
    return diffs


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        d = compare(json.load(open(sys.argv[2])), json.load(open(sys.argv[3])))
        print("\n".join(d) if d else "identical numeric fingerprint and versions")
        sys.exit(1 if any(x.startswith("numeric.") for x in d) else 0)
    rec = write(sys.argv[1]) if len(sys.argv) > 1 else collect()
    print(json.dumps({k: rec[k] for k in ("system", "cpu", "packages", "numeric", "collect_s")}, indent=1))
