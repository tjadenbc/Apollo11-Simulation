/*
 * fast_kernels.c -- compiled kernels for apollo11.py (see fast_kernels.py).
 *
 * Every routine here repeats, operation for operation, the double-precision
 * arithmetic of the Python code it replaces, so its results are bit-identical
 * to that code. Two rules keep it so:
 *   - build with -O2 -ffp-contract=off and never -ffast-math: a fused
 *     multiply-add rounds once where the Python code rounds twice;
 *   - keep every expression in the order Python evaluates it (left to right,
 *     with the same parentheses). Reordering "harmless" sums changes bits.
 * fmod, floor and copysign are exact in IEEE arithmetic, and sqrt is correctly
 * rounded, so they give the same bits as CPython's math module and numpy.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* 1. DE440 Chebyshev evaluation from a cached slice of records.        */
/*    Repeats jplephem 2.24 Segment.generate() for a scalar tdb and      */
/*    tdb2 = 0 (data types 2 and 3, position components only).           */
/* ------------------------------------------------------------------ */

#define T0 2451545.0
#define S_PER_DAY 86400.0
#define NSEG 4                       /* 0 Moon/EMB, 1 Earth/EMB, 2 Sun/SSB, 3 EMB/SSB */

typedef struct {
    int loaded;
    double init, intlen;             /* segment start (s past J2000) and record length (s) */
    long n;                          /* records in the whole segment */
    long rec0, nrec;                 /* the cached slice: records rec0 .. rec0+nrec-1 */
    int ncoef, ncomp;
    double *coef;                    /* [rec][k][comp], k in jplephem's reversed order */
} seg_t;

static seg_t SEG[NSEG];

/* numpy's npy_divmod for float64 (numpy/_core/src/npymath/npy_math_internal.h.src) */
static double npy_divmod(double a, double b, double *modulus)
{
    double div, mod, floordiv;
    mod = fmod(a, b);
    if (!b) {
        *modulus = mod;
        return a / b;
    }
    div = (a - mod) / b;
    if (mod) {
        if ((b < 0) != (mod < 0)) {
            mod += b;
            div -= 1.0;
        }
    } else {
        mod = copysign(0.0, b);
    }
    if (div) {
        floordiv = floor(div);
        if (div - floordiv > 0.5)
            floordiv += 1.0;
    } else {
        floordiv = copysign(0.0, a / b);
    }
    *modulus = mod;
    return floordiv;
}

int fk_seg_load(int which, double init, double intlen, long n, long rec0, long nrec,
                int ncoef, int ncomp, const double *coef)
{
    if (which < 0 || which >= NSEG) return -1;
    seg_t *s = &SEG[which];
    free(s->coef);
    s->coef = (double *)malloc(sizeof(double) * (size_t)nrec * ncoef * ncomp);
    if (!s->coef) { s->loaded = 0; return -2; }
    memcpy(s->coef, coef, sizeof(double) * (size_t)nrec * ncoef * ncomp);
    s->init = init; s->intlen = intlen; s->n = n; s->rec0 = rec0; s->nrec = nrec;
    s->ncoef = ncoef; s->ncomp = ncomp; s->loaded = 1;
    return 0;
}

/* Position (and, if vel != NULL, rate per day) of segment `which` at tdb, in km and
 * km/day, as jplephem returns them. Returns 0, or -1 if tdb is outside the slice
 * (the caller then asks jplephem itself). */
static int seg_eval(int which, double tdb, double *pos, double *vel)
{
    const seg_t *sg = &SEG[which];
    if (!sg->loaded) return -1;
    double intlen = sg->intlen;
    double offset1, offset2, offset;
    double index1 = npy_divmod((tdb - T0) * S_PER_DAY - sg->init, intlen, &offset1);
    double index2 = npy_divmod(0.0 * S_PER_DAY, intlen, &offset2);
    double index3 = npy_divmod(offset1 + offset2, intlen, &offset);
    long index = (long)(index1 + index2 + index3);
    if (index < 0 || index > sg->n) return -1;
    if (index == sg->n) {            /* jplephem's "omegas": the last instant of the segment */
        index -= 1;
        offset += intlen;
    }
    if (index < sg->rec0 || index >= sg->rec0 + sg->nrec) return -1;
    const int nc = sg->ncoef, nk = sg->ncomp;
    const double *c = sg->coef + (size_t)(index - sg->rec0) * nc * nk;
    double s = 2.0 * offset / intlen - 1.0;
    double s2 = 2.0 * s;
    for (int j = 0; j < nk && j < 3; j++) {
        double w0 = 0.0, w1 = 0.0, w2;
        double wl[64];
        for (int k = 0; k < nc - 1; k++) {
            w2 = w1;
            w1 = w0;
            w0 = c[k * nk + j] + (s2 * w1 - w2);
            wl[k] = w1;
        }
        pos[j] = c[(nc - 1) * nk + j] + (s * w0 - w1);
        if (vel) {
            double dw0 = 0.0, dw1 = 0.0, dw2;
            for (int k = 0; k < nc - 1; k++) {
                dw2 = dw1;
                dw1 = dw0;
                dw0 = 2.0 * wl[k] + dw1 * s2 - dw2;
            }
            double rate = w0 + s * dw0 - dw1;
            rate /= intlen;
            rate *= 2.0;
            rate *= S_PER_DAY;
            vel[j] = rate;
        }
    }
    return 0;
}

/* (moon - earth) * 1000.0 [m] and (vm - ve) * 1000.0 / 86400.0 [m/s], J2000 axes,
 * as _moon_state_jpl forms them before the precession. */
int fk_moon_rv(double jde, double *r, double *v)
{
    double rm[3], vm[3], re[3], ve[3];
    if (seg_eval(0, jde, rm, vm) || seg_eval(1, jde, re, ve)) return -1;
    for (int j = 0; j < 3; j++) {
        r[j] = (rm[j] - re[j]) * 1000.0;
        v[j] = (vm[j] - ve[j]) * 1000.0 / 86400.0;
    }
    return 0;
}

/* (moon - earth) * 1000.0 [m], as _moon_eci_m_jpl forms it. */
int fk_moon_r(double jde, double *r)
{
    double rm[3], re[3];
    if (seg_eval(0, jde, rm, NULL) || seg_eval(1, jde, re, NULL)) return -1;
    for (int j = 0; j < 3; j++)
        r[j] = (rm[j] - re[j]) * 1000.0;
    return 0;
}

/* (sun - (emb + earth)) * 1000.0 [m], as _sun_eci_m_jpl forms it. */
int fk_sun_r(double jde, double *r)
{
    double rs[3], rb[3], re[3];
    if (seg_eval(2, jde, rs, NULL) || seg_eval(3, jde, rb, NULL) || seg_eval(1, jde, re, NULL))
        return -1;
    for (int j = 0; j < 3; j++)
        r[j] = (rs[j] - (rb[j] + re[j])) * 1000.0;
    return 0;
}

/* Raw segment evaluation, for the tests: position and rate of one segment. */
int fk_seg_eval(int which, double tdb, double *pos, double *vel)
{
    return seg_eval(which, tdb, pos, vel);
}

/* ------------------------------------------------------------------ */
/* 2. Degree-N lunar field: _lunar_sh_accel_body (Cunningham V/W        */
/*    recursion on unnormalized coefficients), body frame.              */
/* ------------------------------------------------------------------ */

#define SH_MAX 16
static int SH_NMAX = -1, SH_NT = 0;
static double SH_R, SH_K;            /* R_MOON_GRAV and MU_MOON / (R * R) */
static int *SH_n, *SH_m;
static double *SH_C, *SH_S, *SH_f, *SH_g;

int fk_sh_load(int nmax, double R, double mu, int nt, const int *n, const int *m,
               const double *C, const double *S, const double *f, const double *g)
{
    if (nmax < 0 || nmax + 2 > SH_MAX) return -1;
    free(SH_n); free(SH_m); free(SH_C); free(SH_S); free(SH_f); free(SH_g);
    SH_n = malloc(sizeof(int) * nt); SH_m = malloc(sizeof(int) * nt);
    SH_C = malloc(sizeof(double) * nt); SH_S = malloc(sizeof(double) * nt);
    SH_f = malloc(sizeof(double) * nt); SH_g = malloc(sizeof(double) * nt);
    if (!SH_n || !SH_m || !SH_C || !SH_S || !SH_f || !SH_g) { SH_NMAX = -1; return -2; }
    memcpy(SH_n, n, sizeof(int) * nt); memcpy(SH_m, m, sizeof(int) * nt);
    memcpy(SH_C, C, sizeof(double) * nt); memcpy(SH_S, S, sizeof(double) * nt);
    memcpy(SH_f, f, sizeof(double) * nt); memcpy(SH_g, g, sizeof(double) * nt);
    SH_NMAX = nmax; SH_NT = nt; SH_R = R;
    SH_K = mu / (R * R);
    return 0;
}

int fk_sh_accel(double x, double y, double z, double *out)
{
    if (SH_NMAX < 0) return -1;
    const double R = SH_R;
    double r2 = x * x + y * y + z * z;
    double rho = R * R / r2;
    double x0 = R * x / r2;
    double y0 = R * y / r2;
    double z0 = R * z / r2;
    const int size = SH_NMAX + 2;
    double V[SH_MAX][SH_MAX], W[SH_MAX][SH_MAX];
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++) { V[i][j] = 0.0; W[i][j] = 0.0; }
    V[0][0] = R / sqrt(r2);
    for (int m = 0; m < size; m++) {
        if (m > 0) {
            V[m][m] = (double)(2 * m - 1) * (x0 * V[m - 1][m - 1] - y0 * W[m - 1][m - 1]);
            W[m][m] = (double)(2 * m - 1) * (x0 * W[m - 1][m - 1] + y0 * V[m - 1][m - 1]);
        }
        if (m + 1 < size) {
            V[m + 1][m] = (double)(2 * m + 1) * z0 * V[m][m];
            W[m + 1][m] = (double)(2 * m + 1) * z0 * W[m][m];
        }
        for (int n = m + 2; n < size; n++) {
            V[n][m] = ((double)(2 * n - 1) * z0 * V[n - 1][m]
                       - (double)(n + m - 1) * rho * V[n - 2][m]) / (double)(n - m);
            W[n][m] = ((double)(2 * n - 1) * z0 * W[n - 1][m]
                       - (double)(n + m - 1) * rho * W[n - 2][m]) / (double)(n - m);
        }
    }
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int t = 0; t < SH_NT; t++) {
        const int n = SH_n[t], m = SH_m[t];
        const double Cn = SH_C[t], Sn = SH_S[t], f = SH_f[t], g = SH_g[t];
        const double *Vn = V[n + 1], *Wn = W[n + 1];
        if (m == 0) {
            ax += -Cn * Vn[1];
            ay += -Cn * Wn[1];
        } else {
            ax += 0.5 * ((-Cn * Vn[m + 1] - Sn * Wn[m + 1])
                         + f * (Cn * Vn[m - 1] + Sn * Wn[m - 1]));
            ay += 0.5 * ((-Cn * Wn[m + 1] + Sn * Vn[m + 1])
                         + f * (-Cn * Wn[m - 1] + Sn * Vn[m - 1]));
        }
        az += g * (-Cn * Vn[m] - Sn * Wn[m]);
    }
    out[0] = SH_K * ax;
    out[1] = SH_K * ay;
    out[2] = SH_K * az;
    return 0;
}

/* ------------------------------------------------------------------ */
/* 3. gravity_earth_moon's arithmetic tail and _earth_zonal_accel       */
/*    (APOLLO_VECTOR_KERNELS=1). The ephemeris, the precession matrix   */
/*    (a BLAS 3x3 product no scalar order reproduces) and the lunar     */
/*    figure stay in Python; everything between them is here.           */
/*                                                                      */
/*    np.linalg.norm(v) is sqrt(dot(v, v)). tests/fpenv_probe.py checks */
/*    per machine that the BLAS dot of a 3-vector equals the left-to-   */
/*    right sum x*x + y*y + z*z, as nrm3() computes it.                 */
/*    Every power goes through libm pow(), the call numpy's scalar      */
/*    power and CPython's float ** both make. It is called through a    */
/*    volatile pointer so the compiler cannot rewrite pow(x, 2.0) as    */
/*    x * x (equal only where libm's pow is correctly rounded).         */
/* ------------------------------------------------------------------ */

static double (*volatile POW)(double, double) = pow;

static int GV_LOADED = 0;
static double GV_MU_E, GV_RE, GV_KJ2, GV_MU_M, GV_MU_S, GV_ZR_MAX;
static double GV_J[7];                 /* EARTH_ZONALS[n], n = 2..6 */
static int GV_HAS[7];                  /* n in EARTH_ZONALS */

/* kj2 = 1.5 * J2 * MU_EARTH * R_EARTH**2, computed by the caller in Python's order. */
int fk_grav_load(double mu_e, double re, double kj2, double mu_m, double mu_s,
                 double zr_max, const double *J, const int *has)
{
    GV_MU_E = mu_e; GV_RE = re; GV_KJ2 = kj2; GV_MU_M = mu_m; GV_MU_S = mu_s;
    GV_ZR_MAX = zr_max;
    for (int n = 0; n < 7; n++) { GV_J[n] = J[n]; GV_HAS[n] = has[n]; }
    GV_LOADED = 1;
    return 0;
}

static double nrm3(double x, double y, double z)
{
    return sqrt(x * x + y * y + z * z);
}

/* _earth_zonal_accel._Uz */
static double uz(double x, double y, double z)
{
    const double rn = nrm3(x, y, z);
    const double u = z / rn;
    double p0 = 1.0, p1 = u, tot = 0.0;
    for (int n = 2; n < 7; n++) {
        const double pn = ((double)(2 * n - 1) * u * p1 - (double)(n - 1) * p0) / (double)n;
        p0 = p1; p1 = pn;
        if (GV_HAS[n])
            tot += GV_J[n] * POW(GV_RE / rn, (double)n) * pn;
    }
    return -(GV_MU_E / rn) * tot;
}

/* _earth_zonal_accel: central difference with h = 1 m; r + e adds 0.0 to the other
   components exactly as the numpy array sum does. */
static void zonal(double x, double y, double z, double *a)
{
    const double h = 1.0;
    const double r[3] = {x, y, z};
    for (int k = 0; k < 3; k++) {
        double p[3], m[3];
        for (int j = 0; j < 3; j++) {
            const double e = (j == k) ? h : 0.0;
            p[j] = r[j] + e;
            m[j] = r[j] - e;
        }
        a[k] = (uz(p[0], p[1], p[2]) - uz(m[0], m[1], m[2])) / (2.0 * h);
    }
}

int fk_zonal_accel(double x, double y, double z, double *out)
{
    if (!GV_LOADED) return -1;
    zonal(x, y, z, out);
    return 0;
}

/* gravity_earth_moon up to (not including) the lunar-figure term.
   flags: 1 = ENABLE_EARTH_HIGHER_ZONALS, 2 = ENABLE_SOLAR_GRAVITY.
   out[0..2] = the acceleration, out[3] = |mr - r| (the lunar-figure range gate). */
int fk_grav_tail(double rx, double ry, double rz, double mx, double my, double mz,
                 double sx, double sy, double sz, int flags, double *out)
{
    if (!GV_LOADED) return -1;
    const double rn = nrm3(rx, ry, rz);
    /* a = -MU_EARTH * r / rn**3 */
    const double rn3 = POW(rn, 3.0);
    double a0 = (-GV_MU_E * rx) / rn3;
    double a1 = (-GV_MU_E * ry) / rn3;
    double a2 = (-GV_MU_E * rz) / rn3;
    /* J2 */
    const double z2_r2 = POW(rz / rn, 2.0);
    const double f = GV_KJ2 / POW(rn, 5.0);
    a0 = a0 + f * (rx * (5.0 * z2_r2 - 1.0));
    a1 = a1 + f * (ry * (5.0 * z2_r2 - 1.0));
    a2 = a2 + f * (rz * (5.0 * z2_r2 - 3.0));
    /* J3..J6 */
    if ((flags & 1) && rn < GV_ZR_MAX) {
        double z[3];
        zonal(rx, ry, rz, z);
        a0 = a0 + z[0]; a1 = a1 + z[1]; a2 = a2 + z[2];
    }
    /* Moon: direct pull minus the pull on the Earth-centered origin */
    const double dx = mx - rx, dy = my - ry, dz = mz - rz;
    const double dn = nrm3(dx, dy, dz);
    const double dn3 = POW(dn, 3.0);
    a0 = a0 + (GV_MU_M * dx) / dn3;
    a1 = a1 + (GV_MU_M * dy) / dn3;
    a2 = a2 + (GV_MU_M * dz) / dn3;
    const double mn3 = POW(nrm3(mx, my, mz), 3.0);
    a0 = a0 - (GV_MU_M * mx) / mn3;
    a1 = a1 - (GV_MU_M * my) / mn3;
    a2 = a2 - (GV_MU_M * mz) / mn3;
    /* Sun, same tidal form */
    if (flags & 2) {
        const double ex = sx - rx, ey = sy - ry, ez = sz - rz;
        const double en3 = POW(nrm3(ex, ey, ez), 3.0);
        a0 = a0 + (GV_MU_S * ex) / en3;
        a1 = a1 + (GV_MU_S * ey) / en3;
        a2 = a2 + (GV_MU_S * ez) / en3;
        const double sn3 = POW(nrm3(sx, sy, sz), 3.0);
        a0 = a0 - (GV_MU_S * sx) / sn3;
        a1 = a1 - (GV_MU_S * sy) / sn3;
        a2 = a2 - (GV_MU_S * sz) / sn3;
    }
    out[0] = a0; out[1] = a1; out[2] = a2; out[3] = dn;
    return 0;
}
