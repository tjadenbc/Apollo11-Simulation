"""
Apollo 11 Mission Simulator — Real Physics
==========================================

Numerical integration of the Apollo 11 mission from Earth-orbit insertion
through Pacific splashdown. Every burn is a finite-thrust integration;
every coast is a 3-body integration. No impulsive approximations.

Initialization: Apollo 11 parking-orbit insertion state from the Saturn V
Flight Evaluation Report MPR-SAT-FE-69-9 and Orloff's "Apollo by the
Numbers" (NASA SP-2000-4029):
   - Insertion at GET = 11:39 (700 s after liftoff)
   - 185.9 x 183.2 km Earth parking orbit
   - 32.521° inclination, 72° launch azimuth from KSC LC-39A

Vehicle masses (Smithsonian + Orloff; SM_DRY re-derived from CSM-107
as-flown mass properties):
   - CSM "Columbia": ~28,795 kg at TLI (CM 5,557 + SM dry 4,825 + SPS prop 18,413)
   - LM "Eagle": 15,103 kg at TLI (descent 10,149 + ascent 4,954)
   - DPS thrust: 45,040 N max, 4,660 N min (10% throttle)
   - APS thrust: 15,700 N
   - SPS thrust: 91,200 N, Isp 314.5 s

Coordinate frame: Earth-Centered Inertial (ECI). Moon position: by default
(ENABLE_REAL_EPHEMERIS) a real July-1969 lunar ephemeris (truncated Meeus
series) with real launch-epoch GMST; the legacy idealized circular Moon
(384,400 km, fixed 28.4° inclination) is the flag-off fallback.

State vector: y = [x, y, z, vx, vy, vz, m]  (m, m/s, kg)

Run with `python3 -c "import apollo11; apollo11.main_parallel(...)"` (NEVER a
stdin heredoc — macOS spawn re-imports __main__). A single trial is ~350 s on
an M-series laptop core / ~625 s on the cluster's EPYC; use main_parallel or
the cluster pipeline (submit_mc.sh) for production runs. See CLAUDE.md.
"""
from __future__ import annotations
import os, time, json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# Physical constants
# ============================================================
G0       = 9.80665
MU_EARTH = 3.986004418e14
R_EARTH  = 6_378_137.0
J2       = 1.0826267e-3
OMEGA_E  = 7.292115e-5

MU_MOON  = 4.9048695e12
R_MOON   = 1_737_400.0
EM_DIST  = 384_399e3
MOON_INC = np.deg2rad(28.4)        # approx incl. wrt Earth equator (Jul 1969)
OMEGA_M  = np.sqrt(MU_EARTH / EM_DIST**3)
G_MOON   = MU_MOON / R_MOON**2

# ---- Real July-1969 lunar ephemeris + launch epoch (flag-gated) -------------
# When ENABLE_REAL_EPHEMERIS is True, moon_state() returns the ACTUAL Moon
# position from a truncated Meeus (Astronomical Algorithms ch.47) series at the
# real Apollo 11 epoch, and the Earth-rotation angle is anchored to the real
# GMST at launch — so the return-plane geometry and splashdown location reflect
# the true July-1969 sky rather than the idealized circular Moon (fixed 28.4 deg
# inclination, node at +X). When False, the legacy circular Moon and the
# theta = OMEGA_E*t convention are reproduced bit-for-bit (MOON_INC/EM_DIST/
# OMEGA_M are used ONLY in moon_state). Validated vs Meeus's 1992-04-12 worked
# example to <1 arcsec. moon_state is the single source of Moon position, so
# gating it propagates the real ephemeris through TLI/LOI/TEI/return.
# STATUS (splashdown correction complete): the nominal completes end-to-end and
# splashes in the CENTRAL PACIFIC at (18.4 N, 176.1 E) — ~1675 km from Apollo's
# 13.3 N / 169.15 W (right hemisphere & ocean), vs ~8000 km in the south Pacific
# for the idealized Moon. Three pieces made it work: (1) the real Meeus ephemeris
# + GMST anchoring (correct return plane -> northern hemisphere); (2) a TEI
# near-miss refinement (3-DOF trf least-squares driving the integrated return
# perigee into the entry corridor — the prior prograde-only TEI left perigee
# ~400 km high and reported "no return"); (3) a ~60 h pre-descent lunar loiter
# (LUNAR_PARK_COAST_S) matching Apollo's GET timeline, which sets the TEI lunar
# declination (latitude) and the splash GMST (longitude) AND fixed the entry
# geometry. 30-trial MC (outputs/apollo11_eph60): full-success 87%, TEI 0/30
# failures, splash median ~1632 km from Apollo, NO entry-g failures (the earlier
# coast=0 run had 5/30; the timeline match resolved them). Residual gap to exact
# Apollo coords: latitude geometry-capped at ~+18 (vs +13.3) and an erratic-vs-
# coast longitude — closing that needs a recovery-point entry steer, not the
# return geometry. DEFAULT ON: "flying like Apollo end-to-end" requires the
# real Moon (the idealized circular Moon cannot reproduce Apollo's return
# plane / splashdown). This is the fidelity-first production config; OFF
# reproduces the idealized model bit-for-bit. The DEFINITIVE headline is the
# 10,000-trial cluster run apollo11_final10000 (84.0%); see CLAUDE.md. Per-trial
# cost ~350 s laptop / ~625 s cluster.
ENABLE_REAL_EPHEMERIS = True
JD_LAUNCH = 2440419.0639   # 1969-07-16 13:32:00 UTC (Apollo 11 liftoff)
# Pre-descent lunar-orbit loiter (real-ephemeris only). Set to APOLLO'S REAL
# TIMELINE, not tuned for splash phasing: Apollo spent 26.7 h between LOI-1 and
# PDI (LOI-1 075:49:50 GET -> PDI 102:33:04). The flown two-burn LOI + DOI coast
# contributes an irreducible ~3.15 h, so 23.55 h of parking-orbit loiter gives the
# faithful 26.7 h LOI->PDI span. With launch continuity + Apollo return-timing
# the full mission is ~8.18 d (vs Apollo 8.16) and the nominal splashes in the
# western Pacific near (13.5 N, 146 E) — latitude within ~0.2 deg of Apollo's
# 13.3 N; the ~13 deg longitude residual is the remaining timeline/geometry gap.
# (An earlier 60 h loiter was a phasing hack, rejected for the faithful timeline.)
# Closing the splash-longitude residual belongs to those legs (and a recovery-
# point entry steer), NOT to re-tuning this constant.
LUNAR_PARK_COAST_S = 23.55 * 3600.0
# Post-rendezvous coast before the TEI opportunity scan (real-ephemeris only):
# Apollo spent ~10.9 h between ascent insertion (124:29 GET) and TEI ignition
# (135:23:42) on docking, LM jettison, and crew prep, firing TEI on a later rev.
# The sim otherwise scans for the best alignment IMMEDIATELY at rendezvous
# (leg ~4-10 h, trial-dependent). Coast ~9.7 h, then scan one orbit (~2.6 h):
# expected leg ~10.9 h, matching Apollo's GET timeline.
POST_RENDEZVOUS_COAST_S = 9.7 * 3600.0

# Principal periodic terms (Meeus 47.A/47.B): (D, M, Mp, F, lon_1e-6deg, dist_1e-3km)
_MOON_LR = [
    (0,0,1,0,6288774,-20905355),(2,0,-1,0,1274027,-3699111),(2,0,0,0,658314,-2955968),
    (0,0,2,0,213618,-569925),(0,1,0,0,-185116,48888),(0,0,0,2,-114332,-3149),
    (2,0,-2,0,58793,246158),(2,-1,-1,0,57066,-152138),(2,0,1,0,53322,-170733),
    (2,-1,0,0,45758,-204586),(0,1,-1,0,-40923,-129620),(1,0,0,0,-34720,108743),
    (0,1,1,0,-30383,104755),(2,0,0,-2,15327,10321),(0,0,1,2,-12528,0),
    (0,0,1,-2,10980,79661),(4,0,-1,0,10675,-34782),(0,0,3,0,10034,-23210),
    (4,0,-2,0,8548,-21636),(2,1,-1,0,-7888,24208),(2,1,0,0,-6766,30824),
    (1,0,-1,0,-5163,-8379),(1,1,0,0,4987,-16675),(2,-1,1,0,4036,-12831),
    (2,0,2,0,3994,-10445),(4,0,0,0,3861,-11650),(2,0,-3,0,3665,14403),
    (0,1,-2,0,-2689,-7003),(2,0,-1,2,-2602,0),(2,-1,-2,0,2390,10056),
    (1,0,1,0,-2348,6322),(2,-2,0,0,2236,-9884),(0,1,2,0,-2120,5751),
    (0,2,0,0,-2069,0),(2,-2,-1,0,2048,-4950),(2,0,1,-2,-1773,4130),
    (2,0,0,2,-1595,0),(4,-1,-1,0,1215,-3958),(0,0,2,2,-1110,0),
    (3,0,-1,0,-892,3258),(2,1,1,0,-810,2616),(4,-1,-2,0,759,-1897),
    (0,2,-1,0,-713,-2117),(2,2,-1,0,-700,2354),(2,1,-2,0,691,0),
    (2,-1,0,-2,596,0),(4,0,1,0,549,-1423),(0,0,4,0,537,-1117),
    (4,-1,0,0,520,-1571),(1,0,-2,0,-487,-1739),(2,1,0,-2,-399,0),
    (0,0,2,-2,-381,-4421),
]
_MOON_B = [
    (0,0,0,1,5128122),(0,0,1,1,280602),(0,0,1,-1,277693),(2,0,0,-1,173237),
    (2,0,-1,1,55413),(2,0,-1,-1,46271),(2,0,0,1,32573),(0,0,2,1,17198),
    (2,0,1,-1,9266),(0,0,2,-1,8822),(2,-1,0,-1,8216),(2,0,-2,-1,4324),
    (2,0,1,1,4200),(2,1,0,-1,-3359),(2,-1,-1,1,2463),(2,-1,0,1,2211),
    (2,-1,-1,-1,2065),(0,1,-1,-1,-1870),(4,0,-1,-1,1828),(0,1,0,1,-1794),
    (0,0,0,3,-1749),(0,1,-1,1,-1565),(1,0,0,1,-1491),(0,1,1,1,-1475),
    (0,1,1,-1,-1410),(0,1,0,-1,-1344),(1,0,0,-1,-1335),(0,0,3,1,1107),
    (4,0,0,-1,1021),(4,0,-1,1,833),
]

def _moon_eci_m(jde):
    """Geocentric Moon position in ECI (equatorial, mean equinox), METERS."""
    T = (jde - 2451545.0) / 36525.0
    d2r = np.deg2rad
    Lp = d2r(218.3164477 + 481267.88123421*T - 0.0015786*T**2 + T**3/538841 - T**4/65194000)
    D  = d2r(297.8501921 + 445267.1114034*T - 0.0018819*T**2 + T**3/545868 - T**4/113065000)
    M  = d2r(357.5291092 + 35999.0502909*T - 0.0001536*T**2 + T**3/24490000)
    Mp = d2r(134.9633964 + 477198.8675055*T + 0.0087414*T**2 + T**3/69699 - T**4/14712000)
    F  = d2r(93.2720950 + 483202.0175233*T - 0.0036539*T**2 - T**3/3526000 + T**4/863310000)
    A1 = d2r(119.75 + 131.849*T); A2 = d2r(53.09 + 479264.290*T); A3 = d2r(313.45 + 481266.484*T)
    E = 1 - 0.002516*T - 0.0000074*T**2
    sl = sr = sb = 0.0
    for d,m,mp,f,cl,cr in _MOON_LR:
        arg = d*D + m*M + mp*Mp + f*F; e = E**abs(m)
        sl += cl*e*np.sin(arg); sr += cr*e*np.cos(arg)
    for d,m,mp,f,cb in _MOON_B:
        sb += cb*(E**abs(m))*np.sin(d*D + m*M + mp*Mp + f*F)
    sl += 3958*np.sin(A1) + 1962*np.sin(Lp - F) + 318*np.sin(A2)
    sb += (-2235*np.sin(Lp) + 382*np.sin(A3) + 175*np.sin(A1-F) + 175*np.sin(A1+F)
           + 127*np.sin(Lp-Mp) - 115*np.sin(Lp+Mp))
    lon = np.deg2rad(np.rad2deg(Lp) + sl/1e6)
    lat = np.deg2rad(sb/1e6)
    dist = (385000.56 + sr/1000.0) * 1000.0   # meters
    eps = np.deg2rad(23.439291 - 0.0130042*T)
    xe = dist*np.cos(lat)*np.cos(lon); ye = dist*np.cos(lat)*np.sin(lon); ze = dist*np.sin(lat)
    return np.array([xe, ye*np.cos(eps)-ze*np.sin(eps), ye*np.sin(eps)+ze*np.cos(eps)])

def _gmst_rad(jd):
    """Greenwich Mean Sidereal Time (radians) at Julian date jd."""
    T = (jd - 2451545.0) / 36525.0
    g = 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T*T - T*T*T/38710000.0
    return np.deg2rad(g % 360.0)

# Greenwich sidereal angle at launch — anchors ECI(vernal-equinox) -> Earth-fixed.
# Zero in legacy mode so theta = OMEGA_E*t is reproduced exactly.
_GMST0 = _gmst_rad(JD_LAUNCH) if ENABLE_REAL_EPHEMERIS else 0.0

# Sites
LAUNCH_LAT, LAUNCH_LON = np.deg2rad(28.6082), np.deg2rad(-80.6041)
LAND_LAT,   LAND_LON   = np.deg2rad(0.67408), np.deg2rad(23.47297)
SPLASH_LAT, SPLASH_LON = np.deg2rad(13.30),   np.deg2rad(-169.15)

# ============================================================
# Vehicle parameters (Apollo 11 actual)
# ============================================================
# CSM
CSM_CM_MASS     = 5_557.0      # kg
# SM "dry" = everything aboard the SM except SPS propellant (structure, RCS
# propellant, helium, consumables). Derived from the Apollo 11 as-flown
# CSM-107 mass properties: CSM injected ~28,806 kg total − CM 5,557 − SPS
# propellant ~18,425 ≈ 4,825 kg. The previous 6,110 carried the stack
# ~1,285 kg heavy and inflated every SPS burn time ~5-7% vs the mission
# report (LOI-1 381 s vs Apollo's 357.5; TEI 171 vs 152).
CSM_SM_DRY      = 4_825.0
SPS_PROP_INIT   = 18_413.0
SPS_THRUST      = 91_200.0
SPS_ISP         = 314.5        # AJ10-137 vacuum spec

# LM
LM_DESC_TOTAL   = 10_149.0
LM_DESC_DRY     = 2_180.0
LM_DESC_PROP    = 8_248.0      # canonical (Smithsonian gives 8,165; we use Orloff)
LM_ASCT_TOTAL   = 4_954.0
LM_ASCT_DRY     = 2_180.0
LM_ASCT_PROP    = 2_353.0
DPS_THRUST_MAX  = 45_040.0     # Orloff
DPS_THRUST_MIN  = 4_660.0
DPS_ISP         = 311.0
APS_THRUST      = 15_700.0     # Smithsonian
APS_ISP         = 311.0

# S-IC first stage (5x F-1 engines)
S_IC_DRY        = 137_000.0       # kg, post-flight reported mass
S_IC_PROP       = 2_141_000.0     # kg propellant (RP-1 + LOX)
F1_THRUST_SL    = 6_770_000.0     # N per engine, sea level
F1_THRUST_VAC   = 7_770_000.0     # N per engine, vacuum
F1_ISP_SL       = 265.0           # s
F1_ISP_VAC      = 304.0           # s
S_IC_BURN_TIME  = 162.0           # s nominal

# S-II second stage (5x J-2 engines, vacuum only)
S_II_DRY        = 40_100.0        # kg
S_II_PROP       = 443_000.0       # kg (LH2 + LOX)
J2_THRUST_S2    = 1_033_100.0     # N per engine, vacuum
J2_ISP_S2       = 421.0           # s
S_II_BURN_TIME  = 384.0           # s nominal

# S-IVB (used only for TLI — we drop it after)
S_IVB_DRY       = 13_300.0
S_IVB_THRUST    = 1_033_100.0
S_IVB_ISP       = 421.0
S_IVB_PROP_TOTAL = 106_604.0      # kg total at S-IVB ignition
# S-IVB does two burns: ~150s for parking insertion, ~347s for TLI

# Initial mass at parking orbit insertion: full stack with S-IVB still attached
# S-IVB wet at this point includes propellant left for TLI burn
# Apollo 11 S-IVB had ~71,200 kg of propellant remaining at TLI start
S_IVB_PROP_AT_TLI = 71_200.0
S_IVB_RESERVE_KG  = 200.0   # flight-performance reserve floor for the TLI burn
STACK_AT_INSERTION = (CSM_CM_MASS + CSM_SM_DRY + SPS_PROP_INIT
                       + LM_DESC_TOTAL + LM_ASCT_TOTAL
                       + S_IVB_DRY + S_IVB_PROP_AT_TLI)

# Saturn V aerodynamics (rough — vehicle is ~10 m diameter, ~110 m tall)
SATV_AREA       = 80.0            # m² cross-section (10 m diameter)
SATV_CD         = 0.5             # subsonic, increases through transonic

# Kennedy Space Center launch pad (LC-39A): 28.608°N, 80.604°W
LAUNCH_LAT_DEG  = 28.608
LAUNCH_LON_DEG  = -80.604
# Launch azimuth (deg E of N). 72.48 = the SOLVED launch window at the real
# epoch: the azimuth whose parking-orbit plane, AFTER the 2.7 h coast to TLI
# ignition (Earth J2 regresses the node ~0.5 deg over that coast), contains
# the Moon-arrival direction at T_arr. Solving at insertion instead gives
# 72.97 and leaves a ~0.43 deg plane error at cutoff that costs ~80 m/s of
# out-of-plane trim. Apollo 11's actual flight azimuth was 72.058 — the model
# reproduces the historical window to ~0.4 deg (residual: the un-yawed
# Earth-rotation plane bias in the ascent steering, absorbed by solving the
# FINAL plane).
LAUNCH_AZIMUTH_DEG = 72.48

# Feature flag: trans-Earth midcourse correction chain (MCC-5/6/7).
# ON by default. phase_transearth_mcc is fully implemented: a per-opportunity
# along-track FPA-targeting solve (project entry FPA, bracket-and-bisect the
# trim if outside the corridor deadband) with the sourced ~0.15 m/s execution
# residual and SM-RCS propellant accounting (see its docstring). Disabling it
# falls back to a plain trans-Earth coast with no corrections.
ENABLE_TRANS_EARTH_MCC = True

# Feature flag: trans-lunar (outbound) midcourse correction chain MCC-1..4.
# ON by default. When enabled (the default), the single TLI+30h correction in
# run_mission is replaced by Apollo's scheduled outbound sequence, which gives
# several opportunities to null the accumulating perilune-targeting error and
# reduces the missed_lunar_soi losses; disabling it falls back to the single
# correction.
#
# Apollo 11 trans-lunar MCC schedule — SOURCED (NASA Apollo 11 Mission Overview;
# Apollo 11 Flight Journal Day 2). Apollo 11 had FOUR scheduled outbound MCC
# opportunities and executed only MCC-2 ("the launch had been so successful
# that the other three were not needed"); MCC-2 was a ~3-second SPS burn near
# GET ~26-27 h that reduced pericynthion from 175 nmi to 60 nmi. This 1-of-4
# execution pattern is exactly the deadband-gated waive behavior modeled here.
#   MCC-1: TLI + ~9 h        (first opportunity; waived on Apollo 11)
#   MCC-2: TLI + ~24-27 h    (the one Apollo 11 actually performed)
#   MCC-3: LOI - ~22 h       (mid-coast refinement; waived on Apollo 11)
#   MCC-4: LOI - ~5 h        (final perilune placement; waived on Apollo 11)
# The PERILUNE DEADBAND (km) that triggers/waives a burn is still an ESTIMATE —
# the schedule is sourced, but the specific perilune tolerance is not stated in
# these sources.
ENABLE_TRANS_LUNAR_MCC = True
TLMCC_TARGET_PERILUNE_KM = 94.0   # nominal LOI perilune target (matches sim)
TLMCC_PERILUNE_DEADBAND_KM = 5.0  # ESTIMATE: skip burn if within this tolerance
TLMCC_SCHEDULE_HRS = (9.0, 24.0)  # sourced offsets (h from TLI) for MCC-1/2;
                                  # MCC-3/4 are LOI-relative (resolved in-code)

# B-plane targeting for the trans-lunar MCC chain. When ON, each correction
# solves a 2-DOF (along-track + cross-track) burn that drives the lunar-approach
# B-plane intercept (B·T, B·R) to the nominal value, controlling BOTH the
# perilune altitude AND the approach-plane orientation (lunar-orbit inclination
# / node) — the standard Apollo target. When OFF, the chain falls back to the
# 1-DOF perilune-altitude solve, which leaves the approach plane uncontrolled.
ENABLE_BPLANE_TLMCC = True
TLMCC_BPLANE_DEADBAND_KM = 10.0   # waive a correction if the B-plane miss is
                                  # already within this (~5-6 km perilune,
                                  # comparable to the perilune deadband, while
                                  # also bounding the plane error; ESTIMATE).
_BPLANE_TARGET = None  # nominal lunar-approach B-plane target: (B·T_km, B·R_km).

# ------------------------------------------------------------------
# Apollo 11-specific descent failure modes.
# ------------------------------------------------------------------
# These model the documented near-miss events of the actual Apollo 11 landing.
# Each is OFF by default and gated by a flag so the baseline simulation is
# unchanged until we deliberately enable them and repeat the 1000-trial run.
# CRITICAL HONESTY NOTE: every numeric rate / penalty / recovery probability
# below is a RESEARCH_TODO placeholder — an engineering estimate, NOT an
# Apollo-sourced figure. The *existence and qualitative behavior* of these
# events is well-documented history; the *numbers* are not yet grounded.
ENABLE_DESCENT_FAILURE_MODES = True

# (1) Contact probes: 68-inch (1.73 m) probes below three of the four footpads
#     triggered the "contact light," prompting engine cutoff just above the
#     surface. Deterministic geometry, not a failure — sharpens touchdown-
#     velocity realism by cutting thrust ~1.7 m up rather than at 1 m. Not
#     gated by a probability.
CONTACT_PROBE_LENGTH_M = 1.73

# (2) Propellant-slosh sensor anomaly: slosh uncovered a propellant-quantity
#     sensor, firing the low-level warning light early. Effect is on WARNING
#     TIMING, not true propellant. Models the compressed crew decision window.
SLOSH_SENSOR_BIAS_MEAN_S = 18.0   # RESEARCH_TODO: A11 light fired ~18 s early
SLOSH_SENSOR_BIAS_STD_S  = 6.0    # RESEARCH_TODO: spread, placeholder

# (3) 1201/1202 AGC executive-overflow alarms (rendezvous-radar switch feeding
#     the computer extra cycles). On the real flight these were RECOVERABLE —
#     the AGC shed low-priority tasks correctly and MCC called "go." Model as a
#     stochastic event with a small guidance penalty; escalates to abort only
#     when co-occurring with an already-marginal state.
PROB_1202_ALARM            = 0.15  # occurrence prob — ESTIMATE. On Apollo 11
                                   # the alarm condition (rendezvous-radar switch
                                   # stealing ~13% of AGC duty cycle) produced 5
                                   # alarms; it was a procedural/design issue, not
                                   # a random per-flight hazard, so this rate is a
                                   # modeling stand-in, not an observed frequency.
PROB_1202_RECOVERS         = 0.97  # recovery prob — anchored to the fact that the
                                   # AGC restart was designed/tested to recover and
                                   # DID on Apollo 11 (all 5 alarms recovered);
                                   # high recovery is correct, exact value an estimate.

# (4) Landing-radar dropout: brief loss of radar altitude/velocity updates,
#     degrading the navigation solution for a few seconds. Matters only if it
#     lands in the terminal phase.
PROB_LR_DROPOUT            = 0.10  # RESEARCH_TODO: per-descent occurrence prob
LR_DROPOUT_DURATION_S      = 8.0   # RESEARCH_TODO: placeholder dropout length

# ------------------------------------------------------------------
# Realistic descent chain: parking orbit + DOI burn + manual-flying reserve.
# ------------------------------------------------------------------
# When OFF, the LM brakes directly from the post-LOI perilune on the unphysical
# ~5 x 413 km capture orbit (the documented legacy behavior). When ON, this
# reproduces the real Apollo architecture:
#   * The CSM is parked in a near-circular ~100 km lunar orbit (representing the
#     LOI-1 capture + short LOI-2 circularization; a single ~390 s finite-thrust
#     SPS burn cannot circularize, so LOI-2 is modeled as a small impulsive trim
#     — physically justified: the real LOI-2 was a ~17 s burn).
#   * The LM performs a discrete DOI (Descent Orbit Insertion) burn — a small
#     retrograde impulse (~19 m/s, charged to the DPS) that lowers ONLY the LM's
#     perilune to ~15 km. PDI then begins from that controlled ~15 km / ~1695 m/s
#     state, matching Apollo, instead of a 5 km / ~1763 m/s eccentric start.
#   * A manual-flying / landing-redesignation fuel penalty: extra low-altitude
#     powered-flight time (Apollo 11's boulder-field overfly was the extreme
#     case) charged at the hover rate. This is the dominant reason the real
#     descent margin was tight, and it keeps descent-fuel exhaustion a genuine
#     (rare-tail) risk rather than the artifact it was on the eccentric orbit.
ENABLE_DOI = True
PARKING_ORBIT_ALT_KM = 100.0   # fallback constructed parking orbit (if two-burn LOI solve fails)
DOI_PERILUNE_ALT_KM  = 15.0    # LM perilune after DOI, where PDI fires (~50,000 ft)
LOI2_CIRC_DV_MS      = 48.0    # impulsive LOI-2 dV for the FALLBACK path only (flown path solves its own)
# Two-burn FLOWN LOI (ENABLE_DOI): LOI-1 ignites this fraction of its burn
# duration before approach-perilune so the frozen-attitude burn arc is centred on
# perilune and PRESERVES it (~95 km) while lowering apolune. Tuned: 0.35 -> flown
# ~98 x 100 km near-circular orbit (Apollo ~111 km), LOI-1 ~898 / LOI-2 ~45 m/s
# (Apollo 889 / 48.5). Too small drags perilune low; too large fails the capture
# solve (safely falls back to the constructed orbit).
LOI1_LEAD_FRAC       = 0.35
# Manual-flying hover penalty ~ gamma(shape, scale), seconds. Fat-tailed by
# design: median ~13 s (most sites need only minor terminal repositioning), ~46 s
# at the 85th percentile (an Apollo-11-scale boulder-field overfly), and a ~2%
# tail past ~90 s that can run the tanks dry. Grounded in "most landing sites are
# benign, occasionally one forces a long hazard-avoidance overfly," NOT dialed to
# a target failure rate. ESTIMATE — no per-site hazard statistics exist for 1969.
MANUAL_FLYING_SHAPE  = 0.9
MANUAL_FLYING_SCALE  = 22.0
# Efficient-braking guidance gains (used only when ENABLE_DOI is on; tuned so the
# nominal lands from the ~15 km PDI with an Apollo-like margin).
BRAKE_SINK_MAX_MS    = 90.0    # max allowed fall rate during braking (m/s)
BRAKE_SINK_DIV       = 100.0   # fall-rate target = -min(SINK_MAX, alt/DIV)
BRAKE_VERT_GAIN      = 0.12    # gain limiting fall toward the target (never thrusts down)

# ------------------------------------------------------------------
# Lunar non-spherical gravity + mascon landing dispersion.
# ------------------------------------------------------------------
# When enabled, the Moon-centered phases (lunar orbit, descent, ascent) use the
# real large-scale lunar figure (C20 + C22; see lunar_nonspherical_accel) on top
# of the point-mass field, AND the powered descent receives a calibrated
# mascon-induced downrange perturbation. OFF by default so the baseline is
# unchanged. The two pieces are deliberately separate and honestly labeled:
#   * C20/C22 = SOURCED large-scale figure (real physics, perturbs orbits) — but
#     does NOT resolve localized mascons.
#   * The mascon descent dispersion below = a CALIBRATED proxy for the localized
#     mascon anomalies the low-degree field cannot capture, anchored to the
#     documented Apollo experience: Apollo 11 landed LONG (downrange), with the
#     gravity "lumpiness" exacerbating it (NASA/Space.com, NTRS A11 landing
#     reconstruction); Apollo 12 then landed pinpoint once the model was
#     improved with A11 tracking. Magnitudes here are ESTIMATES anchored to that
#     qualitative record (the sources do not give a clean km figure for the
#     mascon-attributable downrange error), so they are calibration, not data.
ENABLE_LUNAR_HARMONICS = True
MASCON_DOWNRANGE_BIAS_M = 4000.0   # ESTIMATE: mean downrange (long) bias, ~few km
MASCON_DOWNRANGE_STD_M  = 2000.0   # ESTIMATE: trial-to-trial spread

# ------------------------------------------------------------------
# LM <-> CSM rendezvous & docking failure mode (LM-crew exposure).
# ------------------------------------------------------------------
# Previously rendezvous was assumed always successful. This models it as a
# physical maneuver: after ascent the LM must spend delta-V to match the CSM's
# orbit (dominated by any orbital-plane mismatch from ascent dispersions, plus
# altitude/phasing matching), checked against the LM ascent stage's remaining
# propellant, followed by a docking-latch step. Exposure is LM-crew-only:
# Collins is safe in the CSM, and a rendezvous failure strands Armstrong &
# Aldrin in lunar orbit (within sight of the CSM but unable to close/dock).
# OFF by default and gated with the other descent/ascent failure modes.
# HONESTY NOTE: the nominal rendezvous ΔV (~30 m/s) and the RCS Isp (290 s) are
# now SOURCED (see citations on the constants below). The usable RCS ΔV budget
# is an estimate anchored to the documented fact that RCS propellant was a
# binding constraint (TN D-7388). The docking-latch failure probability remains
# an ESTIMATE — no sourced rate exists. Phase 8b inserts the ascent INTO the
# CSM plane (timed liftoff), so the LM-vs-CSM plane angle is physical (driven by
# ascent yaw-steering dispersion); the plane-change delta-V is charged directly
# and a large mismatch can drive a rendezvous_insufficient_propellant failure.
# LM <-> CSM rendezvous & docking. Values below are now grounded in primary
# sources where available:
#  - LM ascent-stage RCS: 16 x 445 N, NTO/Aerozine-50, Isp 290 s
#    (Grumman LM specs / braeunig.us). Confirms MCC_RCS_ISP_S = 290.
#  - Nominal coelliptic rendezvous total ~30 m/s (Sostaric, AAS lunar ascent &
#    rendezvous trajectory design; consistent with Apollo 16 crew debrief:
#    insertion residuals ~2 ft/s, largest midcourse 0.9 ft/s, braking 29 ft/s).
#  - RCS ΔV was a genuine binding constraint: NASA TN D-7388 (Apollo rendezvous
#    experience report) states the CSM parking orbit was lowered specifically
#    because of limited LM RCS and APS propellant. So a modest usable rendezvous
#    RCS budget is realistic (estimate, anchored to that constraint).
#  - Docking unrecovered-failure probability: SOURCED DECOMPOSITION from the
#    program record (replaces the former flat 1% estimate). The probe-and-
#    drogue mechanism flew ~21 docking events across 13 missions (1969-75)
#    with TWO severe capture anomalies — Apollo 14 (five failed capture
#    attempts; succeeded on the 6th by thrusting against the drogue with the
#    probe retracted, i.e. the hard-dock workaround; NASA SMA anomaly report
#    "Failure to Achieve Docking Probe Capture Latch Engagement"; Apollo 14
#    Mission Report ch.14) and Skylab 2 (capture failures, recovered via
#    interlock bypass) — and ZERO unrecovered failures. The recovery path is
#    mechanically INDEPENDENT of the failed subsystem (the 12 main ring
#    latches never failed in the program), so:
#      P(capture anomaly)              = 2/21  ~ 0.095   (empirical)
#      P(workaround fails | anomaly)   ~ 0.10            (estimate: 0-for-2
#                                          observed + mechanical independence)
#      P(unrecovered, per docking)     ~ 0.0095
#    Applied at BOTH docking events per mission — transposition & docking
#    after TLI (failure aborts the landing; crew returns on a healthy CSM)
#    and the ascent-rendezvous docking (failure falls to the documented
#    suited-EVA crew-transfer contingency). See crew_survival.py for the
#    per-event survival consequences.
LM_RCS_DV_BUDGET_MS    = 60.0   # usable RCS dV for rendezvous (estimate; TN D-7388
                                # confirms RCS propellant was a binding limit)
RENDEZVOUS_NOMINAL_DV_MS = 30.0 # sourced: nominal coelliptic rendezvous ~30 m/s
PROB_DOCKING_FAILURE   = 0.0095 # per-docking unrecovered failure (sourced, above)

# ------------------------------------------------------------------
# SM in-flight SYSTEMS failure (the Apollo 13 mode) — SOURCED.
# ------------------------------------------------------------------
# The CSM service module flew 15 crewed missions (Apollo 7-17, Skylab 2-4,
# ASTP). SERIOUS in-flight SM systems anomalies: 4 —
#   Apollo 13 (cryo O2 tank 2 explosion; mission lost, crew saved via the
#     LM lifeboat — NASA Apollo 13 Review Board),
#   Apollo 15 (SPS Delta-V Thrust switch short; worked around with revised
#     burn procedures — A15 Mission Report ch.14),
#   Apollo 16 (SPS TVC secondary yaw oscillation; landing delayed ~6 h and
#     nearly aborted — A16 Mission Report),
#   Skylab 3 (TWO SM RCS quad leaks; rescue mission prepared, completed
#     normally).
# => serious-anomaly rate ~4/15 ~ 27%/mission; CATASTROPHIC class
# (mission-ending regardless of workaround) 1/15 ~ 6.7%/mission. The three
# recoverable anomalies cost delays/procedures, not missions, and are not
# separately modeled. The catastrophic event is drawn per mission and
# strikes at a uniform fraction of the reference timeline (cryo/fuel-cell
# duty is roughly continuous); the consequence depends on LM availability
# at that moment (see run_mission checks + crew_survival.py).
ENABLE_SM_SYSTEMS_FAILURES = True
PROB_SM_CATASTROPHIC   = 1.0 / 15.0   # empirical: Apollo 13, 1 of 15 flights
SM_MISSION_REF_DURATION_S = 195.0 * 3600.0   # reference mission length

# ------------------------------------------------------------------
# Surface operations failure modes — SOURCED (previously a pure time-advance).
# ------------------------------------------------------------------
# (a) EVA suit/PLSS: ZERO failures in 28 man-EVAs on the lunar surface
#     (Apollo 11-17; the OPS emergency backup was never once used — Apollo
#     Experience Report: Development of the EMU, NTRS 19760003073). For
#     Apollo 11's single short EVA (2x ~2.5 h, within ~60 m of the LM, no
#     cumulative dust degradation), terminal-anomaly rate ~0.5%/man-EVA
#     (Jeffreys-class on 0/28) x 2 crew, with the OPS + immediate-repress
#     path making most anomalies survivable EVA aborts; the FATAL branch
#     (~10% of anomalies) is ~0.1%/mission.
# (b) LM surface electrical/switchgear: 1 serious incident in 6 landings —
#     Apollo 11's own snapped ascent-engine arming circuit breaker (closed
#     with a felt-tip pen). ~17% anomaly rate, with ground-procedure
#     workarounds demonstrated; unrecoverable fraction ~5% -> ~0.8%/mission
#     (stranded: ascent stage cannot be armed).
# (c) Touchdown tip-over/structural: 0 in 6 landings against a ~12 deg
#     static stability spec (Apollo 15 landed at ~11 deg, the worst case);
#     ~0.5%/landing estimate anchored to the spec margin and the record.
PROB_EVA_SUIT_FATALITY  = 0.001   # per mission (see (a))
PROB_LM_SURFACE_ELEC    = 0.17 * 0.05   # anomaly x unrecoverable (see (b))
PROB_LM_TIPOVER         = 0.005   # per landing (see (c))

# ------------------------------------------------------------------
# OFFLINE-OPTIMIZED 6.5-g ENTRY REFERENCE PROFILE (resolves the peak-g
# residual the five online-HUNTEST variants could not — see the
# ENABLE_HUNTEST_PROFILE corpus). Generated by /tmp/ref_profile_opt.py:
# a bank-vs-velocity profile optimized OFFLINE against the FINE-fidelity
# landing prediction from the nominal entry interface (no coarse-prediction
# bias anywhere), peaking at 6.78 g with a ~108 km open-loop residual that
# the subcircular predictor-corrector trims in flight. Flown OPEN-LOOP while
# supercircular, only for deliveries inside the FPA band (out-of-band steep
# tails keep the legacy g-aware PC + 9.5-g guard).
# CORRIDOR-VALIDATED NOT-READY (6th variant, 3rd architecture — keep OFF):
# the open-loop profile never engaged at its design point (FPA band
# mis-centered vs the air-relative entry FPA), engaged out-of-band instead,
# and produced 5.62 g (benign) at dFPA -0.2 but a SKIP-OUT FAILURE at -0.4.
# Off-design fragility is intrinsic to open-loop profiles; closing the
# 8.5-vs-6.5 g residual requires the full Apollo apparatus (an FPA-indexed
# family of reference profiles + closed-loop drag-vs-velocity tracking with
# derived gains). The residual stays a documented limitation; legacy
# guidance (1-15 km, 6.8-9.4 g across the corridor) remains production.
ENABLE_REF_PROFILE_ENTRY = False
ENTRY_REF_VGRID  = [7600.0, 8200.0, 8800.0, 9400.0, 10000.0, 10600.0, 11200.0]
ENTRY_REF_BANKS  = [120.0, 120.0, 120.0, 42.51, 0.0, 8.6, 0.0]
ENTRY_REF_FPA_CENTER = -6.51
ENTRY_REF_FPA_BAND   = 0.15
APS_EXHAUST_VELOCITY   = None   # (computed from APS_ISP at use site)

# Reentry
CM_AREA = 12.0
CM_CD   = 1.2
CM_LD   = 0.30                  # Apollo CM lift-to-drag from offset CG
# Splashdown targeting. NOTE: the simulation's idealized TLI/TEI geometry does
# not reproduce Apollo 11's specific return plane, so the CM naturally arrives
# at an entry interface ~30 deg of latitude away from Apollo 11's actual Pacific
# splashdown (13.3N, 169.15W). The CM's L/D=0.30 gives only ~hundreds of km of
# crossrange authority, so steering to Apollo 11's literal coordinates is
# physically impossible and measuring "miss" against them is apples-to-oranges.
# We therefore target the ACHIEVABLE nominal landing point (the point the
# unperturbed trajectory reaches) and measure splashdown accuracy as DISPERSION
# of perturbed trials around it — which is the meaningful targeting metric.
# This constant is the nominal landing point for the all-systems configuration,
# determined from an unperturbed run; the entry guidance steers to it and
# splash_miss_km is computed relative to it. So splash_miss_km ~= the TEI-
# targeting dispersion (distance from the nominal), NOT an offset from a separate
# recovery point. IMPORTANT: this must be re-derived whenever the return geometry
# changes — the B-plane TLMCC + DOI work moved the nominal splashdown, and a
# stale value here inflates splash_miss_km into a meaningless ~8000 km "miss".
# Flag-aware: each config's SPLASH_TARGET is its own nominal, keeping
# splash_miss_km ~= the targeting dispersion. The ACTIVE value in the default
# config is set by the ENABLE_LAUNCH_CONTINUITY override block further below
# (western Pacific, ~13.48 N / 146.22 E — latitude within ~0.2 deg of Apollo's
# 13.3 N). The real-ephemeris-only value set in the `if ENABLE_REAL_EPHEMERIS`
# block just below is a SUPERSEDED intermediate (it predates launch continuity)
# and is overwritten when continuity is on. NOTE: a target must be derived in a
# FRESH process — a prior run in the same process leaves a captured _EI_TARGET
# that contaminates the nominal. Idealized value is the south-Pacific
# idealized-Moon nominal.
if ENABLE_REAL_EPHEMERIS:
    # GUIDED-ENTRY recovery target: the guided nominal's own landing point at
    # the SHORT (direct-range) end of the corridor. Apollo flew the same design
    # choice (EI-to-splash ~2,780 km, dug-in early): short-range profiles are
    # intrinsically dispersion-insensitive — the guided corridor sweep lands
    # within ~60 km of this point across +/-0.4 deg of entry-FPA dispersion
    # (vs thousands of km of skip-range scatter when targeting the lofted
    # long-skip point). With guidance ON, splash_miss_km is a true
    # guidance-accuracy metric against this point.
    # Re-derived after the navigation-robustness fixes (the TEI corridor gate
    # commits a better-scoring rev-1 candidate, moving the return geometry).
    SPLASH_TARGET_LAT_DEG = 37.311
    SPLASH_TARGET_LON_DEG = -55.546
else:
    SPLASH_TARGET_LAT_DEG = -26.556
    SPLASH_TARGET_LON_DEG = -103.946
# Closed-loop guided entry. When ON: (1) a predictive lift-vector-up GUARD
# rolls up when the predicted dip peak nears the 9.5 g guidance limit — this
# is a guard, NOT a 6.5 g hold; the nominal peaks ~8.6 g (the short
# dispersion-robust direct profile trades g for accuracy, and the steepest
# deliveries reach 9.5-12 g; 12 g is the structural-failure threshold);
# (2) within that envelope, a
# NUMERICAL PREDICTOR-CORRECTOR root-solves the bank magnitude on the full
# predicted trajectory to landing (_predict_landing/_solve_bank — the
# prediction includes the skip, so the bank controls skip range and landing
# point together; the numerical analogue of Apollo's HUNTEST/UPCONTROL
# exit-velocity targeting, which is what the earlier analytic final-phase
# attempts lacked — they had no authority over the dip/skip where range is
# actually set); (3) crossrange via velocity-scaled bank reversals.
# The recovery target is SPLASH_TARGET, making splash_miss_km a true guidance
# metric when this flag is on. OFF flies the legacy unguided fixed profile
# (peak ~9.6 g, no steering) bit-identically.
# Validated (corridor sweep 2-4 km miss across +/-0.4 deg FPA at 7.1-9.5 g;
# end-to-end nominal miss 1.9 km at 8.9 g): default ON. Reproducing pre-guided
# runs (e.g. the idealized apollo11_doi1000 headline) requires setting False.
ENABLE_SKIP_ENTRY_GUIDANCE = True
# EXPERIMENTAL (validated NOT-READY — keep OFF): HUNTEST-style multi-regime
# bank profile inside the predictor-corrector. Corridor-sweep corpus on the
# continuity EI state (dFPA -0.4..+0.4):
#   legacy constant-bank PC:  miss 0.8-2.4 km, peak 6.4-9.3 g (nominal 8.15)
#   two-segment (dip+range):  peak unchanged ~8.05, miss 45-544 km — a
#     constant lift-up dip bank floats the supercircular phase long;
#   three-regime, FIXED 5-g drag-reference: peak 6.50 g at nominal — the
#     LOAD goal achieved — but miss 500-2,500 km (reference blind to
#     range-to-go);
#   three-regime, RANGE-AWARE drag-reference (signed-miss root each update),
#     re-deciding law each solve: nominal 7.32 g / 99 km; steep +0.2 case
#     7.65 g / 1.5 km (vs legacy 9.29 g) — the best variant;
#   + sticky law commitment / dref<->bank2 consistency iteration: WORSE
#     (1,372 km nominal). Root cause isolated: with a FEEDBACK LAW inside
#     the landing prediction, the coarse-fidelity predictions the
#     supercircular solver can afford diverge systematically from the flown
#     fine-integration trajectory — the drag-reference is rooted on a biased
#     model, and the re-deciding variant only worked because mid-flight
#     reversions to legacy kept rescuing the range. Fixing it needs
#     fine-fidelity predictions throughout (~3x entry cost, unusable in MC)
#     or an analytic drag-energy range law (real HUNTEST's closed forms);
#   ANALYTIC drag-energy reference (closed-form D_ref from range-to-go,
#     receding horizon, no predictor in the supercircular loop): the commit
#     gate (scored on the coarse prediction) falls back to legacy across the
#     whole core corridor (peaks identical 6.84-8.90), and at the +0.4 steep
#     edge where it DID commit it flew 2,384 km off at 9.22 g — the coarse
#     model mis-scores the law in both directions. Five variants tested;
#     none flight-worthy without fine predictions (~3x cost) or a true
#     reference-trajectory architecture.
# The honest residual when OFF: nominal peak ~8.6 g vs Apollo's as-flown
# 6.5 g, with Apollo-grade landing accuracy. The g residual does not affect
# survival outcomes (the 9.5-g guard / 12-g structural bound govern those).
ENABLE_HUNTEST_PROFILE = False
# Derivation/testing aid: when set (degrees), the PC is bypassed and this fixed
# bank flies wherever the g-limiter allows — used to find the guided profile's
# natural landing point when re-deriving the recovery target. None in production.
ENTRY_PC_NEUTRAL_BANK = None

# Authentic Saturn V ascent guidance: closed-loop linear-tangent steering
# (Lawden's bilinear-tangent law, the core of MSFC's Iterative Guidance Mode)
# for the S-IVB parking-orbit insertion, replacing the open-loop altitude-rate
# hold + fixed-time cutoff that under-inserts into an eccentric 118 x 638 km
# orbit. When ON, tan(pitch) varies linearly in time and the two coefficients
# are solved so the burn reaches the target radius at circular speed with zero
# flight-path angle (cutoff at FPA = 0). Gate: ~185 km circular parking orbit.
ENABLE_IGM_ASCENT = True

# ============================================================
# Physics
# ============================================================
def moon_state(t):
    """Moon position and velocity in ECI [m, m/s] at time t."""
    if ENABLE_REAL_EPHEMERIS:
        jde = JD_LAUNCH + t / 86400.0
        r = _moon_eci_m(jde)
        dt = 60.0  # s; velocity by central difference (Moon moves ~13 deg/day)
        v = (_moon_eci_m(JD_LAUNCH + (t + dt) / 86400.0)
             - _moon_eci_m(JD_LAUNCH + (t - dt) / 86400.0)) / (2 * dt)
        return r, v
    theta = OMEGA_M * t
    ci, si = np.cos(MOON_INC), np.sin(MOON_INC)
    r = EM_DIST * np.array([np.cos(theta),
                             np.sin(theta) * ci,
                             np.sin(theta) * si])
    v = EM_DIST * OMEGA_M * np.array([-np.sin(theta),
                                       np.cos(theta) * ci,
                                       np.cos(theta) * si])
    return r, v


def gravity_earth_moon(r, t):
    """Earth (with J2) + Moon point-mass acceleration in ECI."""
    rn = np.linalg.norm(r)
    # Earth point mass
    a = -MU_EARTH * r / rn**3
    # J2
    z2_r2 = (r[2] / rn)**2
    f = 1.5 * J2 * MU_EARTH * R_EARTH**2 / rn**5
    a = a + f * np.array([r[0]*(5*z2_r2-1), r[1]*(5*z2_r2-1), r[2]*(5*z2_r2-3)])
    # Moon (with frame correction so we're computing total inertial accel)
    mr, _ = moon_state(t)
    dr = mr - r
    dr_norm = np.linalg.norm(dr)
    a = a + MU_MOON * dr / dr_norm**3
    a = a - MU_MOON * mr / np.linalg.norm(mr)**3
    # Moon non-spherical figure (C20+C22): only meaningful near the Moon (falls
    # as ~1/r^4). Reuse the Moon-relative distance just computed to SKIP the call
    # entirely when far away — exactly equivalent to the function's own early-out
    # (it returned zeros beyond this range), but avoids the per-call overhead on
    # the long coasts / MCC projection arcs where this is evaluated millions of
    # times. -dr = r - mr is the Moon-relative position.
    if dr_norm < 5.0e7:
        a = a + lunar_nonspherical_accel(-dr, t)
    return a


# Lunar large-scale non-spherical gravity: degree-2 zonal (C20 = -J2) and
# sectoral (C22) harmonics. SOURCED coefficients (Konopliv et al. / GRAIL-era,
# reference radius R = 1738 km; corroborated by Williams & Dickey and the
# GRAIL degree-2 determinations):
#   J2_moon  = -C20 = 2.0323e-4   (oblateness)
#   C22_moon =        2.2382e-5   (equatorial ellipticity; for the Moon this is
#                                  only ~1 order of magnitude below J2, unlike
#                                  Earth where it is negligible — so it matters)
# IMPORTANT HONESTY NOTE: C20 and C22 are the Moon's LARGE-SCALE FIGURE, NOT the
# localized mascons. The mascon anomalies under the maria live in HIGH-degree
# harmonics; published work shows even degree-8 fields cannot model mascons
# (core.ac.uk/works/24860856). So this term adds real, sourced large-scale
# non-sphericity (orbital-plane/periapsis perturbation, eccentricity drift) but
# does NOT reproduce the localized mascon "pull". The localized mascon effect on
# the landing point is handled separately as a calibrated downrange dispersion
# (see the mascon descent perturbation), anchored to Apollo's documented long
# landing rather than emerging from these two coefficients.
J2_MOON   = 2.0323e-4
C22_MOON  = 2.2382e-5
R_MOON_GRAV = 1.738e6   # reference radius matching the coefficient convention

# ------------------------------------------------------------------
# Higher-degree lunar gravity (GRAIL GRGM1200A, truncated).
# ------------------------------------------------------------------
# When ON, lunar_nonspherical_accel evaluates a real degree-LUNAR_SH_DEGREE
# spherical-harmonic field (coefficients in lunar_gravity_coeffs.py, downloaded
# from NASA PDS; degree-2 terms match J2_MOON/C22_MOON to 0.003%) instead of the
# degree-2 closed form, so the lunar parking orbit EVOLVES physically (perilune/
# apolune drift ~5-20 km/day at ~100 km altitude — the Apollo-documented
# behavior; LOI-2 was deliberately left elliptical to let mascon-driven drift
# pull the orbit toward 60 nm circular). HONESTY: deliberately NOT named a
# "mascon field" — degree 8-12 captures the large-scale figure and the real
# orbit evolution, but localized maria mascons live at degree >= 50, so the
# calibrated mascon landing-dispersion proxy (MASCON_DOWNRANGE_*) remains in
# use. The Moon-fixed frame is the existing synchronous-lock approximation
# (x toward instantaneous Earth): pole off by ~6.7 deg from the true spin axis
# and ~7 deg optical librations are ignored — tolerable for the zonal-dominated
# drift modeled here, NOT adequate for future mascon-grade landing dynamics
# (that needs an IAU/WGCCRE orientation model first). When OFF, the legacy
# degree-2 closed form runs untouched (bit-identical).
# Validated (test_sh_field.py all-pass; flag-off bit-compat confirmed; field-on
# nominal flies end-to-end with the orbit evolving ~+5.7 km/day): default ON.
ENABLE_LUNAR_SH_FIELD = True
LUNAR_SH_DEGREE = 8   # converged for 1-day stays (deg 12 changes 26.7 h drift <10%);
                      # the unnormalized recursion below is safe only to deg ~25


def _build_lunar_sh_terms(nmax):
    """Denormalize the embedded GRGM1200A 4-pi-normalized coefficients to
    degree nmax and flatten to a term list [(n, m, C, S, f, g)] for the
    Cunningham recursion (f, g are the acceleration-formula integer factors).
    Skips zero terms. Asserts degree-2 consistency with the sourced constants."""
    import math as _m
    from lunar_gravity_coeffs import C_NORM, S_NORM
    terms = []
    for n in range(2, nmax + 1):
        for m in range(0, n + 1):
            Nnm = _m.sqrt((2.0 if m else 1.0) * (2 * n + 1)
                          * _m.factorial(n - m) / _m.factorial(n + m))
            Cn = Nnm * C_NORM[n][m]
            Sn = Nnm * S_NORM[n][m]
            if abs(Cn) < 1e-15 and abs(Sn) < 1e-15:
                continue
            terms.append((n, m, Cn, Sn,
                          float((n - m + 2) * (n - m + 1)), float(n - m + 1)))
    # Tie the table to the already-sourced degree-2 constants (catches any
    # normalization/sign/tide-convention mix-up the moment the module imports).
    _c20 = next(t for t in terms if t[0] == 2 and t[1] == 0)
    _c22 = next(t for t in terms if t[0] == 2 and t[1] == 2)
    assert abs(-_c20[2] - J2_MOON) < 0.01 * J2_MOON, "SH table C20 vs J2_MOON mismatch"
    assert abs(_c22[2] - C22_MOON) < 0.01 * C22_MOON, "SH table C22 vs C22_MOON mismatch"
    return terms


_LUNAR_SH_TERMS = _build_lunar_sh_terms(LUNAR_SH_DEGREE)
_LUNAR_SH_NMAX = LUNAR_SH_DEGREE


def _lunar_sh_accel_body(x, y, z):
    """Perturbing acceleration (m/s^2, body frame) of the truncated lunar field
    via the Cunningham V/W recursion on UNNORMALIZED coefficients — Cartesian
    in/out, singularity-free at the poles. Sums n>=2 only (a perturbation on
    top of the central -MU_MOON*p/|p|^3 term). Pure-Python hot path: measured
    ~3x faster than a numpy-fancy-indexed kernel at this size (~50 cells)."""
    import math as _m
    R = R_MOON_GRAV
    r2 = x * x + y * y + z * z
    rho = R * R / r2
    x0 = R * x / r2
    y0 = R * y / r2
    z0 = R * z / r2
    size = _LUNAR_SH_NMAX + 2          # degree-n accel needs V/W at n+1
    V = [[0.0] * size for _ in range(size)]
    W = [[0.0] * size for _ in range(size)]
    V[0][0] = R / _m.sqrt(r2)
    for m in range(size):
        if m > 0:
            V[m][m] = (2 * m - 1) * (x0 * V[m - 1][m - 1] - y0 * W[m - 1][m - 1])
            W[m][m] = (2 * m - 1) * (x0 * W[m - 1][m - 1] + y0 * V[m - 1][m - 1])
        if m + 1 < size:
            V[m + 1][m] = (2 * m + 1) * z0 * V[m][m]
            W[m + 1][m] = (2 * m + 1) * z0 * W[m][m]
        for n in range(m + 2, size):
            V[n][m] = ((2 * n - 1) * z0 * V[n - 1][m]
                       - (n + m - 1) * rho * V[n - 2][m]) / (n - m)
            W[n][m] = ((2 * n - 1) * z0 * W[n - 1][m]
                       - (n + m - 1) * rho * W[n - 2][m]) / (n - m)
    ax = ay = az = 0.0
    for n, m, Cn, Sn, f, g in _LUNAR_SH_TERMS:
        Vn = V[n + 1]
        Wn = W[n + 1]
        if m == 0:
            ax += -Cn * Vn[1]
            ay += -Cn * Wn[1]
        else:
            ax += 0.5 * ((-Cn * Vn[m + 1] - Sn * Wn[m + 1])
                         + f * (Cn * Vn[m - 1] + Sn * Wn[m - 1]))
            ay += 0.5 * ((-Cn * Wn[m + 1] + Sn * Vn[m + 1])
                         + f * (-Cn * Wn[m - 1] + Sn * Vn[m - 1]))
        az += g * (-Cn * Vn[m] - Sn * Wn[m])
    k = MU_MOON / (R * R)
    return k * ax, k * ay, k * az


def lunar_nonspherical_accel(p, t):
    """Perturbing acceleration (m/s^2) from the Moon's degree-2 figure (C20,
    C22) at Moon-relative position `p` (ECI-aligned, Moon-centered), at time t.

    Returns ONLY the perturbation to be added on top of the -MU_MOON*p/|p|^3
    central term. Negligible far from the Moon (it falls as 1/r^4), so it is
    applied only in the Moon-centered phases (lunar orbit, descent, ascent).

    Implementation: build the lunar body frame (spin axis ~ Moon orbital
    angular-momentum direction; long/x-axis ~ the Moon->Earth direction in the
    equatorial plane, since the Moon is tidally locked with its long axis toward
    Earth), evaluate the closed-form C20 and C22 accelerations in that frame,
    and rotate back to ECI. Assumptions (stated for honesty): lunar obliquity to
    its orbit (~1.5 deg) is neglected, and the long axis is taken exactly toward
    Earth; both are good approximations at this model's fidelity.
    """
    if not globals().get("ENABLE_LUNAR_HARMONICS", False):
        return np.zeros(3)
    # Early-out: the figure perturbation falls as (R/r)^2 relative to the Moon's
    # central term and is utterly negligible beyond lunar-orbit range. Skip the
    # (relatively expensive) body-frame construction when far from the Moon —
    # this keeps the function near-free on the long trans-lunar/trans-Earth
    # coasts and the iterative MCC solves, where gravity_earth_moon is called
    # millions of times but the spacecraft is tens of thousands of km away.
    p_dist = np.dot(p, p)   # squared distance (avoid sqrt)
    if p_dist > 5.0e7**2:   # > 50,000 km from Moon center
        return np.zeros(3)
    mr, mv = moon_state(t)
    # Lunar body frame
    z_b = np.cross(mr, mv); z_b = z_b / np.linalg.norm(z_b)   # spin axis ~ orbit normal
    earth_dir = -mr                                            # Moon -> Earth
    x_b = earth_dir - np.dot(earth_dir, z_b) * z_b
    nx = np.linalg.norm(x_b)
    if nx < 1e-6:
        return np.zeros(3)
    x_b = x_b / nx
    y_b = np.cross(z_b, x_b)
    # Spacecraft Moon-relative position in body coords
    x = np.dot(p, x_b); y = np.dot(p, y_b); z = np.dot(p, z_b)
    r = np.sqrt(x*x + y*y + z*z)
    if r < 1.0:
        return np.zeros(3)
    # Higher-degree field (flag-gated): full degree-N evaluation only within
    # ~3,500 km of the Moon's center (~1,760 km altitude — covers parking orbit,
    # LOI/DOI, descent, ascent, rendezvous with margin). Beyond that the n>2
    # terms are below integrator tolerance, so control falls through to the
    # legacy degree-2 closed form (which also remains the flag-OFF path).
    if globals().get("ENABLE_LUNAR_SH_FIELD", False) and r < 3.5e6:
        ax, ay, az = _lunar_sh_accel_body(x, y, z)
        return ax * x_b + ay * y_b + az * z_b
    mu = MU_MOON; R = R_MOON_GRAV
    # C20 (= -J2) acceleration, same closed form as the Earth J2 term:
    #   a = 1.5 J2 mu R^2 / r^5 * [x(5 z^2/r^2 -1), y(5 z^2/r^2 -1), z(5 z^2/r^2 -3)]
    z2r2 = (z/r)**2
    fz = 1.5 * J2_MOON * mu * R**2 / r**5
    a_c20 = fz * np.array([x*(5*z2r2-1), y*(5*z2r2-1), z*(5*z2r2-3)])
    # C22 (S22 ~ 0): from U22 = 3 mu R^2 C22 (x^2-y^2)/r^5, a = grad U22:
    K = 3.0 * mu * R**2 * C22_MOON
    r5 = r**5; r7 = r**7; d = (x*x - y*y)
    a_c22 = K * np.array([2*x/r5 - 5*x*d/r7,
                          -2*y/r5 - 5*y*d/r7,
                          -5*z*d/r7])
    a_body = a_c20 + a_c22
    # Rotate body-frame acceleration back to ECI
    return a_body[0]*x_b + a_body[1]*y_b + a_body[2]*z_b


def atm_density(altitude_m):
    """Simple exponential atmosphere calibrated to USSA-76 below 150 km."""
    if altitude_m < 0:    return 1.225
    if altitude_m < 25_000:
        return 1.225 * np.exp(-altitude_m / 7500.0)
    if altitude_m < 100_000:
        return 0.040 * np.exp(-(altitude_m - 25_000) / 7100.0)
    if altitude_m < 200_000:
        return 1e-6 * np.exp(-(altitude_m - 100_000) / 45_000.0)
    return 0.0


def eci_to_latlon(r, t):
    """ECI position → (lat, lon) in degrees at time t."""
    theta = OMEGA_E * t + _GMST0
    x = np.cos(theta)*r[0] + np.sin(theta)*r[1]
    y = -np.sin(theta)*r[0] + np.cos(theta)*r[1]
    z = r[2]
    rn = np.sqrt(x*x + y*y + z*z)
    return np.rad2deg(np.arcsin(z/rn)), np.rad2deg(np.arctan2(y, x))


def latlon_alt_to_eci(lat_deg, lon_deg, alt, t):
    """Lat/lon/alt → ECI position at time t."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    theta = OMEGA_E * t + _GMST0
    R = R_EARTH + alt
    xe = R*np.cos(lat)*np.cos(lon)
    ye = R*np.cos(lat)*np.sin(lon)
    ze = R*np.sin(lat)
    x =  np.cos(theta)*xe - np.sin(theta)*ye
    y =  np.sin(theta)*xe + np.cos(theta)*ye
    return np.array([x, y, ze])


# ============================================================
# Lambert solver (universal variables, Bate-Mueller-White / Vallado)
# Given r1, r2, time of flight, finds v1, v2 for a two-body transfer
# arc under central-force gravity.
# ============================================================
def _stumpff(psi):
    """Stumpff functions c2, c3 for universal variable psi."""
    if psi > 1e-6:
        s = np.sqrt(psi)
        return (1 - np.cos(s))/psi, (s - np.sin(s))/(s*s*s)
    if psi < -1e-6:
        s = np.sqrt(-psi)
        return (1 - np.cosh(s))/psi, (np.sinh(s) - s)/(s*s*s)
    # Series expansion near zero
    return 0.5 - psi/24.0 + psi*psi/720.0, \
           1/6.0 - psi/120.0 + psi*psi/5040.0


def lambert_uv(r1, r2, tof, mu=MU_EARTH, prograde=True):
    """Universal-variable Lambert solver.

    Returns (v1, v2) tuple of velocity vectors that take you from r1 at t=0
    to r2 at t=tof under central-force gravity with parameter mu.
    Returns None if no solution converges.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    r1n = np.linalg.norm(r1)
    r2n = np.linalg.norm(r2)
    cos_dnu = float(np.dot(r1, r2) / (r1n * r2n))
    cos_dnu = max(-1.0, min(1.0, cos_dnu))

    # Direction of motion: sign of z-component of r1 × r2 chooses short/long way
    z_cross = float(r1[0]*r2[1] - r1[1]*r2[0])
    if prograde:
        dm = 1.0 if z_cross >= 0 else -1.0
    else:
        dm = -1.0 if z_cross >= 0 else 1.0

    A = dm * np.sqrt(r1n * r2n * (1.0 + cos_dnu))
    if abs(A) < 1e-6:
        return None

    # Bisection bracket on psi
    psi_low, psi_up = -4.0*np.pi, 4.0*np.pi*np.pi
    psi = 0.0
    c2, c3 = 0.5, 1.0/6.0
    last_y = None

    for _ in range(200):
        y = r1n + r2n + A*(psi*c3 - 1.0)/np.sqrt(c2)
        # If A > 0 and y < 0, raise psi_low to keep y positive
        if A > 0 and y < 0:
            tries = 0
            while y < 0 and tries < 50:
                psi_low = psi
                psi = 0.5*(psi + psi_up)
                c2, c3 = _stumpff(psi)
                y = r1n + r2n + A*(psi*c3 - 1.0)/np.sqrt(c2)
                tries += 1
            if y < 0:
                return None
        if c2 <= 0:
            psi = 0.5*(psi_low + psi)
            c2, c3 = _stumpff(psi)
            continue
        chi = np.sqrt(y / c2)
        t_iter = (chi**3 * c3 + A * np.sqrt(y)) / np.sqrt(mu)
        if abs(t_iter - tof) < 1e-4:
            last_y = y
            break
        if t_iter < tof:
            psi_low = psi
        else:
            psi_up = psi
        psi = 0.5*(psi_low + psi_up)
        c2, c3 = _stumpff(psi)
        last_y = y

    if last_y is None or last_y <= 0:
        return None

    f = 1.0 - last_y/r1n
    g = A * np.sqrt(last_y/mu)
    gdot = 1.0 - last_y/r2n
    if abs(g) < 1e-9:
        return None
    v1 = (r2 - f*r1) / g
    v2 = (gdot*r2 - r1) / g
    return v1, v2


# ============================================================
# Initialize state at post-TLI (Lambert-targeted + 3-body refined)
#
# Real Apollo TLI was a finite-thrust S-IVB burn with closed-loop guidance.
# We bypass the burn dynamics and start at the post-TLI state, using
# Lambert targeting + 3-body shooting to find v1 such that the actual
# (Earth+Moon) trajectory passes Moon at ~110 km altitude (Apollo 11
# nominal periselene).
#
# The Lambert solution under pure-Earth gravity aims at Moon center, then
# 3-body refinement adjusts v1 perpendicular to the trajectory until the
# real trajectory's periapsis matches the target. This mirrors how real
# trajectory designers iterate using a high-fidelity propagator.
# ============================================================
_CACHED_V1_NOMINAL = None
_CACHED_R1 = None

def _compute_nominal_post_tli(t_tli, tof_to_moon, r_park):
    """Compute nominal post-TLI state by Lambert + 3-body grid refinement.
    Cached after first call (geometry depends only on t_tli, tof, r_park)."""
    global _CACHED_V1_NOMINAL, _CACHED_R1
    if _CACHED_V1_NOMINAL is not None:
        return _CACHED_R1.copy(), _CACHED_V1_NOMINAL.copy()

    r_moon_arr, _ = moon_state(t_tli + tof_to_moon)
    moon_hat = r_moon_arr / np.linalg.norm(r_moon_arr)
    plane_z = np.cross(np.array([1.0, 0.0, 0.0]), r_moon_arr)
    plane_z /= np.linalg.norm(plane_z)
    perp = np.cross(plane_z, moon_hat)
    perp /= np.linalg.norm(perp)
    # r1 placement (relative to Moon-arrival direction in the transfer plane).
    # LEGACY (-45 deg): chosen so the lunar approach plane is nearly co-planar
    # with the Moon's orbital plane (Earth-favorable TEI geometry) — but a 45
    # deg transfer angle is a near-radial LOB: FPA ~66 deg at r1, transfer
    # perigee ~5,300 km INSIDE the Earth — physically unflyable from a parking
    # orbit (known-limitation #1). FLYABLE (~-160 deg, used when
    # ENABLE_INTEGRATED_TLI is on): a realistic Apollo-class transfer angle
    # with near-tangential departure (perigee at r1), reachable by a real
    # S-IVB burn. The transfer PLANE is identical either way (it is set by the
    # x-axis x Moon-arrival construction above, not by this angle), so the
    # prograde-arrival property is preserved; only the in-plane shape changes.
    angle = np.deg2rad(globals().get("TLI_TRANSFER_ANGLE_DEG", -45.0))
    r1_dir = np.cos(angle) * moon_hat + np.sin(angle) * perp
    r1 = r_park * r1_dir

    # Lambert initial guess
    result = lambert_uv(r1, r_moon_arr, tof_to_moon,
                         mu=MU_EARTH, prograde=True)
    if result is None:
        raise RuntimeError("Lambert solver failed during nominal targeting")
    v1_lambert, _ = result

    # Build perturbation basis
    v1_hat = v1_lambert / np.linalg.norm(v1_lambert)
    h_traj = np.cross(r1, v1_lambert)
    n_in = np.cross(h_traj, v1_hat); n_in /= np.linalg.norm(n_in)
    n_out = h_traj / np.linalg.norm(h_traj)

    target_dist = R_MOON + 110_000.0  # 110 km altitude

    def closest_dist(v):
        s = np.concatenate([r1, v, [1.0]])
        def rhs(t, y):
            return np.concatenate([y[3:6], gravity_earth_moon(y[:3], t), [0]])
        sol = solve_ivp(rhs, (t_tli, t_tli+tof_to_moon+10*3600), s,
                        method='RK45', rtol=1e-9, atol=1e-1, max_step=900.0,
                        dense_output=True)
        ts = np.linspace(t_tli, t_tli+tof_to_moon+10*3600, 3000)
        states = sol.sol(ts)
        d = np.array([np.linalg.norm(states[:3, i] - moon_state(t)[0])
                      for i, t in enumerate(ts)])
        i_min = int(np.argmin(d))
        t_near = np.linspace(max(ts[0], ts[i_min]-1800),
                              min(ts[-1], ts[i_min]+1800), 500)
        states_near = sol.sol(t_near)
        d_near = np.array([np.linalg.norm(states_near[:3, i]
                            - moon_state(t)[0])
                           for i, t in enumerate(t_near)])
        return float(np.min(d_near))

    # In-plane-only refinement: searching ONLY the n_in direction preserves
    # the Lambert orbit plane (which is aligned with Moon's orbital plane).
    # Adding out-of-plane velocity rotates the orbit plane, which previously
    # caused the lunar arrival to flip to retrograde orientation around Moon.
    # The Lambert solution alone tends to impact the Moon's surface; a small
    # in-plane radial correction is sufficient to achieve target periselene
    # while keeping the orbit prograde.
    best_dvi = 0.0
    best_err = 1e20
    # Coarse search
    for dvi in [-80, -60, -40, -30, -20, -10, 0, 10, 20, 40, 80, 120]:
        d = closest_dist(v1_lambert + dvi*n_in)
        err = abs(d - target_dist)
        if err < best_err:
            best_err = err
            best_dvi = dvi
    # Refinement around best
    for dvi in np.linspace(best_dvi - 8, best_dvi + 8, 9):
        d = closest_dist(v1_lambert + dvi*n_in)
        err = abs(d - target_dist)
        if err < best_err:
            best_err = err
            best_dvi = float(dvi)

    v1_nominal = v1_lambert + best_dvi*n_in
    _CACHED_R1 = r1.copy()
    _CACHED_V1_NOMINAL = v1_nominal.copy()
    print(f"  Nominal Lambert+3body refinement: "
          f"in-plane correction = {best_dvi:+.1f} m/s, "
          f"final periselene err = {best_err/1e3:.1f} km")
    return r1.copy(), v1_nominal.copy()


def initial_state_post_tli(perturb=None,
                            t_tli=9856.0,
                            tof_to_moon=73*3600.0,
                            r_park=R_EARTH + 185_000):
    """Set up state at TLI cutoff with nominal Lambert+3-body refined velocity,
    plus per-trial perturbations."""
    perturb = perturb or {}
    r1, v1 = _compute_nominal_post_tli(t_tli, tof_to_moon, r_park)

    # Apply perturbations to r1 (alt err, lat/lon err)
    alt_err = perturb.get("insertion_alt_err_m", 0.0)
    if abs(alt_err) > 1e-9:
        r1 = r1 * (1.0 + alt_err / np.linalg.norm(r1))
    # Lat/lon perturbations: small rotations of r1
    lat_err = perturb.get("insertion_lat_err_deg", 0.0)
    lon_err = perturb.get("insertion_lon_err_deg", 0.0)
    if abs(lat_err) > 1e-9 or abs(lon_err) > 1e-9:
        # Rotate r1 around Earth axis by lon (z-axis) and elevation by lat
        lon_rad = np.deg2rad(lon_err)
        lat_rad = np.deg2rad(lat_err)
        # Apply z-axis rotation
        Rz = np.array([[np.cos(lon_rad), -np.sin(lon_rad), 0],
                        [np.sin(lon_rad),  np.cos(lon_rad), 0],
                        [0, 0, 1]])
        r1 = Rz @ r1
        # Apply tilt for lat (rotate around y-axis)
        Ry = np.array([[np.cos(lat_rad), 0, np.sin(lat_rad)],
                        [0, 1, 0],
                        [-np.sin(lat_rad), 0, np.cos(lat_rad)]])
        r1 = Ry @ r1

    # Apply perturbations to v1 (which act like residual TLI execution errors)
    pt_err = perturb.get("tli_pointing_rad", np.zeros(3))
    if np.linalg.norm(pt_err) > 1e-12:
        v_mag = np.linalg.norm(v1)
        v_hat = v1 / v_mag
        perp_err = pt_err - np.dot(pt_err, v_hat) * v_hat
        v1 = v1 + v_mag * perp_err

    dv_bias = perturb.get("tli_dv_bias_ms", 0.0)
    if dv_bias != 0:
        v1 = v1 * (1.0 + dv_bias / np.linalg.norm(v1))

    v1 = v1 + perturb.get("insertion_v_err", np.zeros(3))

    # Implied TLI ΔV (relative to parking-orbit circular speed at r1)
    v_park = np.sqrt(MU_EARTH / np.linalg.norm(r1))
    dv_tli = np.linalg.norm(v1) - v_park

    csm_lm_mass = (CSM_CM_MASS + CSM_SM_DRY + SPS_PROP_INIT
                   + LM_DESC_TOTAL + LM_ASCT_TOTAL)
    state = np.concatenate([r1, v1, [csm_lm_mass]])
    return state, t_tli, dv_tli


# ------------------------------------------------------------------
# Physically integrated TLI burn (fixes "Lambert-targeted post-TLI" shortcut).
# ------------------------------------------------------------------
# When ON, the S-IVB second burn is FLOWN: the ignition state on the parking
# orbit is back-solved from the Lambert post-TLI cutoff (prograde finite-thrust,
# duration chosen so backward integration reaches local circular speed exactly
# when the propellant budget closes), then each trial integrates the burn
# FORWARD with its own J-2 Isp/thrust dispersions, a small thrust-pointing
# misalignment (consuming the existing tli_pointing_rad draw), and a
# velocity-magnitude guidance cutoff with sensor residual (consuming
# tli_dv_bias_ms). Parking-insertion dispersions (insertion_*) are applied at
# IGNITION, where they physically belong. Cutoff dispersions thus EMERGE from
# engine physics through the burn, and S-IVB underperformance can genuinely
# starve the burn (tli_propellant_depleted) — propellant margin is honest and
# tight (~3,280 m/s available vs ~3,150 needed). RNG NOTE: all perturbation
# draws are unchanged; only their consumption moves. Residual (documented):
# the ignition state is Lambert-consistent, not launch-state-continuous — the
# launch's actual parking-orbit plane/phase still does not feed TLI (that needs
# launch-window modeling). When OFF, the legacy Lambert shortcut is bit-identical.
# Validated: back-solved ignition at 298 km / circular speed; flown nominal
# matches the transfer target to 0.000 m/s; full mission completes end-to-end
# on the flyable -170 deg geometry. Default ON (fixes known-limitation #1).
ENABLE_INTEGRATED_TLI = True
# Transfer-angle pairing: the integrated burn requires the FLYABLE transfer
# geometry; the legacy -45 lob is kept when OFF for bit-compat with prior runs.
TLI_TRANSFER_ANGLE_DEG = -170.0 if ENABLE_INTEGRATED_TLI else -45.0
# LAUNCH-STATE CONTINUITY (the final piece of known-limitation #1): when ON,
# the TLI burn ignites from the TRIAL'S OWN LAUNCHED PARKING ORBIT — launch ->
# parking coast to the geometric ignition point -> prograde S-IVB burn to the
# nominal-derived guidance cutoff speed — so launch dispersions propagate
# PHYSICALLY through TLI into the transfer (the MCC chain absorbs them) and
# the pad-to-splashdown trajectory is one continuous flown path. Requires the
# azimuth-capable ascent + the solved launch window (LAUNCH_AZIMUTH_DEG below:
# the window root where the J2-coasted plane at TLI ignition contains the
# Moon-arrival direction; solves to 72.48 deg at the real epoch — Apollo's
# actual was 72.058, an independent ~0.4 deg validation of the
# ephemeris+GMST+ascent stack). When OFF,
# the back-solved Lambert-consistent ignition (above) is used.
ENABLE_LAUNCH_CONTINUITY = True

# Launch-continuity SPLASH_TARGET (overrides the block above; this flag is
# defined after that block, and the entry function binds the target as a
# default argument at import, so the override must run here at module level).
# Value = the continuity nominal's own achievable landing point (fresh-process
# fixed point). The return geometry differs from the integrated-TLI config
# because the transfer now derives from the LAUNCHED orbit (J2-coasted plane,
# steered cutoff) and the lunar capture is RETROGRADE (Apollo's actual
# handedness) with the TEI departure on the inbound perigee-nulling branch.
if ENABLE_LAUNCH_CONTINUITY and ENABLE_REAL_EPHEMERIS:
    # PRIMARY RECOVERY ZONE (the per-opportunity zone the on-time nominal
    # aims for; per-trial zones are constructed from each trial's own EI).
    # With APOLLO RETURN-TIMING TARGETING (TEI TOF-targeted to 59.6 h, dv
    # ~995 vs Apollo's 1,008; rev-1 TEI commit; total mission 8.18 d vs
    # Apollo's 8.16) the zone sits in the CENTRAL PACIFIC at ~2,780 km
    # short-corridor range from the nominal EI (1.53 N, 154.78 E):
    # latitude within 0.16 deg of Apollo 11's actual recovery point
    # (13.3 N, 169.15 W); the ~13.7 deg longitude residual (~1,480 km) is
    # transfer-plane geometry, not timing. History of this constant: the
    # first continuity fixed point (-26.70, -59.51) was a BACKWARDS
    # long-skip artifact (target behind the entry point); the NE-Africa
    # value (15.449, 33.712) was the short-corridor fix but carried the
    # +10 h timeline residual (slipped TEI rev + minimum-energy return)
    # that rotated the zone ~150 deg east of the Pacific.
    # Re-derived after the as-flown mass corrections (CSM_SM_DRY 6,110 ->
    # 4,825 kg): the lighter stack changed the TLI cutoff solution and
    # shifted arrival timing ~3 h, moving the zone from (13.14, 177.18) to
    # the western Pacific. Latitude still matches Apollo's 13.3 N to ~0.2
    # deg; the ~4,800 km longitude offset is the transfer-plane geometry
    # residual (deferred — closing it means re-deriving the TLI transfer
    # construction against Apollo's actual trans-lunar trajectory).
    SPLASH_TARGET_LAT_DEG = 13.480
    SPLASH_TARGET_LON_DEG = 146.222

_CACHED_LAUNCH_TLI = None   # (t_ign_angle_deg, vcut_ms) nominal guidance preset

_CACHED_TLI_IGN = None


def _solve_tli_ignition(t_tli=9856.0, tof_to_moon=73*3600.0,
                        r_park=R_EARTH + 185_000):
    """Back-solve the TLI ignition state from the Lambert cutoff (cached)."""
    global _CACHED_TLI_IGN
    if _CACHED_TLI_IGN is not None:
        return _CACHED_TLI_IGN
    r1, v1 = _compute_nominal_post_tli(t_tli, tof_to_moon, r_park)
    m0_stack = (CSM_CM_MASS + CSM_SM_DRY + SPS_PROP_INIT + LM_DESC_TOTAL
                + LM_ASCT_TOTAL + S_IVB_DRY + S_IVB_PROP_AT_TLI)
    c = S_IVB_ISP * G0
    mdot = S_IVB_THRUST / c
    tau_max = (S_IVB_PROP_AT_TLI - 500.0) / mdot   # keep 500 kg residual

    def _back(tau, want_state=False):
        m_cut = m0_stack - mdot * tau
        def rhs_back(tt, y):
            r = y[:3]; v = y[3:6]; m = y[6]
            vhat = v / np.linalg.norm(v)
            a = gravity_earth_moon(r, t_tli - tt) + (S_IVB_THRUST / m) * vhat
            return np.concatenate([-v, -a, [mdot]])
        s = solve_ivp(rhs_back, (0.0, tau), np.concatenate([r1, v1, [m_cut]]),
                      method='RK45', rtol=1e-9, atol=1e-3, max_step=2.0)
        y = s.y[:, -1]
        if want_state:
            return y
        return float(np.linalg.norm(y[3:6])
                     - np.sqrt(MU_EARTH / np.linalg.norm(y[:3])))

    from scipy.optimize import brentq
    lo, hi = 150.0, tau_max
    if _back(lo) > 0 > _back(hi):
        tau_b = brentq(_back, lo, hi, xtol=0.1, maxiter=30)
    else:
        # Cannot reach exactly circular within the propellant budget: use the
        # closest achievable (slightly super-circular parking speed).
        grid = np.linspace(lo, hi, 12)
        tau_b = float(min(grid, key=lambda x: abs(_back(x))))
    y_ign = _back(tau_b, want_state=True)
    _CACHED_TLI_IGN = (y_ign, float(t_tli - tau_b), float(tau_b),
                       float(np.linalg.norm(v1)))
    return _CACHED_TLI_IGN


def phase_tli_burn(perturb=None):
    """Fly the S-IVB TLI burn forward from the back-solved ignition state.

    Returns (state, t_cutoff, dv_tli) matching initial_state_post_tli's
    contract, or None if the burn starves before guidance cutoff."""
    perturb = perturb or {}
    y_ign, t_ign, tau_nom, vcut_nom = _solve_tli_ignition()
    y0 = y_ign.copy()

    # Parking-insertion dispersions -> ignition state (same construction the
    # legacy path applied to the post-TLI state; physically they belong here).
    alt_err = perturb.get("insertion_alt_err_m", 0.0)
    if abs(alt_err) > 1e-9:
        y0[:3] = y0[:3] * (1.0 + alt_err / np.linalg.norm(y0[:3]))
    lat_err = perturb.get("insertion_lat_err_deg", 0.0)
    lon_err = perturb.get("insertion_lon_err_deg", 0.0)
    if abs(lat_err) > 1e-9 or abs(lon_err) > 1e-9:
        lon_rad = np.deg2rad(lon_err); lat_rad = np.deg2rad(lat_err)
        Rz = np.array([[np.cos(lon_rad), -np.sin(lon_rad), 0],
                       [np.sin(lon_rad),  np.cos(lon_rad), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(lat_rad), 0, np.sin(lat_rad)], [0, 1, 0],
                       [-np.sin(lat_rad), 0, np.cos(lat_rad)]])
        y0[:3] = Ry @ (Rz @ y0[:3])
    y0[3:6] = y0[3:6] + np.asarray(perturb.get("insertion_v_err", np.zeros(3)))

    isp = S_IVB_ISP * perturb.get("s_ivb_isp_factor", 1.0)
    T   = S_IVB_THRUST * perturb.get("s_ivb_thrust_factor", 1.0)
    c = isp * G0
    pt = np.asarray(perturb.get("tli_pointing_rad", np.zeros(3)), dtype=float)
    v_cut = vcut_nom + float(perturb.get("tli_dv_bias_ms", 0.0))
    m_floor = y0[6] - (S_IVB_PROP_AT_TLI - 200.0)

    def rhs(tt, y):
        r = y[:3]; v = y[3:6]; m = y[6]
        vhat = v / np.linalg.norm(v)
        tdir = vhat + np.cross(pt, vhat)        # small-angle misalignment
        tdir = tdir / np.linalg.norm(tdir)
        return np.concatenate([v, gravity_earth_moon(r, tt) + (T/m)*tdir,
                               [-T/c]])

    def cutoff(tt, y):
        return np.linalg.norm(y[3:6]) - v_cut
    cutoff.terminal = True; cutoff.direction = +1

    def starve(tt, y):
        return y[6] - m_floor
    starve.terminal = True; starve.direction = -1

    s = solve_ivp(rhs, (t_ign, t_ign + 1.6 * tau_nom + 120.0), y0,
                  method='RK45', rtol=1e-9, atol=1e-3, max_step=2.0,
                  events=[cutoff, starve])
    if len(s.t_events[0]) == 0:
        return None                              # starved before cutoff
    y = s.y_events[0][0]; t_cut = float(s.t_events[0][0])
    # Diagnostic: S-IVB propellant margin above the reserve floor at cutoff
    # (negative would have starved). Module global, overwritten per call.
    globals()["_LAST_TLI_MARGIN_KG"] = float(y[6] - m_floor)
    csm_lm_mass = (CSM_CM_MASS + CSM_SM_DRY + SPS_PROP_INIT
                   + LM_DESC_TOTAL + LM_ASCT_TOTAL)   # S-IVB jettison
    state = np.concatenate([y[:3], y[3:6], [csm_lm_mass]])
    dv_tli = float(np.linalg.norm(y[3:6])
                   - np.sqrt(MU_EARTH / np.linalg.norm(y[:3])))
    return state, t_cut, dv_tli


def _inplane_angle_to_arrival(r, v, r_arr_hat):
    """Signed in-plane angle (deg) from the current position to the
    Moon-arrival direction, measured ALONG the direction of motion."""
    h = np.cross(r, v); h_hat = h / np.linalg.norm(h)
    a_in = r_arr_hat - np.dot(r_arr_hat, h_hat) * h_hat
    a_in /= max(np.linalg.norm(a_in), 1e-12)
    r_hat = r / np.linalg.norm(r)
    cosang = float(np.clip(np.dot(r_hat, a_in), -1.0, 1.0))
    sgn = np.sign(np.dot(np.cross(r_hat, a_in), h_hat)) or 1.0
    return float(np.rad2deg(np.arccos(cosang)) * sgn)


def _coast_to_ignition(state_ins, t_ins, ign_angle_deg):
    """Coast the launched parking orbit (sampled at 20 s) until the in-plane
    angle-to-arrival, measured along the motion, crosses ign_angle_deg.
    Robust unwrapped-phase search over up to 2.2 orbits, then linear
    refinement. Returns (y_ign[:6], t_ign) or None. Deterministic given
    (state_ins, t_ins, ign_angle_deg) — cacheable by preset solvers."""
    r_arr, _ = moon_state(9856.0 + 73 * 3600.0)
    r_arr_hat = r_arr / np.linalg.norm(r_arr)
    y0 = np.concatenate([state_ins[:6], [state_ins[6]]])
    def rhs_coast(tt, y):
        return np.concatenate([y[3:6], gravity_earth_moon(y[:3], tt), [0.0]])
    span = 2.2 * 2 * np.pi * np.sqrt((R_EARTH + 185e3) ** 3 / MU_EARTH)
    sol = solve_ivp(rhs_coast, (t_ins, t_ins + span), y0, method='RK45',
                    rtol=1e-9, atol=1e-3, max_step=60.0, dense_output=True)
    ts = np.arange(t_ins + 600.0, sol.t[-1], 20.0)
    crossings = []
    prev = None
    for tt in ts:
        y = sol.sol(tt)
        ang = _inplane_angle_to_arrival(y[:3], y[3:6], r_arr_hat)
        if prev is not None:
            # WRAP-SAFE crossing detection: the ignition angle (179.2 deg)
            # sits right at the +/-180 wrap, where the raw signed angle jumps
            # ~360 between samples. The old guard (|a0-a1| < 30 on raw
            # angles) rejected legitimate crossings whenever the sample pair
            # straddled the wrap — a numerical cliff that hit ~6% of trials
            # (version-sensitively) and was MISLABELED as TLI propellant
            # starvation. Detect on the wrapped difference to the target
            # instead: d = ((ang - ign) + 180) mod 360 - 180 is continuous
            # through the wrap; a sign change of d with a small step is a
            # true crossing.
            d0 = ((prev[1] - ign_angle_deg + 180.0) % 360.0) - 180.0
            d1 = ((ang - ign_angle_deg + 180.0) % 360.0) - 180.0
            if d0 * d1 <= 0 and abs(d1 - d0) < 30.0:
                frac = -d0 / (d1 - d0) if d1 != d0 else 0.0
                crossings.append(prev[0] + frac * (tt - prev[0]))
        prev = (tt, ang)
    if not crossings:
        globals()["_TLI_FAIL_REASON"] = "no_ignition_crossing"
        return None
    # Apollo ignited TLI on the second parking orbit (~2:44 GET after ~1.5
    # checkout revs): pick the crossing nearest the historical ignition GET.
    best_t = min(crossings, key=lambda x: abs(x - 9580.0))
    return sol.sol(best_t)[:6].copy(), float(best_t)


def _fly_launched_tli(state_ins, t_ins, perturb, ign_angle_deg, vcut_ms,
                      steer_pitch_deg=0.0, steer_yaw_deg=0.0, _ign_cache=None):
    """Coast the launched parking orbit to the geometric ignition point and fly
    the STEERED S-IVB burn to the guidance cutoff speed. The thrust direction
    is the instantaneous prograde tilted by constant (pitch, yaw) angles in the
    local orbital frame — pitch toward the in-plane normal, yaw toward the
    orbit normal. This models IGM's cutoff-STATE steering: the preset angles
    (solved on the nominal in _solve_launch_tli) shape the cutoff velocity
    DIRECTION so the open-loop transfer actually reaches the Moon, at a cost of
    only cosine losses on the burn (~0.1%), exactly as the real instrument unit
    flew it. Returns (state, t_cut, dv_tli) with the CSM+LM mass (S-IVB
    jettisoned), or None if the burn starves."""
    perturb = perturb or {}
    ign = (_ign_cache if _ign_cache is not None
           else _coast_to_ignition(state_ins, t_ins, ign_angle_deg))
    if ign is None:
        return None
    y_ign6, best_t = ign
    # Stack mass at TLI ignition (S-IVB still attached, TLI propellant aboard).
    m0_stack = (CSM_CM_MASS + CSM_SM_DRY + SPS_PROP_INIT + LM_DESC_TOTAL
                + LM_ASCT_TOTAL + S_IVB_DRY + S_IVB_PROP_AT_TLI)
    y_ign = np.concatenate([y_ign6[:6], [m0_stack]])

    isp = S_IVB_ISP * perturb.get("s_ivb_isp_factor", 1.0)
    T = S_IVB_THRUST * perturb.get("s_ivb_thrust_factor", 1.0)
    c = isp * G0
    pt = np.asarray(perturb.get("tli_pointing_rad", np.zeros(3)), dtype=float)
    v_cut = vcut_ms + float(perturb.get("tli_dv_bias_ms", 0.0))
    m_floor = m0_stack - (S_IVB_PROP_AT_TLI - S_IVB_RESERVE_KG)
    _cp, _sp = np.cos(np.deg2rad(steer_pitch_deg)), np.sin(np.deg2rad(steer_pitch_deg))
    _cy, _sy = np.cos(np.deg2rad(steer_yaw_deg)), np.sin(np.deg2rad(steer_yaw_deg))

    def rhs_burn(tt, y):
        r = y[:3]; v = y[3:6]; m = y[6]
        vhat = v / np.linalg.norm(v)
        hh = np.cross(r, v)
        hh = hh / np.linalg.norm(hh)
        nin = np.cross(hh, vhat)
        tdir = _cy * (_cp * vhat + _sp * nin) + _sy * hh
        tdir = tdir + np.cross(pt, tdir)
        tdir /= np.linalg.norm(tdir)
        return np.concatenate([v, gravity_earth_moon(r, tt) + (T / m) * tdir,
                               [-T / c]])
    def cutoff(tt, y):
        return np.linalg.norm(y[3:6]) - v_cut
    cutoff.terminal = True; cutoff.direction = +1
    def starve(tt, y):
        return y[6] - m_floor
    starve.terminal = True; starve.direction = -1
    s = solve_ivp(rhs_burn, (best_t, best_t + 600.0), y_ign, method='RK45',
                  rtol=1e-9, atol=1e-3, max_step=2.0, events=[cutoff, starve])
    if len(s.t_events[0]) == 0:
        globals()["_TLI_FAIL_REASON"] = "propellant_starved"
        return None
    y = s.y_events[0][0]; t_cut = float(s.t_events[0][0])
    # Diagnostic: S-IVB propellant margin above the reserve floor at cutoff
    # (negative would have starved). Module global, overwritten per call.
    globals()["_LAST_TLI_MARGIN_KG"] = float(y[6] - m_floor)
    csm_lm_mass = (CSM_CM_MASS + CSM_SM_DRY + SPS_PROP_INIT
                   + LM_DESC_TOTAL + LM_ASCT_TOTAL)
    state = np.concatenate([y[:3], y[3:6], [csm_lm_mass]])
    dv_tli = float(np.linalg.norm(y[3:6])
                   - np.sqrt(MU_EARTH / np.linalg.norm(y[:3])))
    return state, t_cut, dv_tli


def _solve_launch_tli():
    """Derive the nominal TLI guidance preset (ignition angle, cutoff speed)
    from the NOMINAL launched orbit: ignite ~188 deg before Moon-arrival
    (170 deg transfer + ~18 deg burn arc), solve the cutoff speed so the
    resulting coast's periselene is ~110 km. Cached."""
    global _CACHED_LAUNCH_TLI
    if _CACHED_LAUNCH_TLI is not None:
        return _CACHED_LAUNCH_TLI
    # Disk cache (config-fingerprinted): the derivation is deterministic given
    # the geometry constants and costs minutes — MC workers must not re-derive.
    import json as _json
    import os as _os
    _fp = f"az{LAUNCH_AZIMUTH_DEG}_ang{TLI_TRANSFER_ANGLE_DEG}_v7"
    _path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "launch_tli_preset.json")
    try:
        with open(_path) as _f:
            _d = _json.load(_f)
        if _d.get("fingerprint") == _fp:
            _CACHED_LAUNCH_TLI = tuple(_d["preset"])
            return _CACHED_LAUNCH_TLI
    except Exception:
        pass
    launch = phase_saturn_v_launch({})
    if not launch["success"]:
        raise RuntimeError("nominal launch failed during TLI derivation")
    state_ins, t_ins = launch["state"], launch["t_insertion"]
    ign_angle = 171.4   # initial guess; solved below (rotates the apse line)

    def peri_and_tclose(vc, ang=None):
        out = _fly_launched_tli(state_ins, t_ins, {},
                                ang if ang is not None else ign_angle, vc)
        if out is None:
            return 1.0e9, 1.0e9
        st, tc, _ = out
        def rhs(tt, y):
            return np.concatenate([y[3:6], gravity_earth_moon(y[:3], tt), [0.0]])
        sol = solve_ivp(rhs, (tc, tc + 95 * 3600.0),
                        np.concatenate([st[:6], [1.0]]), method='RK45',
                        rtol=1e-9, atol=1e-1, max_step=900.0, dense_output=True)
        ts = np.linspace(tc, sol.t[-1], 3500)
        ys = sol.sol(ts)
        d = np.array([np.linalg.norm(ys[:3, i] - moon_state(t)[0])
                      for i, t in enumerate(ts)])
        i0 = int(np.argmin(d))
        return float(d[i0] - R_MOON) / 1.0e3, float(ts[i0])

    def periselene_for_vcut(vc):
        return peri_and_tclose(vc)[0]

    # NESTED 2-D SOLVE. Two in-plane targets need two knobs:
    #  - INNER (vcut -> arrival-time phasing): the cutoff speed sets the
    #    transfer's arrival time (~0.27 h per m/s; the Moon crosses its own
    #    diameter in ~1.75 h, so phasing is a ~±3 m/s valley). Solved by
    #    brentq on t_close = T_arr, warm-started from the previous angle.
    #  - OUTER (ignition angle -> encounter radius): cutoff happens on the
    #    LAUNCHED orbit (FPA ~+6 deg, not at perigee), so the apse line
    #    rotates with the ignition point; the angle is solved so the
    #    phase-matched transfer's closest approach reaches the Moon
    #    (periselene -> ~110 km). The B-plane MCC chain + MCC-4b trim the
    #    per-trial residual downstream.
    from scipy.optimize import brentq
    T_arr = 9856.0 + 73 * 3600.0
    _warm = [10_905.0]

    def phased_peri(ang):
        def terr(vc):
            return peri_and_tclose(vc, ang)[1] - T_arr
        v0 = _warm[0]
        lo, hi = v0 - 20.0, v0 + 20.0
        try:
            if terr(lo) > 0 > terr(hi):
                vp = brentq(terr, lo, hi, xtol=0.08, maxiter=24)
            else:
                lo, hi = 10_860.0, 10_960.0
                vp = brentq(terr, lo, hi, xtol=0.08, maxiter=28)
        except Exception:
            grid = np.arange(10_860.0, 10_960.1, 10.0)
            vp = float(min(grid, key=lambda v: abs(terr(v))))
        _warm[0] = vp
        # micro-scan the phasing valley for the periselene minimum
        fine = np.arange(vp - 1.2, vp + 1.21, 0.3)
        vals = [(v, peri_and_tclose(v, ang)[0]) for v in fine]
        v_b, p_b = min(vals, key=lambda c: c[1])
        return p_b, v_b

    # Outer scan on ignition angle (apse-line rotation), then refine.
    best = (1e18, ign_angle, _warm[0])
    for ang in np.arange(150.0, 192.1, 6.0):
        p, v = phased_peri(float(ang))
        if p < best[0]:
            best = (p, float(ang), v)
    for ang in np.arange(best[1] - 5.0, best[1] + 5.01, 1.5):
        p, v = phased_peri(float(ang))
        if p < best[0]:
            best = (p, float(ang), v)
    for ang in np.arange(best[1] - 1.2, best[1] + 1.21, 0.4):
        p, v = phased_peri(float(ang))
        if p < best[0]:
            best = (p, float(ang), v)
    p_best, ign_angle, vcut = best
    print(f"  Launch-TLI 2-D solve: periselene {p_best:.0f} km")

    # STEERED-CUTOFF SOLVE. The 2-D (angle, vcut) family above burns pure
    # prograde, so the cutoff velocity DIRECTION is fixed by the launched
    # orbit — and the family's achievable manifold misses the Moon by a
    # ~12,000 km floor that no (angle, vcut) combination closes. Real IGM
    # steered the full cutoff STATE: model that with constant (pitch, yaw)
    # tilts of the thrust off prograde (in-plane normal / orbit normal),
    # solved here on the nominal and flown per-trial through each trial's own
    # burn physics. Cost is only cosine losses (~0.1% propellant) — no
    # fictitious post-cutoff impulse, nothing charged to the SPS.
    _ign_cached = _coast_to_ignition(state_ins, t_ins, float(ign_angle))
    if _ign_cached is None:
        raise RuntimeError("nominal ignition-point coast failed")

    def fly_eval(dvc, pitch, yaw, full=False):
        out = _fly_launched_tli(state_ins, t_ins, {}, float(ign_angle),
                                float(vcut) + dvc, steer_pitch_deg=pitch,
                                steer_yaw_deg=yaw, _ign_cache=_ign_cached)
        if out is None:
            return (1.0e9, 1.0e9, 1.0e9) if full else 1.0e9
        st_c, t_c = out[0], out[1]
        def rhs(tt, y):
            return np.concatenate([y[3:6], gravity_earth_moon(y[:3], tt), [0.0]])
        sol = solve_ivp(rhs, (t_c, t_c + 95 * 3600.0),
                        np.concatenate([st_c[:6], [1.0]]), method='RK45',
                        rtol=1e-9, atol=1e-1, max_step=900.0, dense_output=True)
        ts = np.linspace(t_c, sol.t[-1], 3500)
        ys = sol.sol(ts)
        d = np.array([np.linalg.norm(ys[:3, i] - moon_state(t)[0])
                      for i, t in enumerate(ts)])
        i0 = int(np.argmin(d))
        peri = float(d[i0] - R_MOON) / 1.0e3
        if not full:
            return peri
        # Decompose the encounter offset at closest approach into transfer-
        # plane components. KEY GEOMETRY: the transfer plane contains the
        # Earth-Moon line by construction (it contains the Earth at the origin
        # AND the targeted Moon), so the capture plane contains the Earth line
        # — which TEI requires — iff the encounter OFFSET is IN-plane. The
        # out-of-plane miss must be driven to ~0; the in-plane offset sets the
        # periselene (~110 km aim).
        mr_ca, _ = moon_state(float(ts[i0]))
        w = ys[:3, i0] - mr_ca
        n_tr = np.cross(st_c[:3], st_c[3:6])
        n_tr /= np.linalg.norm(n_tr)
        oop_km = float(np.dot(w, n_tr)) / 1.0e3
        return peri, oop_km, float(ts[i0])

    # Knob structure. (dvc, pitch) -> (arrival time, periselene) is STRONGLY
    # coupled — energy and in-plane thrust direction both move arrival time
    # AND the in-plane aim — so coordinate descent ping-pongs (each phasing
    # re-null blows the periselene solution away; observed 33,000 km stall).
    # Solve that pair with a damped 2x2 Newton on the residual vector
    # (arrival slip, periselene error). yaw -> out-of-plane is decoupled once
    # slip is held in the residuals, so it is nulled separately by brentq.
    #
    # ROOT CHOICE: periselene-vs-pitch has roots on BOTH sides of the impact
    # valley with OPPOSITE capture handedness. The NEGATIVE-pitch root (the
    # steering equivalent of the validated -120 m/s in-plane impulse,
    # sin(pitch) ~ dv_n/vcut -> ~-2.2 deg) gives the RETROGRADE,
    # Apollo-handed capture whose plane TEI needs; the positive-side root
    # arrives over the opposite lunar face with a wildly different flight
    # time. Seed the Newton at -2.2 deg so it converges into the correct
    # basin.
    from scipy.optimize import brentq as _bq
    dvc_b, pitch_b, yaw_b = 0.0, -2.2, 0.0

    def _null_yaw():
        def oop_of(yw):
            return fly_eval(dvc_b, pitch_b, yw, full=True)[1]
        y0 = yaw_b
        try:
            lo, hi = y0 - 4.0, y0 + 4.0
            if oop_of(lo) * oop_of(hi) < 0:
                return _bq(oop_of, lo, hi, xtol=0.005, maxiter=20)
            g = np.linspace(lo, hi, 13)
            return float(min(g, key=lambda x: abs(oop_of(x))))
        except Exception:
            return y0

    def _resid(dvc, pitch):
        p, o, tca = fly_eval(dvc, pitch, yaw_b, full=True)
        if p >= 1.0e8:
            return None, None
        return np.array([(tca - T_arr) / 3600.0, (p - 110.0) / 1000.0]), p

    converged = False
    for _outer in range(3):
        yaw_b = _null_yaw()
        r0, p0 = _resid(dvc_b, pitch_b)
        for _it in range(14):
            if r0 is None:
                break
            if abs(r0[0]) < 0.05 and abs(r0[1]) < 0.02:
                break
            dd, dp = 0.5, 0.05
            r1, _ = _resid(dvc_b + dd, pitch_b)
            r2, _ = _resid(dvc_b, pitch_b + dp)
            if r1 is None or r2 is None:
                break
            J = np.column_stack([(r1 - r0) / dd, (r2 - r0) / dp])
            try:
                step = np.linalg.solve(J, -r0)
            except np.linalg.LinAlgError:
                break
            step[0] = float(np.clip(step[0], -20.0, 20.0))
            step[1] = float(np.clip(step[1], -1.0, 1.0))
            # backtracking line search on the residual norm
            accepted = False
            for damp in (1.0, 0.5, 0.25):
                r_try, p_try = _resid(dvc_b + damp * step[0],
                                      pitch_b + damp * step[1])
                if r_try is not None and (np.linalg.norm(r_try)
                                          < np.linalg.norm(r0)):
                    dvc_b += damp * step[0]
                    pitch_b += damp * step[1]
                    r0, p0 = r_try, p_try
                    accepted = True
                    break
            if not accepted:
                break
        p_chk, oop_chk, t_chk = fly_eval(dvc_b, pitch_b, yaw_b, full=True)
        if (abs(p_chk - 110.0) < 25.0 and abs(oop_chk) < 60.0
                and abs(t_chk - T_arr) < 0.3 * 3600.0):
            converged = True
            break
    if not converged:
        print("  WARNING: steered-cutoff Newton did not fully converge")
    vcut = float(vcut) + dvc_b
    p_final, oop_final, t_final = fly_eval(0.0, pitch_b, yaw_b, full=True)
    print(f"  Steered cutoff: pitch {pitch_b:+.3f} deg, yaw {yaw_b:+.3f} deg, "
          f"dvcut {dvc_b:+.1f} m/s -> periselene {p_final:.0f} km, "
          f"out-of-plane miss {oop_final:+.0f} km, "
          f"arrival slip {(t_final - T_arr) / 3600.0:+.2f} h")
    _CACHED_LAUNCH_TLI = (float(ign_angle), float(vcut),
                          float(pitch_b), float(yaw_b))
    try:
        with open(_path, "w") as _f:
            _json.dump({"fingerprint": _fp,
                        "preset": list(_CACHED_LAUNCH_TLI)}, _f)
    except Exception:
        pass
    print(f"  Launch-TLI preset: ignition angle {ign_angle:.1f} deg, "
          f"cutoff {vcut:.1f} m/s")
    return _CACHED_LAUNCH_TLI


# ============================================================
# Phase: Saturn V launch (S-IC + S-II + S-IVB first burn → parking orbit)
# ============================================================
def phase_saturn_v_launch(perturb=None, t_liftoff=0.0):
    """Simulate Saturn V ascent from launch pad through parking orbit insertion.

    Models:
      - S-IC stage (5x F-1) with atmospheric drag and pitch program
      - S-II stage (5x J-2) with closed-loop pitch toward orbit
      - S-IVB stage first burn for parking orbit insertion
      - Engine-out failure modes
      - Aerodynamic max-Q load monitoring

    Returns dict with:
      - success: bool
      - failure_reason: str or None
      - state: post-insertion ECI state [r, v, m]
      - t_insertion: insertion time
      - trajectory_t, trajectory_y: ascent trajectory (if captured)
      - max_q_pa, max_g, peak_alt_km, peak_speed_ms
      - engine_failures: list of engine failures during ascent
    """
    perturb = perturb or {}
    result = {"success": True, "failure_reason": None,
              "engine_failures": [], "max_q_pa": 0.0, "max_g": 0.0,
              "peak_alt_km": 0.0}

    # Set up initial state on launch pad
    # Apollo 11 launched at LC-39A, 28.608°N, 80.604°W
    lat = np.deg2rad(LAUNCH_LAT_DEG)
    lon = np.deg2rad(LAUNCH_LON_DEG)
    theta0 = OMEGA_E * t_liftoff + _GMST0
    # ECEF launch position
    R_pad = R_EARTH
    xe = R_pad * np.cos(lat) * np.cos(lon)
    ye = R_pad * np.cos(lat) * np.sin(lon)
    ze = R_pad * np.sin(lat)
    # Rotate to ECI
    x0 = np.cos(theta0)*xe - np.sin(theta0)*ye
    y0 = np.sin(theta0)*xe + np.cos(theta0)*ye
    z0 = ze
    r0 = np.array([x0, y0, z0])
    # Initial velocity is Earth's surface velocity at the pad (in ECI)
    v0 = np.cross(np.array([0, 0, OMEGA_E]), r0)
    # Total mass at liftoff
    m_full = (S_IC_DRY + S_IC_PROP + S_II_DRY + S_II_PROP
               + S_IVB_DRY + S_IVB_PROP_TOTAL
               + CSM_CM_MASS + CSM_SM_DRY + SPS_PROP_INIT
               + LM_DESC_TOTAL + LM_ASCT_TOTAL)
    state = np.concatenate([r0, v0, [m_full]])
    t = t_liftoff

    # Target orbital plane from the LAUNCH AZIMUTH (deg east of north at the
    # pad). The ascent steers its downrange thrust IN this plane, so the
    # insertion plane responds to azimuth — previously the steering followed
    # the ECI velocity, which starts ~400 m/s due-east from Earth rotation and
    # locked every launch into the due-east (minimum-inclination) plane
    # regardless of LAUNCH_AZIMUTH_DEG. The Earth-rotation kick leaves a ~1 deg
    # plane bias vs the ideal azimuth formula; the launch-window solve targets
    # the FINAL plane numerically, so the bias is absorbed.
    _az = np.deg2rad(globals().get("LAUNCH_AZIMUTH_DEG", 72.0))
    _r_hat0 = r0 / np.linalg.norm(r0)
    _east0 = np.cross(np.array([0.0, 0.0, 1.0]), _r_hat0)
    _east0 /= np.linalg.norm(_east0)
    _north0 = np.cross(_r_hat0, _east0)
    _head0 = np.sin(_az) * _east0 + np.cos(_az) * _north0
    _n_target = np.cross(_r_hat0, _head0)
    _n_target /= np.linalg.norm(_n_target)

    def _downrange_in_plane(r_hat):
        d = np.cross(_n_target, r_hat)
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else _head0

    # Sample engine-out events from perturbations
    # 1969-era F-1 reliability: ~99% per engine, but engine-out at S-IC was
    # survivable (Apollo 6 had S-II engine fail too). Mission could continue
    # if at least 4 of 5 F-1 fired full duration.
    n_f1_failures = int(perturb.get("n_f1_failures", 0))
    f1_failure_time = float(perturb.get("f1_failure_time_s", 1e9))
    n_j2_s2_failures = int(perturb.get("n_j2_s2_failures", 0))
    j2_s2_failure_time = float(perturb.get("j2_s2_failure_time_s", 1e9))
    s_ivb_first_burn_fail = bool(perturb.get("s_ivb_first_burn_fail", False))

    # Pitch program for S-IC (open-loop): vertical for 12s, then gravity turn
    # Standard Saturn V program: vertical 0-12s, then start pitch over
    # Pitch rate ~0.5°/s from 12-110s, then flatten to 0° at horizon by S-II
    def pitch_angle_s_ic(t_since_liftoff):
        """Return pitch angle from local vertical at time t_since_liftoff."""
        if t_since_liftoff < 12.0:
            return 0.0    # vertical
        elif t_since_liftoff < 130.0:
            # Linear ramp 0° → 60° from 12s to 130s
            return (t_since_liftoff - 12.0) / 118.0 * 60.0
        else:
            return 60.0

    # =====  S-IC burn  =====
    isp_f1 = (F1_ISP_VAC + F1_ISP_SL) / 2 * perturb.get("s_ic_isp_factor", 1.0)
    thr_factor_f1 = perturb.get("s_ic_thrust_factor", 1.0)

    def get_f1_thrust(alt):
        """F-1 thrust varies with altitude (vacuum higher than sea level)."""
        if alt < 0:
            return F1_THRUST_SL
        if alt > 80_000:
            return F1_THRUST_VAC
        # Linear interpolation
        f = alt / 80_000.0
        return F1_THRUST_SL + f * (F1_THRUST_VAC - F1_THRUST_SL)

    def rhs_s_ic(t_now, y):
        r = y[:3]; v = y[3:6]; m = y[6]
        rn = np.linalg.norm(r)
        alt = rn - R_EARTH
        result["peak_alt_km"] = max(result["peak_alt_km"], alt/1000)

        # Gravity (Earth + Moon, but Moon negligible at this altitude)
        a_grav = -MU_EARTH * r / rn**3

        # Engine thrust
        n_engines = 5 - (n_f1_failures if (t_now - t_liftoff) >= f1_failure_time else 0)
        n_engines = max(0, n_engines)
        if n_engines == 0:
            T_total = 0
            mdot = 0
        else:
            T_per = get_f1_thrust(alt) * thr_factor_f1
            T_total = T_per * n_engines
            mdot = T_total / (isp_f1 * G0)

        # Thrust direction: pitch program in vertical plane
        # Local "up" is radial outward
        r_hat = r / rn
        # Need a local "downrange" direction. Use the launch azimuth + Earth rotation.
        # Surface velocity direction at pad gives initial East direction.
        # Better: maintain pitch in plane defined by initial r and v vectors
        # Initial v0 was Earth surface velocity → mostly East
        # For simplicity, build a "downrange" unit vector from initial conditions
        # downrange ⊥ r_hat in plane of r0 and v_initial_horizontal

        pitch_deg = pitch_angle_s_ic(t_now - t_liftoff)
        pitch_rad = np.deg2rad(pitch_deg)

        # Use velocity direction to determine downrange (after pitch starts)
        v_mag = np.linalg.norm(v)
        # Steer downrange IN the azimuth-defined target plane (see _n_target).
        downrange_hat = _downrange_in_plane(r_hat)

        thrust_dir = (np.cos(pitch_rad) * r_hat
                       + np.sin(pitch_rad) * downrange_hat)
        a_thrust = T_total * thrust_dir / max(m, 1.0)

        # Atmospheric drag
        v_air = np.cross(np.array([0, 0, OMEGA_E]), r)
        v_rel = v - v_air
        v_rel_mag = np.linalg.norm(v_rel)
        rho = atm_density(alt)
        q = 0.5 * rho * v_rel_mag**2
        result["max_q_pa"] = max(result["max_q_pa"], q)
        if v_rel_mag > 1 and rho > 1e-8:
            cd_eff = SATV_CD * (1.0 + 1.5 * np.exp(-((v_rel_mag - 380)/250)**2))
            a_drag = -q * cd_eff * SATV_AREA * v_rel / (v_rel_mag * max(m, 1))
        else:
            a_drag = np.zeros(3)

        a_total = a_grav + a_thrust + a_drag
        # Track g-load
        g_load = np.linalg.norm(a_thrust + a_drag) / G0
        result["max_g"] = max(result["max_g"], g_load)

        return np.concatenate([v, a_total, [-mdot]])

    # Burn S-IC until propellant runs out (or n_engines = 0)
    # Approximate burn time: full S-IC propellant / mdot at 5 engines
    nominal_mdot = 5 * (F1_THRUST_SL+F1_THRUST_VAC)/2 / (isp_f1 * G0)
    eff_burn_time = S_IC_PROP / nominal_mdot

    def s_ic_burnout(t_now, y):
        # S-IC empty: total mass - (S_IC_DRY) ≤ what's left after S-IC drops
        return y[6] - (m_full - S_IC_PROP)
    s_ic_burnout.terminal = True
    s_ic_burnout.direction = -1

    try:
        sol_s_ic = solve_ivp(rhs_s_ic, (t, t + 200), state, method='RK45',
                              rtol=1e-7, atol=1e-1, max_step=2.0,
                              events=s_ic_burnout, dense_output=True)
    except Exception as e:
        result["success"] = False
        result["failure_reason"] = f"s_ic_integration_error: {e}"
        return result

    # Check for catastrophic failure: descent during S-IC means not enough thrust
    final_state = sol_s_ic.y[:, -1]
    final_alt = np.linalg.norm(final_state[:3]) - R_EARTH
    if final_alt < 1000:    # Below 1 km after burn → crashed
        result["success"] = False
        result["failure_reason"] = "s_ic_underperformance_crash"
        result["state"] = final_state
        result["t_insertion"] = sol_s_ic.t[-1]
        result["trajectory_t"] = sol_s_ic.t
        result["trajectory_y"] = sol_s_ic.y
        return result

    # Max-Q check: structural limit ~50 kPa for Saturn V
    if result["max_q_pa"] > 60_000:
        result["success"] = False
        result["failure_reason"] = "structural_failure_max_q_exceeded"
        result["state"] = final_state
        result["t_insertion"] = sol_s_ic.t[-1]
        result["trajectory_t"] = sol_s_ic.t
        result["trajectory_y"] = sol_s_ic.y
        return result

    # Stage separation: drop S-IC dry mass
    state = final_state.copy()
    state[6] = state[6] - S_IC_DRY
    t = sol_s_ic.t[-1]
    result["t_s_ic_end"] = t
    result["alt_s_ic_end_km"] = (np.linalg.norm(state[:3]) - R_EARTH) / 1000.0
    result["v_s_ic_end_ms"] = np.linalg.norm(state[3:6])

    # =====  S-II burn  =====
    # Closed-loop guidance: pitch the vehicle to drive vertical velocity to
    # zero while raising specific orbital energy. Target: 185 km circular.
    isp_j2_s2 = J2_ISP_S2 * perturb.get("s_ii_isp_factor", 1.0)
    thr_factor_s2 = perturb.get("s_ii_thrust_factor", 1.0)

    def rhs_s_ii(t_now, y):
        r = y[:3]; v = y[3:6]; m = y[6]
        rn = np.linalg.norm(r)
        alt = rn - R_EARTH

        a_grav = -MU_EARTH * r / rn**3

        # Engines
        n_engines = 5 - (n_j2_s2_failures
                         if (t_now - result["t_s_ic_end"]) >= j2_s2_failure_time
                         else 0)
        n_engines = max(0, n_engines)
        if n_engines == 0:
            T_total = 0
            mdot = 0
        else:
            T_per = J2_THRUST_S2 * thr_factor_s2
            T_total = T_per * n_engines
            mdot = T_total / (isp_j2_s2 * G0)

        # Closed-loop pitch: target orbit altitude ~185 km
        # We use a "linear tangent" steering approximation: pitch above the
        # horizontal early to gain altitude, then pitch over toward horizontal
        # to accelerate. The target apogee/perigee should converge to 185 km.
        r_hat = r / rn
        v_mag = np.linalg.norm(v)
        v_radial = np.dot(v, r_hat)

        # Estimate when we'd run out of fuel at current consumption (rough)
        # to decide how aggressively to pitch over
        target_alt = 185_000.0
        # If we're below target altitude and going up, keep pitched up
        # If at or above target altitude, level off
        alt_err = (target_alt - alt) / 1000.0   # km
        # Pitch above horizontal: higher when far below target, zero at target
        if alt_err > 50:
            pitch_above_horiz = 20.0
        elif alt_err > 0:
            pitch_above_horiz = max(0, alt_err / 50.0 * 20.0)
        else:
            pitch_above_horiz = max(-5.0, alt_err / 20.0 * 5.0)
        # Add small correction to keep v_radial bounded
        v_rad_max = 200.0   # max ascent rate m/s
        if v_radial > v_rad_max:
            pitch_above_horiz -= 5.0
        elif v_radial < -50:
            pitch_above_horiz += 5.0
        pitch_above_horiz = np.clip(pitch_above_horiz, -10, 30)

        # Build thrust direction
        # Steer downrange IN the azimuth-defined target plane (see _n_target).
        downrange_hat = _downrange_in_plane(r_hat)

        pitch_rad = np.deg2rad(pitch_above_horiz)
        thrust_dir = (np.cos(pitch_rad) * downrange_hat
                       + np.sin(pitch_rad) * r_hat)
        a_thrust = T_total * thrust_dir / max(m, 1.0)

        # Drag (negligible at S-II altitude but include for completeness)
        v_air = np.cross(np.array([0,0,OMEGA_E]), r)
        v_rel = v - v_air
        v_rel_mag = np.linalg.norm(v_rel)
        rho = atm_density(alt)
        if v_rel_mag > 1 and rho > 1e-10:
            q = 0.5 * rho * v_rel_mag**2
            result["max_q_pa"] = max(result["max_q_pa"], q)
            a_drag = -q * SATV_CD * SATV_AREA * v_rel / (v_rel_mag * max(m,1))
        else:
            a_drag = np.zeros(3)

        return np.concatenate([v, a_grav + a_thrust + a_drag, [-mdot]])

    def s_ii_burnout(t_now, y):
        return y[6] - (state[6] - S_II_PROP)
    s_ii_burnout.terminal = True
    s_ii_burnout.direction = -1

    try:
        sol_s_ii = solve_ivp(rhs_s_ii, (t, t + 500), state, method='RK45',
                              rtol=1e-7, atol=1e-1, max_step=3.0,
                              events=s_ii_burnout, dense_output=True)
    except Exception as e:
        result["success"] = False
        result["failure_reason"] = f"s_ii_integration_error: {e}"
        return result

    state = sol_s_ii.y[:, -1].copy()
    state[6] = state[6] - S_II_DRY    # drop S-II dry mass
    t = sol_s_ii.t[-1]
    result["t_s_ii_end"] = t
    result["alt_s_ii_end_km"] = (np.linalg.norm(state[:3]) - R_EARTH) / 1000.0
    result["v_s_ii_end_ms"] = np.linalg.norm(state[3:6])

    # =====  S-IVB first burn (parking orbit insertion)  =====
    # Target: circular orbit at 185 km
    isp_sivb = S_IVB_ISP * perturb.get("s_ivb_isp_factor", 1.0)
    thr_factor_sivb = perturb.get("s_ivb_thrust_factor", 1.0)
    T_sivb = S_IVB_THRUST * thr_factor_sivb

    if s_ivb_first_burn_fail:
        # S-IVB failed to ignite for parking orbit
        result["success"] = False
        result["failure_reason"] = "s_ivb_first_ignition_failure"
        result["state"] = state
        result["t_insertion"] = t
        return result

    # Closed-loop guidance: target circular orbit at 185 km using a
    # linear-tangent steering approximation. We pitch up to climb, then
    # pitch over as altitude rises, with a feedback term on radial velocity.
    def rhs_sivb1(t_now, y):
        r = y[:3]; v = y[3:6]; m = y[6]
        rn = np.linalg.norm(r)
        alt = rn - R_EARTH
        a_grav = -MU_EARTH * r / rn**3

        T = T_sivb
        mdot = T / (isp_sivb * G0)

        r_hat = r / rn
        v_radial = np.dot(v, r_hat)
        v_perp = v - v_radial * r_hat
        v_perp_mag = np.linalg.norm(v_perp)
        # Steer downrange IN the azimuth-defined target plane (see _n_target).
        downrange_hat = _downrange_in_plane(r_hat)

        target_alt = 185_000.0
        alt_err = target_alt - alt   # positive if below

        # Very gentle radial control: pitch slightly up if very low, slightly
        # down if very high. Most of the burn is horizontal acceleration.
        v_rad_des = np.clip(alt_err / 200.0, -30, 50)   # m/s
        v_rad_err = v_rad_des - v_radial
        pitch_above_horiz = np.clip(v_rad_err / 20.0, -5, 8)
        pitch_rad = np.deg2rad(pitch_above_horiz)
        thrust_dir = (np.cos(pitch_rad) * downrange_hat
                       + np.sin(pitch_rad) * r_hat)
        a_thrust = T * thrust_dir / max(m, 1.0)

        return np.concatenate([v, a_grav + a_thrust, [-mdot]])

    # Compute target burn time analytically. We need to add ΔV from the
    # S-II burnout state to reach circular at 185 km.
    target_v = np.sqrt(MU_EARTH / (R_EARTH + 185_000))
    current_v = np.linalg.norm(state[3:6])
    current_rn = np.linalg.norm(state[:3])
    # Energy at S-II burnout
    E_current = 0.5*current_v**2 - MU_EARTH/current_rn
    E_target  = -MU_EARTH / (2 * (R_EARTH + 185_000))
    dE = E_target - E_current
    # Approximate ΔV needed (ignoring gravity loss): v^2 = 2*(E_target + MU/r)
    # at r = 185 km altitude
    v_target_at_alt = np.sqrt(2*(E_target + MU_EARTH/(R_EARTH + 185_000)))
    # Initial guess: ΔV equals difference between current speed and
    # required speed at target altitude. Real Apollo S-IVB first burn
    # was ~150 seconds for ~960 m/s ΔV.
    dv_needed = max(v_target_at_alt - current_v, 200.0) + 200.0  # +200 for gravity loss
    isp = isp_sivb
    mp_needed = state[6] * (1 - np.exp(-dv_needed / (isp * G0)))
    burn_time_target = mp_needed / (T_sivb / (isp * G0))
    burn_time_target = float(np.clip(burn_time_target, 50.0, 300.0))

    # ---- Authentic linear-tangent (IGM-core) closed-loop guidance --------
    # Solve Lawden's linear-tangent coefficients (initial pitch chi0 and the
    # tangent rate) so the S-IVB burn reaches the 185 km target radius at
    # circular speed with zero flight-path angle, cutting off at FPA = 0. This
    # is the authentic optimal exo-atmospheric steering MSFC's IGM is built on.
    igm_done = False
    if globals().get("ENABLE_IGM_ASCENT", False):
        r_T = R_EARTH + 185_000.0
        v_T = np.sqrt(MU_EARTH / r_T)
        t_ign_sivb = t
        state_ign_sivb = state.copy()
        mdot_sivb = T_sivb / (isp_sivb * G0)

        def _fly_lt(chi0_deg, tan_rate):
            tan0 = np.tan(np.deg2rad(chi0_deg))
            def rhs_lt(tt, y):
                r = y[:3]; v = y[3:6]; m = y[6]; rn = np.linalg.norm(r)
                a_grav = -MU_EARTH * r / rn**3
                r_hat = r / rn
                # Steer in the azimuth-defined target plane (see _n_target).
                dh = _downrange_in_plane(r_hat)
                tanchi = tan0 + tan_rate * (tt - t_ign_sivb)
                chi = np.arctan(tanchi)
                tdir = np.cos(chi) * dh + np.sin(chi) * r_hat
                a_th = T_sivb * tdir / max(m, 1.0)
                return np.concatenate([v, a_grav + a_th, [-mdot_sivb]])
            def fpa_zero(tt, y):
                r = y[:3]; v = y[3:6]; rn = np.linalg.norm(r)
                return float(np.dot(v, r / rn))
            fpa_zero.terminal = True; fpa_zero.direction = -1
            sol = solve_ivp(rhs_lt, (t_ign_sivb, t_ign_sivb + 700.0),
                            state_ign_sivb, method='RK45', rtol=1e-7,
                            atol=1e-1, max_step=2.0, events=fpa_zero)
            if len(sol.t_events[0]) > 0:
                yf = sol.y_events[0][0]; tf = sol.t_events[0][0]; cut = True
            else:
                yf = sol.y[:, -1]; tf = sol.t[-1]; cut = False
            return yf, tf, np.linalg.norm(yf[:3]), np.linalg.norm(yf[3:6]), cut, sol

        def _resid(p):
            _, _, rn_f, sp_f, _, _ = _fly_lt(p[0], p[1])
            return [(rn_f - r_T) / 1000.0, sp_f - v_T]

        try:
            from scipy.optimize import fsolve
            p_sol, _info, ier, _msg = fsolve(_resid, [15.0, -0.0015],
                                             full_output=True)
            yf, tf, rn_f, sp_f, cut, sol_lt = _fly_lt(p_sol[0], p_sol[1])
            E_f = 0.5 * sp_f**2 - MU_EARTH / rn_f
            if E_f < 0:
                a_f = -MU_EARTH / (2 * E_f)
                h_f = np.linalg.norm(np.cross(yf[:3], yf[3:6]))
                ecc_f = np.sqrt(max(0.0, 1 - h_f**2 / (MU_EARTH * a_f)))
                peri_f = a_f * (1 - ecc_f) - R_EARTH
                if ier == 1 and cut and 150_000 < peri_f < 220_000 and ecc_f < 0.02:
                    state = yf.copy(); t = tf
                    sol_sivb1 = sol_lt
                    result["parking_insertion_method"] = "igm_lineartangent"
                    result["igm_chi0_deg"] = float(p_sol[0])
                    result["igm_tan_rate"] = float(p_sol[1])
                    igm_done = True
        except Exception:
            igm_done = False

    if not igm_done:
        def time_seco(t_now, y):
            return (t_now - t) - burn_time_target
        time_seco.terminal = True
        time_seco.direction = +1

        def orbit_event(t_now, y):
            # Backup termination: SECO if we somehow escape Earth
            r = y[:3]; v = y[3:6]
            rn = np.linalg.norm(r)
            E = 0.5*np.linalg.norm(v)**2 - MU_EARTH/rn
            return -E   # crosses zero from positive to negative when E > 0
        orbit_event.terminal = True
        orbit_event.direction = -1

        try:
            sol_sivb1 = solve_ivp(rhs_sivb1, (t, t + 400), state, method='RK45',
                                    rtol=1e-7, atol=1e-1, max_step=3.0,
                                    events=[time_seco, orbit_event],
                                    dense_output=True)
        except Exception as e:
            result["success"] = False
            result["failure_reason"] = f"s_ivb_integration_error: {e}"
            return result

        if len(sol_sivb1.t_events[0]) > 0:
            state = sol_sivb1.y_events[0][0].copy()
            t = sol_sivb1.t_events[0][0]
            result["parking_insertion_method"] = "time_seco"
        elif len(sol_sivb1.t_events[1]) > 0:
            state = sol_sivb1.y_events[1][0].copy()
            t = sol_sivb1.t_events[1][0]
            result["parking_insertion_method"] = "energy_escape"
        else:
            state = sol_sivb1.y[:, -1].copy()
            t = sol_sivb1.t[-1]
            result["parking_insertion_method"] = "timeout"

    # Check insertion accuracy
    rn = np.linalg.norm(state[:3])
    alt = rn - R_EARTH
    v_actual = np.linalg.norm(state[3:6])
    v_circ = np.sqrt(MU_EARTH / rn)
    E = 0.5*v_actual**2 - MU_EARTH/rn

    if E >= 0:
        # Hyperbolic: escaped Earth (bad!)
        result["success"] = False
        result["failure_reason"] = "parking_insertion_overshoot_escape"
        result["state"] = state
        result["t_insertion"] = t
        return result

    a = -MU_EARTH / (2*E)
    h = np.linalg.norm(np.cross(state[:3], state[3:6]))
    p = h**2 / MU_EARTH
    ecc = np.sqrt(max(0, 1 - p/a))
    perigee = (a*(1-ecc) - R_EARTH) / 1000.0
    apogee = (a*(1+ecc) - R_EARTH) / 1000.0

    result["parking_perigee_km"] = perigee
    result["parking_apogee_km"] = apogee
    result["alt_insertion_km"] = alt / 1000.0

    if perigee < 80:    # would re-enter atmosphere on perigee pass
        result["success"] = False
        result["failure_reason"] = "parking_orbit_decays_into_atmosphere"
        result["state"] = state
        result["t_insertion"] = t
        return result

    if apogee > 1000:
        result["success"] = False
        result["failure_reason"] = "parking_insertion_overshoot"
        result["state"] = state
        result["t_insertion"] = t
        return result

    result["state"] = state
    result["t_insertion"] = t
    result["trajectory_t"] = np.concatenate([sol_s_ic.t, sol_s_ii.t, sol_sivb1.t])
    result["trajectory_y"] = np.hstack([sol_s_ic.y, sol_s_ii.y, sol_sivb1.y])
    return result


# ============================================================
# Phase: parking orbit coast (1.5 orbits before TLI)
# ============================================================
def phase_parking_orbit(state, t0, t_end):
    def rhs(t, y):
        return np.concatenate([y[3:6], gravity_earth_moon(y[:3], t), [0]])
    sol = solve_ivp(rhs, (t0, t_end), state, method='RK45',
                    rtol=1e-9, atol=1e-2, max_step=120.0)
    return sol.y[:, -1], sol.t[-1], sol


# ============================================================
# Phase: TLI burn (S-IVB second burn — finite duration)
# ============================================================
def phase_tli(state, t0, perturb=None):
    perturb = perturb or {}
    isp_f = perturb.get("s_ivb_isp_factor", 1.0)
    thr_f = perturb.get("s_ivb_thrust_factor", 1.0)
    pt_err = perturb.get("tli_pointing_rad", np.zeros(3))
    # Burn time: target nominal 3,153 m/s of ΔV
    isp = S_IVB_ISP * isp_f
    T   = S_IVB_THRUST * thr_f
    m0  = state[6]
    dv_target = 3153.4 + perturb.get("tli_dv_bias_ms", 0.0)
    mp = m0 * (1 - np.exp(-dv_target / (isp*G0)))
    burn_time = mp / (T / (isp*G0))

    # Direction: along velocity (prograde) at start, fixed inertially
    v_hat = state[3:6] / np.linalg.norm(state[3:6])
    dir_inertial = v_hat + pt_err
    dir_inertial /= np.linalg.norm(dir_inertial)

    def rhs(t, y):
        r = y[:3]; v = y[3:6]; m = y[6]
        a = gravity_earth_moon(r, t)
        # Update direction to track prograde
        vh = v / np.linalg.norm(v)
        dvec = vh + pt_err
        dvec /= np.linalg.norm(dvec)
        a = a + T * dvec / m
        return np.concatenate([v, a, [-T/(isp*G0)]])

    sol = solve_ivp(rhs, (t0, t0 + burn_time), state, method='RK45',
                    rtol=1e-9, atol=1e-2, max_step=2.0)
    return sol.y[:, -1], sol.t[-1], burn_time


# ============================================================
# Phase: Translunar coast (3-day, 3-body)
# After TLI, drop the S-IVB. Optional mid-course correction.
# ============================================================
def phase_translunar_coast(state, t0, duration=3*86400):
    """3-day translunar coast with Earth+Moon gravity."""
    # Drop S-IVB mass — what remains is CSM+LM
    state = state.copy()
    csm_lm_mass = (CSM_CM_MASS + CSM_SM_DRY + SPS_PROP_INIT
                   + LM_DESC_TOTAL + LM_ASCT_TOTAL)
    state[6] = csm_lm_mass

    def rhs(t, y):
        return np.concatenate([y[3:6], gravity_earth_moon(y[:3], t), [0]])
    sol = solve_ivp(rhs, (t0, t0+duration), state, method='RK45',
                    rtol=1e-9, atol=1e-1, max_step=600.0, dense_output=True)
    # Find closest approach to Moon
    times = np.linspace(t0, t0+duration, 3000)
    states = sol.sol(times)
    dists = np.zeros(len(times))
    for i, t in enumerate(times):
        mr, _ = moon_state(t)
        dists[i] = np.linalg.norm(states[:3, i] - mr)
    i_min = int(np.argmin(dists))
    return {
        "final_state": sol.y[:, -1],
        "final_t":     sol.t[-1],
        "sol": sol,
        "log_t": times,
        "log_states": states,
        "moon_distances_km": dists / 1000.0,
        "closest_approach_km": dists[i_min] / 1000.0,
        "t_closest": times[i_min],
    }


# ============================================================
# Mid-course correction: solve for SPS burn to target lunar periapsis
# ============================================================
def midcourse_correction(state, t0, target_alt_km=110.0):
    """
    Find a small SPS burn that targets a specific lunar periapsis altitude.
    Uses 1D shooting in the 'normal to v' direction.
    """
    def predict_periapsis(s_trial):
        def rhs(t, y):
            return np.concatenate([y[3:6], gravity_earth_moon(y[:3], t), [0]])
        sol = solve_ivp(rhs, (t0, t0 + 4*86400), s_trial, method='RK45',
                        rtol=1e-8, atol=1e-1, max_step=600.0, dense_output=True)
        times = np.linspace(t0, t0+4*86400, 3000)
        states = sol.sol(times)
        min_d = 1e15
        for i, t in enumerate(times):
            mr, _ = moon_state(t)
            d = np.linalg.norm(states[:3, i] - mr)
            if d < min_d:
                min_d = d
        return (min_d - R_MOON) / 1000.0

    v = state[3:6]; r = state[:3]
    h = np.cross(r, v)
    n = np.cross(h, v); n /= np.linalg.norm(n)   # in-plane, perp to v

    p_current = predict_periapsis(state)
    if abs(p_current - target_alt_km) < 20:
        return state.copy(), 0.0

    # Binary search dv ∈ [-25, 25] m/s
    lo, hi = -25.0, 25.0
    for _ in range(20):
        mid = 0.5*(lo+hi)
        s = state.copy()
        s[3:6] += mid * n
        p = predict_periapsis(s)
        if p > target_alt_km:
            hi = mid
        else:
            lo = mid
        if abs(p - target_alt_km) < 3:
            break

    new = state.copy()
    new[3:6] += mid * n
    # SPS propellant cost
    isp = SPS_ISP
    prop = new[6] * (1 - np.exp(-abs(mid)/(isp*G0)))
    new[6] -= prop
    return new, abs(mid)


# ============================================================
# B-plane intercept coordinates of a hyperbolic approach
# ============================================================
def b_plane_coords(r, v, mu):
    """B-plane intercept (B·T, B·R) of a hyperbolic approach, from a body-
    relative state (r, v) [m, m/s] and gravitational parameter mu. Returns
    (BdotT_m, BdotR_m, b_mag_m, e, a) or None if the state is not hyperbolic.

    Construction (standard): the incoming-asymptote unit vector S is the normal
    of the B-plane; the B-vector is the impact-parameter vector (from the focus
    to where the incoming asymptote pierces the B-plane), magnitude
    |a|*sqrt(e^2-1). The in-plane reference axis is T = (S x k)/|S x k| with k
    the inertial z-axis, and R = S x T. The (B·T, B·R) pair encodes both the
    closest-approach distance (~|B|) and the approach-plane orientation, and is
    computed identically for the nominal target and each trial, so only the
    consistency of the sign convention matters (not which asymptote branch)."""
    r = np.asarray(r, float); v = np.asarray(v, float)
    rn = np.linalg.norm(r)
    h_vec = np.cross(r, v); h = np.linalg.norm(h_vec)
    if h < 1.0 or rn < 1.0:
        return None
    h_hat = h_vec / h
    e_vec = np.cross(v, h_vec) / mu - r / rn
    e = np.linalg.norm(e_vec)
    if e <= 1.0 + 1e-6:
        return None  # not hyperbolic -> B-plane undefined
    e_hat = e_vec / e
    n_hat = np.cross(h_hat, e_hat)              # in-plane, 90 deg ahead of periapsis
    st = np.sqrt(e * e - 1.0)
    S_hat = (e_hat + st * n_hat) / e            # incoming asymptote (consistent)
    energy = 0.5 * np.dot(v, v) - mu / rn
    a = -mu / (2.0 * energy)                    # < 0 for a hyperbola
    b_mag = abs(a) * st                         # impact parameter |B|
    B_hat = np.cross(S_hat, h_hat)              # in-plane, perpendicular to S
    B_vec = b_mag * B_hat
    k = np.array([0.0, 0.0, 1.0])
    T_hat = np.cross(S_hat, k)
    if np.linalg.norm(T_hat) < 1e-9:
        T_hat = np.cross(S_hat, np.array([0.0, 1.0, 0.0]))
    T_hat /= np.linalg.norm(T_hat)
    R_hat = np.cross(S_hat, T_hat)
    return (float(np.dot(B_vec, T_hat)), float(np.dot(B_vec, R_hat)),
            float(b_mag), float(e), float(a))


# ============================================================
# Phase: Trans-lunar (outbound) midcourse corrections MCC-1..MCC-4
# ============================================================
# Generalizes the single TLI+30h correction to Apollo's scheduled outbound
# sequence. At each scheduled opportunity we coast the current state to the
# point of correction, project the resulting lunar perilune, and — if it is
# outside the deadband — apply a small along-track-plus-radial SPS burn that
# nulls the perilune error (in-plane shooting, identical in spirit to the
# midcourse_correction helper). Corrections inside the deadband are waived,
# matching how most Apollo outbound MCCs were skipped.
#
# HONESTY NOTE: the SCHEDULE is now sourced (Apollo 11 Flight Journal / Mission
# Overview — see the TLMCC constants block), as is the execution residual
# (shared MCC_EXEC_RESIDUAL_MS, ~0.15 m/s, from the Apollo 11 Flight Journal).
# The perilune DEADBAND tolerance remains an estimate. The structure (four
# scheduled outbound trims with deadband-gated waive logic, ~1 executed) is
# faithful to Apollo. The function returns enough diagnostics to audit each burn.
def phase_translunar_mcc(state, t0, tlc, perturb,
                          target_perilune_km=TLMCC_TARGET_PERILUNE_KM,
                          deadband_km=TLMCC_PERILUNE_DEADBAND_KM):
    """Apply scheduled outbound midcourse corrections.

    Parameters
    ----------
    state : (7,) ndarray   post-TLI ECI state at t0
    t0    : float          time at TLI cutoff (start of trans-lunar coast)
    tlc   : dict           output of phase_translunar_coast (for log grid)
    perturb : dict|None    perturbation draw for this trial

    Returns
    -------
    dict with:
        final_state, final_t     state handed to LOI targeting (at/near
                                  the corrected perilune neighborhood)
        perilune_km              projected perilune after corrections
        mcc_burns                list of {name, t, dv_ms, peri_before, peri_after}
        mcc_total_dv_ms          sum of correction magnitudes
        sps_prop_used_kg         propellant consumed by the chain
    """
    perturb = perturb or {}
    isp_mcc = SPS_ISP * perturb.get("sps_isp_factor", 1.0)

    def rhs_em(t, y):
        return np.concatenate([y[3:6], gravity_earth_moon(y[:3], t), [0]])

    def project_perilune(s_trial, t_start):
        # Returns (perilune_altitude_km, t_perilune). If the trajectory strikes
        # the Moon, perilune is reported as a NEGATIVE altitude equal to the
        # (negative) depth of closest approach below the surface, which makes
        # the perilune-vs-deltaV objective monotonic through the impact region
        # (deeper impact -> more negative), keeping the root-finder well-posed.
        def peri_evt(tt, y):
            mr_e = moon_state(tt)[0]; mv_e = moon_state(tt)[1]
            r = y[:3] - mr_e; v = y[3:6] - mv_e
            return float(np.dot(r, v) / np.linalg.norm(r))
        peri_evt.terminal = True
        peri_evt.direction = +1
        sol = solve_ivp(rhs_em, (t_start, t_start + 5*86400), s_trial,
                        method='RK45', rtol=1e-7, atol=1.0, max_step=600.0,
                        events=peri_evt, dense_output=True)
        if len(sol.t_events[0]) > 0:
            s_peri = sol.y_events[0][0]; t_peri = sol.t_events[0][0]
            d = np.linalg.norm(s_peri[:3] - moon_state(t_peri)[0])
            return (d - R_MOON) / 1000.0, t_peri
        # No radial-velocity sign change captured (e.g. the vehicle plunged
        # straight into the Moon, or never reached perilune in window). Use the
        # sampled minimum distance; if it is below the surface, that already
        # yields a negative altitude, preserving monotonicity.
        ts = np.linspace(t_start, t_start + 5*86400, 2000)
        rs = sol.sol(ts)
        dists = np.array([np.linalg.norm(rs[:3, i] - moon_state(tt)[0])
                          for i, tt in enumerate(ts)])
        j = int(np.argmin(dists))
        return (dists[j] - R_MOON) / 1000.0, ts[j]

    def solve_burn(s_at_burn, t_burn):
        """Solve the along-velocity delta-V that drives projected perilune to
        target. Returns (dv_mag, v_hat, converged_bool).

        Robustness measures over the previous version:
          * The search is SCALED TO THE ACTUAL SENSITIVITY. Perilune is highly
            sensitive to along-track delta-V (a few m/s moves perilune by
            hundreds of km), so we start with a NARROW fine bracket (+/-3 m/s)
            and only widen if no sign change is found, instead of a coarse
            10-m/s grid over +/-40 m/s that could straddle the impact region
            and miss or mis-bracket the root.
          * The objective clamps the impact region: once the trajectory
            impacts the Moon, perilune "altitude" is reported as a large
            NEGATIVE number that DECREASES monotonically as the miss deepens,
            so the objective stays monotone and brentq stays well-posed.
          * Returns an explicit convergence flag so the caller never silently
            treats a failed solve as a zero-delta-V correction.
        """
        v = s_at_burn[3:6]
        v_hat = v / np.linalg.norm(v)

        def peri_for(dv_mag):
            s = s_at_burn.copy()
            s[3:6] = v + dv_mag * v_hat
            return project_perilune(s, t_burn)[0]

        def err(dv_mag):
            return peri_for(dv_mag) - target_perilune_km

        from scipy.optimize import brentq

        # Adaptive bracket: widen the search window until a sign change in the
        # perilune error is found (or we exhaust a sane maximum).
        brackets = [3.0, 8.0, 20.0, 50.0]
        for half in brackets:
            # Sample finely across this window (step ~ half/6).
            grid = np.linspace(-half, half, 13)
            samples = [(dv, err(dv)) for dv in grid]
            # Look for the sign change closest to zero delta-V (cheapest burn).
            sign_changes = []
            for i in range(len(samples) - 1):
                a, fa = samples[i]; b, fb = samples[i+1]
                if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                    sign_changes.append((a, b, min(abs(a), abs(b))))
            if sign_changes:
                sign_changes.sort(key=lambda sc: sc[2])
                a, b, _ = sign_changes[0]
                try:
                    root = brentq(err, a, b, xtol=0.05, maxiter=20)
                    return float(root), v_hat, True
                except Exception:
                    pass
            # No bracket at this window size — try wider.
        # Could not bracket a solution anywhere: report best-effort smallest
        # error from the widest grid, flagged as NOT converged so the caller
        # can decide what to do.
        grid = np.linspace(-brackets[-1], brackets[-1], 25)
        best = min(((dv, abs(err(dv))) for dv in grid), key=lambda s: s[1])
        return float(best[0]), v_hat, False

    # B-plane targeting mode: ON only once the nominal target is captured. The
    # nominal reference run itself uses the perilune-altitude solve (below) to
    # define the approach, then records its B-plane as the target for trials.
    use_bplane = (globals().get("ENABLE_BPLANE_TLMCC", False)
                  and globals().get("_BPLANE_TARGET") is not None)
    is_nominal = not perturb

    def b_plane_of(s_trial, t_start):
        """Coast to perilune; return (B·T_km, B·R_km, peri_alt_km) of the
        lunar-approach hyperbola there, or None if no perilune / not hyperbolic."""
        def peri_evt(tt, y):
            mr_e = moon_state(tt)[0]; mv_e = moon_state(tt)[1]
            rr = y[:3] - mr_e; vv = y[3:6] - mv_e
            return float(np.dot(rr, vv) / np.linalg.norm(rr))
        peri_evt.terminal = True; peri_evt.direction = +1
        sol = solve_ivp(rhs_em, (t_start, t_start + 5*86400), s_trial,
                        method='RK45', rtol=1e-7, atol=1.0, max_step=600.0,
                        events=peri_evt, dense_output=True)
        if len(sol.t_events[0]) == 0:
            return None
        s_peri = sol.y_events[0][0]; t_peri = sol.t_events[0][0]
        mr, mv = moon_state(t_peri)
        bp = b_plane_coords(s_peri[:3] - mr, s_peri[3:6] - mv, MU_MOON)
        if bp is None:
            return None
        BdotT, BdotR, b_mag, e, a = bp
        peri_alt = (abs(a) * (e - 1.0) - R_MOON) / 1000.0
        return BdotT / 1000.0, BdotR / 1000.0, peri_alt

    def solve_burn_bplane(s_at_burn, t_burn, tBT, tBR):
        """2-DOF (along-track + cross-track) burn driving the approach B-plane
        intercept to (tBT, tBR) [km]. Returns (dv_vec(3,), converged)."""
        rb = s_at_burn[:3]; vb = s_at_burn[3:6]
        v_hat = vb / np.linalg.norm(vb)
        hgeo = np.cross(rb, vb)
        n_hat = hgeo / np.linalg.norm(hgeo)     # geocentric orbit normal (cross-track)

        def resid(c):
            s = s_at_burn.copy()
            s[3:6] = vb + c[0] * v_hat + c[1] * n_hat
            bp = b_plane_of(s, t_burn)
            if bp is None:
                return [1.0e4, 1.0e4]
            return [bp[0] - tBT, bp[1] - tBR]

        from scipy.optimize import least_squares
        sol = least_squares(resid, [0.0, 0.0], method='trf',
                            bounds=([-50.0, -50.0], [50.0, 50.0]),
                            x_scale='jac', xtol=1e-3, ftol=1e-2, max_nfev=30)
        dv_vec = sol.x[0] * v_hat + sol.x[1] * n_hat
        converged = float(np.linalg.norm(sol.fun)) < 75.0   # within ~75 km B-plane
        return dv_vec, converged

    # Build the schedule (absolute times). MCC-1/2 are TLI-relative; MCC-3/4
    # are LOI-relative and resolved against the current projected perilune time.
    peri_km0, t_peri0 = project_perilune(state, t0)
    schedule = [(f"MCC-{k+1}", t0 + h*3600.0)
                for k, h in enumerate(TLMCC_SCHEDULE_HRS)]
    # MCC-3/4 relative to projected LOI (perilune) time, if they fit.
    for name, dt_before in [("MCC-3", 22.0), ("MCC-4", 5.0)]:
        t_corr = t_peri0 - dt_before*3600.0
        if t_corr > schedule[-1][1] + 3600.0:   # keep them ordered & spaced
            schedule.append((name, t_corr))

    mcc_burns = []
    total_dv = 0.0
    prop_used = 0.0
    cur = state.copy()
    cur_t = t0

    # Per-burn execution residual (sourced: Apollo MCC residuals ~0.15 m/s).
    rng_exec = np.random.default_rng(
        int(abs(hash((float(t0), float(state[0])))) % 2**31))

    for name, t_corr in schedule:
        if t_corr <= cur_t:
            continue
        # Coast to the correction time.
        sol = solve_ivp(rhs_em, (cur_t, t_corr), cur, method='RK45',
                        rtol=1e-8, atol=1e-1, max_step=600.0)
        cur = sol.y[:, -1].copy(); cur_t = sol.t[-1]

        if use_bplane:
            # --- 2-DOF B-plane solve: drive (B·T, B·R) to the nominal target,
            #     controlling perilune altitude AND approach-plane orientation.
            tBT, tBR = globals()["_BPLANE_TARGET"]
            bp_before = b_plane_of(cur, cur_t)
            if bp_before is None:
                mcc_burns.append({"name": name, "t": cur_t, "dv_ms": 0.0,
                                  "peri_before": float("nan"),
                                  "peri_after": float("nan"), "waived": False,
                                  "converged": False})
                continue
            peri_before = bp_before[2]
            miss_before = float(np.hypot(bp_before[0] - tBT, bp_before[1] - tBR))
            if miss_before <= TLMCC_BPLANE_DEADBAND_KM:
                mcc_burns.append({"name": name, "t": cur_t, "dv_ms": 0.0,
                                  "peri_before": peri_before,
                                  "peri_after": peri_before, "waived": True,
                                  "converged": True,
                                  "bplane_miss_km": miss_before})
                continue
            dv0, converged = solve_burn_bplane(cur, cur_t, tBT, tBR)
            if not converged and np.linalg.norm(dv0) < 1e-3:
                mcc_burns.append({"name": name, "t": cur_t, "dv_ms": 0.0,
                                  "peri_before": peri_before,
                                  "peri_after": peri_before, "waived": False,
                                  "converged": False,
                                  "bplane_miss_km": miss_before})
                continue
            dv_vec = dv0 + rng_exec.normal(0.0, MCC_EXEC_RESIDUAL_MS, size=3)
            trial = cur.copy()
            trial[3:6] = trial[3:6] + dv_vec
            bp_after = b_plane_of(trial, cur_t)
            miss_after = (float(np.hypot(bp_after[0] - tBT, bp_after[1] - tBR))
                          if bp_after is not None else 1.0e9)
            # Commit only if the B-plane miss actually improved.
            if miss_after >= miss_before:
                mcc_burns.append({"name": name, "t": cur_t, "dv_ms": 0.0,
                                  "peri_before": peri_before,
                                  "peri_after": peri_before, "waived": False,
                                  "converged": False,
                                  "bplane_miss_km": miss_before})
                continue
            dv_actual = float(np.linalg.norm(dv_vec))
            m_now = cur[6]
            dprop = m_now * (1 - np.exp(-dv_actual/(isp_mcc*G0)))
            cur = trial
            cur[6] = m_now - dprop
            prop_used += dprop
            total_dv += dv_actual
            mcc_burns.append({"name": name, "t": cur_t, "dv_ms": dv_actual,
                              "peri_before": peri_before,
                              "peri_after": bp_after[2], "waived": False,
                              "converged": True, "bplane_miss_km": miss_after})
            continue

        # --- Fallback: 1-DOF perilune-altitude solve (B-plane mode off, or the
        #     nominal reference run before its target is captured). ---
        peri_before, _ = project_perilune(cur, cur_t)
        if abs(peri_before - target_perilune_km) <= deadband_km:
            mcc_burns.append({"name": name, "t": cur_t, "dv_ms": 0.0,
                              "peri_before": peri_before,
                              "peri_after": peri_before, "waived": True,
                              "converged": True})
            continue
        dv_mag, v_hat, converged = solve_burn(cur, cur_t)
        if not converged and abs(dv_mag) < 1e-3:
            # Solver could not find a useful correction at all — record the
            # attempt honestly and leave the state unchanged rather than
            # pretending a zero burn was a correction.
            mcc_burns.append({"name": name, "t": cur_t, "dv_ms": 0.0,
                              "peri_before": peri_before,
                              "peri_after": peri_before, "waived": False,
                              "converged": False})
            continue
        # Apply sourced execution residual (magnitude + small pointing error).
        dv_vec = dv_mag * v_hat + rng_exec.normal(0.0, MCC_EXEC_RESIDUAL_MS,
                                                  size=3)
        trial = cur.copy()
        trial[3:6] = trial[3:6] + dv_vec
        peri_after, _ = project_perilune(trial, cur_t)
        # Verify the burn actually moved perilune toward target; if it made
        # things worse (poor bracket / nonlinearity), skip applying it.
        if abs(peri_after - target_perilune_km) >= abs(peri_before - target_perilune_km):
            mcc_burns.append({"name": name, "t": cur_t, "dv_ms": 0.0,
                              "peri_before": peri_before,
                              "peri_after": peri_before, "waived": False,
                              "converged": False})
            continue
        # Commit the burn.
        dv_actual = float(np.linalg.norm(dv_vec))
        m_now = cur[6]
        dprop = m_now * (1 - np.exp(-dv_actual/(isp_mcc*G0)))
        cur = trial
        cur[6] = m_now - dprop
        prop_used += dprop
        total_dv += dv_actual
        mcc_burns.append({"name": name, "t": cur_t, "dv_ms": dv_actual,
                          "peri_before": peri_before, "peri_after": peri_after,
                          "waived": False, "converged": True})

    # MCC-4b — CLOSED-LOOP perilune verification (navigation-robustness fix).
    # The B-plane solve drives (B·T, B·R) to the NOMINAL's target, but the same
    # B maps to a different perilune for a perturbed arrival v-infinity: at
    # 500-trial scale, 14% of trials satisfied the B-plane deadband while headed
    # BELOW the lunar surface (projected perilune -52..-517 km) — something
    # Apollo's tracking-verified navigation never allowed (0 missed approaches
    # in 9 lunar missions). So after the B-plane chain, VERIFY the projected
    # perilune and, if it is outside a safe corridor, run the 1-DOF
    # perilune-altitude solve as a final placement trim (Apollo's MCC-4 role).
    if use_bplane:
        # ITERATED verify -> trim -> re-verify (up to 5 passes). One open-loop
        # pass is not enough: at MCC-4's epoch the perilune sensitivity to
        # along-track dv is hundreds of km per m/s, so the 0.15 m/s/axis
        # execution residual re-scatters even a CONVERGED trim by ~200 km —
        # at 48-trial scale that parked trials at -120..-364 km (impact side,
        # just outside the -50 km LOI-recoverable gate) after a trim that
        # aimed at 94 km. Apollo's tracking-based navigation verified every
        # burn and re-trimmed; iterating is that practice, not a free lunch —
        # each pass costs its own (noisy) burn. Large arrivals also benefit:
        # a bracket-limited first pass (e.g. 9,900 km high) re-solves from
        # its improved state. Passes stop on in-band, no-improvement, or 5
        # (a 9,900 km-high arrival took 3 passes to 920 km — still outside
        # the band; tracking would have finished the job).
        # (rng_exec is phase-local, so extra draws here cannot shift any
        # other phase's stream; trials that previously stayed in-band take
        # the identical draw sequence.)
        for _pass in range(5):
            peri_chk, _ = project_perilune(cur, cur_t)
            if 40.0 <= peri_chk <= 400.0:
                break
            dv_mag, v_hat, _conv = solve_burn(cur, cur_t)
            dv_vec = dv_mag * v_hat + rng_exec.normal(0.0, MCC_EXEC_RESIDUAL_MS,
                                                      size=3)
            trial = cur.copy()
            trial[3:6] = trial[3:6] + dv_vec
            peri_after, _ = project_perilune(trial, cur_t)
            if abs(peri_after - target_perilune_km) >= abs(peri_chk - target_perilune_km):
                break   # no improvement — stop rather than random-walk on noise
            dv_actual = float(np.linalg.norm(dv_vec))
            m_now = cur[6]
            dprop = m_now * (1 - np.exp(-dv_actual/(isp_mcc*G0)))
            cur = trial
            cur[6] = m_now - dprop
            prop_used += dprop
            total_dv += dv_actual
            mcc_burns.append({"name": f"MCC-4b.{_pass + 1}", "t": cur_t,
                              "dv_ms": dv_actual,
                              "peri_before": peri_chk,
                              "peri_after": peri_after,
                              "waived": False, "converged": bool(_conv)})

    peri_final, t_peri_final = project_perilune(cur, cur_t)

    # On the nominal reference run, record the achieved approach B-plane as the
    # target for perturbed trials (mirrors the entry-interface target capture).
    if (globals().get("ENABLE_BPLANE_TLMCC", False)
            and globals().get("_BPLANE_TARGET") is None and is_nominal):
        bp_nom = b_plane_of(cur, cur_t)
        if bp_nom is not None:
            globals()["_BPLANE_TARGET"] = (bp_nom[0], bp_nom[1])

    return {
        "final_state": cur, "final_t": cur_t,
        "perilune_km": peri_final, "t_perilune": t_peri_final,
        "mcc_burns": mcc_burns,
        "mcc_total_dv_ms": total_dv,
        "sps_prop_used_kg": prop_used,
    }


# ============================================================
# Phase: LOI burn — SPS retrograde, finite duration
# Burn direction fixed inertially at ignition (real Apollo SPS practice).
# ============================================================
def phase_loi(state, t0, dv_target, perturb=None):
    perturb = perturb or {}
    isp = SPS_ISP * perturb.get("sps_isp_factor", 1.0)
    T   = SPS_THRUST * perturb.get("sps_thrust_factor", 1.0)
    m0  = state[6]
    mp  = m0 * (1 - np.exp(-dv_target/(isp*G0)))
    burn_time = mp / (T/(isp*G0))

    # Fix burn direction at ignition: retrograde wrt Moon at start of burn
    mr0, mv0 = moon_state(t0)
    v_rel0 = state[3:6] - mv0
    burn_dir = -v_rel0 / np.linalg.norm(v_rel0)   # retrograde, inertial fixed

    def rhs(t, y):
        r = y[:3]; v = y[3:6]; m = y[6]
        rl = r - moon_state(t)[0]
        a = -MU_MOON * rl / np.linalg.norm(rl)**3
        a = a + lunar_nonspherical_accel(rl, t)  # C20+C22 (flag-gated)
        a = a - MU_EARTH * r / np.linalg.norm(r)**3
        a = a + T * burn_dir / m
        return np.concatenate([v, a, [-T/(isp*G0)]])

    sol = solve_ivp(rhs, (t0, t0+burn_time), state, method='RK45',
                    rtol=1e-9, atol=1e-2, max_step=2.0)
    return sol.y[:, -1], sol.t[-1], burn_time


# ============================================================
# Phase: Powered Descent (the headline integration)
#
# Real Apollo descent had three sub-phases:
#   Braking phase (~10 min): aggressive deceleration from orbital velocity
#   Approach phase (~90 s):  pitch-up for landing visibility, controlled descent
#   Terminal phase (~30 s):  hover & touchdown
#
# We integrate Moon-centered inertial dynamics with throttle-modulated DPS
# and a feedback guidance law. Events: touchdown (alt → 0) or fuel exhausted.
# ============================================================
def phase_powered_descent(state_eci, t0, perturb=None, m0_override=None):
    perturb = perturb or {}
    isp_f = perturb.get("dps_isp_factor", 1.0)
    thr_f = perturb.get("dps_thrust_factor", 1.0)
    hover_seek_s = perturb.get("hover_seek_s", 0.0)   # Apollo 11 used ~40s

    isp = DPS_ISP * isp_f
    T_max = DPS_THRUST_MAX * thr_f
    T_min = DPS_THRUST_MIN * thr_f

    # Convert to Moon-centered inertial
    mr, mv = moon_state(t0)
    r0 = state_eci[:3] - mr
    v0 = state_eci[3:6] - mv
    # Full LM at descent start, less any DOI propellant already spent (m0_override).
    m0_lm = m0_override if m0_override is not None else (LM_DESC_TOTAL + LM_ASCT_TOTAL)
    state_mc = np.concatenate([r0, v0, [m0_lm]])

    # Track diagnostics
    diag = {"min_alt": 1e9, "max_decel": 0, "throttle_history": []}

    def rhs(t, y):
        r = y[:3]; v = y[3:6]; m = y[6]
        rn = np.linalg.norm(r)
        alt = rn - R_MOON
        diag["min_alt"] = min(diag["min_alt"], alt)
        r_hat = r / rn

        a_grav = -MU_MOON * r / rn**3
        a_grav = a_grav + lunar_nonspherical_accel(r, t)  # C20+C22 (flag-gated)

        v_radial = float(np.dot(v, r_hat))
        v_horiz_vec = v - v_radial * r_hat
        v_horiz_mag = np.linalg.norm(v_horiz_vec)
        h_hat = v_horiz_vec / v_horiz_mag if v_horiz_mag > 1e-3 else np.zeros(3)

        # Three-phase descent guidance (Apollo-style).
        if v_horiz_mag > 10.0:
            # ---- BRAKING ----
            # Full throttle, anti-velocity bias. Target sink rate is
            # altitude-dependent to avoid crashing while still horizontal.
            T = T_max
            a_total = T / m
            a_centripetal = v_horiz_mag**2 / rn
            if globals().get("ENABLE_DOI", False):
                # Efficient, P63-like braking that works from the real ~15 km PDI
                # altitude. Key idea: NEVER thrust downward to chase descent
                # (that "dives" and wastes propellant). Point thrust mostly
                # anti-horizontal-velocity and let GRAVITY provide the descent;
                # use vertical thrust only to SUPPORT against gravity (minus the
                # centripetal lift of the orbital speed) and to LIMIT the fall to
                # a gentle, altitude-scaled target. This minimizes gravity loss by
                # spending thrust on braking while orbital lift still offsets
                # weight, so the LM arrives near the surface as its speed bleeds.
                a_support = G_MOON - a_centripetal
                target_sink = -min(BRAKE_SINK_MAX_MS, alt / BRAKE_SINK_DIV)
                a_vert_needed = a_support - BRAKE_VERT_GAIN * (v_radial - target_sink)
                a_vert_needed = float(np.clip(a_vert_needed, 0.0, a_total))
            else:
                # Legacy braking (unchanged): allows mild downward thrust.
                # Sink rate: -40 m/s at high alt, slows to 0 near surface.
                # At 1 km alt: target = -8 m/s; at 200m alt: target = -2 m/s.
                target_v_radial_brake = -min(40.0, alt / 250.0)
                vert_err = v_radial - target_v_radial_brake
                a_vert_needed = (G_MOON - a_centripetal) - 0.5 * vert_err
                a_vert_needed = max(a_vert_needed, -2.5)
            if a_vert_needed >= a_total:
                t_dir = r_hat
            else:
                a_horiz_avail = np.sqrt(a_total**2 - a_vert_needed**2)
                t_dir = a_vert_needed * r_hat - a_horiz_avail * h_hat
                t_dir = t_dir / np.linalg.norm(t_dir)

        elif alt > 200.0:
            # ---- APPROACH ----
            # Aggressive descent. Throttle modulates to track target rate.
            target_v_radial = -(3.0 + 0.005 * alt)
            target_v_radial = max(target_v_radial, -35.0)
            vert_err = v_radial - target_v_radial
            a_vert_req = G_MOON - 0.07 * vert_err
            a_vert_req = max(a_vert_req, 0.0)
            a_horiz_brake = v_horiz_mag / 5.0
            a_req_mag = np.sqrt(a_vert_req**2 + a_horiz_brake**2)
            T = float(np.clip(m * a_req_mag, T_min, T_max))
            t_dir = (a_vert_req * r_hat - a_horiz_brake * h_hat)
            nrm = np.linalg.norm(t_dir)
            t_dir = t_dir/nrm if nrm > 1e-6 else r_hat

        else:
            # ---- TERMINAL ----
            # Hover & land. Smooth descent profile.
            target_v_radial = -(0.5 + 0.012 * alt)
            target_v_radial = max(target_v_radial, -10.0)
            vert_err = v_radial - target_v_radial
            hover_factor = 1.0 + (hover_seek_s / 500.0) if alt < 150 else 1.0
            a_vert_req = G_MOON * hover_factor - 0.4 * vert_err
            a_vert_req = max(a_vert_req, 0.3)
            a_horiz_brake = v_horiz_mag / 4.0 if v_horiz_mag > 0.3 else 0.0
            T = float(np.clip(m * np.sqrt(a_vert_req**2 + a_horiz_brake**2),
                                T_min, T_max))
            t_dir = (a_vert_req * r_hat - a_horiz_brake * h_hat)
            nrm = np.linalg.norm(t_dir)
            t_dir = t_dir/nrm if nrm > 1e-6 else r_hat

        a_thrust = T * t_dir / m
        a = a_grav + a_thrust
        mdot = -T / (isp * G0)
        diag["max_decel"] = max(diag["max_decel"], np.linalg.norm(a_thrust))
        return np.concatenate([v, a, [mdot]])

    # Events
    # With contact-probe modeling enabled, engine cutoff is commanded at
    # footpad-probe contact (~1.7 m), and the LM free-falls the remaining
    # distance. Baseline keeps the original 1 m touchdown threshold.
    cutoff_alt = (CONTACT_PROBE_LENGTH_M
                  if globals().get("ENABLE_DESCENT_FAILURE_MODES", False)
                  else 1.0)
    def touchdown(t, y):
        return np.linalg.norm(y[:3]) - R_MOON - cutoff_alt
    touchdown.terminal = True
    touchdown.direction = -1

    dry_floor = LM_DESC_DRY + LM_ASCT_TOTAL
    def fuel_out(t, y):
        return y[6] - dry_floor
    fuel_out.terminal = True
    fuel_out.direction = -1

    sol = solve_ivp(rhs, (t0, t0 + 1500), state_mc, method='RK45',
                    rtol=1e-7, atol=1e-2, max_step=0.5,
                    events=[touchdown, fuel_out])

    if len(sol.t_events[1]) > 0 and len(sol.t_events[0]) == 0:
        # Fuel exhausted before touchdown event fired
        s_fo = sol.y_events[1][0]
        t_fo = sol.t_events[1][0]
        r_fo = s_fo[:3]; v_fo = s_fo[3:6]
        rn = np.linalg.norm(r_fo)
        alt_fo = rn - R_MOON
        r_hat_fo = r_fo / rn
        v_rad_fo = float(np.dot(v_fo, r_hat_fo))
        v_horiz_fo = np.linalg.norm(v_fo - v_rad_fo * r_hat_fo)

        # Was this a survivable crash? Apollo 11 actually had ~25s of hover
        # fuel margin so this case is the failure mode. Report it honestly.
        return {
            "success": False, "reason": "descent_propellant_exhausted",
            "final_alt_m": alt_fo,
            "touchdown_speed_v_ms": v_rad_fo,
            "touchdown_speed_h_ms": v_horiz_fo,
            "prop_remaining_kg": 0.0,
            "fuel_margin_s": -1.0,   # negative = ran out before landing
            "trajectory_t": sol.t, "trajectory_y": sol.y,
            "final_t": t_fo,
        }

    if len(sol.t_events[0]) > 0:
        final_state = sol.y_events[0][0]
        final_t = sol.t_events[0][0]
        rfinal = final_state[:3]
        vfinal = final_state[3:6]
        r_hat = rfinal / np.linalg.norm(rfinal)
        v_radial = float(np.dot(vfinal, r_hat))
        v_horiz  = np.linalg.norm(vfinal - v_radial * r_hat)
        prop_rem = final_state[6] - dry_floor

        # Manual-flying / landing-redesignation fuel penalty (flag-gated). The
        # crew took semi-manual control near the surface and flew level/downrange
        # to clear terrain (Apollo 11's boulder field), burning extra hover
        # propellant — the dominant reason the real descent margin was tight.
        # Charge that extra low-altitude powered flight at the hover rate; if it
        # exhausts the reserve, the landing runs the tanks dry (a real risk).
        if globals().get("ENABLE_DOI", False):
            manual_s = float(perturb.get("manual_flying_s", 0.0))
            if manual_s > 0.0:
                m_h0 = LM_DESC_DRY + LM_ASCT_TOTAL + max(prop_rem, 0)
                mdot_h0 = m_h0 * G_MOON / (DPS_ISP * G0)
                prop_rem -= mdot_h0 * manual_s
                if prop_rem <= 0.0:
                    return {
                        "success": False, "reason": "descent_propellant_exhausted",
                        "final_alt_m": 0.0,
                        "touchdown_speed_v_ms": v_radial,
                        "touchdown_speed_h_ms": v_horiz,
                        "prop_remaining_kg": 0.0,
                        "fuel_margin_s": -1.0,
                        "trajectory_t": sol.t, "trajectory_y": sol.y,
                        "final_t": final_t,
                    }

        # Find lat/lon of landing site (in Moon-fixed frame)
        # Treat r_hat in inertial as roughly fixed lunar coords (slow Moon rotation)
        lat = np.rad2deg(np.arcsin(r_hat[2]))
        lon = np.rad2deg(np.arctan2(r_hat[1], r_hat[0]))

        # Hover-time fuel margin: how long can remaining descent prop hover
        # the full LM (which still includes the ascent stage on top)?
        # Hover thrust = m_total * g_moon; m_dot = thrust / (isp * g0)
        m_hover = LM_DESC_DRY + LM_ASCT_TOTAL + max(prop_rem, 0)
        mdot_hover = m_hover * G_MOON / (DPS_ISP * G0)
        fuel_margin = max(0.0, prop_rem) / mdot_hover

        result = {
            "success": True, "reason": "touchdown",
            "touchdown_speed_v_ms": v_radial,
            "touchdown_speed_h_ms": v_horiz,
            "prop_remaining_kg": prop_rem,
            "fuel_margin_s": fuel_margin,
            "land_lat_deg": lat, "land_lon_deg": lon,
            "trajectory_t": sol.t, "trajectory_y": sol.y,
            "final_t": final_t,
            "min_alt_m": diag["min_alt"],
            "max_decel_ms2": diag["max_decel"],
        }

        # Landing-accuracy outcome: mascon-induced downrange error vs target.
        # The C20/C22 field perturbs the trajectory physically (folded into the
        # landing already); this calibrated term adds the localized-mascon
        # downrange bias the low-degree field cannot resolve. Recorded as
        # accuracy, not charged to fuel (Apollo accepted landing long).
        if globals().get("ENABLE_LUNAR_HARMONICS", False):
            result["land_downrange_err_m"] = float(
                perturb.get("mascon_downrange_m", MASCON_DOWNRANGE_BIAS_M))

        # ---- Apollo 11-specific descent events (flag-gated) -------------
        if globals().get("ENABLE_DESCENT_FAILURE_MODES", False):
            # (1) Contact-probe free-fall: engine was cut at cutoff_alt, so the
            #     LM drops that height under lunar gravity before footpad
            #     contact, adding to the true vertical touchdown velocity.
            v_touch = np.sqrt(max(v_radial, 0.0)**2 + 2.0 * G_MOON * cutoff_alt)
            result["touchdown_speed_v_ms"] = -v_touch
            # Hard-landing check against the SOURCED LM touchdown velocity
            # envelope (NASA SP-2013-605, Apollo program review; corroborated by
            # LM chief engineer T. Kelly): the gear honeycomb was designed for a
            # sink (vertical) velocity up to 10 ft/s (3.05 m/s), with allowable
            # horizontal velocity coupled to sink speed — max sink 10 ft/s at
            # 0 ft/s horizontal, and for sink <= 7 ft/s (2.13 m/s) the horizontal
            # limit is 4 ft/s (1.22 m/s); between 7 and 10 ft/s sink the
            # horizontal allowance ramps down linearly from 4 to 0 ft/s. We flag
            # a hard landing when the (vertical, horizontal) pair falls outside
            # this envelope.
            V_SINK_MAX   = 3.05   # 10 ft/s
            V_SINK_KNEE  = 2.13   # 7 ft/s
            V_HORIZ_MAX  = 1.22   # 4 ft/s
            if v_touch <= V_SINK_KNEE:
                h_allow = V_HORIZ_MAX
            elif v_touch <= V_SINK_MAX:
                # linear ramp 4 ft/s (at 7 ft/s sink) -> 0 ft/s (at 10 ft/s sink)
                h_allow = V_HORIZ_MAX * (V_SINK_MAX - v_touch) / (V_SINK_MAX - V_SINK_KNEE)
            else:
                h_allow = 0.0
            result["hard_landing"] = bool(v_touch > V_SINK_MAX or v_horiz > h_allow)

            # (2) Slosh-sensor warning bias: when the low-level light fired
            #     relative to TRUE remaining fuel. Reported as a diagnostic;
            #     it compresses the crew window but does not change real prop.
            slosh_bias = float(perturb.get(
                "slosh_sensor_bias_s",
                SLOSH_SENSOR_BIAS_MEAN_S))
            result["lowlevel_light_early_s"] = slosh_bias
            result["effective_margin_s"] = fuel_margin - 0.0  # true margin

            # (3)/(4) Recoverable AGC alarms and landing-radar dropout. These
            #     are surfaced as event flags; escalation to abort is handled
            #     in run_mission only when co-occurring with a marginal state.
            result["agc_1202_alarm"] = bool(perturb.get("agc_1202_fired", False))
            result["agc_1202_recovered"] = bool(
                perturb.get("agc_1202_recovered", True))
            result["lr_dropout"] = bool(perturb.get("lr_dropout_fired", False))

        return result
    # Timeout
    return {
        "success": False, "reason": "descent_timeout",
        "final_alt_m": np.linalg.norm(sol.y[:3, -1]) - R_MOON,
        "trajectory_t": sol.t, "trajectory_y": sol.y,
        "final_t": sol.t[-1],
    }


# ============================================================
# Phase: LM ascent — APS burn from surface to lunar orbit
# ============================================================
def phase_ascent_burn(t0, perturb=None, csm_plane_normal=None):
    """LM ascent to lunar orbit, inserting INTO THE CSM ORBITAL PLANE.

    Real Apollo timed the LM liftoff to the CSM's overhead pass and steered
    during powered ascent so the ascent stage inserted nearly coplanar with the
    CSM (small residual yaw-steering error only). The previous version instead
    launched "due east" from a fixed lunar-equatorial site, fixing the LM plane
    to the equator regardless of the CSM's actual (inclined) plane — producing
    a large, ARTIFACTUAL plane mismatch at rendezvous.

    This version takes the CSM orbital-plane normal and:
      * places the launch site at the actual landing-site direction PROJECTED
        into the CSM plane (the closest in-plane point — i.e. the timed-liftoff
        geometry that puts the CSM ground track over the site), and
      * builds the ascent steering frame from the in-plane "up" and the in-plane
        prograde direction (h x up), so thrust stays in the CSM plane and the
        nominal insertion is coplanar (zero nominal plane error).
    A small `ascent_yaw_err_deg` perturbation tilts the launch azimuth out of
    plane, producing a PHYSICAL (small) plane error that the rendezvous model
    then legitimately pays for. If no CSM plane is supplied, falls back to the
    old equatorial-east launch (keeps the function usable standalone).
    """
    perturb = perturb or {}
    isp = APS_ISP * perturb.get("aps_isp_factor", 1.0)
    T   = APS_THRUST * perturb.get("aps_thrust_factor", 1.0)
    m0  = LM_ASCT_TOTAL

    mr, mv = moon_state(t0)

    # Landing-site direction (Moon-relative), approximate Tranquility.
    lat = np.deg2rad(0.674)
    lon = np.deg2rad(23.473)
    u_site = np.array([np.cos(lat)*np.cos(lon),
                       np.cos(lat)*np.sin(lon),
                       np.sin(lat)])

    if csm_plane_normal is not None and np.linalg.norm(csm_plane_normal) > 0:
        h_hat = np.asarray(csm_plane_normal, dtype=float)
        h_hat = h_hat / np.linalg.norm(h_hat)
        # Project the landing-site direction into the CSM plane (remove the
        # out-of-plane component) -> closest in-plane launch point.
        u_in = u_site - np.dot(u_site, h_hat) * h_hat
        if np.linalg.norm(u_in) < 1e-6:
            # Degenerate (site near the plane pole): pick any in-plane axis.
            tmp = np.array([1.0, 0.0, 0.0])
            u_in = tmp - np.dot(tmp, h_hat) * h_hat
        up0 = u_in / np.linalg.norm(u_in)
        # In-plane prograde (direction of CSM motion at this point): h x up.
        prograde0 = np.cross(h_hat, up0)
        prograde0 /= np.linalg.norm(prograde0)
        # Optional out-of-plane yaw-steering error (degrees): tilt the launch
        # azimuth toward the plane normal by the error angle.
        yaw_err = np.deg2rad(perturb.get("ascent_yaw_err_deg", 0.0))
        if yaw_err != 0.0:
            prograde0 = (np.cos(yaw_err) * prograde0
                         + np.sin(yaw_err) * h_hat)
            prograde0 /= np.linalg.norm(prograde0)
    else:
        # Fallback: original equatorial-east launch.
        up0 = u_site / np.linalg.norm(u_site)
        east = np.cross(np.array([0.0, 0.0, 1.0]), up0)
        prograde0 = east / np.linalg.norm(east) if np.linalg.norm(east) > 1e-3 \
            else np.array([1.0, 0.0, 0.0])

    r_mc = R_MOON * up0
    v_mc = np.zeros(3)   # at rest in Moon frame
    state_mc = np.concatenate([r_mc, v_mc, [m0]])

    def rhs(t, y):
        r = y[:3]; v = y[3:6]; m = y[6]
        rn = np.linalg.norm(r)
        r_hat = r / rn
        a_grav = -MU_MOON * r / rn**3
        a_grav = a_grav + lunar_nonspherical_accel(r, t)  # C20+C22 (flag-gated)
        # Apollo LM ascent pitch program (matches AS-506 ascent profile):
        #   0-10s:   vertical climb
        #   10-60s:  smooth pitch from vertical to ~50° from vertical
        #   60s+:    hold ~50° from vertical (gravity turn target)
        # Cutoff is event-driven on circular orbital velocity at altitude.
        # The downrange (pitch) direction is the in-plane prograde direction
        # projected to stay perpendicular to the current radius, so the whole
        # ascent remains in the CSM orbital plane (up to the yaw-error tilt).
        tau = t - t0
        if tau < 10:
            pitch = 0.0
        elif tau < 60:
            pitch = np.deg2rad(50.0 * (tau - 10) / 50.0)
        else:
            pitch = np.deg2rad(50.0)

        up = r_hat
        # Downrange direction: remove any radial component from prograde0 and
        # renormalize, so it is horizontal at the current position.
        downrange = prograde0 - np.dot(prograde0, up) * up
        dn = np.linalg.norm(downrange)
        downrange = downrange / dn if dn > 1e-6 else prograde0
        t_dir = np.cos(pitch) * up + np.sin(pitch) * downrange

        if m > LM_ASCT_DRY:
            a_thr = T * t_dir / m
            mdot = -T / (isp * G0)
        else:
            a_thr = np.zeros(3)
            mdot = 0.0
        return np.concatenate([v, a_grav + a_thr, [mdot]])

    # Event: stop burn when orbital energy reaches that of a 60 km circular orbit
    target_radius = R_MOON + 60_000.0
    target_energy = -MU_MOON / (2 * target_radius)
    def insertion_event(t, y):
        r = y[:3]; v = y[3:6]
        if t - t0 < 30:    # don't trigger too early
            return 1.0
        rn = np.linalg.norm(r)
        # Wait until altitude is at least 15 km
        if rn - R_MOON < 15_000:
            return 1.0
        E = 0.5 * np.dot(v, v) - MU_MOON / rn
        return target_energy - E
    insertion_event.terminal = True
    insertion_event.direction = -1

    # Also fail-safe: stop when propellant runs out
    def fuel_event(t, y):
        return y[6] - LM_ASCT_DRY
    fuel_event.terminal = True
    fuel_event.direction = -1

    # Integrate up to 800s
    sol = solve_ivp(rhs, (t0, t0 + 800.0), state_mc, method='RK45',
                    rtol=1e-8, atol=1e-2, max_step=2.0,
                    events=[insertion_event, fuel_event])

    # Final state in ECI
    mr_f, mv_f = moon_state(sol.t[-1])
    final_eci = np.concatenate([sol.y[:3,-1]+mr_f, sol.y[3:6,-1]+mv_f, [sol.y[6,-1]]])
    final_alt = np.linalg.norm(sol.y[:3,-1]) - R_MOON
    return {
        "final_state": final_eci, "final_t": sol.t[-1],
        "final_alt_km": final_alt/1e3,
        "prop_remaining_kg": sol.y[6,-1] - LM_ASCT_DRY,
        "success": final_alt > 5_000,
    }


# ============================================================
# Phase: TEI — SPS prograde burn (escape Moon, head for Earth)
# ============================================================
def phase_tei(state, t0, dv_target=1008.0, perturb=None):
    perturb = perturb or {}
    isp = SPS_ISP * perturb.get("sps_isp_factor", 1.0)
    T   = SPS_THRUST * perturb.get("sps_thrust_factor", 1.0)

    m0 = state[6]
    mp = m0 * (1 - np.exp(-dv_target/(isp*G0)))
    burn_time = mp / (T/(isp*G0))

    # Fix burn direction at ignition: prograde wrt Moon at start of burn
    # (inertially fixed for the duration of the burn — real SPS practice)
    mr0, mv0 = moon_state(t0)
    v_rel0 = state[3:6] - mv0
    burn_dir = v_rel0 / np.linalg.norm(v_rel0)

    def rhs(t, y):
        r = y[:3]; v = y[3:6]; m = y[6]
        rl = r - moon_state(t)[0]
        a = -MU_MOON * rl / np.linalg.norm(rl)**3
        a = a + lunar_nonspherical_accel(rl, t)  # C20+C22 (flag-gated)
        a = a - MU_EARTH * r / np.linalg.norm(r)**3
        a = a + T * burn_dir / m
        return np.concatenate([v, a, [-T/(isp*G0)]])

    sol = solve_ivp(rhs, (t0, t0+burn_time), state, method='RK45',
                    rtol=1e-9, atol=1e-2, max_step=2.0)
    return sol.y[:, -1], sol.t[-1], burn_time


# ============================================================
# Phase: Trans-Earth coast — terminates at entry interface
# ============================================================
def phase_transearth_coast(state, t0, duration=7*86400):
    def rhs(t, y):
        return np.concatenate([y[3:6], gravity_earth_moon(y[:3], t), [0]])

    def entry(t, y):
        return np.linalg.norm(y[:3]) - (R_EARTH + 121_920)
    entry.terminal = True
    entry.direction = -1

    sol = solve_ivp(rhs, (t0, t0+duration), state, method='RK45',
                    rtol=1e-9, atol=1e-1, max_step=600.0,
                    events=entry, dense_output=True)

    if len(sol.t_events[0]) > 0:
        t_ei = sol.t_events[0][0]
        y_ei = sol.y_events[0][0]
        r = y_ei[:3]; v = y_ei[3:6]
        r_hat = r/np.linalg.norm(r); v_mag = np.linalg.norm(v)
        v_rad = np.dot(v, r_hat)
        fpa = np.rad2deg(np.arcsin(v_rad / v_mag))
        return {
            "final_state": y_ei, "final_t": t_ei,
            "fpa_at_entry_deg": fpa, "entry_speed_ms": v_mag,
            "sol": sol,
            "reached_entry": True,
        }
    return {
        "final_state": sol.y[:,-1], "final_t": sol.t[-1],
        "fpa_at_entry_deg": None, "entry_speed_ms": None,
        "reached_entry": False, "sol": sol,
    }


# ============================================================
# Phase: Trans-Earth midcourse corrections (MCC-5 / MCC-6 / MCC-7)
# ============================================================
#
# SCAFFOLD — built ahead of the Apollo-TEI-methodology research landing.
# This is the dominant robustness mechanism Apollo used: rather than relying
# on TEI precision alone, the trans-Earth coast included scheduled midcourse
# correction opportunities that measured the projected entry flight-path angle
# (FPA) from the current state vector and applied a small RCS/SPS delta-V to
# walk it into the reentry corridor. Small, late corrections cost very little
# delta-V because the lever arm to entry shrinks as the spacecraft nears Earth.
#
# This function is IMPLEMENTED (not a scaffold): it performs a real targeting
# solve — at each scheduled opportunity it projects the entry FPA via the
# trans-Earth coast integrator and, if outside the deadband, solves for the
# along-track delta-V (bracket-and-bisect) that drives the projected FPA to the
# corridor center. The deadband, execution residual, and RCS Isp below are
# SOURCED (NASA TN D-6725 and the Apollo 11 Flight Journal; see per-constant
# notes). The correction SCHEDULE (MCC-5/6/7 offsets) is still an ESTIMATE.
#
# Apollo trans-Earth MCC schedule (offsets are ESTIMATES; magnitudes/cadence
# approximate the real GET-based plan):
#   MCC-5: TEI + ~15 h      (first cleanup once tracking re-converges)
#   MCC-6: EI  - ~22 h      (mid-coast trim)
#   MCC-7: EI  - ~3 h       (final corridor placement)
# Most Apollo flights only needed one or two of these; many were waived when
# the projected corridor was already acceptable. The function models all three
# as conditional burns gated on the deadband.
#
# Corridor target & deadband are SOURCED below (NASA TN D-6725): Apollo's entry
# corridor was specified as inertial FPA at the entry interface (~400,000 ft /
# 121,920 m), nominal -6.5 deg, with an asymmetric correction deadband.
MCC_TARGET_FPA_DEG    = -6.5    # Apollo 11 nominal entry FPA (NASA TN D-6725)
# Apollo midcourse-correction logic did NOT correct if the entry flight-path
# angle was within 0.1 deg on the steep side and 0.2 deg on the shallow side
# of target (NASA TN D-6725). Asymmetric deadband, replacing the prior 0.25
# symmetric placeholder.
MCC_FPA_DEADBAND_STEEP_DEG   = 0.10   # steep-side half-band
MCC_FPA_DEADBAND_SHALLOW_DEG = 0.20   # shallow-side half-band
# Post-burn execution residual observed on Apollo 11 (Flight Journal, MCC-2:
# residuals "on the order of a half a foot a second or less"). ~0.15 m/s.
MCC_EXEC_RESIDUAL_MS  = 0.15    # 1-sigma-class execution error magnitude
# Primary trans-Earth corrections were trimmed with the SM RCS thrusters;
# the large SPS was the backup mode (NASA TN D-6725).
MCC_RCS_ISP_S         = 290.0   # SM RCS Isp (approx) for small MCC trims


ENABLE_TEI_TARGETING = True   # B2 (faithful): solve the 3-DOF TEI burn VECTOR
                   # (robust trf least-squares, Jacobian-scaled, CONDITIONAL
                   # target homotopy) to drive the trans-Earth trajectory
                   # through the nominal entry point + corridor, so the trans-
                   # Earth MCCs become small trims. VALIDATED: splashdown
                   # dispersion collapses from ~5860 km to ~40-200 km with
                   # trans-Earth MCC ~0-6 m/s (Apollo 11 actual 1.5) and FPA in
                   # the corridor, robust across seeds (no stalls).
                   # PERF: a cheap single solve handles small/moderate
                   # dispersion (~20 s/trial, vs ~15 s baseline mission); the
                   # expensive homotopy fallback (N=6 warm-started steps) runs
                   # ONLY for large-dispersion trials (~75 s). The tight coast
                   # (rtol 1e-9) is REQUIRED for convergence — loosening it
                   # reintroduces local-min stalls — so the homotopy path is
                   # inherently integration-bound. A full 1000-trial MC with
                   # this on is ~12 h (overnight); ~200-300 trials give a solid
                   # dispersion estimate in a few hours. Default ON (= True
                   # above); set False to fall back to the 1-DOF magnitude
                   # search.
ENABLE_ENTRY_TARGETING = False  # B2: 3-DOF differential corrector at MCC-6 that
                   # perturbed trials through the nominal entry point. Default
                   # OFF: mechanism is validated (collapses EI/splashdown
                   # dispersion) but not yet faithful — correcting the large EI
                   # dispersion late (MCC-6) needs ~200 m/s vs Apollo's ~1.5,
                   # and convergence is not yet robust. See session notes.
_EI_TARGET = None  # nominal entry-interface target: dict(lat, lon, fpa).


def phase_transearth_mcc(post_burn_state, t_post_burn, perturb,
                          target_fpa_deg=MCC_TARGET_FPA_DEG,
                          deadband_steep_deg=MCC_FPA_DEADBAND_STEEP_DEG,
                          deadband_shallow_deg=MCC_FPA_DEADBAND_SHALLOW_DEG):
    """Apply scheduled trans-Earth midcourse corrections to walk the projected
    entry FPA into the corridor.

    IMPLEMENTED: performs a real along-track targeting solve at each scheduled
    opportunity (project entry FPA -> if outside deadband, bracket-and-bisect on
    delta-V to drive FPA to target), applies the sourced execution residual, and
    accounts propellant on the SM RCS Isp. The deadband/residual/Isp are sourced;
    the schedule offsets are estimates (see module comment above).

    Parameters
    ----------
    post_burn_state : (7,) ndarray   ECI position/velocity/mass just after TEI
    t_post_burn     : float          time (s) at end of TEI burn
    perturb         : dict|None       perturbation draw for this trial
    target_fpa_deg  : float          corridor center to aim the coast at
    deadband_steep_deg   : float     skip correction if FPA is within this on
                                      the steep (more-negative) side of target
    deadband_shallow_deg : float     skip correction if FPA is within this on
                                      the shallow (less-negative) side of target

    Returns
    -------
    dict with keys:
        final_state, final_t        entry-interface state/time (or last coast)
        reached_entry               bool
        fpa_at_entry_deg            achieved inertial FPA at EI
        entry_speed_ms
        mcc_burns                   list of {name, t, dv_ms, fpa_before, fpa_after}
        mcc_total_dv_ms             scalar sum of correction magnitudes
        sol                         final coast solution (for trajectory capture)
    """
    if perturb is None:
        perturb = {}

    isp_mcc = MCC_RCS_ISP_S * perturb.get("sps_isp_factor", 1.0)

    def rhs_em(t, y):
        return np.concatenate([y[3:6], gravity_earth_moon(y[:3], t), [0]])

    # --- helper: project the entry interface from an arbitrary coast state ---
    # Reuses the existing trans-Earth coast integrator so the projection and
    # the final propagation share identical dynamics.
    def project_entry(state, t0):
        return phase_transearth_coast(state, t0)

    # --- helper: solve the delta-V (along velocity) that drives the projected
    # inertial entry FPA to want_fpa_deg ---------------------------------------
    # Apollo trans-Earth corrections were small trims targeting the entry
    # corridor; the entry FPA is monotonic in along-track delta-V near the
    # corridor (a small prograde nudge raises pericenter -> shallower entry; a
    # retrograde nudge lowers it -> steeper). We bracket on delta-V magnitude
    # and bisect on the projected-FPA error. This is the same shooting approach
    # validated on the outbound (perilune) chain, with entry FPA as the
    # controlled variable instead of perilune altitude. (A full RTCC solution
    # used B-plane partials; a 1-D along-track shoot is the dominant term and
    # is sufficient at this model's fidelity.)
    def solve_correction(state, t0, want_fpa_deg):
        v = state[3:6]
        v_hat = v / np.linalg.norm(v)

        def fpa_err(dv_mag):
            s = state.copy()
            s[3:6] = v + dv_mag * v_hat
            c = project_entry(s, t0)
            if not c["reached_entry"] or c["fpa_at_entry_deg"] is None:
                # No entry within window: treat a skip-out (too shallow) as a
                # large positive error and an impact (too steep) as large
                # negative, so the bracket still has a sign to chase.
                return 90.0
            return c["fpa_at_entry_deg"] - want_fpa_deg

        # Sample a modest along-track range (these are small trims; entry FPA
        # is very sensitive, so a few m/s spans the corridor and beyond).
        grid = np.linspace(-8.0, 8.0, 9)
        samples = [(dv, fpa_err(dv)) for dv in grid]
        # Find a sign change to bracket the root.
        best_dv = 0.0
        bracket = None
        for i in range(len(samples) - 1):
            a, fa = samples[i]; b, fb = samples[i+1]
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                bracket = (a, b); break
        if bracket is not None:
            try:
                from scipy.optimize import brentq
                best_dv = brentq(fpa_err, bracket[0], bracket[1],
                                 xtol=0.05, maxiter=12)
            except Exception:
                best_dv = min(samples, key=lambda s: abs(s[1]))[0]
        else:
            # No sign change in range: take the smallest-|error| sample.
            best_dv = min(samples, key=lambda s: abs(s[1]))[0]
        return float(best_dv), v_hat

    # --- helper: inject execution error into a commanded correction ---------
    # Apollo MCC post-burn residuals were ~0.15 m/s (Apollo 11 Flight Journal,
    # MCC-2: "half a foot a second or less"). Model that as a small random
    # residual in magnitude and direction left over after the burn.
    rng_exec = np.random.default_rng(
        int(abs(hash((float(t_post_burn), float(post_burn_state[0])))) % 2**31))
    def apply_execution_error(dv_vec):
        resid = rng_exec.normal(0.0, MCC_EXEC_RESIDUAL_MS, size=3)
        return dv_vec + resid

    # --- inside/outside the asymmetric corridor deadband --------------------
    def inside_deadband(fpa_deg):
        # Steep side = more negative than target; shallow side = less negative.
        if fpa_deg is None:
            return False
        err = fpa_deg - target_fpa_deg            # <0 steeper, >0 shallower
        if err < 0:
            return (-err) <= deadband_steep_deg
        return err <= deadband_shallow_deg

    # --- execute one correction at the current coast state ------------------
    def do_correction(name, state, t0):
        nonlocal_burn = {"name": name, "t": t0}
        c0 = project_entry(state, t0)
        fpa_before = c0["fpa_at_entry_deg"]
        if inside_deadband(fpa_before):
            nonlocal_burn.update({"dv_ms": 0.0, "fpa_before": fpa_before,
                                  "fpa_after": fpa_before, "waived": True})
            return state, 0.0, 0.0, nonlocal_burn
        dv_mag, v_hat = solve_correction(state, t0, target_fpa_deg)
        dv_vec = apply_execution_error(dv_mag * v_hat)
        s = state.copy()
        s[3:6] = s[3:6] + dv_vec
        m0 = s[6]
        dv_actual = np.linalg.norm(dv_vec)
        dprop = m0 * (1 - np.exp(-dv_actual / (isp_mcc * G0)))
        s[6] = m0 - dprop
        c1 = project_entry(s, t0)
        nonlocal_burn.update({"dv_ms": float(dv_actual),
                              "fpa_before": fpa_before,
                              "fpa_after": c1["fpa_at_entry_deg"],
                              "waived": False})
        return s, dv_actual, dprop, nonlocal_burn

    # --- 3-DOF entry-interface targeting (authentic precise entry targeting) -
    # Apollo's MCCs targeted the full entry-interface conditions, not just FPA;
    # the entry guidance then steered within a few-hundred-km footprint to the
    # recovery point. A 1-DOF along-track FPA solve leaves the entry POINT
    # uncorrected, so the entry point (and splashdown) disperses by thousands of
    # km. This corrector drives the trajectory through the nominal EI position
    # at the nominal EI time via the 3-DOF correction delta-V (a numerical
    # B-plane/Lambert solve: Newton with a numerical 3x3 sensitivity Jacobian).
    def _solve_ei_targeting(state, t0, tgt):
        import sys
        v0 = state[3:6].copy()
        lat_t, lon_t, fpa_t = tgt["lat"], tgt["lon"], tgt["fpa"]

        def err(dv):
            # Error vector at the EI crossing: geographic lat/lon miss (scaled to
            # metres) and FPA miss. Targeting the Earth-fixed entry point makes
            # it time-decoupled (we don't care WHEN it crosses, only WHERE +
            # corridor), so the splashdown — flown relative to the rotating
            # Earth — is consistent. 3 independent constraints, 3-DOF dV.
            s = state.copy(); s[3:6] = v0 + dv
            c = project_entry(s, t0)
            if not c["reached_entry"] or c["fpa_at_entry_deg"] is None:
                return None
            la, lo = eci_to_latlon(c["final_state"][:3], c["final_t"])
            e_lat = (la - lat_t) * (np.pi / 180.0) * R_EARTH
            dlon = ((lo - lon_t + 180.0) % 360.0) - 180.0
            e_lon = dlon * (np.pi / 180.0) * R_EARTH * np.cos(np.radians(la))
            e_fpa = (c["fpa_at_entry_deg"] - fpa_t) * 1.0e5   # 0.1 deg ~ 10 km
            return np.array([e_lat, e_lon, e_fpa])

        e0 = err(np.zeros(3))
        if e0 is None:
            return None
        err0 = np.linalg.norm(e0)

        def jac(at, base):
            J = np.zeros((3, 3))
            for k in range(3):
                pert = np.zeros(3); pert[k] = 1.0     # 1 m/s probe per axis
                ek = err(at + pert)
                if ek is None:
                    return None
                J[:, k] = (ek - base) / 1.0
            return J

        dv = np.zeros(3); lam = 1e-3; it = 0
        J = jac(dv, e0)
        if J is None:
            return None
        for it in range(15):
            if np.linalg.norm(e0) < 3000.0:           # ~3 km combined residual
                break
            JTJ = J.T @ J; JTe = J.T @ e0; stepped = False
            for _try in range(8):                     # adaptive LM damping
                A_lm = JTJ + lam * np.diag(np.diag(JTJ) + 1e-9)
                try:
                    step = np.linalg.solve(A_lm, -JTe)
                except np.linalg.LinAlgError:
                    lam *= 10; continue
                en = err(dv + step)
                if en is not None and np.linalg.norm(en) < np.linalg.norm(e0):
                    dv = dv + step; e0 = en
                    lam = max(lam / 3.0, 1e-7); stepped = True; break
                lam *= 10
            if not stepped:                            # stalled: refresh Jacobian
                J = jac(dv, e0)
                if J is None or lam > 1e6:
                    break
            if np.linalg.norm(dv) > 300.0:             # implausible -> bail
                return None
        return dv

    def do_ei_correction(name, state, t0):
        eit = globals().get("_EI_TARGET")
        brec = {"name": name, "t": t0}
        dv_vec = _solve_ei_targeting(state, t0, eit)
        if dv_vec is None:
            return do_correction(name, state, t0)  # fall back to FPA trim
        c0 = project_entry(state, t0)
        dv_vec = apply_execution_error(dv_vec)
        s = state.copy()
        m0 = s[6]
        s[3:6] = s[3:6] + dv_vec
        dv_actual = np.linalg.norm(dv_vec)
        dprop = m0 * (1 - np.exp(-dv_actual / (isp_mcc * G0)))
        s[6] = m0 - dprop
        c1 = project_entry(s, t0)
        brec.update({"dv_ms": float(dv_actual),
                     "fpa_before": c0["fpa_at_entry_deg"],
                     "fpa_after": c1["fpa_at_entry_deg"], "waived": False,
                     "ei_targeted": True})
        return s, dv_actual, dprop, brec

    # --- scheduled correction opportunities ---------------------------------
    # MCC-5 at TEI+15h; MCC-6/7 resolved relative to the projected entry time
    # (EI-22h, EI-3h). Each is waived if the projected FPA is already inside
    # the corridor deadband, matching how most Apollo MCCs were skipped.
    mcc_burns = []
    mcc_total_dv = 0.0
    prop_used = 0.0
    cur = post_burn_state.copy()
    cur_t = t_post_burn

    # First projection to locate the entry interface time for EI-relative burns.
    proj0 = project_entry(cur, cur_t)
    t_ei = proj0["final_t"] if proj0["reached_entry"] else cur_t + 3*86400

    schedule = [("MCC-5", t_post_burn + 15*3600.0),
                ("MCC-6", t_ei - 22*3600.0),
                ("MCC-7", t_ei - 3*3600.0)]

    for name, t_corr in schedule:
        if not (cur_t < t_corr < t_ei):
            continue
        # Coast to the correction time.
        sol = solve_ivp(rhs_em, (cur_t, t_corr), cur, method='RK45',
                        rtol=1e-8, atol=1e-1, max_step=600.0)
        cur = sol.y[:, -1].copy(); cur_t = sol.t[-1]
        if (globals().get("ENABLE_ENTRY_TARGETING", False)
                and globals().get("_EI_TARGET") is not None
                and name == "MCC-6"):
            cur, dv, dprop, brec = do_ei_correction(name, cur, cur_t)
        else:
            cur, dv, dprop, brec = do_correction(name, cur, cur_t)
        mcc_burns.append(brec)
        mcc_total_dv += dv
        prop_used += dprop
        # Refresh entry-time estimate after a real burn (it can shift).
        if not brec["waived"]:
            pj = project_entry(cur, cur_t)
            if pj["reached_entry"]:
                t_ei = pj["final_t"]

    coast = project_entry(cur, cur_t)
    # Capture the nominal entry-interface target on the first (reference) run so
    # subsequent perturbed runs can be steered through the same entry point at
    # the same time (collapsing entry-interface position dispersion).
    if (globals().get("ENABLE_ENTRY_TARGETING", False)
            and globals().get("_EI_TARGET") is None and coast["reached_entry"]
            and coast.get("final_state") is not None):
        _la, _lo = eci_to_latlon(coast["final_state"][:3], coast["final_t"])
        globals()["_EI_TARGET"] = {"lat": float(_la), "lon": float(_lo),
                                   "fpa": float(coast["fpa_at_entry_deg"])}

    return {
        "final_state":      coast["final_state"],
        "final_t":          coast["final_t"],
        "reached_entry":    coast["reached_entry"],
        "fpa_at_entry_deg": coast["fpa_at_entry_deg"],
        "entry_speed_ms":   coast["entry_speed_ms"],
        "mcc_burns":        mcc_burns,
        "mcc_total_dv_ms":  mcc_total_dv,
        "sps_prop_used_kg": prop_used,
        "sol":              coast["sol"],
    }


# ============================================================
# Predictive peak-load-factor estimator for the entry 10g limiter
# ============================================================
def _predict_peak_g_liftup(r, v_eci, m, max_time_s=120.0, dt=2.0,
                           bank_deg=0.0):
    """Predict the peak aerodynamic load factor (in g) of the current entry
    dip ASSUMING a constant-bank attitude is held from now on (default 0 =
    full lift-vector-up, the limiter's assumption; nonzero bank scales the
    vertical lift by cos(bank) — the dominant dip-depth effect — letting the
    HUNTEST dip-shaping solve "the largest bank1 whose dip stays under the
    load cap" on this same cheap propagator).

    This implements the prediction underlying the Apollo entry-guidance 10g
    limiter (NASA TN D-6725): the limiter rolls to lift-vector-up when the
    navigated state would otherwise exceed a 10g peak. Rather than a fragile
    closed-form, we cheaply forward-propagate a point-mass CM under drag +
    full-up lift + Earth gravity (no Earth rotation; the dip is short and the
    relative-velocity correction is negligible for a peak-g estimate), and
    return the maximum drag encountered until the vehicle stops descending or
    the time budget expires.

    It is intentionally lightweight (fixed-step, ~a few hundred steps) because
    it is evaluated inside the guidance loop. It does not need to be a precise
    trajectory — only an accurate-enough peak estimate to trigger the limiter.
    """
    cd_a = CM_CD * CM_AREA
    ld = CM_LD
    r = np.array(r, dtype=float)
    v = np.array(v_eci, dtype=float)
    peak_g = 0.0
    n_steps = int(max_time_s / dt)
    for _ in range(n_steps):
        rn = np.linalg.norm(r)
        alt = rn - R_EARTH
        if alt < 0:
            break
        vmag = np.linalg.norm(v)
        if vmag < 1.0:
            break
        rho = atm_density(alt)
        q = 0.5 * rho * vmag**2
        drag_g = q * cd_a / m / G0
        if drag_g > peak_g:
            peak_g = drag_g
        # Stop once the vehicle is clearly climbing back out of the dip: the
        # peak load factor has been passed.
        v_radial = np.dot(v, r / rn)
        if v_radial > 0 and drag_g < 0.5:
            break
        # Accelerations: drag opposite velocity, lift along +radial (full up),
        # gravity toward Earth center.
        v_hat = v / vmag
        a_drag = -q * cd_a / m * v_hat
        # Lift magnitude = (L/D) * |drag accel|, directed "up" (away from Earth)
        # in the plane perpendicular to velocity.
        r_hat = r / rn
        up_perp = r_hat - np.dot(r_hat, v_hat) * v_hat
        up_norm = np.linalg.norm(up_perp)
        if up_norm > 1e-6:
            up_perp /= up_norm
        a_lift = (ld * np.linalg.norm(a_drag)
                  * np.cos(np.deg2rad(bank_deg)) * up_perp)
        a_grav = -MU_EARTH * r / rn**3
        a = a_drag + a_lift + a_grav
        # Semi-implicit Euler step
        v = v + a * dt
        r = r + v * dt
    return peak_g


# ============================================================
# Phase: Atmospheric entry — full integration with drag + lift
# ============================================================
_ENTRY_REF = None  # cached reference entry trajectory: dict(v[], R[], bank[],
                   # D[], lat_deg, lon_deg). Built from the first (nominal)
                   # g-limited entry; the authentic Apollo drag-tracking law
                   # then steers dispersed entries back to this g-safe reference
                   # profile, collapsing splashdown dispersion toward nominal.
_ENTRY_REF_LOG = None  # transient (velocity, bank, drag, range) log captured
                       # during reference generation.


def phase_entry(state, t0, perturb=None,
                  target_lat_deg=SPLASH_TARGET_LAT_DEG,
                  target_lon_deg=SPLASH_TARGET_LON_DEG):
    """Apollo Earth-entry guidance (programs P64/P65/P66/P67 simplified).

    The Apollo Command Module enters Earth's atmosphere at ~11 km/s on a
    -6.5° flight path angle. To dissipate hyperbolic-class kinetic energy
    while keeping peak deceleration to crew-survivable ~6.5 g and to land
    accurately, Apollo used a sophisticated bank-angle modulation scheme
    documented in the AGC source (Comanche055/REENTRY_CONTROL.agc).

    The guidance has 4 phases (in addition to pre-entry P63):
      P64 Post-0.05g:   First atmospheric pass. Bank angle modulated to
                        control altitude/g-load and range.
      P65 Up-Control:   Skip phase (vehicle climbing back out of atmosphere).
                        Bank tuned so skip apogee is right for re-entry.
      P66 Ballistic:    High-altitude coast (no aero control).
      P67 Final Phase:  Second entry. Apollo's final-phase guidance law:
                        bank chosen so predicted landing is at target.

    My implementation uses a predictor-corrector for P64/P67: at each
    guidance update, propagate forward at current bank and compute where
    we'll land; adjust bank to move predicted landing toward target.
    Lateral guidance reverses bank sign when crossrange error grows too
    large, mimicking Apollo's lateral logic.
    """
    perturb = perturb or {}
    cd_a = CM_CD * CM_AREA * perturb.get("cd_factor", 1.0)
    ld = CM_LD * perturb.get("ld_factor", 1.0)

    # Target landing site in ECEF
    target_lat = np.deg2rad(target_lat_deg)
    target_lon = np.deg2rad(target_lon_deg)
    target_ecef = R_EARTH * np.array([np.cos(target_lat)*np.cos(target_lon),
                                       np.cos(target_lat)*np.sin(target_lon),
                                       np.sin(target_lat)])

    # State for guidance — mutable across rhs calls
    bank_cmd = [60.0]     # bank angle in degrees (initial guess)
    bank_sign = [+1.0]    # bank sign (+1 = lift component to North, -1 = to South)
    phase = ["P64"]       # current Apollo program
    # Predictor-corrector cached state (skip-entry guidance): the range-solving
    # bank is expensive, so we solve it infrequently and hold it between solves.
    pc_state = {"bank": 45.0, "t_last": -1.0e18}
    max_g = [0.0]
    max_q = [0.0]
    min_alt_seen = [1e9]
    last_guidance_t = [t0]
    GUIDANCE_INTERVAL = 5.0    # seconds between guidance updates

    # Earth rotation rate vector (ECI z-axis)
    omega_e_vec = np.array([0, 0, OMEGA_E])

    # --- Entry-interface conditions, captured once at the start ----------
    # The corridor width is governed by how well the P64 drag reference is
    # matched to the actual entry flight-path angle. A single fixed reference
    # only captures a narrow band of entry angles; Apollo's guidance adapted
    # the reference (and the depth of the first dip) to the entry conditions.
    # We compute the entry FPA here and use it to schedule an adaptive drag
    # reference in get_phase_and_bank, which is what widens the survivable
    # corridor from ~0.5 deg to the Apollo-class ~2 deg.
    _r0 = state[:3]; _v0 = state[3:6]
    _rn0 = np.linalg.norm(_r0)
    _vair0 = np.cross(omega_e_vec, _r0)
    _vrel0 = _v0 - _vair0
    _vrel0_mag = np.linalg.norm(_vrel0)
    entry_fpa_deg = float(np.rad2deg(np.arcsin(
        np.clip(np.dot(_vrel0, _r0 / _rn0) / max(_vrel0_mag, 1.0), -1, 1))))
    entry_speed_ms = float(_vrel0_mag)
    # Entry-point anchor in ECEF (unit) + its arc to the target: reference
    # geometry for SIGNED along-track miss in the landing predictions.
    _th0 = OMEGA_E * t0 + _GMST0
    _rot0 = np.array([[ np.cos(_th0), np.sin(_th0), 0],
                      [-np.sin(_th0), np.cos(_th0), 0],
                      [0, 0, 1]])
    _ei_ecef_hat = (_rot0 @ _r0) / _rn0
    _arc_ei_to_tgt = float(np.arccos(np.clip(
        np.dot(_ei_ecef_hat, target_ecef / np.linalg.norm(target_ecef)),
        -1, 1)))

    def aero_forces(r, v_eci, m):
        """Return (drag accel, lift_dir, q, v_rel) for current state."""
        rn = np.linalg.norm(r)
        alt = rn - R_EARTH
        v_air = np.cross(omega_e_vec, r)
        v_rel = v_eci - v_air
        v_rel_mag = np.linalg.norm(v_rel)
        rho = atm_density(alt)
        q = 0.5 * rho * v_rel_mag**2
        if v_rel_mag < 1 or rho < 1e-10:
            return np.zeros(3), np.zeros(3), q, v_rel
        # Drag along -v_rel
        a_drag = -q * cd_a * v_rel / (v_rel_mag * m)
        # Lift perpendicular to v_rel, in the local-vertical plane
        r_hat = r / rn
        v_rel_hat = v_rel / v_rel_mag
        # "Up" component perpendicular to v_rel
        lift_up = r_hat - np.dot(r_hat, v_rel_hat) * v_rel_hat
        if np.linalg.norm(lift_up) > 1e-3:
            lift_up /= np.linalg.norm(lift_up)
        else:
            lift_up = np.zeros(3)
        return a_drag, lift_up, q, v_rel

    def lift_accel(r, v_eci, m, bank_deg, bank_sign_val):
        """Compute lift acceleration with given bank angle and sign.

        Bank angle 0° = full lift up; 90° = lift horizontal; 180° = lift down.
        Bank sign controls left/right component of horizontal lift.
        """
        a_drag, lift_up, q, v_rel = aero_forces(r, v_eci, m)
        v_rel_mag = np.linalg.norm(v_rel)
        if v_rel_mag < 1 or q < 1e-3:
            return a_drag
        # Build out-of-plane direction (perpendicular to lift_up and v_rel)
        v_rel_hat = v_rel / v_rel_mag
        side = np.cross(lift_up, v_rel_hat)
        side_norm = np.linalg.norm(side)
        if side_norm > 1e-6:
            side /= side_norm
        # Lift magnitude is L = D * L/D
        L_mag = -np.linalg.norm(a_drag) * ld
        # Resolve into vertical (cos(bank)) + horizontal (sin(bank))
        bank_rad = np.deg2rad(bank_deg)
        lift_vec = (L_mag * np.cos(bank_rad) * lift_up
                     + L_mag * np.sin(bank_rad) * bank_sign_val * side)
        return a_drag - lift_vec   # subtract because lift_vec is in opposite sign convention

    # ---- Closed-loop predictor-corrector skip-entry guidance ---------------
    # Forward-propagate a COARSE trajectory to landing under a constant bank
    # magnitude (and the given crossrange sign), returning predicted landing
    # lat/lon, downrange-to-target, crossrange, and peak g. This is the core of
    # numerical entry guidance: it naturally includes the skip, so root-solving
    # the bank magnitude on predicted range controls the skip and the landing
    # point together. Coarse steps keep it cheap enough to call many times.
    def _predict_landing(t_now, r0, v0, m0, bank_deg, bank_sign_val,
                         max_t=5400.0, fine=False, bank2_deg=None,
                         vswitch=7600.0, drag_ref=None, bank_profile=None):
        """Predict landing under a constant bank — or, when bank2_deg is
        given, a HUNTEST-style THREE-REGIME profile:
          (1) bank_deg through the first dip (shapes the peak load);
          (2) once the dip peak has passed and while still supercircular, a
              CONSTANT-DRAG feedback law holding ~drag_ref g (Apollo's
              supercircular energy-bleed segment — what lets a shaved peak
              still make the design range without floating long);
          (3) bank2_deg once subcircular (< vswitch): final-phase range trim.
        A constant lift-up bank through the whole supercircular phase floats
        the vehicle long (hundreds of km of overfly); the drag-reference
        regime is the missing dof that makes low-peak AND on-range
        simultaneously feasible."""
        peak_g_pred = [0.0]
        def rhs_pred(t, y):
            r = y[:3]; v = y[3:6]
            rn = np.linalg.norm(r)
            if rn < R_EARTH:
                return np.concatenate([v, gravity_earth_moon(r, t), [0]])
            a_g = gravity_earth_moon(r, t)
            bk = bank_deg
            if bank_profile is not None:
                # Bank-vs-velocity reference profile (offline-optimized);
                # overrides the constant bank while supercircular.
                vrel_p = np.linalg.norm(v - np.cross(omega_e_vec, r))
                if vrel_p > vswitch:
                    bk = float(np.interp(vrel_p, bank_profile[0],
                                         bank_profile[1]))
                elif bank2_deg is not None:
                    bk = bank2_deg
            elif bank2_deg is not None:
                v_rel_p = v - np.cross(omega_e_vec, r)
                vrel_n = np.linalg.norm(v_rel_p)
                if vrel_n <= vswitch:
                    bk = bank2_deg
                elif drag_ref is not None:
                    rho_p = atm_density(rn - R_EARTH)
                    dg = 0.5 * rho_p * vrel_n**2 * cd_a / m0 / G0
                    if (peak_g_pred[0] > drag_ref * 1.05
                            and dg < peak_g_pred[0] * 0.98):
                        # past the dip peak, supercircular: hold drag at ref
                        bk = float(np.clip(55.0 - (dg - drag_ref) * 25.0,
                                           0.0, 120.0))
            a_a = lift_accel(r, v, m0, bk, bank_sign_val)
            _gl = np.linalg.norm(a_a) / G0
            if _gl > peak_g_pred[0]:
                peak_g_pred[0] = _gl
            return np.concatenate([v, a_g + a_a, [0]])
        def land_evt(t, y):
            return np.linalg.norm(y[:3]) - R_EARTH - 7_300.0
        land_evt.terminal = True; land_evt.direction = -1
        # Terminal phase uses a finer prediction (the remaining arc is short, so
        # this stays cheap); coarse otherwise.
        _rtol, _mstep = (1e-5, 3.0) if fine else (1e-4, 8.0)
        s = solve_ivp(rhs_pred, (t_now, t_now + max_t),
                      np.concatenate([r0, v0, [m0]]),
                      method='RK45', rtol=_rtol, atol=5.0, max_step=_mstep,
                      events=land_evt)
        if len(s.t_events[0]) > 0:
            yf = s.y_events[0][0]; tf = s.t_events[0][0]
            latf, lonf = eci_to_latlon(yf[:3], tf)
            theta = OMEGA_E * tf + _GMST0
            rotm = np.array([[ np.cos(theta), np.sin(theta), 0],
                             [-np.sin(theta), np.cos(theta), 0],
                             [0,0,1]])
            r_ec = rotm @ yf[:3]; r_h = r_ec/np.linalg.norm(r_ec)
            t_h = target_ecef/np.linalg.norm(target_ecef)
            ang = np.arccos(np.clip(np.dot(r_h, t_h), -1, 1))
            # SIGNED along-track miss: + = overfly (landed beyond the target's
            # arc from the entry point), - = short. Used by the HUNTEST drag-
            # reference root-solve, which needs direction, not just distance.
            arc_land = np.arccos(np.clip(np.dot(_ei_ecef_hat, r_h), -1, 1))
            signed_m = (arc_land - _arc_ei_to_tgt) * R_EARTH
            return R_EARTH*ang, latf, lonf, True, peak_g_pred[0], signed_m
        return 1e8, None, None, False, peak_g_pred[0], 0.0

    # HUNTEST analytic drag-energy reference. Calibrated subcircular
    # final-phase range (from v_switch to splash): the energy law budgets the
    # supercircular glide against the range REMAINING after the final phase.
    # The receding-horizon re-solve (every guidance call) absorbs calibration
    # residuals — as the switch approaches, the formula re-budgets whatever
    # is left, and the validated subcircular PC owns the endgame.
    HX_R_FINAL_M = 900e3
    HX_VSWITCH = 7600.0

    def _analytic_dref(r_now, v_now, t_now):
        th_ = OMEGA_E * t_now + _GMST0
        rot_ = np.array([[np.cos(th_), np.sin(th_), 0],
                         [-np.sin(th_), np.cos(th_), 0],
                         [0, 0, 1]])
        r_ec_ = rot_ @ r_now[:3]
        rh_ = r_ec_ / np.linalg.norm(r_ec_)
        tt_ = target_ecef / np.linalg.norm(target_ecef)
        rtogo_ = R_EARTH * np.arccos(np.clip(np.dot(rh_, tt_), -1, 1))
        vrel_ = np.linalg.norm(v_now - np.cross(omega_e_vec, r_now[:3]))
        r_super_ = max(rtogo_ - HX_R_FINAL_M, 50e3)
        dref_ = ((max(vrel_, HX_VSWITCH) ** 2 - HX_VSWITCH ** 2)
                 / (2.0 * r_super_) / G0)
        # Clip to the load-cap band: the cap IS the design point (6.4 g dip
        # shaping); a demand above it means the geometry can't make range
        # within the load limit — fly the cap and let the final phase trim.
        return float(np.clip(dref_, 2.0, 6.4))

    def _solve_bank(t_now, r, v_eci, m, bank_sign_val, fine=False):
        """Pick the bank profile minimizing predicted miss WITHIN a g-aware
        envelope (Apollo HUNTEST traded range against load the same way).
        Returns (bank_now, bank_post_dip_or_None, score).

        LEGACY (ENABLE_HUNTEST_PROFILE off): a single constant bank. Scoring:
        predicted miss + ~400 km per predicted g above 7.0, hard-reject above
        9.5 (the guidance arrest guard; Apollo's guidance limit was 10 g per
        NASA TN D-6725, structural bound 12 g). A constant bank CANNOT fly
        the 2,780 km design range from a -6.4 deg FPA below ~8 g — flatter
        banks overfly — which is why the legacy nominal peaks ~8.6 g.

        HUNTEST (flag on): a TWO-SEGMENT profile — bank1 through the
        supercircular first dip (shapes the peak load), bank2 after (makes
        the range), the dof Apollo's HUNTEST/UPCONTROL actually had. The
        g-penalty knee drops to 6.5 (Apollo's as-flown nominal peak), which
        the extra dof makes reachable at the design range. Once the vehicle
        is subcircular the dip is history and the solve collapses to the
        single-knob legacy form automatically.

        BANK_MAX 160 deg allows ~87% lift-DOWN (the authority Apollo's
        up-control used to suppress the skip; a 95-deg cap left the shallow
        corridor edge unflyable)."""
        BANK_MIN, BANK_MAX = 12.0, 160.0
        _huntest = bool(globals().get("ENABLE_HUNTEST_PROFILE", False))
        _vrel_now = np.linalg.norm(v_eci - np.cross(omega_e_vec, r))
        _two_seg = _huntest and _vrel_now > 7600.0
        _knee = 6.5 if _huntest else 7.0

        def miss(bank, bank2=None):
            dr, la, lo, landed, pk, _sg = _predict_landing(
                t_now, r, v_eci, m, bank, bank_sign_val, fine=fine,
                bank2_deg=bank2,
                drag_ref=(pc_state.get("dref", 5.0) if bank2 is not None
                          else None))
            if not landed:
                return 1e8
            if pk > 9.5:
                return 1e9
            # g-penalty 400 km per g above the knee: low-dispersion entries
            # stay near the knee (no miss to trade), while shallow corridor
            # edges may dig to ~9-9.5 g rather than overfly by thousands of km
            # (the earlier 1500 km/g weight made an 1800 km miss score better
            # than +1.4 g — backwards vs Apollo's 10-g guidance limit).
            return dr + max(0.0, pk - _knee) * 4.0e5

        def solve_b2(bank1=None):
            # 1-D search over the (post-dip) bank: 9-point grid + 4 refines
            # when single-segment; 7-point + 3 when inner to the dip loop.
            n_grid, n_ref = (7, 3) if bank1 is not None else (9, 4)
            banks = np.linspace(BANK_MIN, BANK_MAX, n_grid)
            vals = [miss(bank1, b) if bank1 is not None else miss(b)
                    for b in banks]
            j = int(np.argmin(vals))
            lo_b = banks[max(0, j-1)]; hi_b = banks[min(len(banks)-1, j+1)]
            best_b = banks[j]; best_v = vals[j]
            for _ in range(n_ref):
                mid = 0.5*(lo_b+hi_b)
                vm = miss(bank1, mid) if bank1 is not None else miss(mid)
                if vm < best_v:
                    best_v = vm; best_b = mid
                if ((miss(bank1, lo_b) if bank1 is not None else miss(lo_b))
                        < (miss(bank1, hi_b) if bank1 is not None else miss(hi_b))):
                    hi_b = mid
                else:
                    lo_b = mid
            return float(np.clip(best_b, BANK_MIN, BANK_MAX)), best_v

        if not _two_seg:
            b, v = solve_b2(None)
            return b, None, v

        # HUNTEST, range-aware. LEGACY-FIRST GATE, decided ONCE at the first
        # supercircular solve and held for the remainder of the entry: if the
        # constant-bank optimum stays at/under 7 g, fly legacy — preserving
        # the validated shallow-corridor behavior — else commit to the
        # multi-regime law. Re-deciding at every solve flip-flopped the law
        # mid-dip (a post-dip constant-bank re-prediction can transiently
        # exceed 7 g even on a shallow entry) and cost 15+ km of miss.
        _mode = pc_state.get("hx_mode")
        if _mode == "legacy":
            b, v = solve_b2(None)
            return b, None, v
        b_leg, v_leg = (None, None)
        if _mode is None:
            b_leg, v_leg = solve_b2(None)
            # Gate on a FINE prediction: the coarse propagator over-reads the
            # dip peak by ~0.5-0.8 g and was committing genuinely-shallow
            # entries (flown 6.4 g) to the multi-regime law.
            _drl, _, _, _ldl, pk_leg, _ = _predict_landing(
                t_now, r, v_eci, m, b_leg, bank_sign_val, fine=True)
            if _ldl and pk_leg <= 7.0:
                pc_state["hx_mode"] = "legacy"
                return b_leg, None, v_leg

        # (1) DIP SHAPING: largest bank1 whose predicted dip peak stays under
        # the 6.4 g cap (more bank = deeper dip = less float to correct
        # later). Solved on the cheap short-horizon dip propagator.
        if _predict_peak_g_liftup(r, v_eci, m, max_time_s=240.0) >= 6.4:
            b1 = 0.0   # even full lift-up exceeds the cap: take the minimum
        else:
            lo1, hi1 = 0.0, 100.0
            for _ in range(6):
                mid1 = 0.5 * (lo1 + hi1)
                if _predict_peak_g_liftup(r, v_eci, m, max_time_s=240.0,
                                          bank_deg=mid1) <= 6.4:
                    lo1 = mid1
                else:
                    hi1 = mid1
            b1 = lo1

        # (2) RANGE-AWARE DRAG REFERENCE — the piece a fixed reference lacks:
        # root-solve the post-peak constant-drag level so the SIGNED
        # along-track miss of the full three-regime prediction is zero (the
        # reference bleeds exactly the energy the range-to-go requires; real
        # HUNTEST computed this each pass). Higher drag = shorter, so the
        # signed miss is monotone-decreasing in the reference.
        # (2) ANALYTIC drag-energy reference (real HUNTEST's approach). The
        # earlier ROOT-SOLVED reference failed because a feedback law inside
        # the COARSE landing prediction diverges from the flown fine
        # integration — the root lands on a biased model (corpus in the flag
        # comment). Closed form instead: the reference is the deceleration
        # that dissipates the supercircular energy over exactly the range
        # available,
        #     D_ref = (v^2 - v_switch^2) / (2 * (R_togo - R_FINAL_CAL)),
        # no predictor in the loop. The IN-FLIGHT command path recomputes it
        # every guidance call (receding horizon), so model error
        # self-corrects as the switch approaches; the validated subcircular
        # PC owns the endgame regardless.
        dref_g = _analytic_dref(r, v_eci, t_now)
        pc_state["dref"] = dref_g
        b2_r, v_hx = solve_b2(b1)
        # Never trade a landing for a load number: at the COMMIT decision
        # only, fall back to legacy if the range-aware design scores worse.
        if _mode is None and v_leg is not None and v_hx > v_leg:
            pc_state["hx_mode"] = "legacy"
            return b_leg, None, v_leg
        pc_state["hx_mode"] = "huntest"
        return b1, b2_r, v_hx

    # Offline-tooling hook (reference-profile generation): when
    # ENTRY_PREDICT_ONLY is set, run ONE fine-fidelity landing prediction
    # under the given bank profile from the entry state and return its
    # outcome immediately — no flight. Lets the offline optimizer evaluate
    # candidate profiles at ~0.4 s each with the EXACT flight physics.
    _po = globals().get("ENTRY_PREDICT_ONLY", None)
    if _po is not None:
        _dr, _la, _lo, _ld, _pk, _sg = _predict_landing(
            t0, _r0.copy(), _v0.copy(), state[6], _po.get("bank", 55.0),
            +1.0, fine=True, bank_profile=_po.get("bank_profile"),
            bank2_deg=_po.get("bank2", 55.0))
        return {"predict_only": True, "miss_m": _dr, "landed": _ld,
                "peak_g": _pk, "signed_m": _sg, "lat": _la, "lon": _lo,
                "success": False}

    def get_phase_and_bank(t, r, v_eci, m):
        """Apollo-style guidance: return (phase_name, bank_deg, bank_sign).

        Phase logic:
          P64: aerodynamic deceleration > 0.05g, vehicle descending
          P65: skip phase (climbing out)
          P66: ballistic coast (load < 0.05g and high alt)
          P67: final phase (drag returning after skip)
        """
        rn = np.linalg.norm(r)
        alt = rn - R_EARTH
        v_air = np.cross(omega_e_vec, r)
        v_rel = v_eci - v_air
        v_rel_mag = np.linalg.norm(v_rel)
        rho = atm_density(alt)
        q = 0.5 * rho * v_rel_mag**2
        a_drag = q * cd_a / m / G0   # drag accel in g
        v_radial = np.dot(v_eci, r/rn)
        # 1. Phase transitions
        if phase[0] == "P64":
            # Stay in P64 until drag drops below 0.05g AND climbing
            if a_drag < 0.05 and v_radial > 0 and alt > 60_000:
                phase[0] = "P66"
        elif phase[0] == "P66":
            # In ballistic coast — re-enter aero when drag rises above 0.05g
            if a_drag > 0.05:
                phase[0] = "P67"
        # P67: stays until splash

        # 2. Compute crossrange and downrange to target in ECEF
        theta = OMEGA_E * t + _GMST0
        rot = np.array([[ np.cos(theta), np.sin(theta), 0],
                         [-np.sin(theta), np.cos(theta), 0],
                         [ 0,             0,             1]])
        r_ecef = rot @ r
        # Vector from current position to target, on Earth's surface
        # Simplified: use angular separation on sphere
        r_hat_surf = r_ecef / np.linalg.norm(r_ecef)
        target_hat = target_ecef / np.linalg.norm(target_ecef)
        # Downrange angle to target
        cos_dr = np.clip(np.dot(r_hat_surf, target_hat), -1, 1)
        downrange_to_go_m = R_EARTH * np.arccos(cos_dr)
        # Velocity in ECEF (relative to Earth)
        v_ecef = rot @ v_rel
        v_horiz_ecef = v_ecef - np.dot(v_ecef, r_hat_surf) * r_hat_surf
        # Heading direction (where we're going on the surface)
        if np.linalg.norm(v_horiz_ecef) > 1.0:
            heading_hat = v_horiz_ecef / np.linalg.norm(v_horiz_ecef)
        else:
            heading_hat = np.zeros(3)
        # Direction toward target on surface
        target_dir_horiz = target_hat - np.dot(target_hat, r_hat_surf) * r_hat_surf
        if np.linalg.norm(target_dir_horiz) > 1e-6:
            target_dir_horiz /= np.linalg.norm(target_dir_horiz)
        # Crossrange: signed angle between heading and target direction
        # (positive if target is to the left of velocity)
        cross_sign_vec = np.cross(heading_hat, target_dir_horiz)
        crossrange_sign = np.sign(np.dot(cross_sign_vec, r_hat_surf))
        cos_cross = np.clip(np.dot(heading_hat, target_dir_horiz), -1, 1)
        crossrange_angle = np.arccos(cos_cross)
        crossrange_m = R_EARTH * np.sin(crossrange_angle) * np.sign(crossrange_sign or 1)

        # --- Closed-loop predictor-corrector branch (flag-gated) ------------
        if globals().get("ENABLE_SKIP_ENTRY_GUIDANCE", False):
            # Crossrange: velocity-scaled deadband. Reverse the lift's lateral
            # sign when crossrange error exceeds the deadband (Apollo's lateral
            # logic). Deadband shrinks as velocity drops (tighter near target).
            xr_deadband_m = max(40_000.0, 0.04 * v_rel_mag * 1000.0)
            if abs(crossrange_m) > xr_deadband_m:
                # SIGN: must match the unguided P64 lateral logic (positive
                # sign), which demonstrably converges (16 km nominal miss).
                # The inherited negative sign steered AWAY from the target
                # laterally, parking trials ~580 km off at the deadband edge.
                bank_sign[0] = np.sign(crossrange_m) if crossrange_m != 0 else bank_sign[0]
            # Only run the (expensive) range solver once aero is meaningful;
            # above 0.05g and supercircular keep lift up to limit the first dip,
            # then let the predictor-corrector fly the range.
            drag_g = q * cd_a / m / G0
            if drag_g < 0.05:
                # Pre-atmospheric / ballistic skip coast: hold a moderate bank.
                return phase[0], 60.0, bank_sign[0]

            # OFFLINE REFERENCE PROFILE (flag-gated): in-band deliveries fly
            # the precomputed 6.5-g bank-vs-velocity profile open-loop while
            # supercircular; the subcircular PC below trims the residual.
            if (globals().get("ENABLE_REF_PROFILE_ENTRY", False)
                    and abs(entry_fpa_deg - ENTRY_REF_FPA_CENTER)
                        <= ENTRY_REF_FPA_BAND
                    and v_rel_mag > 7600.0):
                _bref = float(np.interp(v_rel_mag, ENTRY_REF_VGRID,
                                        ENTRY_REF_BANKS))
                return phase[0], _bref, bank_sign[0]

            # G-LIMIT PRIORITY + NUMERICAL PREDICTOR-CORRECTOR range control.
            #
            # The earlier analytic final-phase laws could not collapse the
            # ~5900 km dispersion because range is SET during the first dip /
            # skip, where the analytic reference had no authority. This law
            # instead root-solves the bank magnitude on the FULL predicted
            # trajectory to landing (_solve_bank/_predict_landing — the
            # prediction naturally includes the skip, so the bank controls the
            # skip range and the landing point together: numerical PC guidance,
            # in the spirit of Apollo's HUNTEST/UPCONTROL exit-velocity
            # targeting rather than its literal analytic implementation).
            # HARD GUARD (not a 6.5-g priority): command lift-vector-up only
            # when the predicted lift-up dip peak nears the GUIDANCE g-limit
            # (~10 g, NASA TN D-6725; structural bound 12 g). The 6.5-g band is
            # achieved by the g-AWARE predictor-corrector scoring instead — a
            # hard 6.5 priority forfeited range control on shallow entries
            # (lift-up is also the max-skip action; the PC never engaged and
            # miss blew up to 2000-7900 km on the shallow corridor edge).
            G_GUARD = 9.5
            peak_pred = _predict_peak_g_liftup(r.copy(), v_eci.copy(), m)
            if peak_pred >= G_GUARD:
                pc_state["t_last"] = -1.0e18   # re-solve once authority returns
                return phase[0], 0.0, bank_sign[0]

            # Derivation/testing aid: fly a fixed neutral bank instead of the
            # PC (used to find the guided profile's natural landing point when
            # re-deriving the recovery target; None in production).
            _nb = globals().get("ENTRY_PC_NEUTRAL_BANK", None)
            if _nb is not None:
                return phase[0], float(_nb), bank_sign[0]

            # Range control: solve every pc_interval seconds of flight (the
            # solve is ~17 predictions); hold the command in between. Cadence
            # and prediction fidelity tighten as range-to-go shrinks — the
            # terminal arc is short, so fine solves there stay cheap.
            _fine = downrange_to_go_m < 1.5e6
            pc_interval = (20.0 if downrange_to_go_m > 2.0e6
                           else 10.0 if downrange_to_go_m > 0.6e6 else 6.0)
            if t - pc_state["t_last"] >= pc_interval:
                nb, nb2, _miss = _solve_bank(t, r.copy(), v_eci.copy(), m,
                                             bank_sign[0], fine=_fine)
                pc_state["bank"] = nb
                pc_state["bank2"] = nb2
                pc_state["t_last"] = t
            # HUNTEST three-regime command: bank1 through the dip;
            # constant-drag feedback at the ANALYTIC drag-energy reference —
            # recomputed EVERY call (receding horizon) so the flown profile
            # re-budgets continuously as range-to-go shrinks — once the dip
            # peak has passed while supercircular; bank2 once subcircular
            # (the next subcircular re-solve replaces it with the
            # single-knob optimum anyway).
            _bcmd = pc_state["bank"]
            if pc_state.get("bank2") is not None:
                if v_rel_mag <= 7600.0:
                    _bcmd = pc_state["bank2"]
                else:
                    _dref = _analytic_dref(r, v_eci, t)
                    pc_state["dref"] = _dref
                    if max_g[0] > _dref * 1.05 and drag_g < max_g[0] * 0.98:
                        _bcmd = float(np.clip(55.0 - (drag_g - _dref) * 25.0,
                                              0.0, 120.0))
            return phase[0], float(_bcmd), bank_sign[0]

        # 3. Bank-angle logic by phase
        if phase[0] == "P64":
            # Apollo P64 (Post-0.05g): the goal is to do an aero-braking dip
            # that decelerates the vehicle from supercircular velocity
            # without exceeding crew g-limit. Apollo's actual technique:
            # roll the vehicle to apply full lift UP (bank near 0°) during
            # the first dip. This is the maximum-lift configuration, which
            # keeps the trajectory from going too deep. After velocity drops
            # below circular velocity, transition to skip-out (P65).
            #
            # Real Apollo guidance used a "constant-drag reference" — keep
            # drag at a specific reference value (typically 4-5g) by rolling.
            # We implement that here.
            target_drag_g = 6.0
            current_drag_g = q * cd_a / m / G0
            drag_err = current_drag_g - target_drag_g
            # Constant-drag-reference feedback. SIGN IS CRITICAL: when drag is
            # too HIGH (drag_err > 0) the vehicle is going too deep, so we must
            # roll lift UP — i.e. DECREASE bank toward 0° (cos(bank) → +1, more
            # vertical lift). The previous code ADDED drag_err to the bank,
            # which rolled lift further DOWN as drag rose — positive feedback
            # that drove a runaway second dip and spiked peak g just shallow of
            # the design point (the instability the entry bench exposed).
            new_bank = 55.0 - drag_err * 5.0
            new_bank = float(np.clip(new_bank, 30.0, 110.0))
            # --- Predictive 10g limiter (NASA TN D-6725) ---------------------
            # Apollo's guidance "predicts the limiting altitude rate at each
            # flight condition that will result in a 10g peak load factor,
            # based upon a lift-vector-up attitude. If the magnitude of the
            # navigated altitude rate exceeds this limit, a lift-vector-up
            # attitude is commanded to minimize the aerodynamic load factor."
            #
            # We implement the prediction directly: cheaply forward-propagate
            # the current state under a FULL-LIFT-UP assumption and find the
            # peak drag (in g) of the resulting dip. If that predicted peak is
            # at/above 10g, command lift-vector-up (bank -> 0) now — early
            # enough to actually arrest the descent, unlike a reactive check.
            # Predictive limiter is evaluated while descending into a dip
            # (the regime where a lift-up command can still arrest the
            # descent). On shallow entries near/over the overshoot boundary
            # the vehicle skips and the limiter cannot prevent the subsequent
            # uncontrolled re-entry — that is the physical corridor edge, not
            # a guidance defect, so no guard change helps there.
            if v_radial < 0.0:
                peak_pred_g = _predict_peak_g_liftup(r.copy(), v_eci.copy(), m)
                if peak_pred_g >= 10.0:
                    new_bank = 0.0
            # Lateral: reverse bank if crossrange grows beyond corridor
            if abs(crossrange_m) > 100_000:
                bank_sign[0] = np.sign(crossrange_m) if crossrange_m != 0 else 1.0
            return phase[0], new_bank, bank_sign[0]

        elif phase[0] == "P66":
            # Ballistic coast — minimal effect anyway since q is tiny
            return phase[0], 90.0, bank_sign[0]

        else:  # P67 final phase
            # Final phase: full closed-loop guidance to target.
            # Predict landing point at current bank; adjust bank to drive
            # predicted landing toward target.
            # Simplified Apollo final-phase law: bank ∝ (downrange_excess)
            # plus crossrange correction.

            # Expected drag profile from here: D₀ × (v/v₀)² × (ρ(h)/ρ(h₀))
            # We don't need to predict explicitly — use feedback on h-dot and v.
            # Apollo final-phase guidance: deviations from reference drive bank.
            # Reference: at v_rel = 4000 m/s, alt ≈ 40 km, descent ~ 100 m/s
            v_target = 4000.0
            # Range-to-velocity gain (rough): at v=4 km/s, ~500 km more range
            # to cover. If we have less, bank more (less lift). If more, bank less.
            # Use range-to-go and current velocity to derive desired bank.
            # Apollo: low velocity (v < 4000): use range-to-go as primary input.
            range_excess = downrange_to_go_m - estimate_range_at_velocity(v_rel_mag)
            # Bank angle: nominal 60°. Range excess → less lift up → bigger bank.
            new_bank = 60.0 + range_excess / 10_000.0   # 10° per 100 km
            new_bank = float(np.clip(new_bank, 30.0, 120.0))
            # Lateral
            if abs(crossrange_m) > 30_000:
                bank_sign[0] = np.sign(crossrange_m) if crossrange_m != 0 else 1.0
            return phase[0], new_bank, bank_sign[0]

    def estimate_range_at_velocity(v_mag):
        """Reference range remaining at given velocity, fit to Apollo data."""
        # At v = 11 km/s, expect ~6000 km of entry range (skip + final)
        # At v = 7 km/s, ~2500 km left
        # At v = 4 km/s, ~500 km left
        # At v = 1 km/s, ~50 km left
        # Quadratic fit: R(v) ≈ a*v² (good enough for guidance reference)
        return 50_000 * (v_mag / 1000) ** 1.8   # 50 km × (v/1km/s)^1.8

    def rhs(t, y):
        r = y[:3]; v_eci = y[3:6]; m = y[6]
        rn = np.linalg.norm(r)
        alt = rn - R_EARTH
        a_grav = gravity_earth_moon(r, t)
        if rn < R_EARTH:
            return np.concatenate([v_eci, a_grav, [0]])

        # Update guidance every GUIDANCE_INTERVAL seconds
        if t - last_guidance_t[0] >= GUIDANCE_INTERVAL or t == t0:
            _, bank_cmd[0], _ = get_phase_and_bank(t, r, v_eci, m)
            last_guidance_t[0] = t

        # Aerodynamic acceleration with current bank
        a_aero = lift_accel(r, v_eci, m, bank_cmd[0], bank_sign[0])
        # Track g/q
        v_air = np.cross(omega_e_vec, r)
        v_rel_mag = np.linalg.norm(v_eci - v_air)
        rho = atm_density(alt)
        q = 0.5 * rho * v_rel_mag**2
        max_q[0] = max(max_q[0], q)
        # Track max_g only during the entry deceleration regime (above 12 km,
        # which is well above drogue-deploy altitude). Below this the
        # parachutes would have deployed and the bare-capsule g-load reported
        # here is unphysical — it's the dynamic pressure we'd experience
        # *if* there were no parachutes.
        if alt > 12_000:
            max_g[0] = max(max_g[0], np.linalg.norm(a_aero) / G0)
        if alt < min_alt_seen[0]:
            min_alt_seen[0] = alt
        return np.concatenate([v_eci, a_grav + a_aero, [0]])

    def splash(t, y):
        # Terminate at drogue parachute deploy altitude (24,000 ft ≈ 7300 m).
        # Below this, parachutes (drogue then main) reduce velocity from
        # ~250 m/s to ~8 m/s before splashdown. Parachute phase is highly
        # reliable and not the source of mission risk, so we don't model
        # the descent below 7.3 km. The 'g loading' below this altitude
        # would be spurious — the parachutes absorb the kinetic energy.
        return np.linalg.norm(y[:3]) - R_EARTH - 7_300.0
    splash.terminal = True
    splash.direction = -1

    sol = solve_ivp(rhs, (t0, t0 + 5400), state, method='RK45',
                    rtol=1e-6, atol=1.0, max_step=2.0, events=splash)

    if len(sol.t_events[0]) > 0:
        ts = sol.t_events[0][0]
        ys = sol.y_events[0][0]
        lat, lon = eci_to_latlon(ys[:3], ts)
        # Survival criterion: peak load factor must stay within the Apollo
        # entry-corridor undershoot boundary. Per NASA TN D-6725 (Apollo
        # entry mission planning, Graves & Harpold 1972), the undershoot
        # boundary is defined by a MAXIMUM AERODYNAMIC LOAD FACTOR OF 12g;
        # the entry-guidance g-limiter separately commands lift-vector-up if
        # the predicted load factor would exceed 10g. We use 12g as the hard
        # structural/crew limit (was a 15g estimate previously).
        survived = (max_g[0] < 12.0 and max_g[0] > 1.0)
        return {
            "success": survived,
            "splash_lat_deg": lat, "splash_lon_deg": lon,
            "max_g": max_g[0], "max_q_pa": max_q[0],
            "splash_t": ts,
            "trajectory_t": sol.t, "trajectory_y": sol.y,
            "reason": None if survived else "high_g_breakup",
        }
    return {
        "success": False, "max_g": max_g[0], "max_q_pa": max_q[0],
        "trajectory_t": sol.t, "trajectory_y": sol.y,
        "reason": "skip_out_or_breakup",
    }


# ============================================================
# Main mission driver
# ============================================================
# Mission phases = segments between consecutive _mark() events in run_mission,
# paired with Apollo 11's actual mission-elapsed durations for comparison
# (derived from the Apollo 11 Flight Journal / Mission Report GET timeline).
PHASE_SEGMENTS = [
    ("liftoff",          "orbit_insertion",   "Launch to orbit"),
    ("orbit_insertion",  "tli",               "Parking orbit + TLI"),
    ("tli",              "loi",               "Translunar coast"),
    ("loi",              "pdi",               "Lunar orbit (LOI->PDI)"),
    ("pdi",              "touchdown",         "Powered descent"),
    ("touchdown",        "lm_liftoff",        "Surface stay"),
    ("lm_liftoff",       "ascent_insertion",  "Ascent to orbit"),
    ("ascent_insertion", "tei",               "Rendezvous to TEI"),
    ("tei",              "entry_interface",   "Trans-earth coast"),
    ("entry_interface",  "splashdown",        "Entry to splashdown"),
]
APOLLO_PHASE_DUR_S = {
    "Launch to orbit":         713.0,   # liftoff -> parking-orbit insertion 00:11:53
    "Parking orbit + TLI":    9143.0,   # insertion -> TLI ignition 02:44:16
    "Translunar coast":     263134.0,   # TLI -> LOI-1 75:49:50 (~73.1 h)
    "Lunar orbit (LOI->PDI)": 96195.0,  # LOI-1 -> PDI 102:33:05 (~26.7 h)
    "Powered descent":         755.0,   # PDI -> touchdown 102:45:40
    "Surface stay":          77780.0,   # touchdown -> LM liftoff 124:22:00 (~21.6 h)
    "Ascent to orbit":         435.0,   # liftoff -> orbit insertion ~124:29:15
    "Rendezvous to TEI":     39267.0,   # ascent insertion -> TEI 135:23:42 (~10.9 h)
    "Trans-earth coast":    214764.0,   # TEI -> entry interface 195:03:06 (~59.7 h)
    "Entry to splashdown":     929.0,   # entry interface -> splashdown 195:18:35
}


def build_phase_timeline(phase_log):
    """Convert raw _mark() events [(event, get_s, wall_s), ...] into a per-phase
    timeline with mission-elapsed duration and compute (wall) time per phase.
    Phases not reached (e.g. after an early failure) are simply omitted."""
    by_event = {ev: (get, wall) for ev, get, wall in (phase_log or [])}
    out = []
    for a, b, label in PHASE_SEGMENTS:
        if a in by_event and b in by_event:
            ga, wa = by_event[a]
            gb, wb = by_event[b]
            out.append({"phase": label,
                        "get_start_s": round(ga, 2),
                        "duration_s": round(gb - ga, 2),
                        "compute_s": round(wb - wa, 4)})
    return out


def run_mission(perturb=None, capture_trajectories=False):
    """Run a single complete mission, return (results_dict, trajectories_dict).

    Architecture:
      1. Lambert-targeted post-TLI initialization (avoids needing
         closed-loop ascent guidance — see initial_state_post_tli)
      2. Translunar coast (3-day, 3-body integration)
      3. MCC (small velocity correction to trim periapsis)
      4. LOI burn (finite-thrust SPS)
      5. Lunar-orbit coast to PDI
      6. Powered descent (the headline integration)
      7. Surface stay (skipped — just advances time)
      8. APS ascent burn to lunar orbit
      9. Rendezvous (assumed successful — geometric)
     10. TEI burn (finite-thrust SPS)
     11. Trans-earth coast
     12. Atmospheric entry & splashdown
    """
    perturb = perturb or {}
    results = {}
    trajectories = {}

    # Per-phase timing log for debugging / per-trial overview: each entry is
    # (event, mission-elapsed GET seconds, compute wall seconds). Phase durations
    # are the diffs between consecutive events. Stored by reference so it captures
    # the timeline up to wherever the mission ends (incl. early failure returns).
    _wall0 = time.time()
    _phase_log = []
    results["_phase_log"] = _phase_log
    def _mark(event, get_s):
        _phase_log.append((event, float(get_s), round(time.time() - _wall0, 4)))
    _mark("liftoff", 0.0)

    # SM SYSTEMS catastrophic failure (sourced; see PROB_SM_CATASTROPHIC):
    # if this trial drew the event, it strikes at a uniformly-drawn fraction
    # of the reference mission timeline. Checked at each phase boundary —
    # the failure label (and hence the crew-survival consequence) depends on
    # WHERE in the mission it lands: with the LM attached it is an
    # Apollo-13-style lifeboat abort; with the LM on the surface or already
    # jettisoned the picture is far worse (see crew_survival.py).
    _sm_t = None
    if (globals().get("ENABLE_SM_SYSTEMS_FAILURES", False)
            and perturb.get("sm_failure", False)):
        _sm_t = (float(perturb.get("sm_failure_frac", 0.5))
                 * SM_MISSION_REF_DURATION_S)
    def _sm_check(t_now, label):
        if _sm_t is not None and _sm_t <= t_now:
            results["full_success"] = False
            results["mission_failure"] = label
            results["sm_failure_get_h"] = _sm_t / 3600.0
            return True
        return False

    # 0. Saturn V launch: pad through parking orbit insertion.
    #    A failure here ends the mission immediately.
    launch = phase_saturn_v_launch(perturb)
    results["launch_success"] = launch["success"]
    results["launch_failure_reason"] = launch.get("failure_reason")
    results["launch_max_q_pa"] = launch.get("max_q_pa", 0.0)
    results["launch_max_g"] = launch.get("max_g", 0.0)
    results["launch_t_insertion_s"] = launch.get("t_insertion", 0.0)
    results["launch_parking_perigee_km"] = launch.get("parking_perigee_km", 0.0)
    results["launch_parking_apogee_km"] = launch.get("parking_apogee_km", 0.0)
    if capture_trajectories and "trajectory_t" in launch:
        trajectories["launch"] = (launch["trajectory_t"], launch["trajectory_y"])
    if not launch["success"]:
        results["full_success"] = False
        results["mission_failure"] = "launch_" + launch.get("failure_reason", "unknown")
        return results, trajectories

    _mark("orbit_insertion", launch.get("t_insertion", 0.0))

    # 1. Initial post-TLI state from Lambert targeting
    #    (in real Apollo this would be computed by the IU from telemetry;
    #    our Lambert+3body refinement gives the same answer for nominal
    #    geometry).
    if globals().get("ENABLE_LAUNCH_CONTINUITY", False):
        # Launch-state continuity: TLI ignites from THIS TRIAL'S launched
        # parking orbit (launch dispersions flow physically into the transfer;
        # the MCC chain absorbs them downstream).
        _preset = _solve_launch_tli()
        _ign_angle, _vcut = _preset[0], _preset[1]
        _pitch = _preset[2] if len(_preset) >= 4 else 0.0
        _yaw = _preset[3] if len(_preset) >= 4 else 0.0
        # The nominal-derived steering (pitch/yaw thrust tilt = IGM's
        # cutoff-state shaping) is flown through THIS TRIAL'S burn physics —
        # trial Isp/thrust/pointing dispersions perturb the steered cutoff
        # exactly as they would have perturbed the real instrument unit.
        _tli = _fly_launched_tli(launch["state"], launch["t_insertion"],
                                 perturb, _ign_angle, _vcut,
                                 steer_pitch_deg=_pitch, steer_yaw_deg=_yaw)
        if _tli is None:
            results["full_success"] = False
            # Distinguish a true S-IVB starve from an ignition-window solver
            # miss (the latter should be ~zero after the wrap-safe crossing
            # fix; kept as a separate label so it can never silently
            # masquerade as propellant physics again).
            results["mission_failure"] = (
                "tli_propellant_depleted"
                if globals().get("_TLI_FAIL_REASON") == "propellant_starved"
                else "tli_ignition_window_missed")
            return results, trajectories
        state, t, tli_dv = _tli
    elif globals().get("ENABLE_INTEGRATED_TLI", False):
        _tli = phase_tli_burn(perturb)
        if _tli is None:
            # S-IVB starved before guidance cutoff (real failure: the TLI
            # propellant margin is honest and tight, ~130 m/s of dv reserve).
            results["full_success"] = False
            results["mission_failure"] = "tli_propellant_depleted"
            return results, trajectories
        state, t, tli_dv = _tli
    else:
        state, t, tli_dv = initial_state_post_tli(perturb)
    results["tli_dv_ms"] = tli_dv
    _mark("tli", t)

    # Transposition & docking (~30 min after TLI): the CSM separates, turns,
    # and docks with the LM atop the S-IVB to extract it. An unrecovered
    # docking failure here (sourced two-docking decomposition, see
    # PROB_DOCKING_FAILURE) means NO LM and therefore no landing — the mission
    # aborts to an Earth return on a fully healthy CSM (this nearly ended
    # Apollo 14's landing). Mission failure, crew survives (crew_survival.py).
    if (globals().get("ENABLE_DESCENT_FAILURE_MODES", False)
            and perturb.get("td_docking_failed", False)):
        results["full_success"] = False
        results["mission_failure"] = "transposition_docking_failure"
        return results, trajectories

    # 2. Translunar coast (3 days, 3-body)
    tlc = phase_translunar_coast(state, t)
    results["closest_approach_km"] = tlc["closest_approach_km"]
    results["periapsis_alt_km"] = tlc["closest_approach_km"] - R_MOON/1000.0
    if capture_trajectories:
        trajectories["translunar"] = (tlc["log_t"], tlc["log_states"])
        trajectories["moon_distances"] = (tlc["log_t"], tlc["moon_distances_km"])

    # SOI-miss pre-screen. NOTE (calibration, Apollo 11 Mission Report): the
    # UNCORRECTED translunar pericynthion is SUPPOSED to be large — Apollo 11's
    # injection gave a pericynthion of 896 miles (~1440 km) and the planned
    # free-return value was hundreds of miles; this was rated "excellent" (1.6
    # ft/s injection accuracy). A small midcourse correction then walks it down
    # to the ~60 nmi (111 km) LOI target. So a few hundred-to-~1500 km
    # uncorrected perilune is NORMAL and fully correctable, NOT a missed SOI.
    # This pre-screen therefore only rejects GROSS misses that no MCC could
    # recover (deep impact, or far outside the lunar SOI ~66,000 km); the real
    # verdict is taken AFTER the MCC chain runs (see post-MCC check below).
    if globals().get("ENABLE_TRANS_LUNAR_MCC", False):
        gross_miss = (results["periapsis_alt_km"] < -1000.0
                      or results["periapsis_alt_km"] > 60000.0)
    else:
        # Without the correction chain, keep the original tight gate (the
        # single-MCC branch below has far less correction authority).
        gross_miss = (results["periapsis_alt_km"] < -200
                      or results["periapsis_alt_km"] > 1500)
    if gross_miss:
        results["mission_failure"] = "missed_lunar_soi"
        results["full_success"] = False
        return results, trajectories

    # 3. MCC (Midcourse Correction). By default (ENABLE_TRANS_LUNAR_MCC) this
    # runs Apollo's scheduled outbound chain MCC-1..4 (see phase_translunar_mcc),
    # each a perilune-targeting prograde burn waived inside a deadband. If the
    # chain is disabled, the else-branch falls back to a single impulsive
    # prograde burn at TLI+30h that nulls the periapsis error via a
    # bracket+brent solve to the nominal 94 km closest approach.
    sps_prop_remaining = SPS_PROP_INIT
    nominal_periapsis_km = 94.0
    actual_periapsis = results["periapsis_alt_km"]

    if globals().get("ENABLE_TRANS_LUNAR_MCC", False):
        # Outbound chain MCC-1..4 replaces the single correction.
        tlmcc = phase_translunar_mcc(state, t, tlc, perturb,
                                      target_perilune_km=nominal_periapsis_km)
        sps_prop_remaining -= tlmcc["sps_prop_used_kg"]
        results["mcc_dv_ms"] = float(tlmcc["mcc_total_dv_ms"])
        results["mcc_n_burns"] = int(sum(1 for b in tlmcc["mcc_burns"]
                                          if not b.get("waived", False)))
        results["periapsis_alt_km"] = tlmcc["perilune_km"]
        # Refresh the coast log around the corrected perilune for downstream
        # LOI targeting, mirroring what the single-MCC branch does.
        def _rhs_em(t, y):
            return np.concatenate([y[3:6], gravity_earth_moon(y[:3], t), [0]])
        t_pf = tlmcc["t_perilune"]
        sol_pf = solve_ivp(_rhs_em, (tlmcc["final_t"], t_pf + 2*3600),
                            tlmcc["final_state"], method='RK45',
                            rtol=1e-9, atol=1e-1, max_step=300.0,
                            dense_output=True)
        ts_pf = np.linspace(tlmcc["final_t"], t_pf + 2*3600, 2000)
        rs_pf = sol_pf.sol(ts_pf)
        tlc["log_states"] = rs_pf
        tlc["log_t"] = ts_pf
        # Refresh moon-distance array to MATCH the new states grid so the
        # downstream idx_closest (argmin over moon_distances_km) indexes a
        # same-length log_states. Leaving this stale caused an IndexError.
        dists_pf = np.array([np.linalg.norm(rs_pf[:3, i] - moon_state(tt)[0])
                             for i, tt in enumerate(ts_pf)])
        tlc["moon_distances_km"] = dists_pf / 1000.0
        tlc["t_closest"] = t_pf
        # Honest SOI-miss verdict, taken AFTER the correction chain. The MCC
        # chain targets 94 km; LOI can absorb a residual perilune error by
        # tuning the insertion burn, so the trial only fails if the corrected
        # perilune is still outside a realistically LOI-recoverable band. A
        # deep impact (< -50 km, i.e. the corrected trajectory still strikes the
        # Moon) or a perilune too high for LOI to capture (> 1500 km) is a
        # genuine miss; otherwise the mission proceeds to LOI.
        if (results["periapsis_alt_km"] < -50.0
                or results["periapsis_alt_km"] > 1500.0):
            results["mission_failure"] = "missed_lunar_soi_after_mcc"
            results["full_success"] = False
            return results, trajectories
    elif abs(actual_periapsis - nominal_periapsis_km) > 2.0:
        t_mcc = t + 30 * 3600
        mcc_states = tlc["log_states"]
        mcc_ts = tlc["log_t"]
        mcc_idx = int(np.argmin(np.abs(mcc_ts - t_mcc)))
        state_mcc = mcc_states[:, mcc_idx].copy()
        t_mcc_actual = mcc_ts[mcc_idx]

        def rhs_em(t, y):
            return np.concatenate([y[3:6], gravity_earth_moon(y[:3], t), [0]])

        v_cur = state_mcc[3:6].copy()
        v_hat = v_cur / np.linalg.norm(v_cur)

        def periapsis_for_dv(dv_mag):
            s_post = state_mcc.copy()
            s_post[3:6] = v_cur + dv_mag * v_hat
            def peri_evt(tt, y):
                mr_e = moon_state(tt)[0]
                r = y[:3] - mr_e
                v = y[3:6] - moon_state(tt)[1]
                return float(np.dot(r, v) / np.linalg.norm(r))
            peri_evt.terminal = True   # stop at first periapsis
            peri_evt.direction = +1
            sol = solve_ivp(rhs_em, (t_mcc_actual, t_mcc_actual + 50*3600),
                            s_post, method='RK45', rtol=1e-7, atol=1.0,
                            max_step=600.0, events=peri_evt)
            if len(sol.t_events[0]) > 0:
                s_peri = sol.y_events[0][0]
                t_peri = sol.t_events[0][0]
                d = np.linalg.norm(s_peri[:3] - moon_state(t_peri)[0])
                return (d - R_MOON) / 1000.0
            # Fallback: minimum over samples
            ts2 = np.linspace(t_mcc_actual, t_mcc_actual + 50*3600, 200)
            if not hasattr(sol, 'sol'):
                return 1000.0
            rs2 = sol.sol(ts2)
            dists = np.array([np.linalg.norm(rs2[:3,i] - moon_state(tt)[0])
                              for i, tt in enumerate(ts2)])
            return (np.min(dists) - R_MOON) / 1000.0

        from scipy.optimize import brentq
        mcc_dv = 0.0
        try:
            # Coarse sample to bracket — fewer points for speed
            samples = [(dv, periapsis_for_dv(dv) - nominal_periapsis_km)
                        for dv in np.linspace(-30, 30, 7)]
            bracketed = False
            for i in range(len(samples)-1):
                if samples[i][1] * samples[i+1][1] < 0:
                    mcc_dv = brentq(
                        lambda dv: periapsis_for_dv(dv) - nominal_periapsis_km,
                        samples[i][0], samples[i+1][0],
                        xtol=0.5, maxiter=8)
                    bracketed = True
                    break
            if not bracketed:
                mcc_dv = min(samples, key=lambda s: abs(s[1]))[0]
        except Exception:
            mcc_dv = 0.0

        results["mcc_dv_ms"] = float(abs(mcc_dv))

        # Apply MCC burn
        state_mcc[3:6] = v_cur + mcc_dv * v_hat
        # Consume SPS propellant
        m_now = state_mcc[6]
        isp_mcc = SPS_ISP * perturb.get("sps_isp_factor", 1.0)
        sps_prop_remaining -= m_now * (1 - np.exp(-abs(mcc_dv)/(isp_mcc*G0)))
        # Re-propagate forward; find TRUE periapsis via event detection
        # (radial velocity sign change), not just sample minimum.
        def periapsis_event(tt, y):
            mr_e = moon_state(tt)[0]
            r = y[:3] - mr_e
            v = y[3:6] - moon_state(tt)[1]
            return float(np.dot(r, v) / np.linalg.norm(r))
        periapsis_event.terminal = False
        periapsis_event.direction = +1     # transitioning from -ve to +ve

        sol_post = solve_ivp(rhs_em, (t_mcc_actual, t_mcc_actual + 80*3600),
                              state_mcc, method='RK45', rtol=1e-9, atol=1e-1,
                              max_step=300.0, dense_output=True,
                              events=periapsis_event)
        # Use the periapsis event time if found, else fall back to sample min
        if len(sol_post.t_events[0]) > 0:
            # Take the first periapsis (closest to Moon)
            best_t_peri = None
            best_dist = float('inf')
            for j, t_peri in enumerate(sol_post.t_events[0]):
                s_peri = sol_post.y_events[0][j]
                d = np.linalg.norm(s_peri[:3] - moon_state(t_peri)[0])
                if d < best_dist:
                    best_dist = d
                    best_t_peri = t_peri
                    best_state_peri = s_peri.copy()
            results["periapsis_alt_km"] = (best_dist - R_MOON) / 1000.0
            tlc["t_closest"] = best_t_peri
            # Also save log_states sampled at fine grid for downstream use
            ts2 = np.linspace(t_mcc_actual, t_mcc_actual + 80*3600, 2000)
            rs2 = sol_post.sol(ts2)
            tlc["log_states"] = rs2
            tlc["log_t"] = ts2
            # Place an idx-aligned entry at best periapsis time in log_states
            # by overriding the closest sample. This way idx_closest in next
            # phase will find the true periapsis state.
            idx_repl = int(np.argmin(np.abs(ts2 - best_t_peri)))
            tlc["log_states"][:, idx_repl] = best_state_peri
            tlc["log_t"][idx_repl] = best_t_peri
            dists2 = np.array([np.linalg.norm(rs2[:3,i] - moon_state(tt)[0])
                                for i, tt in enumerate(ts2)])
            dists2[idx_repl] = best_dist
            tlc["moon_distances_km"] = dists2 / 1000.0
        else:
            # Fallback: sample minimum
            ts2 = np.linspace(t_mcc_actual, t_mcc_actual + 80*3600, 2000)
            rs2 = sol_post.sol(ts2)
            dists2 = np.array([np.linalg.norm(rs2[:3,i] - moon_state(tt)[0])
                                for i, tt in enumerate(ts2)])
            idx_min = int(np.argmin(dists2))
            results["periapsis_alt_km"] = (dists2[idx_min] - R_MOON) / 1000.0
            tlc["t_closest"] = ts2[idx_min]
            tlc["log_states"] = rs2
            tlc["log_t"] = ts2
            tlc["moon_distances_km"] = dists2 / 1000.0
    else:
        results["mcc_dv_ms"] = 0.0

    # SM systems check: event during launch->translunar coast (LM attached,
    # Apollo-13-style lifeboat abort available). NOTE: `t` still holds the
    # TLI cutoff epoch here — the coast's end time lives in tlc["t_closest"].
    if _sm_check(float(tlc.get("t_closest", t)), "sm_failure_translunar"):
        return results, trajectories

    # 4. LOI — physically FLOWN two-burn lunar orbit insertion (Apollo-faithful):
    #    LOI-1 captures into a ~314 x 113 km ellipse (Apollo: 889 m/s, 169.6x60.9
    #    nm), then ~1 rev later LOI-2 at perilune circularizes to ~111 km (Apollo:
    #    48.5 m/s). A single ~360 s SPS burn cannot circularize (its frozen
    #    inertial attitude sweeps a large arc -> e floors ~0.08 with sub-surface
    #    perilune), so two burns are required. Robustness keys: target APOLUNE for
    #    LOI-1 (a steep, interior brentq root, vs the fragile low-perilune target),
    #    ignite ~half a burn before approach-perilune so the burn arc is centred on
    #    perilune (preserves perilune), and pick LOI-2 by MINIMIZING apo-peri
    #    SPREAD (a fixed-apolune bracket fails at the peri/apo role-swap). Any
    #    failure falls back to the legacy single burn so a trial degrades, never
    #    crashes. Gated on ENABLE_DOI (the realistic chain).
    idx_closest = int(np.argmin(tlc["moon_distances_km"]))
    t_closest = tlc["t_closest"]
    stack_mass = (CSM_CM_MASS + CSM_SM_DRY + sps_prop_remaining
                  + LM_DESC_TOTAL + LM_ASCT_TOTAL)

    def rhs_lunar(t, y):
        return np.concatenate([y[3:6], gravity_earth_moon(y[:3], t), [0]])

    def perilune_event(tt, y):
        mr_e, mv_e = moon_state(tt)
        r_lun = y[:3] - mr_e
        v_lun = y[3:6] - mv_e
        return float(np.dot(r_lun, v_lun) / np.linalg.norm(r_lun))
    perilune_event.direction = +1
    perilune_event.terminal = True

    def _peri_apo_ecc(s_out, t_out):
        """(perilune_km, apolune_km, ecc) of the Moon-relative orbit, or None if
        not gravitationally captured (E >= 0)."""
        mr_o, mv_o = moon_state(t_out)
        r_l = s_out[:3] - mr_o
        v_l = s_out[3:6] - mv_o
        E = 0.5 * np.linalg.norm(v_l)**2 - MU_MOON / np.linalg.norm(r_l)
        if E >= 0:
            return None
        a = -MU_MOON / (2 * E)
        h = np.linalg.norm(np.cross(r_l, v_l))
        ecc = np.sqrt(max(0.0, 1 - (h*h / MU_MOON) / a))
        return ((a*(1-ecc) - R_MOON) / 1000.0,
                (a*(1+ecc) - R_MOON) / 1000.0, ecc)

    from scipy.optimize import brentq
    loi_flown = False
    if globals().get("ENABLE_DOI", False):
        try:
            # LOI-1: ignite ~half a burn-duration before approach-perilune.
            _ve = SPS_ISP * G0
            _bt_est = (stack_mass * (1 - np.exp(-825.0/_ve))) / (SPS_THRUST/_ve)
            lead_s = globals().get("LOI1_LEAD_FRAC", 0.5) * _bt_est
            _logt = np.asarray(tlc["log_t"])
            jig = int(np.argmin(np.abs(_logt - (t_closest - lead_s))))
            t_loi = float(_logt[jig])
            ig_state = tlc["log_states"][:, jig].copy()
            ig_state[6] = stack_mass

            def _apo1(dv):
                s_o, t_o, _ = phase_loi(ig_state.copy(), t_loi, dv, perturb)
                pae = _peri_apo_ecc(s_o, t_o)
                return 99999.0 if pae is None else pae[1]
            # Adaptive bracket (navigation-robustness fix): the fixed [650,950]
            # bracket failed to straddle the 314 km apolune target on ~24% of
            # dispersed real-ephemeris arrivals, dumping those trials onto the
            # constructed-orbit fallback — where 17.5% later died at TEI (vs
            # 0.8% for flown orbits). Widen the bracket when the endpoints do
            # not straddle; if the target is genuinely unreachable, capture at
            # the nearest achievable apolune (a bound elliptical orbit is what
            # LOI-2 + DOI need — the exact apolune is secondary).
            _lo, _hi = 650.0, 950.0
            _alo, _ahi = _apo1(_lo), _apo1(_hi)
            if not (_alo > 314.0 > _ahi):
                _lo, _hi = 550.0, 1100.0
                _alo, _ahi = _apo1(_lo), _apo1(_hi)
            if _alo > 314.0 > _ahi:
                loi1_dv = float(np.clip(
                    brentq(lambda d: _apo1(d) - 314.0, _lo, _hi,
                           xtol=1.0, maxiter=12), 600.0, 1050.0))
            else:
                # Target not bracketed: pick the endpoint/grid dv whose CAPTURED
                # apolune is closest to target (must be bound and above perilune).
                _grid = np.linspace(_lo, _hi, 12)
                _cands = [(dv, _apo1(dv)) for dv in _grid]
                _cands = [(dv, ap) for dv, ap in _cands if 120.0 < ap < 5000.0]
                if not _cands:
                    raise RuntimeError("LOI-1 cannot capture a usable orbit")
                loi1_dv = float(min(_cands, key=lambda c: abs(c[1] - 314.0))[0])
            _mark("loi", t_loi)
            m_pre1 = ig_state[6]
            state, t, loi1_burn = phase_loi(ig_state, t_loi, dv_target=loi1_dv,
                                            perturb=perturb)
            sps_prop_remaining = max(0.0, sps_prop_remaining - (m_pre1 - state[6]))
            results["loi_dv_ms"] = loi1_dv
            results["loi_burn_time_s"] = loi1_burn

            pae1 = _peri_apo_ecc(state, t)
            if pae1 is None or pae1[0] < 30.0:
                raise RuntimeError("LOI-1 did not capture into a valid orbit")

            # Coast ~1 rev to the capture-ellipse perilune (LOI-2 ignition point).
            sol_c = solve_ivp(rhs_lunar, (t, t + 4*3600), state, method='RK45',
                              rtol=1e-9, atol=1e-2, max_step=60.0,
                              events=perilune_event)
            if len(sol_c.t_events[0]) > 0:
                state = sol_c.y_events[0][0].copy(); t = float(sol_c.t_events[0][0])
            else:
                state = sol_c.y[:, -1].copy(); t = float(sol_c.t[-1])

            # LOI-2: short burn at perilune; choose dv that MINIMIZES apo-peri spread.
            _dv_grid = np.linspace(0.0, 120.0, 25)
            _spread = []
            for _d in _dv_grid:
                _s, _to, _ = phase_loi(state.copy(), t, _d, perturb)
                _pae = _peri_apo_ecc(_s, _to)
                _spread.append(1e9 if _pae is None else abs(_pae[1] - _pae[0]))
            loi2_dv = float(_dv_grid[int(np.argmin(_spread))])
            _mark("loi2", t)
            m_pre2 = state[6]
            state, t, loi2_burn = phase_loi(state, t, dv_target=loi2_dv, perturb=perturb)
            sps_prop_remaining = max(0.0, sps_prop_remaining - (m_pre2 - state[6]))
            results["loi2_circ_dv_ms"] = loi2_dv
            results["loi2_burn_time_s"] = loi2_burn
            loi_flown = True
        except Exception:
            loi_flown = False   # degrade to the legacy single burn below

    if not loi_flown:
        # Legacy single-burn LOI (target ~5 km perilune): fallback, or ENABLE_DOI off.
        t_loi = t_closest
        state = tlc["log_states"][:, idx_closest].copy()
        state[6] = stack_mass

        def perilune_alt_for_loi_dv(dv_try):
            s_out, t_out, _ = phase_loi(state.copy(), t_loi, dv_try, perturb)
            pae = _peri_apo_ecc(s_out, t_out)
            return 99999.0 if pae is None else pae[0]

        loi_dv = 890.0
        try:
            peri_at_890 = perilune_alt_for_loi_dv(890.0)
            if abs(peri_at_890 - 5.0) > 2.0:
                peri_low = perilune_alt_for_loi_dv(870.0)
                peri_high = perilune_alt_for_loi_dv(940.0)
                if peri_low > 5.0 > peri_high:
                    loi_dv = brentq(lambda d: perilune_alt_for_loi_dv(d) - 5.0,
                                    870.0, 940.0, xtol=1.0, maxiter=8)
        except Exception:
            loi_dv = 890.0
        loi_dv = float(np.clip(loi_dv, 850.0, 970.0))
        _mark("loi", t_loi)
        m_pre_loi = state[6]
        state, t, loi_burn = phase_loi(state, t_loi, dv_target=loi_dv, perturb=perturb)
        sps_prop_remaining = max(0.0, sps_prop_remaining - (m_pre_loi - state[6]))
        results["loi_dv_ms"] = loi_dv
        results["loi_burn_time_s"] = loi_burn

    results["loi_flown"] = bool(loi_flown)

    # Parking-orbit characterization from the ACHIEVED orbit.
    _pae_post = _peri_apo_ecc(state, t)
    if _pae_post is not None:
        results["lunar_orbit_periselene_km"] = _pae_post[0]
        results["lunar_orbit_aposelene_km"]  = _pae_post[1]

    t_loi_end = t
    if globals().get("ENABLE_DOI", False):
        # Pre-descent lunar-orbit loiter (real-ephemeris timeline match): Apollo
        # orbited ~27 h between LOI and powered descent. Advancing the timeline
        # here puts TEI at the correct lunar declination (-> splash latitude) and
        # splashdown at the correct GMST (-> longitude). Coast the captured orbit.
        _coast = LUNAR_PARK_COAST_S if ENABLE_REAL_EPHEMERIS else 0.0
        if _coast > 0:
            sol_loiter = solve_ivp(rhs_lunar, (t, t + _coast), state,
                                   method='RK45', rtol=1e-9, atol=1e-2, max_step=120.0)
            state = sol_loiter.y[:, -1].copy()
            t = float(sol_loiter.t[-1])
            t_loi_end = t
        # --- Parking orbit + DOI burn -----------------------------------------
        mr_p, mv_p = moon_state(t)
        r_l = state[:3] - mr_p
        v_l = state[3:6] - mv_p
        r_hat = r_l / np.linalg.norm(r_l)
        h_hat = np.cross(r_l, v_l); h_hat = h_hat / np.linalg.norm(h_hat)
        v_hat = np.cross(h_hat, r_hat)          # prograde, in-plane

        if loi_flown:
            # Parking orbit is the FLOWN near-circular orbit (two-burn LOI). The
            # CSM stays in it; the LM does DOI from the real flown radius/speed.
            r_park = float(np.linalg.norm(r_l))
            v_circ = np.sqrt(MU_MOON / r_park)
            csm_state_at_loi = state.copy()
            csm_state_at_loi[6] = CSM_CM_MASS + CSM_SM_DRY + sps_prop_remaining
        else:
            # Fallback (two-burn solve failed / single-burn eccentric capture):
            # construct a ~100 km near-circular parking orbit in the achieved
            # plane + an impulsive LOI-2 trim, so the trial still proceeds.
            r_park = R_MOON + PARKING_ORBIT_ALT_KM * 1000.0
            v_circ = np.sqrt(MU_MOON / r_park)
            m_stack = state[6]
            loi2_prop = m_stack * (1 - np.exp(-LOI2_CIRC_DV_MS / (SPS_ISP * G0)))
            sps_prop_remaining = max(0.0, sps_prop_remaining - loi2_prop)
            results["loi2_circ_dv_ms"] = LOI2_CIRC_DV_MS
            csm_state_at_loi = np.concatenate([
                mr_p + r_hat * r_park,
                mv_p + v_hat * v_circ,
                [CSM_CM_MASS + CSM_SM_DRY + sps_prop_remaining]])
            results["lunar_orbit_periselene_km"] = PARKING_ORBIT_ALT_KM
            results["lunar_orbit_aposelene_km"]  = PARKING_ORBIT_ALT_KM

        # DOI: a small retrograde DPS impulse lowering ONLY the LM perilune to
        # ~15 km, where PDI fires (the current point becomes apoapsis of the
        # ~15 x 100 km descent orbit). Charged to the DPS.
        r_doi = R_MOON + DOI_PERILUNE_ALT_KM * 1000.0
        a_doi = 0.5 * (r_park + r_doi)
        v_apo_doi = np.sqrt(MU_MOON * (2.0 / r_park - 1.0 / a_doi))
        doi_dv = v_circ - v_apo_doi
        results["doi_dv_ms"] = float(doi_dv)
        m_lm_full = LM_DESC_TOTAL + LM_ASCT_TOTAL
        doi_prop = m_lm_full * (1 - np.exp(-doi_dv / (DPS_ISP * G0)))
        descent_m0 = m_lm_full - doi_prop

        lm_state = np.concatenate([
            mr_p + r_hat * r_park,
            mv_p + v_hat * v_apo_doi,
            [descent_m0]])

        # Coast the LM down to the ~15 km perilune for PDI.
        sol_lo = solve_ivp(rhs_lunar, (t, t + 3*3600), lm_state,
                           method='RK45', rtol=1e-9, atol=1e-2,
                           max_step=60.0, events=perilune_event,
                           dense_output=capture_trajectories)
    else:
        # --- Legacy: freeze CSM at post-LOI; brake directly from post-LOI peri -
        csm_state_at_loi = state.copy()
        csm_state_at_loi[6] = CSM_CM_MASS + CSM_SM_DRY + sps_prop_remaining
        descent_m0 = None
        sol_lo = solve_ivp(rhs_lunar, (t, t + 3*3600), state,
                           method='RK45', rtol=1e-9, atol=1e-2,
                           max_step=60.0, events=perilune_event,
                           dense_output=capture_trajectories)

    if len(sol_lo.t_events[0]) > 0:
        state = sol_lo.y_events[0][0]
        t = sol_lo.t_events[0][0]
    else:
        state, t = sol_lo.y[:, -1], sol_lo.t[-1]
    if capture_trajectories:
        trajectories["lunar_orbit"] = (sol_lo.t, sol_lo.y)

    # 6. Powered descent
    # SM systems check: event during LOI/parking-orbit phase (LM attached).
    if _sm_check(t, "sm_failure_lunar_orbit"):
        return results, trajectories

    _mark("pdi", t)
    desc = phase_powered_descent(state, t, perturb, m0_override=descent_m0)
    results["descent_success"] = desc["success"]
    if desc.get("success"):
        _mark("touchdown", desc.get("final_t", t))
    results["descent_reason"] = desc["reason"]
    if capture_trajectories:
        trajectories["descent"] = (desc["trajectory_t"], desc["trajectory_y"])
    if not desc["success"]:
        results["full_success"] = False
        results["mission_failure"] = "descent_" + str(desc.get("reason", "unknown"))
        return results, trajectories

    results["fuel_margin_s"]         = desc["fuel_margin_s"]
    results["prop_remaining_kg"]     = desc["prop_remaining_kg"]
    results["touchdown_v_radial_ms"] = desc["touchdown_speed_v_ms"]
    results["touchdown_v_horiz_ms"]  = desc["touchdown_speed_h_ms"]
    results["land_lat_deg"]          = desc["land_lat_deg"]
    results["land_lon_deg"]          = desc["land_lon_deg"]
    if "land_downrange_err_m" in desc:
        results["land_downrange_err_m"] = desc["land_downrange_err_m"]
    t = desc["final_t"]

    # ---- Apollo 11-specific descent events: record + conditional escalation
    # These events were RECOVERABLE on the actual flight. We only escalate a
    # successful landing to a loss when an event goes UNRECOVERED *and* the
    # landing was already marginal (so the model never overstates lethality of
    # the historically-survived events). All thresholds are RESEARCH_TODO.
    if globals().get("ENABLE_DESCENT_FAILURE_MODES", False):
        results["agc_1202_alarm"]      = desc.get("agc_1202_alarm", False)
        results["agc_1202_recovered"]  = desc.get("agc_1202_recovered", True)
        results["lr_dropout"]          = desc.get("lr_dropout", False)
        results["hard_landing"]        = desc.get("hard_landing", False)
        results["lowlevel_light_early_s"] = desc.get("lowlevel_light_early_s")

        marginal = (desc["fuel_margin_s"] is not None
                    and desc["fuel_margin_s"] < 5.0)

        # (a) Hard landing (excess touchdown velocity) is a structural loss.
        if desc.get("hard_landing", False):
            results["full_success"] = False
            results["mission_failure"] = "descent_hard_landing"
            return results, trajectories

        # (b) Unrecovered AGC alarm during a marginal landing → abort/loss.
        if (desc.get("agc_1202_alarm", False)
                and not desc.get("agc_1202_recovered", True)
                and marginal):
            results["full_success"] = False
            results["mission_failure"] = "descent_agc_alarm_unrecovered"
            return results, trajectories

        # (c) Radar dropout that coincides with a marginal terminal phase →
        #     degraded touchdown; treat as loss only in that conjunction.
        if desc.get("lr_dropout", False) and marginal:
            results["full_success"] = False
            results["mission_failure"] = "descent_radar_dropout_marginal"
            return results, trajectories

    # 7. Surface stay (21.6 hr — Armstrong/Aldrin EVA + sleep)
    # Surface-operations failure modes (sourced; gated with the other
    # Apollo-specific failure modes). Checked in hazard order: touchdown
    # tip-over (strikes at landing), LM electrical (pre-ascent arming),
    # EVA suit fatality (during the EVA window).
    if globals().get("ENABLE_DESCENT_FAILURE_MODES", False):
        if perturb.get("lm_tipover", False):
            results["full_success"] = False
            results["mission_failure"] = "surface_lm_tipover"
            return results, trajectories
        if perturb.get("eva_suit_fatality", False):
            results["full_success"] = False
            results["mission_failure"] = "surface_eva_suit_fatality"
            return results, trajectories
        if perturb.get("lm_surface_elec_failed", False):
            results["full_success"] = False
            results["mission_failure"] = "surface_lm_electrical_failure"
            return results, trajectories
    t += 21.6 * 3600
    # SM systems check: event while the LM is ON THE SURFACE — the worst
    # geometry short of the return coast (emergency liftoff + rendezvous with
    # a dying CSM; Collins alone aboard it).
    if _sm_check(t, "sm_failure_surface"):
        return results, trajectories
    _mark("lm_liftoff", t)

    # 8. LM ascent — insert into the CSM orbital plane (timed liftoff).
    # Propagate the CSM to ascent time and extract its Moon-relative orbital-
    # plane normal so the ascent steers into that plane.
    sol_csm_asc = solve_ivp(rhs_lunar, (t_loi_end, t), csm_state_at_loi,
                            method='RK45', rtol=1e-9, atol=1e-2, max_step=120.0)
    csm_asc = sol_csm_asc.y[:, -1]
    mr_asc, mv_asc = moon_state(t)
    r_cs_asc = csm_asc[:3] - mr_asc
    v_cs_asc = csm_asc[3:6] - mv_asc
    h_csm = np.cross(r_cs_asc, v_cs_asc)
    csm_plane_normal = h_csm / np.linalg.norm(h_csm)

    asc = phase_ascent_burn(t, perturb, csm_plane_normal=csm_plane_normal)
    results["ascent_success"]        = asc["success"]
    results["ascent_alt_km"]         = asc["final_alt_km"]
    results["aps_prop_remaining_kg"] = asc["prop_remaining_kg"]
    if not asc["success"]:
        results["full_success"] = False
        results["mission_failure"] = "ascent_" + str(asc.get("reason", "unknown"))
        return results, trajectories
    state, t = asc["final_state"], asc["final_t"]
    _mark("ascent_insertion", t)

    # 8b. Rendezvous & docking (physical, flag-gated).
    # The ascent now inserts INTO the CSM plane (timed liftoff), so the LM-vs-CSM
    # plane angle is no longer an artifact — at nominal it is ~0, and any
    # mismatch reflects the physical ascent yaw-steering dispersion. We
    # therefore charge the plane-change delta-V directly (2 v sin(di/2)), plus
    # the nominal coelliptic rendezvous budget, and check against available
    # LM propellant (APS residual + RCS budget). A failure is LM-crew-only.
    if globals().get("ENABLE_DESCENT_FAILURE_MODES", False):
        sol_csm_rdv = solve_ivp(rhs_lunar, (t_loi_end, t), csm_state_at_loi,
                                 method='RK45', rtol=1e-9, atol=1e-2,
                                 max_step=120.0)
        csm_state_rdv = sol_csm_rdv.y[:, -1]
        mr_rdv, mv_rdv = moon_state(t)
        r_lm = state[:3] - mr_rdv;  v_lm = state[3:6] - mv_rdv
        r_cs = csm_state_rdv[:3] - mr_rdv; v_cs = csm_state_rdv[3:6] - mv_rdv

        # Plane mismatch (now physical: driven by ascent yaw dispersion).
        h_lm = np.cross(r_lm, v_lm); h_cs = np.cross(r_cs, v_cs)
        cos_i = np.clip(np.dot(h_lm, h_cs) /
                        (np.linalg.norm(h_lm) * np.linalg.norm(h_cs)), -1, 1)
        d_incl = np.arccos(cos_i)
        v_orb = np.linalg.norm(v_lm)
        dv_plane = 2.0 * v_orb * np.sin(d_incl / 2.0)

        dv_rendezvous = RENDEZVOUS_NOMINAL_DV_MS + dv_plane

        # Available LM delta-V: APS residual (rocket equation) + RCS budget.
        m_after_ascent = state[6]
        m_dry = LM_ASCT_DRY
        if m_after_ascent > m_dry:
            dv_aps_avail = APS_ISP * G0 * np.log(m_after_ascent / m_dry)
        else:
            dv_aps_avail = 0.0
        dv_avail = dv_aps_avail + LM_RCS_DV_BUDGET_MS

        results["rendezvous_dv_required_ms"] = float(dv_rendezvous)
        results["rendezvous_dv_available_ms"] = float(dv_avail)
        results["rendezvous_plane_angle_deg"] = float(np.rad2deg(d_incl))

        if dv_rendezvous > dv_avail:
            results["full_success"] = False
            results["mission_failure"] = "rendezvous_insufficient_propellant"
            return results, trajectories

        if perturb.get("docking_failed", False):
            results["full_success"] = False
            results["mission_failure"] = "rendezvous_docking_failure"
            return results, trajectories

    # 9. Rendezvous & docking — (baseline path / post-success) The CSM has been
    # orbiting the Moon in the LOI orbit while the LM descended/landed/ascended.
    # We use the CSM's natural orbit at the post-ascent time as the docked
    # vehicle's state, then transfer to CSM-only mass for TEI.
    t_rendezvous = t   # post-ascent time
    sol_csm = solve_ivp(rhs_lunar, (t_loi_end, t_rendezvous), csm_state_at_loi,
                         method='RK45', rtol=1e-9, atol=1e-2, max_step=120.0)
    state = sol_csm.y[:, -1].copy()
    state[6] = CSM_CM_MASS + CSM_SM_DRY + sps_prop_remaining
    t = t_rendezvous

    # Post-rendezvous coast (docking / LM jettison / crew prep — Apollo's
    # ascent-to-TEI leg was ~10.9 h; see POST_RENDEZVOUS_COAST_S). The TEI
    # opportunity scan then covers ONE orbit instead of 10 h, so TEI fires at
    # the best alignment of the rev after the coast, like Apollo's rev-31 TEI.
    _rdv_coast = POST_RENDEZVOUS_COAST_S if ENABLE_REAL_EPHEMERIS else 0.0
    if _rdv_coast > 0:
        sol_prep = solve_ivp(rhs_lunar, (t, t + _rdv_coast), state,
                             method='RK45', rtol=1e-9, atol=1e-2, max_step=120.0)
        state = sol_prep.y[:, -1].copy()
        t = float(sol_prep.t[-1])
    # TEI candidates from the FIRST rev are preferred (keeps Apollo's ~10.9 h
    # ascent->TEI leg); later revs in the 10 h scan are the REV-SLIP fallback,
    # used only when rev-1 geometry can't produce an acceptable return (a
    # hard one-rev window caused tei_no_earth_return_found on perturbed trials).
    _tei_first_rev_s = 2.6 * 3600.0 if _rdv_coast > 0 else 10 * 3600.0

    # Pre-TEI capture hook (debug harness, mirrors _ENTRY_CAPTURE_HOOK):
    # everything upstream of here is unaffected by TEI/return/entry changes,
    # so capturing (state, t, sps_prop) lets TEI+entry iterations replay from
    # this point in ~1-2 min instead of re-flying the whole ~6-10 min mission.
    # Default None -> zero overhead.
    _tei_hook = globals().get("_TEI_CAPTURE_HOOK", None)
    if _tei_hook is not None:
        _tei_hook.append((state.copy(), float(t), dict(perturb),
                          float(sps_prop_remaining)))

    # SM systems check: event between ascent and TEI (LM attached for most
    # of this window; jettisoned shortly before TEI).
    if _sm_check(t, "sm_failure_lunar_orbit"):
        return results, trajectories

    # 10. TEI — Trans-Earth Injection, PHYSICALLY INTEGRATED 3-body burn.
    #
    # The CSM's lunar-orbit plane contains the Earth direction at TEI time, so
    # a physical finite-thrust SPS burn can reach the entry corridor. NOTE on
    # handedness: under ENABLE_LAUNCH_CONTINUITY (default) the capture is
    # RETROGRADE (Apollo's actual sense), so candidate selection scores the
    # predicted two-body departure ASYMPTOTE against the perigee-nulling
    # inbound direction (handedness-agnostic) rather than a prograde
    # velocity-alignment heuristic; the legacy (continuity-off) path is the
    # prograde Lambert-derived orbit the velocity-alignment scan assumes.
    #
    # Algorithm (legacy/prograde path; the continuity path overrides scoring):
    #   1. Scan the next lunar orbit (~2.2h) to find the moment where the
    #      prograde-Moon velocity direction aligns with Earth direction.
    #   2. Execute a finite-thrust SPS burn in prograde-Moon direction with
    #      ΔV iteratively tuned to give entry FPA close to -6.5°.
    #   3. The trans-Earth coast then runs as full 3-body physics.
    isp_sps = SPS_ISP * perturb.get("sps_isp_factor", 1.0)
    T_sps  = SPS_THRUST * perturb.get("sps_thrust_factor", 1.0)
    mdot_sps = T_sps / (isp_sps * G0)

    # Step 1: scan up to 4 orbital periods for candidate TEI moments
    def rhs_lun_tei(ti, y):
        return np.concatenate([y[3:6], gravity_earth_moon(y[:3], ti), [0]])

    sol_scan = solve_ivp(rhs_lun_tei, (t, t + 10*3600), state,
                          method='RK45', rtol=1e-9, atol=1e-2, max_step=30.0,
                          dense_output=True)
    scan_t = np.arange(t, t + 10*3600, 30.0)
    scan_t = scan_t[scan_t <= sol_scan.t[-1]]
    scan_y = sol_scan.sol(scan_t)

    # Candidate-quality metric at each scan point.
    #
    # LEGACY (prograde-validated): cos(prograde velocity, Earth direction).
    #
    # LAUNCH-CONTINUITY (handedness-agnostic, physically targeted): two fixes
    # over the legacy heuristic, both required for the RETROGRADE
    # (Apollo-handed) capture the continuity transfer produces.
    #  (a) The right DEPARTURE DIRECTION is not "toward Earth": leaving the
    #      SOI along the Moon->Earth line keeps the Moon's ~1 km/s tangential
    #      orbital velocity as geocentric angular momentum (perigee ~200,000
    #      km — no entry at ANY dv). TEI must depart nearly ANTI-PARALLEL to
    #      the Moon's orbital velocity to cancel it. Solve ONCE for the
    #      in-plane departure angle phi* whose post-escape geocentric perigee
    #      hits the entry interface, in a (-v_moon, in-plane) basis.
    #  (b) The post-burn escape asymptote departs rotated ~(theta_inf - 90
    #      deg) ~ 45 deg from the burn-point velocity IN THE DIRECTION OF
    #      MOTION, so velocity-alignment peaks sit ~45 deg from the correct
    #      burn point — and for a retrograde orbit the dv-bisect sweeps the
    #      asymptote the WRONG way from there (the observed
    #      4-candidates/no-solutions failure). Score by the PREDICTED
    #      ASYMPTOTE of the analytic two-body lunar hyperbola for a nominal
    #      ~950 m/s prograde burn: cos(outgoing asymptote, departure target).
    _asym_metric = bool(globals().get("ENABLE_LAUNCH_CONTINUITY", False))
    _phi_star = 0.0
    if _asym_metric:
        mr0_, mv0_ = moon_state(float(scan_t[0]))
        rr0_ = scan_y[:3, 0] - mr0_
        vv0_ = scan_y[3:6, 0] - mv0_
        hcap0_ = np.cross(rr0_, vv0_)
        hcap0_ /= np.linalg.norm(hcap0_)
        vinf0_ = np.sqrt(max((np.linalg.norm(vv0_) + 950.0) ** 2
                             - 2 * MU_MOON / np.linalg.norm(rr0_), 1.0))
        e1_ = -(mv0_ - hcap0_ * np.dot(mv0_, hcap0_))
        e1_ /= np.linalg.norm(e1_)
        e2_ = np.cross(hcap0_, e1_)

        def _peri_geo(phi):
            u_ = np.cos(phi) * e1_ + np.sin(phi) * e2_
            vg_ = mv0_ + vinf0_ * u_
            hg_ = np.cross(mr0_, vg_)
            hn_ = np.linalg.norm(hg_)
            epsg_ = 0.5 * np.dot(vg_, vg_) - MU_EARTH / np.linalg.norm(mr0_)
            ecg_ = np.sqrt(max(0.0, 1.0 + 2 * epsg_ * hn_ * hn_
                               / MU_EARTH ** 2))
            if epsg_ < 0:
                return (-MU_EARTH / (2 * epsg_)) * (1 - ecg_)
            return hn_ * hn_ / MU_EARTH / (1 + ecg_)

        _tgt_r = R_EARTH + 30e3
        _phis = np.deg2rad(np.arange(-75.0, 75.1, 2.0))
        _peris = np.array([_peri_geo(p) for p in _phis])
        # all sign-change crossings of (perigee - target); take the one
        # nearest pure anti-velocity (phi = 0)
        _cross = []
        for jj in range(len(_phis) - 1):
            if ((_peris[jj] - _tgt_r) * (_peris[jj + 1] - _tgt_r)) <= 0:
                try:
                    from scipy.optimize import brentq as _bqt
                    _cross.append(float(_bqt(
                        lambda p: _peri_geo(p) - _tgt_r,
                        _phis[jj], _phis[jj + 1], xtol=1e-4)))
                except Exception:
                    pass
        # Both perigee-crossings (+-~43 deg) null the geocentric perigee, but
        # the OUTBOUND-radial one coasts to a ~550,000 km apogee first and
        # falls back to its perfect perigee only after ~10+ days — outside any
        # entry window and nothing like a 2.5-day trans-Earth coast. Keep only
        # crossings whose departure velocity points EARTH-WARD (radial-in).
        _inb = []
        for _c in _cross:
            _u_c = np.cos(_c) * e1_ + np.sin(_c) * e2_
            _vg_c = mv0_ + vinf0_ * _u_c
            if np.dot(_vg_c, mr0_) < 0.0:
                _inb.append(_c)
        if _inb:
            _phi_star = min(_inb, key=abs)
        elif _cross:
            _phi_star = min(_cross, key=abs)
        else:
            _phi_star = float(_phis[int(np.argmin(np.abs(_peris - _tgt_r)))])

    alignments = np.zeros(len(scan_t))
    for i in range(len(scan_t)):
        mr_, mv_ = moon_state(scan_t[i])
        v_rel_ = scan_y[3:6, i] - mv_
        v_rel_n = np.linalg.norm(v_rel_)
        if v_rel_n < 100:
            alignments[i] = -2
            continue
        prograde_ = v_rel_ / v_rel_n
        earth_dir_ = -scan_y[:3, i] / np.linalg.norm(scan_y[:3, i])
        if not _asym_metric:
            alignments[i] = float(np.dot(prograde_, earth_dir_))
            continue
        r_rel_ = scan_y[:3, i] - mr_
        r_rel_n = np.linalg.norm(r_rel_)
        v_b_ = v_rel_ + 950.0 * prograde_
        eps_ = 0.5 * np.dot(v_b_, v_b_) - MU_MOON / r_rel_n
        if eps_ <= 0:
            alignments[i] = -2
            continue
        h_vec_ = np.cross(r_rel_, v_b_)
        h_hat_ = h_vec_ / np.linalg.norm(h_vec_)
        ev_ = np.cross(v_b_, h_vec_) / MU_MOON - r_rel_ / r_rel_n
        e_ = np.linalg.norm(ev_)
        if e_ <= 1.0 + 1e-9:
            alignments[i] = -2
            continue
        ehat_ = ev_ / e_
        phat_ = np.cross(h_hat_, ehat_)
        nu_inf_ = np.arccos(-1.0 / e_)
        u_inf_ = np.cos(nu_inf_) * ehat_ + np.sin(nu_inf_) * phat_
        # departure target at THIS epoch: phi* in the (-v_moon, in-plane)
        # basis built from this point's Moon velocity and capture plane
        e1i_ = -(mv_ - h_hat_ * np.dot(mv_, h_hat_))
        e1n_ = np.linalg.norm(e1i_)
        if e1n_ < 1.0:
            alignments[i] = -2
            continue
        e1i_ /= e1n_
        e2i_ = np.cross(h_hat_, e1i_)
        u_tgt_ = np.cos(_phi_star) * e1i_ + np.sin(_phi_star) * e2i_
        alignments[i] = float(np.dot(u_inf_, u_tgt_))

    best_i = int(np.argmax(alignments))
    best_score = float(alignments[best_i])

    t_tei = scan_t[best_i]
    state_tei = scan_y[:, best_i].copy()
    results["tei_alignment_cos"] = float(best_score)
    # Diagnostics: lunar-orbit handedness vs the Moon's orbital motion
    # (+1 = prograde-around-Moon in the Moon's orbital sense; Apollo's real
    # lunar orbit was RETROGRADE).
    _mr_d, _mv_d = moon_state(float(scan_t[0]))
    _rrel = scan_y[:3, 0] - _mr_d
    _vrel = scan_y[3:6, 0] - _mv_d
    _h_lun = np.cross(_rrel, _vrel)
    _h_moon = np.cross(_mr_d, _mv_d)
    results["lunar_orbit_handedness"] = float(np.sign(
        np.dot(_h_lun, _h_moon)))

    # Find all local maxima of alignment cos (peaks where prograde best aligns
    # with Earth direction). There are typically 3-5 peaks per orbital period.
    # Keep top 3 by alignment quality for speed (each peak adds ~3s to runtime).
    align_peaks = []
    for i in range(1, len(scan_t) - 1):
        if alignments[i] > 0.95 and alignments[i] >= alignments[i-1] and alignments[i] >= alignments[i+1]:
            align_peaks.append(i)
    # Split into first-rev and later-rev (rev-slip fallback) candidates.
    _rev1_end = t + _tei_first_rev_s
    _rev1 = sorted((i for i in align_peaks if scan_t[i] <= _rev1_end),
                   key=lambda i: -alignments[i])
    _later = sorted((i for i in align_peaks if scan_t[i] > _rev1_end),
                    key=lambda i: -alignments[i])
    align_peaks = _rev1[:3] + _later[:3]

    if not align_peaks:
        # No high-quality alignment peaks — use absolute best
        best_i = int(np.argmax(alignments))
        align_peaks = [best_i]

    # We'll search ΔV at each candidate peak, picking the one that gives
    # entry FPA closest to -6.5°
    target_fpa = -6.5
    best_overall_sim = None
    best_overall_score = float("inf")
    best_t_tei = None
    best_burn_dir = None
    best_state_tei = None

    isp_sps_local = isp_sps    # for reuse below
    _tei_dbg = []              # per-candidate sweep diagnostics

    for peak_i in align_peaks:
        t_tei_candidate = scan_t[peak_i]
        # Rev-slip gate (corridor-gated): TEI fires on the Apollo-timeline rev
        # when rev-1 produced an IN-CORRIDOR return (entry FPA within the gate
        # of -6.5 — a band the trans-Earth MCC chain + the 3-DOF EI-targeting
        # refinement can genuinely trim). Otherwise later revs ARE evaluated:
        # at 500-trial scale, committing to any "valid" rev-1 return delivered
        # entry FPAs out to -25.7 deg and 10% of all trials died at entry g —
        # corridors no real navigation would have committed to. The best
        # candidate overall then wins (slipped revs trade the timeline for a
        # flyable corridor, exactly Apollo's backup-opportunity design).
        # Under LAUNCH CONTINUITY the gate widens to 1.5 deg: the
        # asymptote-targeted candidates are far better conditioned, the
        # EI-refinement demonstrably recovers errors of this size (the >1.5
        # hard-reject removal showed it recovering ALL of them), and the
        # 0.5-deg gate was rev-slipping even the NOMINAL by ~3 revs (+7 h vs
        # Apollo's 10.9 h rendezvous->TEI leg, rotating the recovery zone
        # ~105 deg east of the mid-Pacific). Apollo committed to the FIRST
        # opportunity; the steep-tail entry-g rate is the honest price,
        # watched in MC.
        # (2.0 deg: the nominal's rev-1 bisect score is 1.75 — the asymptote-
        # targeted candidates start that close — and the refinement recovers
        # 2-deg-class errors; 1.5 still rev-slipped the nominal.)
        _rev1_gate = (2.0 if globals().get("ENABLE_LAUNCH_CONTINUITY", False)
                      else 0.5)
        if (best_overall_sim is not None and best_overall_score < _rev1_gate
                and t_tei_candidate > _rev1_end):
            break
        state_tei_candidate = scan_y[:, peak_i].copy()
        mr_c, mv_c = moon_state(t_tei_candidate)
        v_rel_c = state_tei_candidate[3:6] - mv_c
        burn_dir_c = v_rel_c / np.linalg.norm(v_rel_c)

        def simulate_at_peak(dv_mag, t_tei_=t_tei_candidate,
                              state_tei_=state_tei_candidate,
                              burn_dir_=burn_dir_c):
            m0 = state_tei_[6]
            if dv_mag <= 0:
                return None
            mp_burn = m0 * (1 - np.exp(-dv_mag / (isp_sps * G0)))
            if mp_burn >= m0 - (CSM_CM_MASS + CSM_SM_DRY) - 50:
                return None
            burn_time = mp_burn / mdot_sps
            def rhs_burn(ti, y):
                r = y[:3]; v = y[3:6]; m = y[6]
                a = gravity_earth_moon(r, ti)
                a = a + T_sps * burn_dir_ / max(m, 1.0)
                return np.concatenate([v, a, [-mdot_sps]])
            sol_b = solve_ivp(rhs_burn, (t_tei_, t_tei_ + burn_time),
                              state_tei_, method='RK45',
                              rtol=1e-7, atol=1.0, max_step=2.0)
            post_burn = sol_b.y[:, -1].copy()
            t_post = sol_b.t[-1]
            # Geocentric osculating perigee of the post-burn orbit — a CONTINUOUS
            # return-quality metric (no entry-event discontinuity), used to refine
            # a near-miss down into the entry corridor.
            _rpb = post_burn[:3]; _vpb = post_burn[3:6]
            _rn0 = np.linalg.norm(_rpb)
            _en = 0.5*np.dot(_vpb, _vpb) - MU_EARTH/_rn0
            _hn = np.linalg.norm(np.cross(_rpb, _vpb))
            _ecc = np.sqrt(max(0.0, 1.0 + 2*_en*_hn*_hn/MU_EARTH**2))
            osc_perigee = (-MU_EARTH/(2*_en))*(1-_ecc) if _en < 0 else _hn*_hn/MU_EARTH/(1+_ecc)
            def rhs_coast(ti, y):
                return np.concatenate([y[3:6], gravity_earth_moon(y[:3], ti), [0]])
            def entry_evt(ti, y):
                return np.linalg.norm(y[:3]) - (R_EARTH + 122_000.0)
            entry_evt.terminal = True; entry_evt.direction = -1
            sol_c = solve_ivp(rhs_coast, (t_post, t_post + 7*86400),
                               post_burn, method='DOP853',
                               rtol=1e-6, atol=100.0, max_step=1800.0,
                               events=entry_evt)
            if len(sol_c.t_events[0]) > 0:
                te = sol_c.t_events[0][0]
                ye = sol_c.y_events[0][0].copy()
                r_e = ye[:3]; v_e = ye[3:6]
                rn = np.linalg.norm(r_e); vn = np.linalg.norm(v_e)
                fpa = np.arcsin(np.dot(r_e, v_e) / (rn * vn))
                return {"entry_state": ye, "entry_t": te,
                         "entry_fpa_deg": float(np.rad2deg(fpa)),
                         "entry_speed_ms": float(vn),
                         "post_burn_state": post_burn,
                         "burn_time_s": burn_time,
                         "t_post_burn": t_post,
                         "dv_mag": dv_mag, "osc_perigee_m": float(osc_perigee)}
            # No atmospheric entry: report the closest approach so the caller can
            # refine a near-miss instead of discarding it.
            r_perigee = float(np.min(np.linalg.norm(sol_c.y[:3], axis=0)))
            return {"entry_state": None, "dv_mag": dv_mag,
                    "perigee_m": r_perigee, "osc_perigee_m": float(osc_perigee)}

        # Coarse sweep (narrower range based on geometry analysis)
        _peak_log = {"t_off_h": float((t_tei_candidate - t) / 3600.0),
                     "align": float(alignments[peak_i]), "sweep": []}
        miss_dvs = []; entry_sims = []
        for dv_try in [800, 900, 1000, 1100]:
            s = simulate_at_peak(dv_try)
            if s is not None:
                if s["entry_state"] is not None:
                    entry_sims.append(s)
                    _peak_log["sweep"].append(
                        (dv_try, "entry", round(s["entry_fpa_deg"], 2)))
                else:
                    miss_dvs.append(dv_try)
                    _peak_log["sweep"].append(
                        (dv_try, "miss_perigee_km",
                         round((s["perigee_m"] - R_EARTH) / 1e3)))
            else:
                _peak_log["sweep"].append((dv_try, "none", None))
        # If no entry at any tested point, try wider
        if not entry_sims:
            for dv_try in [1200, 1400, 1600]:
                s = simulate_at_peak(dv_try)
                _peak_log["sweep"].append(
                    (dv_try, "none", None) if s is None else
                    ((dv_try, "entry", round(s["entry_fpa_deg"], 2))
                     if s["entry_state"] is not None else
                     (dv_try, "miss_perigee_km",
                      round((s["perigee_m"] - R_EARTH) / 1e3))))
                if s is not None and s["entry_state"] is not None:
                    entry_sims.append(s)
                    break
        _tei_dbg.append(_peak_log)

        if not entry_sims:
            # Near-miss refinement: with off-nominal Moon geometry a prograde burn
            # can bottom out a few hundred km ABOVE the entry interface (perigee
            # never crosses 122 km, so no entry event fires). Close that gap with a
            # 3-DOF burn-VECTOR least-squares that drives the ACTUAL (3-body
            # integrated) return perigee into the entry corridor, seeded from the
            # lowest-perigee prograde dv. (The near-Moon osculating perigee is not
            # a valid proxy — the Moon bends the trajectory over the coast.)
            from scipy.optimize import least_squares as _lsq_peri

            def _coast_perigee_vec(dvv):
                n = float(np.linalg.norm(dvv))
                if n < 1.0:
                    return None, n
                bdir = dvv / n
                m0 = state_tei_candidate[6]
                mp = m0 * (1 - np.exp(-n / (isp_sps * G0)))
                if mp >= m0 - (CSM_CM_MASS + CSM_SM_DRY) - 50:
                    return None, n
                bt = mp / mdot_sps
                def rhs_b(ti, y):
                    a = gravity_earth_moon(y[:3], ti) + T_sps * bdir / max(y[6], 1.0)
                    return np.concatenate([y[3:6], a, [-mdot_sps]])
                sb = solve_ivp(rhs_b, (t_tei_candidate, t_tei_candidate + bt),
                               state_tei_candidate, method='RK45',
                               rtol=1e-6, atol=10.0, max_step=10.0)
                pb = sb.y[:, -1]
                def rhs_c(ti, y):
                    return np.concatenate([y[3:6], gravity_earth_moon(y[:3], ti), [0]])
                sc = solve_ivp(rhs_c, (sb.t[-1], sb.t[-1] + 6*86400), pb,
                               method='DOP853', rtol=1e-5, atol=1000.0, max_step=3600.0)
                return float(np.min(np.linalg.norm(sc.y[:3], axis=0))), n

            probes = [p for p in (simulate_at_peak(dv) for dv in
                                  (950, 1000, 1050, 1100, 1150, 1200))
                      if p is not None and p.get("perigee_m")]
            if probes:
                seed = min(probes, key=lambda p: p["perigee_m"])
                if seed["perigee_m"] < R_EARTH + 8000e3:   # within ~8000 km: refine
                    dv0v = seed["dv_mag"] * burn_dir_c
                    def _peri_resid(dvv):
                        peri, _ = _coast_perigee_vec(dvv)
                        if peri is None:
                            return [1e6]
                        return [(peri - (R_EARTH + 30000.0)) / 1000.0]
                    try:
                        solp = _lsq_peri(_peri_resid, dv0v, method='trf',
                                         x_scale='jac', max_nfev=40,
                                         xtol=1e-8, ftol=1e-8, diff_step=1e-3)
                        npv = float(np.linalg.norm(solp.x))
                        s = simulate_at_peak(npv, burn_dir_=solp.x / npv)
                        if s is not None and s["entry_state"] is not None:
                            entry_sims.append(s)
                    except Exception:
                        pass
            if not entry_sims:
                continue   # this peak still doesn't give entry

        # Find shallowest entry by bisecting from largest miss to smallest entry
        smallest_entry_dv = min(s["dv_mag"] for s in entry_sims)
        largest_miss_dv = max([dv for dv in miss_dvs if dv < smallest_entry_dv]
                                or [0.5 * smallest_entry_dv])
        dv_lo = largest_miss_dv; dv_hi = smallest_entry_dv
        peak_best = min(entry_sims, key=lambda s: abs(s["entry_fpa_deg"] - target_fpa))
        for _ in range(10):
            dv_mid = (dv_lo + dv_hi) / 2
            s = simulate_at_peak(dv_mid)
            if s is not None and s["entry_state"] is not None:
                dv_hi = dv_mid
                if abs(s["entry_fpa_deg"] - target_fpa) < abs(peak_best["entry_fpa_deg"] - target_fpa):
                    peak_best = s
            else:
                dv_lo = dv_mid
            if dv_hi - dv_lo < 0.2:
                break

        peak_score = abs(peak_best["entry_fpa_deg"] - target_fpa)
        _peak_log["bisect_score"] = round(float(peak_score), 3)
        _peak_log["bisect_fpa"] = round(float(peak_best["entry_fpa_deg"]), 3)
        if peak_score < best_overall_score:
            best_overall_score = peak_score
            best_overall_sim = peak_best
            best_t_tei = t_tei_candidate
            best_burn_dir = burn_dir_c
            best_state_tei = state_tei_candidate

    # NOTE on corridor acceptance: do NOT hard-reject on the pre-refinement
    # bisect score here — the 3-DOF entry-interface refinement below (with its
    # large-dispersion homotopy) exists precisely to walk big initial corridor
    # errors to the nominal EI, and the corridor-gated candidate CHOICE above
    # already hands it the best available start. (A score>1.5 reject at this
    # point ballooned tei_no_earth_return to 29% while the refinement would
    # have recovered most of them.) Residual steep arrivals are judged honestly
    # by entry physics downstream.
    if best_overall_sim is None:
        results["full_success"] = False
        results["mission_failure"] = "tei_no_earth_return_found"
        results["tei_dv_ms"] = None
        results["tei_diag_n_candidates"] = len(align_peaks)
        results["tei_diag_best_score"] = (None if best_overall_score == float("inf")
                                          else float(best_overall_score))
        results["tei_diag_sweeps"] = repr(_tei_dbg)
        return results, trajectories

    best_sim = best_overall_sim
    t_tei = best_t_tei
    burn_dir = best_burn_dir
    state_tei = best_state_tei

    # Reusable simulator for perturbations (in case of execution bias)
    def simulate_tei(dv_mag, _t_tei=best_t_tei, _state_tei=best_state_tei, _burn_dir=best_burn_dir):
        m0 = _state_tei[6]
        if dv_mag <= 0:
            return None
        mp_burn = m0 * (1 - np.exp(-dv_mag / (isp_sps * G0)))
        if mp_burn >= m0 - (CSM_CM_MASS + CSM_SM_DRY) - 50:
            return None
        burn_time = mp_burn / mdot_sps
        def rhs_burn(ti, y):
            r = y[:3]; v = y[3:6]; m = y[6]
            a = gravity_earth_moon(r, ti)
            a = a + T_sps * _burn_dir / max(m, 1.0)
            return np.concatenate([v, a, [-mdot_sps]])
        sol_b = solve_ivp(rhs_burn, (_t_tei, _t_tei + burn_time),
                          _state_tei, method='RK45',
                          rtol=1e-7, atol=1.0, max_step=2.0)
        post_burn = sol_b.y[:, -1].copy()
        t_post = sol_b.t[-1]
        def rhs_coast(ti, y):
            return np.concatenate([y[3:6], gravity_earth_moon(y[:3], ti), [0]])
        def entry_evt(ti, y):
            return np.linalg.norm(y[:3]) - (R_EARTH + 122_000.0)
        entry_evt.terminal = True; entry_evt.direction = -1
        sol_c = solve_ivp(rhs_coast, (t_post, t_post + 7*86400),
                           post_burn, method='DOP853',
                           rtol=1e-6, atol=100.0, max_step=1800.0,
                           events=entry_evt)
        if len(sol_c.t_events[0]) > 0:
            te = sol_c.t_events[0][0]
            ye = sol_c.y_events[0][0].copy()
            r_e = ye[:3]; v_e = ye[3:6]
            rn = np.linalg.norm(r_e); vn = np.linalg.norm(v_e)
            fpa = np.arcsin(np.dot(r_e, v_e) / (rn * vn))
            return {"entry_state": ye, "entry_t": te,
                     "entry_fpa_deg": float(np.rad2deg(fpa)),
                     "entry_speed_ms": float(vn),
                     "post_burn_state": post_burn,
                     "burn_time_s": burn_time,
                     "t_post_burn": t_post,
                     "dv_mag": dv_mag}
        return None

    results["tei_alignment_cos"] = float(alignments[align_peaks[0]])

    # Apply small TEI execution perturbations
    tei_dv_bias = float(perturb.get("tei_dv_bias_ms", 0.0))
    if abs(tei_dv_bias) > 0.01 and not globals().get("ENABLE_TEI_TARGETING", False):
        s_perturbed = simulate_tei(best_sim["dv_mag"] + tei_dv_bias)
        if s_perturbed is not None and s_perturbed["entry_state"] is not None:
            best_sim = s_perturbed

    # === Faithful TEI-level entry-interface targeting (3-DOF, robust solver) ==
    # Apollo put the entry precision in the TEI/early solve; the trans-Earth
    # MCCs were then small corridor trims (Apollo 11: 4.8 ft/s). Here we solve
    # the full 3-DOF TEI burn VECTOR with a robust least-squares solver so the
    # trajectory passes through the nominal entry POINT (lat/lon) + FPA from the
    # perturbed pre-TEI state. The big upstream dispersion is absorbed by the
    # (already-large) injection burn; only the TEI execution residual is left
    # for the MCCs. Impulsive burn keeps the solve fast and makes targeting and
    # applied burn consistent. Gated OFF by default.
    if globals().get("ENABLE_TEI_TARGETING", False):
        from scipy.optimize import least_squares

        def simulate_burn_vec(dv_vec):
            dvm = float(np.linalg.norm(dv_vec))
            if dvm < 1.0:
                return None
            bdir = dv_vec / dvm
            m0 = best_state_tei[6]
            mp = m0 * (1 - np.exp(-dvm / (isp_sps * G0)))
            if mp >= m0 - (CSM_CM_MASS + CSM_SM_DRY) - 50:
                return None
            bt = mp / mdot_sps

            def rhs_burn(ti, y):
                a = gravity_earth_moon(y[:3], ti) + T_sps * bdir / max(y[6], 1.0)
                return np.concatenate([y[3:6], a, [-mdot_sps]])
            sb = solve_ivp(rhs_burn, (best_t_tei, best_t_tei + bt), best_state_tei,
                           method='RK45', rtol=1e-7, atol=1.0, max_step=5.0)
            pb = sb.y[:, -1].copy(); tp = sb.t[-1]

            def rhs_coast(ti, y):
                return np.concatenate([y[3:6], gravity_earth_moon(y[:3], ti), [0]])

            def evt(ti, y):
                return np.linalg.norm(y[:3]) - (R_EARTH + 122_000.0)
            evt.terminal = True; evt.direction = -1
            sc = solve_ivp(rhs_coast, (tp, tp + 7 * 86400), pb, method='DOP853',
                           rtol=1e-9, atol=1.0, max_step=1800.0, events=evt)
            if len(sc.t_events[0]) == 0:
                return None
            te = sc.t_events[0][0]; ye = sc.y_events[0][0].copy()
            r_e = ye[:3]; v_e = ye[3:6]
            rn = np.linalg.norm(r_e); vn = np.linalg.norm(v_e)
            fpa = float(np.degrees(np.arcsin(np.dot(r_e, v_e) / (rn * vn))))
            la, lo = eci_to_latlon(r_e, te)
            return {"entry_state": ye, "entry_t": te, "entry_fpa_deg": fpa,
                    "entry_speed_ms": float(vn), "post_burn_state": pb,
                    "burn_time_s": float(bt), "t_post_burn": tp,
                    "dv_mag": dvm, "lat": float(la), "lon": float(lo)}

        dv0 = best_sim["dv_mag"] * best_burn_dir
        eit = globals().get("_EI_TARGET")
        is_nominal = not perturb            # None or empty dict -> nominal run

        def resid_to(dv_vec, la_tg, lo_tg, fp_tg):
            s = simulate_burn_vec(dv_vec)
            if s is None:
                return [1e7, 1e7, 1e7]
            dl = ((s["lon"] - lo_tg + 180.0) % 360.0) - 180.0
            return [(s["lat"] - la_tg) * 111000.0,
                    dl * 111000.0 * np.cos(np.radians(s["lat"])),
                    (s["entry_fpa_deg"] - fp_tg) * 1.0e5]

        if eit is None and is_nominal:
            if globals().get("ENABLE_LAUNCH_CONTINUITY", False):
                # APOLLO RETURN-TIMING TARGETING (nominal only, captured into
                # the EI target so every trial inherits it via the homotopy):
                # the corridor-grazing minimum-energy return takes ~63.6 h vs
                # Apollo 11's 59.6 h TEI->EI — and since landing longitude is
                # set by Earth rotation under the return, those 4 h (plus the
                # rev-slip leg) put the recovery zone in NE Africa instead of
                # the mid-Pacific. Bisect dv UP on return time-of-flight to
                # Apollo's 59.6 h (real TEI targeted arrival conditions, not
                # minimum energy: 1,008 m/s vs our corridor-grazing 946), then
                # 3-DOF-refine the burn vector to pull entry FPA back to -6.5
                # at the TOF-correct geography before capturing the target.
                # Raising dv along a FIXED burn direction rotates the escape
                # asymptote off the perigee-nulling aim (the return stops
                # intersecting the atmosphere at all — measured: +220 m/s ->
                # no entry event). Energy and aim must be solved JOINTLY, as
                # the RTCC did: 3-DOF least-squares on the burn VECTOR with
                # residuals (entry FPA -> -6.5 deg, TOF -> 59.6 h, hold the
                # corridor solution's entry latitude to pin the return plane).
                _TOF_TGT = 59.6 * 3600.0
                _lat_hold, _ = eci_to_latlon(best_sim["entry_state"][:3],
                                             best_sim["entry_t"])
                _lat_hold = float(_lat_hold)
                def _resid_tof(dv_vec):
                    s_ = simulate_burn_vec(dv_vec)
                    if s_ is None:
                        return [1e7, 1e7, 1e7]
                    return [(s_["entry_fpa_deg"] + 6.5) * 1.0e5,
                            ((s_["entry_t"] - t_tei) - _TOF_TGT) * 30.0,
                            (s_["lat"] - _lat_hold) * 5.0e3]
                try:
                    _sol_t = least_squares(_resid_tof, dv0, method='trf',
                                           x_scale='jac', ftol=1e-12,
                                           xtol=1e-12, gtol=1e-12,
                                           diff_step=1e-4, max_nfev=60)
                    _s_tof = simulate_burn_vec(_sol_t.x)
                    _ok_t = (_s_tof is not None
                             and abs(_s_tof["entry_fpa_deg"] + 6.5) < 0.3
                             and abs((_s_tof["entry_t"] - t_tei) - _TOF_TGT)
                                 < 0.5 * 3600.0)
                    if globals().get("_TEI_DEBUG"):
                        if _s_tof is not None:
                            print(f"  TOF dbg: dv={np.linalg.norm(_sol_t.x):.0f}"
                                  f" fpa={_s_tof['entry_fpa_deg']:.2f}"
                                  f" TOF={(_s_tof['entry_t']-t_tei)/3600:.2f} h"
                                  f" accepted={_ok_t}")
                        else:
                            print("  TOF dbg: joint solve returned no entry")
                    if _ok_t:
                        best_sim = _s_tof
                except Exception:
                    pass
            # NOMINAL: capture the entry target from the finite-thrust result.
            # (Capture ONLY from a nominal run, so a perturbed trial can never
            # corrupt the target with an off-nominal landing point.)
            _la, _lo = eci_to_latlon(best_sim["entry_state"][:3],
                                     best_sim["entry_t"])
            globals()["_EI_TARGET"] = {"lat": float(_la), "lon": float(_lo),
                                       "fpa": float(best_sim["entry_fpa_deg"])}
        elif eit is not None:
            # Robust target HOMOTOPY (least-squares per step): walk the target
            # from the natural entry point to the nominal in N warm-started
            # steps; each small target move keeps the solve in its basin of
            # attraction (converges even for large initial dispersions). The
            # coast tolerance is moderately loosened (1e-8) and the burn step
            # widened for speed without losing the dispersion collapse.
            lat_t, lon_t, fpa_t = eit["lat"], eit["lon"], eit["fpa"]
            # 1) Cheap single solve from dv0 — converges for small/moderate
            #    dispersion, which is most trials, skipping the homotopy.
            try:
                sol_ls = least_squares(
                    lambda dv: resid_to(dv, lat_t, lon_t, fpa_t), dv0,
                    method='trf', x_scale='jac', ftol=1e-12, xtol=1e-12,
                    gtol=1e-12, diff_step=1e-4, max_nfev=40)
                dv_solved = sol_ls.x
                cost = 2.0 * sol_ls.cost          # sum of squared residuals
            except Exception:
                dv_solved = dv0; cost = np.inf
            # 2) HOMOTOPY fallback ONLY when the single solve stalled (residual
            #    above ~50 km equivalent) — the expensive path runs just for the
            #    large-dispersion trials that need it.
            if cost > (5.0e4) ** 2:
                s0 = simulate_burn_vec(dv0)
                if s0 is not None:
                    lat_n, lon_n, fpa_n = s0["lat"], s0["lon"], s0["entry_fpa_deg"]
                    dlon_total = ((lon_t - lon_n + 180.0) % 360.0) - 180.0
                    dv_cur = dv0.copy()
                    N = 6
                    for k in range(1, N + 1):
                        a = k / N
                        la_i = lat_n + a * (lat_t - lat_n)
                        lo_i = lon_n + a * dlon_total
                        fp_i = fpa_n + a * (fpa_t - fpa_n)
                        try:
                            sol_ls = least_squares(
                                lambda dv: resid_to(dv, la_i, lo_i, fp_i),
                                dv_cur, method='trf', x_scale='jac',
                                ftol=1e-12, xtol=1e-12, gtol=1e-12,
                                diff_step=1e-4, max_nfev=40)
                            dv_cur = sol_ls.x
                        except Exception:
                            break
                    dv_solved = dv_cur
            # TEI execution residual (along the burn dir) -> small MCC workload.
            bias = float(perturb.get("tei_dv_bias_ms", 0.0))
            if abs(bias) > 1e-6:
                dv_solved = dv_solved + bias * (dv_solved / np.linalg.norm(dv_solved))
            s1 = simulate_burn_vec(dv_solved)
            if s1 is not None:
                best_sim = s1

    # Record TEI metrics
    if globals().get("_TEI_DEBUG"):
        results["tei_diag_sweeps"] = repr(_tei_dbg)
    results["tei_dv_ms"]        = best_sim["dv_mag"]
    results["tei_burn_time_s"]  = best_sim["burn_time_s"]
    results["tei_ignition_t_s"] = t_tei
    _mark("tei", t_tei)
    results["fpa_at_entry_deg"] = best_sim["entry_fpa_deg"]
    results["entry_speed_ms"]   = best_sim["entry_speed_ms"]
    results["reached_entry"]    = True

    # --- Trans-Earth midcourse correction chain (MCC-5/6/7) ---------------
    # Implemented along-track FPA-targeting trims to walk the projected entry
    # FPA into the corridor (ON by default; see ENABLE_TRANS_EARTH_MCC and the
    # phase_transearth_mcc docstring). Disabling it falls back to a plain coast.
    if globals().get("ENABLE_TRANS_EARTH_MCC", False):
        mcc = phase_transearth_mcc(best_sim["post_burn_state"],
                                    best_sim["t_post_burn"], perturb)
        results["mcc_total_dv_ms"] = mcc["mcc_total_dv_ms"]
        results["mcc_n_burns"]     = len(mcc["mcc_burns"])
        if not mcc["reached_entry"]:
            results["full_success"] = False
            results["mission_failure"] = "transearth_no_entry_after_mcc"
            return results, trajectories
        # Adopt the (corrected) entry interface state/FPA
        best_sim = dict(best_sim)
        best_sim["entry_state"]   = mcc["final_state"]
        best_sim["entry_t"]       = mcc["final_t"]
        best_sim["entry_fpa_deg"] = mcc["fpa_at_entry_deg"]
        results["fpa_at_entry_deg"] = mcc["fpa_at_entry_deg"]
        results["entry_speed_ms"]   = mcc["entry_speed_ms"]

    state = best_sim["entry_state"].copy()
    t = best_sim["entry_t"]

    if capture_trajectories:
        def rhs_coast_cap(ti, y):
            return np.concatenate([y[3:6], gravity_earth_moon(y[:3], ti), [0]])
        sol_te = solve_ivp(rhs_coast_cap,
                            (best_sim["t_post_burn"], best_sim["entry_t"]),
                            best_sim["post_burn_state"],
                            method='DOP853',
                            rtol=1e-7, atol=10.0, max_step=600.0,
                            dense_output=True)
        ts_te = np.linspace(sol_te.t[0], sol_te.t[-1], 1500)
        trajectories["transearth"] = (ts_te, sol_te.sol(ts_te))

    # 12. Atmospheric entry — CM only (SM jettisoned)
    state[6] = CSM_CM_MASS
    # Optional capture hook for the entry test-bench: records the genuine
    # entry-interface (state, time, perturb) that phase_entry actually
    # receives. Default None → zero overhead and no behavior change.
    _hook = globals().get("_ENTRY_CAPTURE_HOOK", None)
    if _hook is not None:
        _hook.append((state.copy(), float(t), dict(perturb)))
    # SM systems check: event during the trans-Earth coast — POST-LM-jettison,
    # no lifeboat; the CM's entry batteries last hours, not days.
    if _sm_check(t, "sm_failure_transearth"):
        return results, trajectories

    _mark("entry_interface", t)
    # Record the delivered entry interface (diagnostic: separates TEI/MCC
    # delivery dispersion from entry-guidance dispersion in the MC).
    _la_ei, _lo_ei = eci_to_latlon(state[:3], t)
    results["ei_lat"] = float(_la_ei)
    results["ei_lon"] = float(_lo_ei)
    results["ei_t_s"] = float(t)
    _tg_lat, _tg_lon = SPLASH_TARGET_LAT_DEG, SPLASH_TARGET_LON_DEG
    if globals().get("ENABLE_LAUNCH_CONTINUITY", False):
        # PER-OPPORTUNITY RECOVERY ZONE (RTCC practice): Apollo pre-planned a
        # recovery zone for EVERY TEI opportunity — a rev-slipped return aimed
        # at ITS OWN zone and the recovery force repositioned. With launch
        # continuity, rev-slipped trials arrive with their EI rotated up to
        # ~110 deg east of nominal; steering them at the PRIMARY zone is
        # geometrically impossible (~10,000 km behind the entry point) and was
        # never the procedure. Place each trial's zone at the calibrated
        # short-corridor range (2,784 km — the nominal EI->SPLASH_TARGET
        # distance, Apollo's own EI-to-splash design range) along the TRIAL'S
        # OWN entry ground track. splash_miss_km is then guidance accuracy vs
        # the zone actually aimed for; recovery_zone_displacement_km records
        # the operational cost of the slip (0 for on-time returns).
        _rhat_ei = state[:3] / np.linalg.norm(state[:3])
        _vrel_ei = state[3:6] - np.cross(np.array([0.0, 0.0, OMEGA_E]),
                                         state[:3])
        _east_ei = np.cross([0.0, 0.0, 1.0], _rhat_ei)
        _east_ei /= np.linalg.norm(_east_ei)
        _north_ei = np.cross(_rhat_ei, _east_ei)
        _vh_ei = _vrel_ei - np.dot(_vrel_ei, _rhat_ei) * _rhat_ei
        _az_ei = np.arctan2(np.dot(_vh_ei, _east_ei),
                            np.dot(_vh_ei, _north_ei))
        _la0 = np.deg2rad(_la_ei); _lo0 = np.deg2rad(_lo_ei)
        _dlt = 2784.0 / 6371.0
        _lat_t = np.arcsin(np.sin(_la0) * np.cos(_dlt)
                           + np.cos(_la0) * np.sin(_dlt) * np.cos(_az_ei))
        _lon_t = _lo0 + np.arctan2(
            np.sin(_az_ei) * np.sin(_dlt) * np.cos(_la0),
            np.cos(_dlt) - np.sin(_la0) * np.sin(_lat_t))
        _tg_lat = float(np.rad2deg(_lat_t))
        _tg_lon = float((np.rad2deg(_lon_t) + 180.0) % 360.0 - 180.0)
        results["recovery_zone_lat"] = _tg_lat
        results["recovery_zone_lon"] = _tg_lon
        _hav = (np.sin(np.deg2rad(_tg_lat - SPLASH_TARGET_LAT_DEG) / 2) ** 2
                + np.cos(np.deg2rad(SPLASH_TARGET_LAT_DEG))
                * np.cos(np.deg2rad(_tg_lat))
                * np.sin(np.deg2rad(_tg_lon - SPLASH_TARGET_LON_DEG) / 2) ** 2)
        results["recovery_zone_displacement_km"] = float(
            R_EARTH / 1000.0 * 2 * np.arcsin(np.sqrt(min(1.0, _hav))))
    entry = phase_entry(state, t, perturb, target_lat_deg=_tg_lat,
                        target_lon_deg=_tg_lon)
    results["entry_success"] = entry["success"]
    results["max_g"]         = entry["max_g"]
    results["max_q_pa"]      = entry.get("max_q_pa", 0.0)
    if capture_trajectories:
        trajectories["entry"] = (entry["trajectory_t"], entry["trajectory_y"])

    if entry["success"]:
        _mark("splashdown", entry.get("splash_t", t))
        results["splash_lat"] = entry["splash_lat_deg"]
        results["splash_lon"] = entry["splash_lon_deg"]
        # ABSOLUTE miss from the fixed recovery target SPLASH_TARGET. NOTE:
        # this is NOT the targeting dispersion — it is dominated by the
        # systematic offset of the nominal splashdown from SPLASH_TARGET
        # (the default non-skip entry flies a fixed profile and does not
        # steer to the target). True TEI-targeting dispersion = distance
        # from the NOMINAL splashdown, computed in generate_outputs.py as
        # splash_dispersion_km.
        lat1 = np.deg2rad(_tg_lat)
        lat2 = np.deg2rad(entry["splash_lat_deg"])
        dlat = lat2 - lat1
        dlon = np.deg2rad(entry["splash_lon_deg"] - _tg_lon)
        dlon = (dlon + np.pi) % (2*np.pi) - np.pi
        a_hav = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c_hav = 2*np.arcsin(np.sqrt(min(1.0, a_hav)))
        results["splash_miss_km"] = R_EARTH/1000.0 * c_hav
    else:
        # Entry failed: too steep, breakup, or skipped out
        # 12g = Apollo undershoot/structural boundary (NASA TN D-6725).
        if entry.get("max_g", 0) > 12:
            results["mission_failure"] = "entry_structural_failure_high_g"
        elif "reason" in entry:
            results["mission_failure"] = "entry_" + str(entry["reason"])
        else:
            results["mission_failure"] = "entry_failed"

    results["full_success"] = entry.get("success", False)
    return results, trajectories


# ============================================================
# Monte Carlo driver
# ============================================================
def sample_perturbation(rng):
    """Generate small realistic perturbation set.

    Engine reliability and dispersions reflect 1969-era Apollo systems
    documented in NASA mission reports and post-flight analyses.
    """
    # Engine-out dispersions. Saturn V flight history: no F-1 ever failed in
    # flight (perfect record), so the F-1 rate below is a conservative
    # reliability estimate rather than an observed frequency; Apollo 6 lost two
    # S-II J-2 engines and its S-IVB failed to reignite, and Apollo 13 lost its
    # S-II center J-2 to pogo. The vehicle still reached orbit in every case.
    p_f1_failure = 0.015          # 1.5% per engine per flight
    p_j2_s2_failure = 0.010
    p_s_ivb_failure = 0.005
    n_f1_failures = int(rng.binomial(5, p_f1_failure))
    f1_failure_time = float(rng.uniform(10, S_IC_BURN_TIME)) if n_f1_failures > 0 else 1e9
    n_j2_s2_failures = int(rng.binomial(5, p_j2_s2_failure))
    j2_s2_failure_time = float(rng.uniform(10, S_II_BURN_TIME)) if n_j2_s2_failures > 0 else 1e9
    s_ivb_first_burn_fail = bool(rng.uniform(0, 1) < p_s_ivb_failure)

    p = {
        # Launch perturbations
        "n_f1_failures": n_f1_failures,
        "f1_failure_time_s": f1_failure_time,
        "n_j2_s2_failures": n_j2_s2_failures,
        "j2_s2_failure_time_s": j2_s2_failure_time,
        "s_ivb_first_burn_fail": s_ivb_first_burn_fail,
        # Engine ISP dispersions (1-σ ≈ 0.3%)
        "s_ic_isp_factor":       rng.normal(1, 0.003),
        "s_ic_thrust_factor":    rng.normal(1, 0.004),
        "s_ii_isp_factor":       rng.normal(1, 0.003),
        "s_ii_thrust_factor":    rng.normal(1, 0.004),
        # Post-TLI insertion dispersions
        "insertion_alt_err_m":   rng.normal(0, 200),
        "insertion_lat_err_deg": rng.normal(0, 0.005),
        "insertion_lon_err_deg": rng.normal(0, 0.01),
        # Post-TLI random velocity error (per axis). Trimmed 0.10 -> 0.08 m/s as
        # part of pinning the TOTAL injection velocity error to a sourced value
        # (see tli_pointing_rad note below).
        "insertion_v_err":       rng.normal(0, 0.08, 3),
        # Engine dispersions (rest of mission)
        "s_ivb_isp_factor":      rng.normal(1, 0.003),
        "s_ivb_thrust_factor":   rng.normal(1, 0.004),
        "sps_isp_factor":        rng.normal(1, 0.003),
        "sps_thrust_factor":     rng.normal(1, 0.004),
        "dps_isp_factor":        rng.normal(1, 0.003),
        "dps_thrust_factor":     rng.normal(1, 0.005),
        "aps_isp_factor":        rng.normal(1, 0.003),
        "aps_thrust_factor":     rng.normal(1, 0.005),
        # TLI execution. CALIBRATED to a sourced injection-accuracy target:
        # the Saturn V IU guidance velocity-error spec at S-IVB cutoff was
        # ±3.3 ft/s per component (Apollo 4 flight evaluation; conventionally a
        # 3σ tolerance) → ~0.34 m/s 1σ/axis, ~0.58 m/s 1σ total over 3 axes.
        # Realized flight accuracy was better — Apollo 10 = 0.5 ft/s (0.15 m/s)
        # and Apollo 11 = 1.6 ft/s (0.49 m/s) total (mission reports). These
        # inputs are tuned so the TOTAL injection velocity-error magnitude is
        # ~0.5 m/s rms — at Apollo 11's realized level and just inside the spec.
        # (Pointing reduced 5e-5 → 2.8e-5 rad; dv-bias kept at 0.2 m/s.)
        "tli_pointing_rad":      rng.normal(0, 0.000028, 3),
        "tli_dv_bias_ms":        rng.normal(0, 0.2),
        # TEI execution
        "tei_dv_bias_ms":        rng.normal(0, 0.5),
        # Hover-seeking time. Recalibrated from gamma(2.0, 8.0) (mean 16 s,
        # which put ~41% of draws above the ~18 s fuel budget and produced an
        # unrealistic ~45% descent-exhaustion rate) to gamma(1.5, 3.0)
        # (mean 4.5 s): most landings need minor repositioning, with a thin
        # tail for the rare Armstrong-style boulder-field search.
        "hover_seek_s":          max(0, rng.gamma(1.5, 3.0)),
        # Entry aero dispersions
        "cd_factor":             rng.normal(1, 0.02),
        "ld_factor":             rng.normal(1, 0.04),
    }

    # Apollo 11-specific descent failure-mode draws (only consumed when
    # ENABLE_DESCENT_FAILURE_MODES is True; harmless otherwise). Provenance is
    # mixed and noted per constant where defined: the 1202-alarm recovery, abort
    # and touchdown survival probabilities are anchored to NASA's LLORM and the
    # Apollo 11 alarm record, and the rendezvous nominal delta-V / RCS Isp are
    # sourced; the slosh-sensor bias, landing-radar dropout, docking-latch
    # probability and ascent yaw-dispersion magnitudes remain ESTIMATES (Apollo
    # flew the LM nine times with zero such failures, so no observed rate exists).
    if globals().get("ENABLE_DESCENT_FAILURE_MODES", False):
        p["slosh_sensor_bias_s"] = float(
            max(0.0, rng.normal(SLOSH_SENSOR_BIAS_MEAN_S,
                                SLOSH_SENSOR_BIAS_STD_S)))
        fired_1202 = rng.random() < PROB_1202_ALARM
        p["agc_1202_fired"] = bool(fired_1202)
        p["agc_1202_recovered"] = bool(
            (not fired_1202) or (rng.random() < PROB_1202_RECOVERS))
        p["lr_dropout_fired"] = bool(rng.random() < PROB_LR_DROPOUT)
        p["docking_failed"] = bool(rng.random() < PROB_DOCKING_FAILURE)
        # Out-of-plane ascent yaw-steering error (deg, 1-sigma). Real Apollo
        # ascent guidance held the CSM plane to a small residual; this models
        # the dispersion that produces a PHYSICAL (small) plane mismatch at
        # rendezvous. ESTIMATE: ~0.3 deg 1-sigma (yaw-steering budget references
        # ~6 m/s per degree, so sub-degree errors are realistic).
        p["ascent_yaw_err_deg"] = float(rng.normal(0.0, 0.3))

    # Mascon-induced landing downrange dispersion (gated on lunar harmonics).
    # Calibrated proxy for the localized mascon anomalies the C20/C22 field
    # cannot resolve; anchored to Apollo's documented long landing. Recorded as
    # a landing-accuracy outcome (downrange error vs target), biased long.
    if globals().get("ENABLE_LUNAR_HARMONICS", False):
        p["mascon_downrange_m"] = float(
            rng.normal(MASCON_DOWNRANGE_BIAS_M, MASCON_DOWNRANGE_STD_M))

    # Manual-flying / landing-redesignation hover time (s); consumed only when
    # ENABLE_DOI is on. Models the extra low-altitude powered flight to select a
    # safe touchdown point (Apollo 11's boulder-field overfly was the extreme
    # case). Drawn LAST so it never shifts the existing perturbation stream.
    p["manual_flying_s"] = float(max(0.0, rng.gamma(MANUAL_FLYING_SHAPE,
                                                    MANUAL_FLYING_SCALE)))

    # Transposition & docking (post-TLI) unrecovered failure — the SECOND
    # docking event of the sourced two-docking decomposition (see the
    # PROB_DOCKING_FAILURE block; the existing docking_failed draw above keeps
    # its stream position and covers the ascent-rendezvous docking). Appended
    # at the very end per the RNG-stream rule.
    p["td_docking_failed"] = bool(rng.random() < PROB_DOCKING_FAILURE)

    # SM systems catastrophic failure (the Apollo 13 mode; see
    # PROB_SM_CATASTROPHIC block): occurrence + uniform timing fraction.
    # Appended after td_docking_failed per the RNG-stream rule.
    p["sm_failure"] = bool(rng.random() < PROB_SM_CATASTROPHIC)
    p["sm_failure_frac"] = float(rng.random())

    # Surface-operations modes (sourced; see PROB_EVA_SUIT_FATALITY /
    # PROB_LM_SURFACE_ELEC / PROB_LM_TIPOVER). Appended after sm_failure
    # per the RNG-stream rule.
    p["eva_suit_fatality"] = bool(rng.random() < PROB_EVA_SUIT_FATALITY)
    p["lm_surface_elec_failed"] = bool(rng.random() < PROB_LM_SURFACE_ELEC)
    p["lm_tipover"] = bool(rng.random() < PROB_LM_TIPOVER)

    return p


def _json_safe(v):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def save_trial_debug(outdir, trial_idx, results):
    """Write a per-trial debug/overview JSON (phase timing + all outcomes) to
    <outdir>/trials/trial_<idx>.json, so any trial can be reviewed individually."""
    import json
    tdir = os.path.join(outdir, "trials")
    os.makedirs(tdir, exist_ok=True)
    rec = {k: _json_safe(v) for k, v in results.items() if k != "_phase_log"}
    rec["trial"] = trial_idx if isinstance(trial_idx, str) else int(trial_idx)
    rec["phase_timeline"] = build_phase_timeline(results.get("_phase_log"))
    with open(os.path.join(tdir, f"trial_{trial_idx}.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)


def main(n=1000,
         outdir="/mnt/user-data/outputs/apollo11_realsim",
         seed=42,
         resume=True):
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "results.csv")
    ei_target_path = os.path.join(outdir, "ei_target.json")
    # TEI targeting needs the nominal entry target. Start clean, then load a
    # persisted target if this is a resume of a run that already captured one
    # (so resume batches don't re-run the nominal just to recapture it).
    globals()["_EI_TARGET"] = None
    if globals().get("ENABLE_TEI_TARGETING", False) and os.path.exists(ei_target_path):
        try:
            import json as _json
            with open(ei_target_path) as _f:
                globals()["_EI_TARGET"] = _json.load(_f)
        except Exception:
            pass
    # B-plane TLMCC targeting needs the nominal lunar-approach B-plane, captured
    # on the nominal run; load a persisted one on resume (same rationale).
    bplane_target_path = os.path.join(outdir, "bplane_target.json")
    globals()["_BPLANE_TARGET"] = None
    if globals().get("ENABLE_BPLANE_TLMCC", False) and os.path.exists(bplane_target_path):
        try:
            import json as _json
            with open(bplane_target_path) as _f:
                globals()["_BPLANE_TARGET"] = tuple(_json.load(_f))
        except Exception:
            pass
    json_path = os.path.join(outdir, "nominal_results.json")
    npz_path  = os.path.join(outdir, "nominal_traj.npz")

    # Resume: load existing results
    existing_results = []
    start_trial = 0
    if resume and os.path.exists(csv_path):
        try:
            prev = pd.read_csv(csv_path)
            existing_results = prev.to_dict('records')
            if 'trial' in prev.columns:
                start_trial = int(prev['trial'].max()) + 1
            else:
                start_trial = len(prev)
            print(f"  Resuming from trial {start_trial} ({len(existing_results)} existing)")
        except Exception:
            pass

    print(f"Running {n} physics-integrated Apollo 11 missions (start={start_trial})...")
    t_start = time.time()
    rng = np.random.default_rng(seed)
    # Advance RNG to skip already-completed trials
    for _ in range(start_trial):
        sample_perturbation(rng)

    results_list = list(existing_results)
    nominal_traj = None
    nominal_results = None

    # Run nominal trajectory if not already saved
    if not os.path.exists(json_path):
        print("  Nominal trajectory (full capture)...")
        nominal_results, nominal_traj = run_mission(perturb=None,
                                                     capture_trajectories=True)
        print(f"  Nominal complete in {time.time()-t_start:.1f}s")
        print(f"  Nominal full_success: {nominal_results.get('full_success')}, "
              f"fuel_margin: {nominal_results.get('fuel_margin_s', 0):.0f}s")
        save_trial_debug(outdir, "nominal", nominal_results)  # nominal phase-timing overview

        import json
        traj_save = {}
        for k, v in nominal_traj.items():
            if isinstance(v, tuple) and len(v) == 2:
                traj_save[k + "_t"] = np.asarray(v[0])
                traj_save[k + "_y"] = np.asarray(v[1])
        np.savez_compressed(npz_path, **traj_save)

        nom_safe = {}
        for k, v in nominal_results.items():
            if k == "_phase_log":
                continue
            if isinstance(v, np.ndarray):
                nom_safe[k] = v.tolist()
            elif isinstance(v, np.bool_):
                nom_safe[k] = bool(v)
            elif isinstance(v, (np.floating, np.integer)):
                nom_safe[k] = float(v)
            else:
                nom_safe[k] = v
        with open(json_path, "w") as f:
            json.dump(nom_safe, f, indent=2, default=str)
    else:
        # Warm Lambert cache
        _ = initial_state_post_tli()
        print("  (skipping nominal; using existing nominal_results.json)")
        # The MC loop runs only PERTURBED trials, but TEI targeting needs the
        # entry target captured from a NOMINAL run. On resume the nominal above
        # is skipped, so capture it here with one nominal mission (discarded).
        need_ei = (globals().get("ENABLE_TEI_TARGETING", False)
                   and globals().get("_EI_TARGET") is None)
        need_bp = (globals().get("ENABLE_BPLANE_TLMCC", False)
                   and globals().get("_BPLANE_TARGET") is None)
        if need_ei or need_bp:
            print("  Capturing nominal target(s) for TEI / B-plane targeting...")
            run_mission(perturb=None, capture_trajectories=False)

    # Persist the captured entry target so resume batches load it instead of
    # re-running the nominal to recapture it.
    if (globals().get("ENABLE_TEI_TARGETING", False)
            and globals().get("_EI_TARGET") is not None
            and not os.path.exists(ei_target_path)):
        try:
            import json as _json
            with open(ei_target_path, "w") as _f:
                _json.dump(globals()["_EI_TARGET"], _f)
        except Exception:
            pass
    if (globals().get("ENABLE_BPLANE_TLMCC", False)
            and globals().get("_BPLANE_TARGET") is not None
            and not os.path.exists(bplane_target_path)):
        try:
            import json as _json
            with open(bplane_target_path, "w") as _f:
                _json.dump(list(globals()["_BPLANE_TARGET"]), _f)
        except Exception:
            pass

    # Run remaining trials
    t0_mc = time.time()
    for i in range(start_trial, n):
        if (i - start_trial) % 10 == 0 or i == n-1:
            done = i - start_trial + 1
            rate = done / max(0.1, time.time() - t0_mc)
            eta = (n - i - 1) / max(0.01, rate)
            print(f"  Trial {i+1}/{n}  ({rate:.2f}/s, ETA {eta:.0f}s)", flush=True)
        t_trial = time.time()
        try:
            perturb = sample_perturbation(rng)
            r, _ = run_mission(perturb=perturb, capture_trajectories=False)
            r["trial"] = i
            r["trial_time_s"] = time.time() - t_trial
            save_trial_debug(outdir, i, r)   # per-trial phase-timing overview
            r.pop("_phase_log", None)         # keep the CSV scalar-only
            results_list.append(r)
        except Exception as e:
            results_list.append({"trial": i, "error": str(e),
                                 "trial_time_s": time.time() - t_trial})
        # Checkpoint after EVERY trial: with TEI targeting on, trials can take
        # ~45-90 s, so coarser checkpointing risks losing a whole batch to the
        # foreground time cap. Per-trial CSV writes are cheap at this N.
        pd.DataFrame(results_list).to_csv(csv_path, index=False)

    elapsed = time.time() - t_start
    new_runs = n - start_trial
    print(f"Done. {new_runs} new trials in {elapsed:.1f}s")

    df = pd.DataFrame(results_list)
    df.to_csv(csv_path, index=False)

    return df, nominal_traj, nominal_results


# ---------------------------------------------------------------------------
# Parallel Monte Carlo
# ---------------------------------------------------------------------------

def _parallel_worker_init(ei_target, bplane_target):
    """Pool initializer: inject nominal targets into each worker's module globals."""
    globals()["_EI_TARGET"] = ei_target
    globals()["_BPLANE_TARGET"] = None if bplane_target is None else tuple(bplane_target)


def _parallel_run_trial(args):
    """Top-level worker function (must be picklable — no closures).

    Records this trial's own wall-clock compute time (each worker runs one trial
    at a time, so this is the per-trial cost, not amortized wall time)."""
    trial_idx, perturb = args
    t_trial = time.time()
    try:
        r, _ = run_mission(perturb=perturb, capture_trajectories=False)
        r["trial"] = trial_idx
        r["trial_time_s"] = time.time() - t_trial
        return r
    except Exception as e:
        return {"trial": trial_idx, "error": str(e),
                "trial_time_s": time.time() - t_trial}


def main_parallel(n=1000,
                  outdir="/mnt/user-data/outputs/apollo11_realsim",
                  seed=42,
                  resume=True,
                  workers=None,
                  indices=None):
    """Monte Carlo driver using multiprocessing.Pool for parallel trials.

    Nominal trajectory and target capture are serial (same as main()).
    MC trials fan out across `workers` processes (default: cpu_count - 1).
    Results are checkpointed to CSV after every trial via imap_unordered.

    `indices` (optional): restrict this invocation to a SUBSET of trial
    indices in [0, n) — the multi-node sharding hook (each cluster node runs
    a disjoint stride into its own outdir; see cluster_run.py). All n
    perturbations are still generated in trial order first, so trial i maps
    to the identical perturbation regardless of sharding. indices=[] runs
    only the nominal + target capture (the shard-setup stage).
    """
    import multiprocessing as mp

    os.makedirs(outdir, exist_ok=True)
    csv_path         = os.path.join(outdir, "results.csv")
    ei_target_path   = os.path.join(outdir, "ei_target.json")
    bplane_target_path = os.path.join(outdir, "bplane_target.json")
    json_path        = os.path.join(outdir, "nominal_results.json")
    npz_path         = os.path.join(outdir, "nominal_traj.npz")

    # --- load persisted targets (resume) ------------------------------------
    import json as _json

    globals()["_EI_TARGET"] = None
    if globals().get("ENABLE_TEI_TARGETING", False) and os.path.exists(ei_target_path):
        try:
            with open(ei_target_path) as _f:
                globals()["_EI_TARGET"] = _json.load(_f)
        except Exception:
            pass

    globals()["_BPLANE_TARGET"] = None
    if globals().get("ENABLE_BPLANE_TLMCC", False) and os.path.exists(bplane_target_path):
        try:
            with open(bplane_target_path) as _f:
                globals()["_BPLANE_TARGET"] = tuple(_json.load(_f))
        except Exception:
            pass

    # --- resume: load existing results --------------------------------------
    # Track the SET of completed trial indices, not just the max. Under
    # imap_unordered the CSV is written in completion order, so an interrupted
    # parallel run can leave non-contiguous completed trials (e.g. trial 50
    # checkpointed while trial 30 was still running). Using max+1 would skip the
    # holes forever; the set lets us dispatch exactly the missing trials.
    existing_results = []
    completed_trials = set()
    if resume and os.path.exists(csv_path):
        try:
            prev = pd.read_csv(csv_path)
            existing_results = prev.to_dict("records")
            if "trial" in prev.columns:
                completed_trials = {int(t) for t in prev["trial"].dropna()}
            print(f"  Resuming: {len(completed_trials)} trial(s) already complete")
        except Exception:
            pass

    # Checkpoint helper: always write the CSV SORTED by trial index. Parallel
    # trials finish out of order, but downstream consumers (e.g. crew_survival,
    # which draws survival outcomes positionally by row) and serial-equivalence
    # both require deterministic trial-order rows.
    def _checkpoint(rl):
        _df = pd.DataFrame(rl)
        if "trial" in _df.columns:
            _df = _df.sort_values("trial").reset_index(drop=True)
        _df.to_csv(csv_path, index=False)
        return _df

    print(f"Running {n} physics-integrated Apollo 11 missions "
          f"({len(completed_trials)} already done)...")
    t_start = time.time()

    # --- nominal trajectory -------------------------------------------------
    if not os.path.exists(json_path):
        print("  Nominal trajectory (full capture)...")
        nominal_results, nominal_traj = run_mission(perturb=None, capture_trajectories=True)
        print(f"  Nominal complete in {time.time()-t_start:.1f}s")
        print(f"  Nominal full_success: {nominal_results.get('full_success')}, "
              f"fuel_margin: {nominal_results.get('fuel_margin_s', 0):.0f}s")
        save_trial_debug(outdir, "nominal", nominal_results)  # nominal phase-timing overview

        traj_save = {}
        for k, v in nominal_traj.items():
            if isinstance(v, tuple) and len(v) == 2:
                traj_save[k + "_t"] = np.asarray(v[0])
                traj_save[k + "_y"] = np.asarray(v[1])
        np.savez_compressed(npz_path, **traj_save)

        nom_safe = {}
        for k, v in nominal_results.items():
            if k == "_phase_log":
                continue
            if isinstance(v, np.ndarray):
                nom_safe[k] = v.tolist()
            elif isinstance(v, np.bool_):
                nom_safe[k] = bool(v)
            elif isinstance(v, (np.floating, np.integer)):
                nom_safe[k] = float(v)
            else:
                nom_safe[k] = v
        with open(json_path, "w") as f:
            _json.dump(nom_safe, f, indent=2, default=str)
    else:
        _ = initial_state_post_tli()
        print("  (skipping nominal; using existing nominal_results.json)")
        need_ei = (globals().get("ENABLE_TEI_TARGETING", False)
                   and globals().get("_EI_TARGET") is None)
        need_bp = (globals().get("ENABLE_BPLANE_TLMCC", False)
                   and globals().get("_BPLANE_TARGET") is None)
        if need_ei or need_bp:
            print("  Capturing nominal target(s) for TEI / B-plane targeting...")
            run_mission(perturb=None, capture_trajectories=False)

    # --- persist targets so resume batches skip re-running nominal ----------
    if (globals().get("ENABLE_TEI_TARGETING", False)
            and globals()["_EI_TARGET"] is not None
            and not os.path.exists(ei_target_path)):
        with open(ei_target_path, "w") as _f:
            _json.dump(globals()["_EI_TARGET"], _f)

    if (globals().get("ENABLE_BPLANE_TLMCC", False)
            and globals()["_BPLANE_TARGET"] is not None
            and not os.path.exists(bplane_target_path)):
        with open(bplane_target_path, "w") as _f:
            _json.dump(list(globals()["_BPLANE_TARGET"]), _f)

    # --- pre-generate perturbations in the main process (deterministic) -----
    # Generate ALL n perturbations in trial order so trial i always maps to the
    # same perturbation as in serial main(), then dispatch only the trials not
    # already present in the CSV. This is gap-safe on resume (fills exactly the
    # holes left by an interrupted parallel run).
    rng = np.random.default_rng(seed)
    all_perturbs = [sample_perturbation(rng) for _ in range(n)]
    _wanted = set(range(n)) if indices is None else set(int(i) for i in indices)
    perturbations = [(i, all_perturbs[i]) for i in range(n)
                     if i in _wanted and i not in completed_trials]

    if not perturbations:
        print("All requested trials already complete."
              if indices is not None else "All trials already complete.")
        return _checkpoint(existing_results), None, None

    # --- parallel MC --------------------------------------------------------
    n_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    print(f"  Dispatching {len(perturbations)} trials across {n_workers} workers...")

    ei_target     = globals()["_EI_TARGET"]
    bplane_target = globals()["_BPLANE_TARGET"]

    results_list = list(existing_results)
    completed    = 0
    t0_mc        = time.time()

    with mp.Pool(
        processes=n_workers,
        initializer=_parallel_worker_init,
        initargs=(ei_target, bplane_target),
    ) as pool:
        for r in pool.imap_unordered(_parallel_run_trial, perturbations):
            if "trial" in r:
                save_trial_debug(outdir, r["trial"], r)  # per-trial phase-timing overview
            r.pop("_phase_log", None)                     # keep the CSV scalar-only
            results_list.append(r)
            completed += 1
            if completed % 5 == 0 or completed == len(perturbations):
                rate = completed / max(0.1, time.time() - t0_mc)
                eta  = (len(perturbations) - completed) / max(0.01, rate)
                print(f"  {completed}/{len(perturbations)} trials done  "
                      f"({rate:.2f}/s, ETA {eta:.0f}s)", flush=True)
            _checkpoint(results_list)

    elapsed = time.time() - t_start
    print(f"Done. {len(perturbations)} new trials in {elapsed:.1f}s")

    df = _checkpoint(results_list)
    return df, None, None


if __name__ == "__main__":
    main(n=10)   # quick smoke test
