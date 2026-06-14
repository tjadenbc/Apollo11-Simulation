"""
Apollo 11 Monte Carlo: post-processing and output generation.

Reads results.csv and the nominal trajectory, produces:
  - results.csv (already exists from main())
  - summary.txt
  - 6 plots (PNG)
  - dashboard.html (consolidates everything)
  - Copy of apollo11.py source
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

OUTDIR = "outputs/apollo11_final10000"
SIMDIR = "."   # apollo11.py lives in the project dir (run from there)

R_EARTH = 6_378_137.0
R_MOON  = 1_737_400.0

# Apollo 11 actual values for comparison
APOLLO_ACTUALS = {
    "tli_dv_ms":          3153,
    "periapsis_alt_km":   110,
    "loi_burn_time_s":    357,
    "fuel_margin_s":      25,
    "touchdown_v_radial_ms": -1.0,
    "touchdown_v_horiz_ms": 0.5,
    "ascent_alt_km":      85,
    "aps_prop_remaining_kg": 380,
    "tei_dv_ms":          1008,
    "tei_burn_time_s":    152,
    "fpa_at_entry_deg":   -6.49,
    "entry_speed_ms":     11032,
    "max_g":              6.5,
    "splash_lat":         13.30,
    "splash_lon":         -169.15,
    "mission_duration_d": 8.16,
}


# Failure mode explanations (in plain language) — keyed by mission_failure value
FAILURE_EXPLANATIONS = {
    "launch_s_ic_underperformance_crash": (
        "Saturn V S-IC stage underperformance",
        "The first stage couldn't generate enough thrust to climb out of the "
        "lower atmosphere — typically caused by multiple F-1 engine failures "
        "early in flight. The vehicle decelerated and fell back."),
    "launch_structural_failure_max_q_exceeded": (
        "Structural failure at max-Q",
        "Aerodynamic loading on the vehicle exceeded the design limit "
        "(~50 kPa) during transonic flight. This can occur when off-nominal "
        "thrust profiles cause the rocket to accelerate too aggressively "
        "through the dense lower atmosphere."),
    "launch_s_ivb_first_ignition_failure": (
        "S-IVB failed to ignite for parking orbit",
        "The J-2 engine on the third stage failed to fire after S-II "
        "burnout. With no third-stage burn, the stack remained on a "
        "suborbital ballistic trajectory. (Apollo 6 had a related J-2 "
        "shutdown anomaly; ground analysis identified igniter line "
        "fuel-leak failure modes.)"),
    "launch_parking_orbit_decays_into_atmosphere": (
        "Parking orbit perigee too low",
        "S-IVB first burn placed the stack on a parking orbit with perigee "
        "below 80 km — the atmosphere would have decayed the orbit before "
        "TLI could be performed. Caused by guidance dispersion + engine "
        "underperformance."),
    "launch_parking_insertion_overshoot": (
        "Parking orbit apogee too high",
        "S-IVB over-burn placed the stack on an excessively elongated orbit "
        "(apogee > 1000 km). TLI from such an orbit wastes propellant and "
        "would generally require an additional correction burn."),
    "launch_parking_insertion_overshoot_escape": (
        "S-IVB overshoot → escape trajectory",
        "S-IVB burned too long; specific orbital energy went positive, "
        "putting the stack on a hyperbolic Earth-escape trajectory. With no "
        "ability to recover, the mission is unrecoverable."),
    "missed_lunar_soi": (
        "Translunar trajectory missed Moon's sphere of influence",
        "After translunar coast and MCC, the trajectory's closest approach "
        "to the Moon was outside the recoverable range (perilune below "
        "-200 km, i.e. would impact the surface, or above +1500 km, i.e. "
        "too high for LOI to capture). Apollo carried fuel for one MCC "
        "but couldn't fix arbitrarily large dispersions."),
    "descent_descent_propellant_exhausted": (
        "LM ran out of fuel during powered descent",
        "The Lunar Module's Descent Propulsion System (DPS) consumed all "
        "available propellant before touchdown. A nominal, anomaly-free descent "
        "from the ~15 km PDI lands with a large reserve (close to Apollo's "
        "planned ~2-minute hover budget), but a long manual landing-site "
        "redesignation — flying level to overfly hazardous terrain, as Apollo 11 "
        "did across a boulder field — combined with DPS performance dispersions "
        "(ISP, thrust) can eat that reserve. Apollo 11 itself touched down within "
        "tens of seconds of the abort threshold; this is the rare tail where the "
        "overfly outlasts the propellant."),
    "descent_breakup_on_landing": (
        "Hard landing — LM destroyed",
        "Touchdown vertical or horizontal velocity exceeded the LM landing "
        "gear's design tolerance (~3 m/s vertical, ~1.5 m/s horizontal). "
        "The vehicle is treated as lost."),
    "ascent_engine_failure": (
        "APS (ascent engine) failed",
        "The Lunar Module ascent stage engine failed to ignite or shut "
        "down prematurely. With no ascent capability, the crew is stranded."),
    "ascent_insufficient_dv": (
        "LM ascent didn't reach lunar orbit",
        "Ascent burn completed but produced insufficient ΔV to reach a "
        "stable lunar orbit for rendezvous with the CSM."),
    "transearth_no_entry": (
        "Trans-Earth coast didn't reach Earth atmosphere",
        "The Earth-return trajectory failed to intersect the atmospheric "
        "entry interface (122 km altitude) within five days of TEI. "
        "Either TEI ΔV was insufficient, or the post-burn trajectory was "
        "deflected away from Earth by Moon's residual gravity."),
    "entry_structural_failure_high_g": (
        "Command Module destroyed by high-g entry",
        "Atmospheric entry exceeded the CM's 12 g structural limit (the "
        "entry_structural_failure_high_g threshold; NASA TN D-6725 cites 10 g "
        "guidance / ~12 g structural). The capsule would have broken up; crew "
        "lost. Apollo 11's actual entry produced ~6.5 g — a steeper "
        "flight-path-angle dispersion can push it past 12 g."),
    "entry_skip_out_or_breakup": (
        "Entry trajectory skipped out or capsule lost",
        "The capsule either skipped off the atmosphere and onto a long "
        "ballistic trajectory (lost in space), or experienced structural "
        "breakup. Skip-out happens with shallow flight path angles; "
        "breakup happens with steep flight path angles."),
    "synthetic_tei_failed": (
        "TEI targeting failed",
        "The TEI burn-vector solver couldn't drive the trajectory to the "
        "entry interface from the post-rendezvous CSM state. Rare — usually "
        "caused by extremely off-nominal lunar orbit geometry."),
    "tei_dv_too_large": (
        "TEI would require excessive ΔV",
        "Required TEI ΔV exceeded SPS capability (>6 km/s). The entry-"
        "targeting solution fell outside the feasible region."),
    "tei_insufficient_propellant": (
        "TEI: not enough SPS propellant",
        "The CSM ran out of usable SPS propellant before TEI could complete. "
        "Usually caused by upstream burns (especially LOI) consuming more "
        "propellant than expected due to ISP/thrust dispersions. The crew "
        "is stranded in lunar orbit."),
    # --- SM-systems catastrophic mode (Apollo 13 class; PROB_SM_CATASTROPHIC
    #     = 1/15 sourced; consequence depends on phase / LM availability) ---
    "sm_failure_translunar": (
        "SM systems failure — translunar (Apollo 13 class)",
        "Catastrophic service-module systems failure (cryo tank / fuel cell) "
        "during the outbound coast, LM attached. Recovered via the "
        "LM-lifeboat abort — the demonstrated Apollo 13 path — at real margin "
        "risk (power, water, CO2, entry corridor)."),
    "sm_failure_lunar_orbit": (
        "SM systems failure — lunar orbit",
        "Catastrophic SM failure in lunar orbit (pre-descent or post-ascent). "
        "LM lifeboat available, but the abort starts from lunar orbit with "
        "tighter consumables than Apollo 13's free-return geometry."),
    "sm_failure_surface": (
        "SM systems failure — LM on surface",
        "Catastrophic SM failure while the LM is on the surface: Collins is "
        "alone in the failing CSM, the LM must launch immediately outside its "
        "window, and the stack attempts a lifeboat return — the worst "
        "pre-jettison geometry."),
    "sm_failure_transearth": (
        "SM systems failure — trans-Earth (no lifeboat)",
        "Catastrophic SM failure during the trans-Earth coast, after the LM has "
        "been jettisoned — there is no lifeboat. The CM's entry batteries and "
        "surge tanks last hours, not the days usually remaining to entry, so it "
        "kills all three in the great majority of cases; ONLY those whole-stack "
        "losses are counted in this row as mission failures. In the minority of "
        "missions where the failure strikes late enough — within the last few "
        "hours before entry — the crew rides the remaining consumables down and "
        "returns alive. Those trials had already landed on the Moon and brought "
        "all three home, so under the mission-success definition (land + return "
        "all alive) they are counted as MISSION SUCCESSES with a recovered "
        "anomaly, NOT as failures in this row."),
    # --- Docking (sourced two-docking decomposition) ---
    "transposition_docking_failure": (
        "Transposition & docking failed (no LM)",
        "After TLI the CSM could not extract the LM from the S-IVB "
        "(unrecoverable docking failure; nearly ended Apollo 14). The landing "
        "is aborted and the crew returns on a fully healthy CSM."),
    "rendezvous_docking_failure": (
        "Ascent-rendezvous docking failed",
        "The LM reached the CSM but the docking mechanism failed to hard-dock "
        "(sourced ~0.95%/docking). The documented contingency is a suited EVA "
        "crew transfer. LM-crew exposure; Collins safe in the CSM."),
    # --- Surface operations (sourced) ---
    "surface_lm_electrical_failure": (
        "LM surface electrical/switchgear failure",
        "An unrecoverable LM electrical fault on the surface prevents arming "
        "the ascent engine (anchored to Apollo 11's snapped arming circuit "
        "breaker). The LM crew is stranded; Collins returns alone."),
    "surface_lm_tipover": (
        "LM tip-over at touchdown",
        "The LM tipped beyond recovery at touchdown (~12° stability limit; "
        "Apollo 15 landed at ~11°). No ascent is possible and no rescue "
        "capability exists."),
    "surface_eva_suit_fatality": (
        "EVA suit/PLSS terminal failure",
        "A terminal suit/PLSS failure during the surface EVA (0 failures in "
        "28 program man-EVAs; the OPS backup was never used). Modeled as one "
        "LM-crew death; the partner and Collins return."),
    # --- TLI guidance/propellant (post wrap-fix; typically zero) ---
    "tli_propellant_depleted": (
        "TLI: S-IVB propellant depleted",
        "The S-IVB ran out of propellant before reaching TLI cutoff speed — a "
        "genuine engine-underperformance starve (distinct from the "
        "now-fixed ignition-window solver miss)."),
    "tli_ignition_window_missed": (
        "TLI: ignition window not found",
        "The guidance could not locate the TLI ignition crossing on the "
        "launched parking orbit (a solver condition, not propellant). Driven "
        "to ~zero by the wrap-safe crossing fix."),
    # --- Navigation (post-MCC SOI miss) ---
    "missed_lunar_soi_after_mcc": (
        "Missed lunar approach after MCC chain",
        "After the full MCC chain (including the closed-loop MCC-4b trim) the "
        "corrected perilune was still outside the LOI-recoverable band — a "
        "deep impact or a too-high approach no midcourse could rescue."),
    "descent_radar_dropout_marginal": (
        "Landing-radar dropout — marginal landing",
        "A brief landing-radar blackout that's harmless on a normal descent, "
        "but on the rare descent that's already scraping the bottom of the "
        "fuel tank, the few seconds of degraded guidance are enough to lose "
        "the landing. (Modeled as a loss only when a radar dropout coincides "
        "with a near-empty terminal phase — fuel margin under ~5 s.)"),
    "crew_return_leg_loss": (
        "Lost on the Earth-return leg (post-abort)",
        "Astronaut(s) who SURVIVED the precipitating failure but died on the "
        "trans-Earth coast + entry — e.g. Collins returning solo in the CSM "
        "after the LM crew was lost, or a crew that aborted successfully but "
        "did not complete the return. Attributed here (the actual cause of "
        "death) rather than to the upstream mode that only killed their "
        "crewmates."),
}


def categorize_failure(reason):
    """Return (category, short_label) for a failure reason."""
    if pd.isna(reason) or reason is None:
        return ("success", "Mission Success")
    s = str(reason)
    if s.startswith("launch_"):
        return ("launch", s)
    if s.startswith("missed_lunar_soi") or s == "missed_lunar_soi":
        return ("translunar", s)
    if s == "tli_propellant_depleted" or s == "tli_ignition_window_missed":
        return ("translunar", s)
    if s == "transposition_docking_failure":
        return ("translunar", s)
    if s == "sm_failure_translunar":
        return ("translunar", s)
    if s == "sm_failure_lunar_orbit" or s == "sm_failure_surface":
        return ("lunar_orbit", s)
    if s.startswith("descent_"):
        return ("descent", s)
    if s.startswith("surface_"):
        return ("surface", s)
    if s.startswith("ascent_"):
        return ("ascent", s)
    if s.startswith("rendezvous_"):
        return ("rendezvous", s)
    if s.startswith("transearth_") or s == "sm_failure_transearth":
        return ("transearth", s)
    if s.startswith("tei_") or s == "synthetic_tei_failed" or s == "tei_dv_too_large":
        return ("tei", s)
    if s.startswith("entry_"):
        return ("entry", s)
    return ("other", s)


def _fmtnum(v, unit, dec):
    s = f"{v:,.{dec}f}"
    if not unit:
        return s
    if unit == "°":
        return f"{s}°"
    return f"{s} {unit}"


# Phase labels + Apollo 11 actual durations: single source of truth in apollo11.py
# (per-trial JSONs carry these same labels), with a local fallback.
try:
    import apollo11 as _a11
    _PHASE_ORDER = [lbl for _, _, lbl in _a11.PHASE_SEGMENTS]
    _APOLLO_PHASE_DUR = dict(_a11.APOLLO_PHASE_DUR_S)
except Exception:
    _PHASE_ORDER = ["Launch to orbit", "Parking orbit + TLI", "Translunar coast",
                    "Lunar orbit (LOI->PDI)", "Powered descent", "Surface stay",
                    "Ascent to orbit", "Rendezvous to TEI", "Trans-earth coast",
                    "Entry to splashdown"]
    _APOLLO_PHASE_DUR = {"Launch to orbit": 713.0, "Parking orbit + TLI": 9143.0,
                         "Translunar coast": 263134.0, "Lunar orbit (LOI->PDI)": 96195.0,
                         "Powered descent": 755.0, "Surface stay": 77780.0,
                         "Ascent to orbit": 435.0, "Rendezvous to TEI": 39267.0,
                         "Trans-earth coast": 214764.0, "Entry to splashdown": 929.0}


def _fmt_dur(s):
    """Human-friendly duration: seconds -> s / min / h / d."""
    if s is None:
        return "—"
    s = float(s)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s/60:.1f} min"
    if s < 172800:
        return f"{s/3600:.1f} h"
    return f"{s/86400:.2f} d"


def phase_timing_stats(outdir):
    """Aggregate per-trial phase MISSION durations from <outdir>/trials/*.json into
    min/avg/max per phase, with the Apollo 11 actual and the avg-vs-Apollo delta."""
    import glob
    files = glob.glob(os.path.join(outdir, "trials", "trial_*.json"))
    durs = {}
    for f in files:
        if os.path.basename(f) == "trial_nominal.json":
            continue   # nominal shown separately; aggregate over MC trials
        try:
            with open(f) as fh:
                rec = json.load(fh)
        except Exception:
            continue
        for ph in rec.get("phase_timeline", []):
            durs.setdefault(ph["phase"], []).append(float(ph["duration_s"]))
    rows = []
    for label in _PHASE_ORDER:
        vals = durs.get(label, [])
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        apollo = _APOLLO_PHASE_DUR.get(label)
        pct = ((avg - apollo) / apollo * 100.0) if apollo else None
        rows.append({"phase": label, "n": len(vals), "min": min(vals),
                     "avg": avg, "max": max(vals), "apollo": apollo, "pct": pct})
    return rows


def build_crosscheck_rows(nominal_results, df):
    """Data-driven Apollo cross-check rows, computed from nominal_results + df
    (no hardcoded sim values, so the table can't drift from the actual run).
    Each comparable row colors by |error| vs APOLLO_ACTUALS (green <5%,
    yellow <30%, red otherwise) unless a curated note explains an expected
    deviation; special rows (parking orbit, splash position, dispersion) are
    computed directly."""
    N = nominal_results

    def err_status(v, ap):
        e = (v - ap) / max(abs(ap), 1e-9) * 100.0
        a = abs(e)
        if a < 5:
            return "#2D5A3D", "✓ within 5%"
        if a < 30:
            return "#F18F01", f"~{a:.0f}% off"
        return "#C73E1D", f"✗ {a:.0f}% off"

    rows = []  # (label, sim_str, actual_str, color, status)
    for label, key, unit, dec, note in [
        ("TLI ΔV",                "tli_dv_ms",          "m/s", 0, None),
        ("LOI burn time",         "loi_burn_time_s",    "s",   0, None),
        ("Descent fuel margin",   "fuel_margin_s",      "s",   0,
         "nominal = anomaly-free reserve; Apollo's 25 s was its as-flown "
         "boulder-field margin (now in the MC tail — see histogram)"),
        ("TEI ΔV",                "tei_dv_ms",          "m/s", 0, None),
        ("TEI burn time",         "tei_burn_time_s",    "s",   0, None),
        ("Entry FPA",             "fpa_at_entry_deg",   "°",   2, None),
        ("Entry interface speed", "entry_speed_ms",     "m/s", 0, None),
        ("Peak entry g-load",     "max_g",              "g",   1,
         "guided entry floors at ~8.6 g for the 2,780 km design range "
         "(constant-bank PC); matching Apollo's 6.5 g needs a range-aware "
         "drag reference (HUNTEST) — documented future work"),
        ("Mission duration",      "mission_duration_d", "d",   2, None),
    ]:
        if not isinstance(N.get(key), (int, float)):
            continue
        v = N[key]
        ap = APOLLO_ACTUALS.get(key)
        if ap is None:
            color, status, ap_s = "#6e6e73", "—", "—"
        elif note is not None:
            color, status, ap_s = "#F18F01", note, _fmtnum(ap, unit, dec)
        else:
            color, status = err_status(v, ap)
            ap_s = _fmtnum(ap, unit, dec)
        rows.append((label, _fmtnum(v, unit, dec), ap_s, color, status))

    # Parking orbit (two keys) — placed just after TLI ΔV
    pe = N.get("launch_parking_perigee_km")
    apo = N.get("launch_parking_apogee_km")
    if isinstance(pe, (int, float)) and isinstance(apo, (int, float)):
        if abs(apo - pe) < 50:
            color, status = "#2D5A3D", "✓ circular (IGM)"
        else:
            color, status = "#C73E1D", "✗ eccentric"
        rows.insert(1, ("Parking orbit (pre-TLI)", f"{pe:.0f} × {apo:.0f} km",
                        "184 × 186 km", color, status))

    # Splash position (nominal) vs Apollo actual — scored by ground distance,
    # axis by axis (percent error is meaningless for coordinates).
    if isinstance(N.get("splash_lat"), (int, float)):
        import math
        def _coord_status(off_km, residual_note):
            if off_km < 100:
                return "#2D5A3D", f"✓ within {off_km:.0f} km"
            if off_km < 2500:
                return "#F18F01", f"~{off_km:.0f} km — {residual_note}"
            return "#C73E1D", f"✗ {off_km:.0f} km off"
        _dlat_km = abs(N["splash_lat"] - APOLLO_ACTUALS["splash_lat"]) * 111.0
        _dlon = abs(((N["splash_lon"] - APOLLO_ACTUALS["splash_lon"]) + 180.0)
                    % 360.0 - 180.0)
        _dlon_km = _dlon * 111.0 * math.cos(math.radians(N["splash_lat"]))
        c, s = _coord_status(_dlat_km, "")
        rows.append(("Splash latitude (nominal)", f"{N['splash_lat']:.1f}°",
                     f"{APOLLO_ACTUALS['splash_lat']:.1f}°", c,
                     s.rstrip(" —")))
        c, s = _coord_status(_dlon_km, "transfer-plane geometry residual "
                             "(timing matches Apollo to ~0.02 d)")
        rows.append(("Splash longitude (nominal)", f"{N['splash_lon']:.1f}°",
                     f"{APOLLO_ACTUALS['splash_lon']:.1f}°", c, s))

    # MC guidance accuracy vs the aimed per-opportunity recovery zone (the
    # metric Apollo's own ~3 km splash miss was measured against). Scatter
    # "around the nominal" is no longer meaningful: rev-slipped returns land
    # in their own displaced contingency zones by design.
    if "splash_miss_km" in df.columns and len(df["splash_miss_km"].dropna()):
        med = float(df["splash_miss_km"].median())
        mx = float(df["splash_miss_km"].max())
        rows.append(("Splash miss vs aimed recovery zone (MC)",
                     f"~{med:.1f} km median (max {mx:.1f})",
                     "~3 km (as flown)",
                     "#2D5A3D" if med < 5 else "#F18F01",
                     "✓ guided entry, per-opportunity zones" if med < 5
                     else "guided entry, per-opportunity zones"))

    # Nominal miss from its recovery zone
    if isinstance(N.get("splash_miss_km"), (int, float)):
        _nm = N["splash_miss_km"]
        rows.append(("Abs. splash miss from recovery zone",
                     f"~{_nm:.0f} km (nominal)", "~3 km (as flown)",
                     "#2D5A3D" if _nm < 5 else "#F18F01",
                     "✓ guided entry steers to the zone" if _nm < 5
                     else "guided entry steers to the zone"))

    html = ""
    for i, (label, sim, act, color, status) in enumerate(rows):
        bg = ' style="background:#fafafa"' if i % 2 else ''
        html += (f'<tr{bg}><td style="padding:6px">{label}</td>'
                 f'<td style="text-align:right">{sim}</td>'
                 f'<td style="text-align:right">{act}</td>'
                 f'<td style="color:{color}">{status}</td></tr>\n')
    return html


def known_limitations(nominal_results, df):
    """Single source of truth for the Known Limitations, with key numbers
    pulled from nominal_results + df so the summary and the dashboard stay
    consistent with the actual run. Returns a list of (title, body) in plain
    text (renderers add their own formatting)."""
    N = nominal_results

    def g(k):
        v = N.get(k)
        return v if isinstance(v, (int, float)) else float("nan")

    tei = g("tei_dv_ms")
    pg = g("max_g")
    absmiss = g("splash_miss_km")
    # Guidance accuracy = miss vs the aimed per-opportunity recovery zone
    # (splash_miss_km). The legacy splash_dispersion_km column (scatter about
    # the nominal point) is no longer written and is NOT this metric, so fall
    # back to splash_miss_km rather than rendering NaN.
    if "splash_miss_km" in df.columns and len(df["splash_miss_km"].dropna()):
        disp = float(df["splash_miss_km"].median())
        disp_max = float(df["splash_miss_km"].max())
    else:
        disp = float("nan")
        disp_max = float("nan")
    pe = N.get("launch_parking_perigee_km")
    apo = N.get("launch_parking_apogee_km")
    park = (f"~{pe:.0f} × {apo:.0f} km"
            if isinstance(pe, (int, float)) and isinstance(apo, (int, float))
            else "~185 × 185 km")
    tlmcc_dv = (float(df["mcc_dv_ms"].median())
                if "mcc_dv_ms" in df.columns and len(df["mcc_dv_ms"].dropna())
                else float("nan"))
    tlmcc_n = (float(df["mcc_n_burns"].median())
               if "mcc_n_burns" in df.columns and len(df["mcc_n_burns"].dropna())
               else float("nan"))
    temcc_dv = (float(df["mcc_total_dv_ms"].median())
                if "mcc_total_dv_ms" in df.columns
                and len(df["mcc_total_dv_ms"].dropna())
                else float("nan"))

    return [
        ("Saturn V ascent — reliability & guidance are estimates",
         "The ascent and TLI ARCHITECTURE is no longer a limitation: the "
         "full stack is physically flown (S-IC/S-II/S-IVB, IGM linear-tangent "
         f"steering to a {park} parking orbit vs Apollo's ~184 × 186 km), and "
         "TLI ignites from each trial's own launched parking orbit with an "
         "IGM-style steered cutoff (ENABLE_INTEGRATED_TLI + "
         "ENABLE_LAUNCH_CONTINUITY, both default ON) — the pad-to-splashdown "
         "trajectory is one continuous flown path. What remains a limitation "
         "is the INPUTS: the engine-out probabilities driving the "
         "launch-failure mode are sourced estimates — F-1 ~98.5% per engine "
         "(conservative; no F-1 ever failed in flight), J-2 ~99%, S-IVB "
         "ignition ~99.5% (informed by the Apollo 6 failures) — and the "
         "ascent steering is a stand-in whose solved launch window (72.48°) "
         "sits ~0.4° from Apollo's as-flown 72.058° (the un-yawed "
         "Earth-rotation plane bias in the gravity-turn model)."),
        ("Navigation is open-loop, not ground-tracked",
         "The deep-space solves (TEI burn vector, midcourse targets) are "
         "computed once from the nominal trajectory and flown open-loop, "
         "rather than continuously re-solved from tracking data as Apollo's "
         "ground network did. Consequences visible in this run: TEI commits "
         "with larger corridor residuals than ground tracking would allow, so "
         f"the trans-Earth MCC chain trims ~{temcc_dv:.0f} m/s median vs "
         "Apollo 11's single 1.5 m/s; and when rev-1 is out of corridor the "
         "TEI slips to a later opportunity, inflating the rendezvous→TEI leg "
         "beyond Apollo's 10.9 h. (The solved nominal TEI ΔV ~"
         f"{tei:.0f} m/s vs Apollo's 1,008 reflects the sim's lunar-orbit "
         "geometry, not a propellant error.)"),
        ("Rendezvous burns are lumped, not individually flown",
         "After LM ascent the rendezvous is modeled as a propellant-budget "
         "check (the LM-vs-CSM plane angle charged as plane-change ΔV plus a "
         "sourced ~30 m/s coelliptic budget against APS + ~60 m/s RCS) rather "
         "than flying the individual Apollo rendezvous burns (CSI / CDH / "
         "terminal phase); phasing- and altitude-matching dispersions are not "
         "modeled. (Docking failure itself is sourced — ~0.95%/docking from "
         "2 capture anomalies in ~21 program dockings — and applied at both "
         "docking events.)"),
        ("Surface operations",
         "The 21.6-hour surface stay now carries three SOURCED failure modes "
         "(ENABLE_DESCENT_FAILURE_MODES): an unrecoverable LM "
         "electrical/switchgear fault (~0.85%/mission, anchored to Apollo "
         "11's own snapped ascent-engine arming circuit breaker — closed with "
         "a felt-tip pen — as 1 serious event in 6 landings × ~5% "
         "unrecoverable), a touchdown tip-over beyond the ~12° stability "
         "limit (~0.5%/landing; Apollo 15 landed at ~11°), and a terminal "
         "EVA suit/PLSS failure (~0.1%/mission, from 0 failures in 28 program "
         "man-EVAs with the OPS backup never used). All three are LM-crew "
         "exposure. Still not modeled: thermal/power dispersions during the "
         "stay, cumulative dust degradation, and per-astronaut EVA workload "
         "differences."),
        ("Service-module systems failure (the Apollo 13 mode)",
         "A catastrophic CSM service-module systems failure (cryo-tank / fuel "
         "cell, the Apollo 13 class) is drawn per mission at "
         "PROB_SM_CATASTROPHIC = 1/15 — the SOURCED rate of 1 such event in "
         "15 crewed CSM flights — and strikes at a uniformly-drawn fraction "
         "of the mission timeline (cryo/fuel-cell duty is roughly "
         "continuous). It is the single largest failure family in the run. "
         "The CONSEQUENCE depends on where it lands: translunar or lunar-orbit "
         "events (LM still attached) fall to the demonstrated LM-lifeboat "
         "abort (Apollo 13 survived exactly this, ~85–90% modeled); a "
         "surface-phase event forces an emergency liftoff to a failing CSM "
         "(~40%); a trans-Earth event is post-LM-jettison with no lifeboat "
         "(~10%, CM batteries last hours not days). The three RECOVERABLE "
         "Apollo SM anomalies (Apollo 15 SPS switch, Apollo 16 TVC, Skylab 3 "
         "RCS quad leaks) are not separately modeled — only the catastrophic "
         "class. Per-mission timing within a phase is uniform, not "
         "duty-cycle-weighted."),
        ("Midcourse corrections use a simplified basis",
         "The outbound MCC chain (Apollo 11's MCC-1..4 schedule, B-plane "
         "targeted) corrects only in an along-track + cross-track basis — the "
         "less-efficient radial component is omitted — and both the ~10 km "
         "B-plane deadband and the ~0.15 m/s per-burn execution residual are "
         "estimates, not sourced dispersions. (The chain itself, with its "
         "closed-loop MCC-4b perilune trim, is faithful; these are the "
         "remaining modeling approximations.)"),
        ("Moon model & lunar gravity",
         "The default configuration (ENABLE_REAL_EPHEMERIS=True) uses a real "
         "July-1969 lunar ephemeris — a truncated Meeus series validated to "
         "<1 arcsec against the published worked example — with Earth rotation "
         "anchored to the real launch-epoch GMST, so the return plane and "
         "splashdown reflect the true sky. Lunar gravity is the real GRAIL "
         "GRGM1200A field truncated to degree 8 (ENABLE_LUNAR_SH_FIELD), so "
         "the parking orbit evolves physically (~5–20 km/day, the documented "
         "Apollo behavior). Honest residuals: the Moon-fixed frame is a "
         "synchronous-lock approximation (pole ~6.7° from the true spin axis; "
         "~7° optical librations ignored) — fine for the zonal-driven orbit "
         "evolution modeled, not for localized mascon dynamics, which live at "
         "degree ≥50 and are therefore still represented by a calibrated "
         "landing-dispersion proxy rather than integrated gravity. Setting "
         "the flags False reproduces the legacy idealized model (circular "
         "Moon at 384,400 km, fixed 28.4° inclination, degree-2 gravity)."),
        ("Entry & splashdown",
         "Entry flies CLOSED-LOOP guidance by default "
         "(ENABLE_SKIP_ENTRY_GUIDANCE=True): a g-aware numerical "
         "predictor-corrector picks the bank angle from full trajectory "
         "predictions (miss scored with a penalty above 7 g, hard guard at "
         "9.5 g — under Apollo's 10 g guidance limit and the 12 g structural "
         "bound), with crossrange managed by deadband bank reversals. The "
         f"nominal lands ~{absmiss:.0f} km from its recovery zone at peak "
         f"~{pg:.1f} g; the Monte Carlo guidance miss is ~{disp:.0f} km "
         "median. RECOVERY TARGETING follows RTCC practice: Apollo "
         "pre-planned a recovery zone for EVERY TEI opportunity, so each "
         "trial's zone is placed at the calibrated ~2,784 km short-corridor "
         "range along the TRIAL'S OWN entry ground track — a rev-slipped "
         "return (trans-Earth timing shifts the entry interface by up to "
         "~110° of longitude) aims at its own zone exactly as the recovery "
         "force would have repositioned. splash_miss_km is therefore true "
         f"guidance accuracy vs the zone aimed for (median ~{disp:.1f} km, "
         f"fleet max ~{disp_max:.0f} km from a rare shallow-tail overfly; "
         "Apollo 11 splashed ~3 km from its aim point), while "
         "recovery_zone_displacement_km records the operational cost of a "
         "slip (~0–200 km on-time, thousands of km for slipped revs). "
         f"Honest residuals: nominal peak g ~{pg:.1f} vs Apollo 11's "
         "as-flown 6.5 g — the short dispersion-robust profile trades g for "
         "accuracy; closing it needs an FPA-indexed family of HUNTEST-style "
         "reference profiles with closed-loop drag tracking (a 6-variant "
         "corpus — online predictor-corrector and an offline open-loop "
         "profile — left it documented future work). Shallow entry-FPA tails "
         "(~+2σ of delivery) can overfly yet survive. The recovery REGION is "
         "the western Pacific near (13.5°N, 146°E) — latitude matching "
         "Apollo 11's 13.3°N to ~0.2°, with the ~13° longitude offset from "
         "Apollo's 169°W a residual of the mission timeline (~8.18 d vs "
         "8.16 d) and the sim's TLI/TEI plane geometry, not a guidance "
         "error."),
    ]


def generate_summary(df, nominal_results, outdir):
    """Write summary.txt with key statistics and known limitations."""
    n = len(df)
    # Headline = landed on the Moon AND all three returned alive (mission_success);
    # `full_success` is the stricter flawless-mission flag, reported as a subset.
    succ = df.get("mission_success", df.get("full_success", pd.Series([]))).fillna(False).astype(bool)
    flawless = df.get("full_success", pd.Series([])).fillna(False).astype(bool)

    lines = []
    lines.append("=" * 70)
    lines.append("APOLLO 11 PHYSICS-INTEGRATED SIMULATION — MONTE CARLO RESULTS")
    lines.append("=" * 70)
    lines.append(f"Total trials: {n}")
    lines.append(f"Mission success (landed + crew returned alive): {succ.sum()}/{n} "
                  f"({100*succ.sum()/max(1,n):.1f}%)")
    lines.append(f"  of which flawless (no in-flight anomaly):     {flawless.sum()}/{n} "
                  f"({100*flawless.sum()/max(1,n):.1f}%)")
    lines.append("")
    lines.append("PHASE SURVIVAL (cumulative — % reaching each phase)")
    lines.append("-" * 70)
    phase_definitions = [
        ("Launch (Saturn V → parking orbit)", "launch_success",       lambda d: d["launch_success"].fillna(False).astype(bool)),
        ("Captured into lunar SOI (post-MCC)", "periapsis_alt_km",    lambda d: d["launch_success"].fillna(False).astype(bool) & (d.get("mission_failure", pd.Series([None]*len(d))) != "missed_lunar_soi")),
        ("LOI + powered descent landing",      "descent_success",     lambda d: d["descent_success"].fillna(False).astype(bool)),
        ("Surface stay + LM ascent",           "ascent_success",      lambda d: d["ascent_success"].fillna(False).astype(bool)),
        ("Rendezvous + TEI + reach entry IF",  "reached_entry",       lambda d: d["reached_entry"].fillna(False).astype(bool)),
        ("Survived atmospheric entry",         "entry_success",       lambda d: d["entry_success"].fillna(False).astype(bool)),
        ("MISSION SUCCESS (landed + returned)", "mission_success",     lambda d: d["mission_success"].fillna(False).astype(bool)),
    ]
    for name, col, fn in phase_definitions:
        if col not in df.columns:
            continue
        ok = int(fn(df).sum())
        lines.append(f"  {name:<40s} {ok:>5d}/{n}  ({100*ok/n:5.1f}%)")
    lines.append("")
    lines.append("FAILURE MODE BREAKDOWN — what went wrong in the {} failed missions".format(n - int(succ.sum())))
    lines.append("(trials where the crew landed AND all returned alive are successes, not failures,")
    lines.append(" even if a recovered anomaly occurred — they are excluded from the counts below)")
    lines.append("-" * 70)
    if "mission_failure" in df.columns:
        # Count only TRUE failures: a recorded mission_failure on a trial that is
        # nonetheless a mission_success (landed + all returned) is a recovered
        # anomaly, not a mission failure, so it is excluded here.
        fc = df.loc[~succ, "mission_failure"].value_counts(dropna=False)
        for reason, count in fc.items():
            if pd.isna(reason):
                continue
            pct = 100 * count / n
            label_explanation = FAILURE_EXPLANATIONS.get(reason, (reason, ""))
            label, expl = label_explanation
            lines.append(f"  {count:>4d} ({pct:5.1f}%)  {label}")
            if expl:
                # word-wrap explanation
                import textwrap
                wrapped = textwrap.wrap(expl, width=64)
                for w in wrapped:
                    lines.append(f"             {w}")
            lines.append("")

    lines.append("=" * 70)
    lines.append("NOMINAL TRAJECTORY vs APOLLO 11 ACTUALS")
    lines.append("=" * 70)
    lines.append(f"{'METRIC':<36s} {'NOMINAL':>14s} {'APOLLO 11':>14s} {'ERROR':>10s}")
    metric_list = [
        ("TLI ΔV (m/s)",                "tli_dv_ms"),
        ("Approach periapsis (km)",      "periapsis_alt_km"),
        ("LOI burn time (s)",            "loi_burn_time_s"),
        ("Descent fuel margin (s)",      "fuel_margin_s"),
        ("Touchdown V_vertical (m/s)",   "touchdown_v_radial_ms"),
        ("Touchdown V_horizontal (m/s)", "touchdown_v_horiz_ms"),
        ("LM ascent altitude (km)",      "ascent_alt_km"),
        ("APS prop remaining (kg)",      "aps_prop_remaining_kg"),
        ("TEI ΔV (m/s)",                 "tei_dv_ms"),
        ("TEI burn time (s)",            "tei_burn_time_s"),
        ("Entry FPA (deg)",              "fpa_at_entry_deg"),
        ("Entry speed (m/s)",            "entry_speed_ms"),
        ("Peak entry g",                 "max_g"),
    ]
    # Metrics whose NOMINAL is intentionally not comparable to Apollo's as-flown
    # actual: show the value but an explanatory note instead of a misleading error.
    import textwrap
    crosscheck_notes = {
        "fuel_margin_s": ("nominal = anomaly-free reserve (~Apollo's planned ~2-min "
                          "hover budget); Apollo's ~25 s was its as-flown boulder-field "
                          "margin, which here lives in the MC tail (see fuel-margin histogram)"),
    }
    for desc, key in metric_list:
        if key not in nominal_results:
            continue
        v = nominal_results[key]
        ap = APOLLO_ACTUALS.get(key, "—")
        note = crosscheck_notes.get(key)
        if note and isinstance(v, (int, float)) and isinstance(ap, (int, float)):
            lines.append(f"  {desc:<34s} {v:>14.2f} {ap:>14.2f} {'(see note)':>10s}")
            for w in textwrap.wrap(note, 64):
                lines.append(f"             {w}")
        elif isinstance(v, (int, float)) and isinstance(ap, (int, float)):
            err = ((v - ap) / max(abs(ap), 1)) * 100
            lines.append(f"  {desc:<34s} {v:>14.2f} {ap:>14.2f} {err:>+8.1f}%")
        else:
            lines.append(f"  {desc:<34s} {str(v):>14s} {str(ap):>14s}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("MONTE CARLO DISPERSION STATISTICS (1-σ where applicable)")
    lines.append("=" * 70)
    for desc, col in [
        ("Saturn V launch max-Q (Pa)",     "launch_max_q_pa"),
        ("Saturn V launch max-g",          "launch_max_g"),
        ("Parking orbit perigee (km)",     "launch_parking_perigee_km"),
        ("Parking orbit apogee (km)",      "launch_parking_apogee_km"),
        ("TLI ΔV (m/s)",                   "tli_dv_ms"),
        ("MCC ΔV (m/s)",                   "mcc_dv_ms"),
        ("LOI ΔV (m/s)",                   "loi_dv_ms"),
        ("Periapsis (km)",                 "periapsis_alt_km"),
        ("Descent fuel margin (s)",        "fuel_margin_s"),
        ("Entry FPA (deg)",                "fpa_at_entry_deg"),
        ("Peak entry g",                   "max_g"),
        ("Splashdown dispersion (km, from nominal)", "splash_dispersion_km"),
        ("Abs. splash miss from recovery target (km)", "splash_miss_km"),
    ]:
        if col in df.columns:
            v = df[col].dropna()
            if len(v) > 0:
                lines.append(f"  {desc:<32s} mean={v.mean():>9.2f} "
                              f"std={v.std():>9.2f} "
                              f"min={v.min():>9.2f} "
                              f"max={v.max():>9.2f}")

    pt_rows = phase_timing_stats(outdir)
    if pt_rows:
        lines.append("")
        lines.append("=" * 70)
        lines.append("PHASE TIMING (mission-elapsed duration across trials)")
        lines.append("=" * 70)
        lines.append(f"{'PHASE':<24s}{'MIN':>10s}{'AVG':>10s}{'MAX':>10s}"
                     f"{'APOLLO 11':>12s}{'AVG vs A11':>12s}")
        for r in pt_rows:
            pct = f"{r['pct']:+.0f}%" if r['pct'] is not None else "—"
            lines.append(f"{r['phase']:<24s}{_fmt_dur(r['min']):>10s}"
                         f"{_fmt_dur(r['avg']):>10s}{_fmt_dur(r['max']):>10s}"
                         f"{_fmt_dur(r['apollo']):>12s}{pct:>12s}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("KNOWN LIMITATIONS")
    lines.append("=" * 70)
    import textwrap
    for _i, (_title, _body) in enumerate(known_limitations(nominal_results, df), 1):
        lines.append("")
        lines.append(f"{_i}. {_title}.")
        lines.append(textwrap.fill(_body, width=70,
                                   initial_indent="   ", subsequent_indent="   "))
    lines.append("")
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ summary.txt ({len(lines)} lines)")


def generate_plots(df, nominal_traj, nominal_results, outdir):
    """Generate plots."""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    n_trials = len(df)

    # 1) Cumulative phase survival, aligned to the 10 TIMELINE PHASES (same
    #    decomposition as the Phase Timing table). Each mission_failure is
    #    mapped to the timeline phase in which it ends the mission; a trial
    #    "clears" phase k if it did not fail in phase k or any earlier phase.
    #    Bar k = fraction of trials still on-track at the END of phase k, so
    #    the final bar equals the full-success rate.
    _PHASES10 = [
        ("Launch\n→ orbit",        ("launch_s_ic", "launch_structural",
                                    "launch_s_ivb_first")),
        ("Parking\norbit + TLI",   ("launch_parking", "tli_", "transposition_docking")),
        ("Translunar\ncoast",      ("sm_failure_translunar", "missed_lunar_soi")),
        ("Lunar orbit\n(LOI→PDI)", ("sm_failure_lunar_orbit",)),
        ("Powered\ndescent",       ("descent_",)),
        ("Surface\nstay",          ("surface_", "sm_failure_surface")),
        ("Ascent\n→ orbit",        ("ascent_",)),
        ("Rendezvous\n→ TEI",      ("rendezvous_", "tei_")),
        ("Trans-earth\ncoast",     ("sm_failure_transearth", "transearth_")),
        ("Entry →\nsplashdown",    ("entry_",)),
    ]
    def _fail_phase_idx(reason):
        if reason is None or (isinstance(reason, float) and pd.isna(reason)):
            return 99  # nominal — clears every phase
        s = str(reason)
        for idx, (_lbl, prefixes) in enumerate(_PHASES10):
            if any(s.startswith(p) for p in prefixes):
                return idx
        return 99  # unmapped → treat as cleared (none expected)
    fr = df.get("mission_failure", pd.Series([None]*n_trials))
    fidx = fr.apply(_fail_phase_idx).to_numpy()
    # A trial that is a MISSION SUCCESS (landed + all returned alive) clears every
    # phase even if it recorded a recovered anomaly — so the final bar equals the
    # mission-success rate and recovered-anomaly trials do not show as drop-outs.
    _ms = df.get("mission_success", pd.Series([False]*n_trials)).fillna(False).astype(bool).to_numpy()
    fidx = np.where(_ms, 99, fidx)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    names = [p[0] for p in _PHASES10]
    counts = [int((fidx > k).sum()) for k in range(len(_PHASES10))]
    rates = [100*c/max(1, n_trials) for c in counts]
    # High-contrast categorical palette (one distinct hue per timeline phase) so
    # adjacent bars are easy to tell apart — replaces the old near-monochrome
    # blue/purple ramp.
    cmap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#393b79']
    bars = ax.bar(names, rates, color=cmap)
    ax.set_ylabel("Cumulative success rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Apollo 11 Monte Carlo: Cumulative Survival by Timeline Phase "
                 f"({n_trials} trials; final bar = mission success)")
    for b, c, r in zip(bars, counts, rates):
        ax.text(b.get_x() + b.get_width()/2, r + 1.5,
                  f"{r:.1f}%", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase_survival.png"), dpi=110)
    plt.close()
    print("  ✓ phase_survival.png")

    # 1b) Failure modes pie/bar — categorized by phase. Count only TRUE failures
    # (exclude recovered-anomaly trials that are mission successes).
    _failed_mask = ~df["mission_success"].fillna(False).astype(bool)
    if "mission_failure" in df.columns:
        fc = df.loc[_failed_mask, "mission_failure"].value_counts(dropna=True)
        if len(fc) > 0:
            # Group by phase
            # High-contrast categorical palette — one clearly distinct colour per
            # mission phase (tab10-style), so the many failure-mode bars no longer
            # blur into shades of blue and purple.
            phase_colors = {
                "launch": "#1f77b4",      # blue
                "translunar": "#ff7f0e",  # orange
                "lunar_orbit": "#2ca02c", # green
                "descent": "#d62728",     # red
                "surface": "#9467bd",     # purple
                "ascent": "#8c564b",      # brown
                "rendezvous": "#e377c2",  # pink
                "tei": "#17becf",         # cyan
                "transearth": "#bcbd22",  # olive
                "entry": "#393b79",       # navy
                "other": "#7f7f7f",       # gray
            }
            # Sort failure modes by phase, then count (chronological)
            phase_order = ["launch", "translunar", "lunar_orbit", "descent",
                            "surface", "ascent", "rendezvous", "tei",
                            "transearth", "entry", "other"]
            sorted_failures = []
            for ph in phase_order:
                for reason, count in fc.items():
                    cat, _ = categorize_failure(reason)
                    if cat == ph:
                        sorted_failures.append((reason, count, ph))
            fig, ax = plt.subplots(figsize=(12, 6))
            labels = []
            counts = []
            colors = []
            for reason, count, ph in sorted_failures:
                label_explanation = FAILURE_EXPLANATIONS.get(reason, (reason, ""))
                short_label = label_explanation[0]
                if len(short_label) > 38:
                    short_label = short_label[:35] + "..."
                labels.append(f"{short_label}\n({count} trials, {100*count/n_trials:.1f}%)")
                counts.append(count)
                colors.append(phase_colors.get(ph, "#888"))
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, counts, color=colors, edgecolor='black', alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel(f"Number of trials (out of {n_trials})")
            ax.set_title(f"Failure Mode Breakdown — what went wrong in the {n_trials - int(df['mission_success'].fillna(False).sum())} failed missions")
            # Phase legend
            from matplotlib.patches import Patch
            legend_handles = [Patch(facecolor=phase_colors[ph], label=ph.replace("_", " ").capitalize())
                                for ph in phase_order if any(pc == ph for _,_,pc in sorted_failures)]
            ax.legend(handles=legend_handles, loc='lower right', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "failure_modes.png"), dpi=110)
            plt.close()
            print("  ✓ failure_modes.png")

    # 2) Entry FPA distribution
    if "fpa_at_entry_deg" in df.columns:
        fpa = df["fpa_at_entry_deg"].dropna()
        if len(fpa) > 0:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(fpa, bins=20, color='#4ECDC4', edgecolor='black', alpha=0.8)
            ax.axvline(APOLLO_ACTUALS["fpa_at_entry_deg"], color='red',
                        linewidth=2, linestyle='--',
                        label=f"Apollo 11 actual: {APOLLO_ACTUALS['fpa_at_entry_deg']}°")
            ax.axvline(fpa.mean(), color='blue', linewidth=2,
                        label=f"Monte Carlo mean: {fpa.mean():.2f}°")
            ax.set_xlabel("Entry flight path angle (deg)")
            ax.set_ylabel("Count")
            ax.set_title(f"Entry FPA Distribution (n={len(fpa)})")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "entry_fpa.png"), dpi=110)
            plt.close()
            print("  ✓ entry_fpa.png")

    # 3) Descent fuel margin histogram
    if "fuel_margin_s" in df.columns:
        f = df["fuel_margin_s"].dropna()
        if len(f) > 0:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(f, bins=20, color='#F7B801', edgecolor='black', alpha=0.8)
            ax.axvline(APOLLO_ACTUALS["fuel_margin_s"], color='red',
                        linewidth=2, linestyle='--',
                        label=f"Apollo 11 actual: ~{APOLLO_ACTUALS['fuel_margin_s']} s")
            ax.axvline(f.mean(), color='blue', linewidth=2,
                        label=f"Monte Carlo mean: {f.mean():.1f} s")
            ax.axvline(0, color='black', linewidth=1.5,
                        label="Crash threshold")
            ax.set_xlabel("Descent fuel margin at touchdown (s)")
            ax.set_ylabel("Count")
            ax.set_title(f"Descent Fuel Margin Distribution (n={len(f)})")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "fuel_margin.png"), dpi=110)
            plt.close()
            print("  ✓ fuel_margin.png")

    # 4) Splashdown dispersion histogram (the TEI-targeting metric:
    #    spread around the nominal splashdown, not absolute miss from target)
    disp_col = "splash_dispersion_km" if "splash_dispersion_km" in df.columns else "splash_miss_km"
    if disp_col in df.columns:
        s = df[disp_col].dropna()
        if len(s) > 0:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(s, bins=20, color='#1F77B4', edgecolor='black', alpha=0.8)
            ax.axvline(s.median(), color='red', linewidth=2,
                        label=f"Median: {s.median():.0f} km")
            ax.axvline(s.mean(), color='blue', linewidth=2, linestyle='--',
                        label=f"Mean: {s.mean():.0f} km")
            if disp_col == "splash_dispersion_km":
                ax.set_xlabel("Splashdown dispersion from nominal (km)")
                ax.set_title(f"Splashdown Dispersion — TEI Targeting (n={len(s)})")
            else:
                ax.set_xlabel("Splash miss from recovery target (km)")
                ax.set_title(f"Splashdown Miss Distance (n={len(s)})")
            ax.set_ylabel("Count")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "splash_miss.png"), dpi=110)
            plt.close()
            print("  ✓ splash_miss.png")

    # 5) Nominal descent profile
    if "descent" in nominal_traj:
        ts, ys = nominal_traj["descent"]
        alts = (np.linalg.norm(ys[:3, :], axis=0) - R_MOON) / 1000  # km
        v_radial = np.array([np.dot(ys[3:6, i], ys[:3, i] / np.linalg.norm(ys[:3, i]))
                              for i in range(len(ts))])
        v_horiz = np.array([np.sqrt(max(0, np.dot(ys[3:6, i], ys[3:6, i]) - v_radial[i]**2))
                             for i in range(len(ts))])
        t_rel = ts - ts[0]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax1.plot(t_rel, alts, 'b-', linewidth=1.8, label="Altitude (km)")
        ax1.set_ylabel("Altitude (km)")
        ax1.legend(loc='upper right')
        ax1.set_title(f"Nominal Lunar Descent Profile "
                       f"(fuel margin = {nominal_results.get('fuel_margin_s', 0):.0f} s)")
        ax2.plot(t_rel, v_horiz, 'g-', linewidth=1.6, label="Horizontal velocity (m/s)")
        ax2.plot(t_rel, np.abs(v_radial), 'r-', linewidth=1.6, label="Descent rate |v_r| (m/s)")
        ax2.set_xlabel("Time from PDI (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.legend(loc='upper right')
        ax2.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "descent_profile.png"), dpi=110)
        plt.close()
        print("  ✓ descent_profile.png")

    # 6) 3D translunar trajectory
    if "translunar" in nominal_traj:
        ts, ys = nominal_traj["translunar"]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Earth
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
        Ex = R_EARTH * np.cos(u) * np.sin(v) / 1e3
        Ey = R_EARTH * np.sin(u) * np.sin(v) / 1e3
        Ez = R_EARTH * np.cos(v) / 1e3
        ax.plot_surface(Ex, Ey, Ez, color='blue', alpha=0.3, edgecolor='none')

        # Trajectory in km
        rs = ys[:3, :] / 1e3
        ax.plot(rs[0], rs[1], rs[2], 'r-', linewidth=1.5, label="Spacecraft")

        # Moon trajectory (sampled at trajectory times)
        from apollo11 import moon_state
        moon_pos = np.array([moon_state(t)[0] for t in ts]) / 1e3
        ax.plot(moon_pos[:, 0], moon_pos[:, 1], moon_pos[:, 2],
                  'k--', alpha=0.5, linewidth=1, label="Moon orbit")

        # Moon at arrival
        moon_arr = moon_pos[-1]
        ax.scatter(moon_arr[0], moon_arr[1], moon_arr[2],
                    color='gray', s=100, label="Moon at arrival")

        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        ax.set_title("Nominal Trans-Lunar Trajectory (ECI frame)")
        ax.legend(loc='upper left')
        try:
            ax.set_box_aspect([1,1,1])
        except Exception:
            pass
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "trajectory_3d.png"), dpi=110)
        plt.close()
        print("  ✓ trajectory_3d.png")


def generate_dashboard(df, nominal_results, outdir):
    """Generate dashboard.html consolidating results."""
    import base64

    def embed_image(filename):
        """Return an <img> tag with the image base64-inlined for portability."""
        path = os.path.join(outdir, filename)
        if not os.path.exists(path):
            return f'<img alt="(missing: {filename})" src="">'
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f'<img src="data:image/png;base64,{data}" alt="{filename}">'

    n = len(df)
    # Headline success = landed on the Moon AND all three returned alive.
    _ms = df.get("mission_success", df.get("full_success", pd.Series([]))).fillna(False).astype(bool)
    succ = int(_ms.sum())
    # Recovered-anomaly missions: a recorded mission_failure but still a success
    # (e.g. EVA crew transfer after a docking failure, or a survivable late SM
    # failure) — surfaced in the definition text for transparency.
    _mf = df.get("mission_failure", pd.Series([None]*n))
    n_recovered = int((_ms & _mf.notna() & (_mf.astype(str) != "")).sum())

    # Add crew-survival data if results_with_survival.csv exists
    survival_csv = os.path.join(outdir, "results_with_survival.csv")
    crew_survived = 0
    has_crew_data = False
    crew_deaths_html = ""
    crew_deaths_summary = ""
    if os.path.exists(survival_csv):
        try:
            df_s = pd.read_csv(survival_csv)
            if 'crew_survived' in df_s.columns:
                crew_survived = int(df_s['crew_survived'].sum())
                has_crew_data = True
            # Astronaut-death accounting (the crew analogue of the failure
            # breakdown): deaths/trial = 3 - n_astronauts_survived, grouped
            # by cause. 3 astronauts x n trials = the population at risk.
            if 'n_astronauts_survived' in df_s.columns:
                s = df_s[df_s['trial'] >= 0] if 'trial' in df_s.columns else df_s
                ns = len(s)
                all3 = int((s['n_astronauts_survived'] == 3).sum())
                deaths_per = (3 - s['n_astronauts_survived'])
                total_deaths = int(deaths_per.sum())
                possible = 3 * ns
                fatal_missions = int((deaths_per > 0).sum())
                # Group by PER-ASTRONAUT death cause when available (dcause_*):
                # a crewman lost on the Earth-return leg is charged to
                # crew_return_leg_loss, not the upstream mode that only killed
                # his crewmates. Falls back to the trial mission_failure if the
                # per-astronaut columns are absent (older survival files).
                _crew = [c for c in ('Armstrong', 'Aldrin', 'Collins')
                         if f'dcause_{c}' in s.columns]
                if _crew:
                    _cc = []
                    for nm in _crew:
                        dc = s.loc[~s[f'survived_{nm}'].astype(bool), f'dcause_{nm}']
                        _cc.extend([c if isinstance(c, str) and c else '(unattributed)'
                                    for c in dc])
                    by = pd.Series(_cc).value_counts()
                else:
                    cause = s['mission_failure'].fillna('(nominal mission)')
                    by = deaths_per.groupby(cause).sum()
                by = by[by > 0].sort_values(ascending=False)
                crew_deaths_summary = (
                    f"All three astronauts returned alive in <strong>{all3:,} of "
                    f"{ns:,}</strong> missions ({100*all3/ns:.1f}%). "
                    f"{fatal_missions:,} missions ({100*fatal_missions/ns:.1f}%) "
                    f"were fatal to at least one crew member. Of "
                    f"<strong>{possible:,} astronaut-missions</strong> "
                    f"(3 crew × {ns:,} trials), <strong>{total_deaths:,} deaths</strong> "
                    f"occurred ({100*total_deaths/possible:.2f}%). What killed "
                    f"astronauts, by cause:")
                rows_h = ""
                # Per-cause LETHALITY pattern: how many astronauts die of this
                # cause in an affected trial. If that count is the same in every
                # affected trial -> "always N"; otherwise "variable". Computed
                # from the per-astronaut dcause columns.
                _leth = {}
                _ntrials = {}   # trials in which this cause killed >=1 astronaut
                if _crew:
                    for c in by.index:
                        per_trial = np.zeros(len(s), dtype=int)
                        for nm in _crew:
                            per_trial += ((~s[f'survived_{nm}'].astype(bool)) &
                                          (s[f'dcause_{nm}'] == c)).to_numpy().astype(int)
                        nz = sorted(set(int(x) for x in per_trial[per_trial > 0]))
                        _leth[c] = ('variable', nz) if len(nz) != 1 else ('fixed', nz[0])
                        _ntrials[c] = int((per_trial > 0).sum())
                _LCOLOR = {1: '#2D7D46', 2: '#E08E0B', 3: '#C73E1D', 'variable': '#6A4C93'}
                def _leth_badge(c):
                    kind, val = _leth.get(c, ('variable', []))
                    if kind == 'fixed':
                        col = _LCOLOR.get(val, '#6A4C93')
                        txt = f'always {val}'
                    else:
                        col = _LCOLOR['variable']
                        rng_txt = f"{min(val)}–{max(val)}" if val else "?"
                        txt = f'variable ({rng_txt})'
                    return (f'<td style="text-align:center"><span style="background:{col};'
                            f'color:#fff;border-radius:4px;padding:2px 7px;font-size:11px;'
                            f'white-space:nowrap">{txt}</span></td>')
                for c, d in by.items():
                    lbl, expl = FAILURE_EXPLANATIONS.get(str(c), (str(c), ""))
                    _nt = _ntrials.get(c, 0)
                    rows_h += (
                        f'<tr><td>{int(d)}<br><span style="color:#6e6e73;'
                        f'font-size:11px">{100*d/total_deaths:.1f}% of deaths</span></td>'
                        f'<td style="text-align:center">{_nt}<br>'
                        f'<span style="color:#6e6e73;font-size:11px">'
                        f'{100*_nt/ns:.1f}% of missions</span></td>'
                        f'{_leth_badge(c)}'
                        f'<td><strong>{lbl}</strong></td>'
                        f'<td style="font-size:13px;color:#3c3c43">{expl}</td></tr>\n')
                _legend = (
                    '<div style="font-size:12px;color:#3c3c43;margin:6px 0">'
                    'Astronauts killed per affected trial: '
                    '<span style="background:#2D7D46;color:#fff;border-radius:4px;padding:1px 6px">always 1</span> '
                    '(one moonwalker — EVA suit) &nbsp; '
                    '<span style="background:#E08E0B;color:#fff;border-radius:4px;padding:1px 6px">always 2</span> '
                    '(both moonwalkers; Collins survives) &nbsp; '
                    '<span style="background:#C73E1D;color:#fff;border-radius:4px;padding:1px 6px">always 3</span> '
                    '(whole stack — shared vehicle) &nbsp; '
                    '<span style="background:#6A4C93;color:#fff;border-radius:4px;padding:1px 6px">variable</span> '
                    '(depends on the return-leg draw).</div>')
                crew_deaths_html = (
                    _legend +
                    '<table><tr><th>Astronaut deaths</th><th>Trials</th>'
                    '<th>Per trial</th><th>Cause</th><th>Explanation</th></tr>\n'
                    + rows_h + '</table>')
        except Exception:
            pass

    nominal_table = ""
    for desc, key in [
        ("TLI ΔV (m/s)",                "tli_dv_ms"),
        ("Approach periapsis (km)",      "periapsis_alt_km"),
        ("LOI burn time (s)",            "loi_burn_time_s"),
        ("Descent fuel margin (s)",      "fuel_margin_s"),
        ("Touchdown V_vertical (m/s)",   "touchdown_v_radial_ms"),
        ("Touchdown V_horizontal (m/s)", "touchdown_v_horiz_ms"),
        ("LM ascent altitude (km)",      "ascent_alt_km"),
        ("APS prop remaining (kg)",      "aps_prop_remaining_kg"),
        ("TEI ΔV (m/s)",                 "tei_dv_ms"),
        ("TEI burn time (s)",            "tei_burn_time_s"),
        ("Entry FPA (deg)",              "fpa_at_entry_deg"),
        ("Entry speed (m/s)",            "entry_speed_ms"),
        ("Peak entry g",                 "max_g"),
        ("Abs. splash miss from target (km)", "splash_miss_km"),
    ]:
        if key not in nominal_results:
            continue
        v = nominal_results[key]
        ap = APOLLO_ACTUALS.get(key, "—")
        if isinstance(v, (int, float)) and isinstance(ap, (int, float)):
            err = ((v - ap) / max(abs(ap), 1)) * 100
            nominal_table += (f"<tr><td>{desc}</td><td>{v:.2f}</td>"
                                f"<td>{ap:.2f}</td><td>{err:+.1f}%</td></tr>\n")
        else:
            nominal_table += (f"<tr><td>{desc}</td><td>{v}</td>"
                                f"<td>{ap}</td><td>—</td></tr>\n")

    # Build failure analysis HTML
    failure_table = ""
    if "mission_failure" in df.columns:
        # TRUE failures only: a recorded mission_failure on a mission_success
        # trial is a recovered anomaly (landed + all returned), not a failure.
        fc = df.loc[~df["mission_success"].fillna(False).astype(bool),
                    "mission_failure"].value_counts(dropna=True)
        phase_order = ["launch", "translunar", "lunar_orbit", "descent",
                        "surface", "ascent", "rendezvous", "tei",
                        "transearth", "entry", "other"]
        # Group failures by phase
        for ph in phase_order:
            phase_failures = []
            for reason, count in fc.items():
                cat, _ = categorize_failure(reason)
                if cat == ph:
                    phase_failures.append((reason, count))
            if not phase_failures:
                continue
            ph_total = sum(c for _, c in phase_failures)
            failure_table += f'<tr class="ph-header"><td colspan="3"><strong>{ph.upper().replace("_", " ")} PHASE — {ph_total} trials ({100*ph_total/n:.1f}%)</strong></td></tr>\n'
            for reason, count in phase_failures:
                label_expl = FAILURE_EXPLANATIONS.get(reason, (reason, ""))
                label, expl = label_expl
                failure_table += f'<tr><td>{count}<br><span style="color:#6e6e73;font-size:11px">{100*count/n:.1f}%</span></td><td><strong>{label}</strong></td><td style="font-size:13px;color:#3c3c43">{expl}</td></tr>\n'

    # Per-trial runtime: mean compute time per trial (present when the run
    # recorded trial_time_s). Each parallel worker runs one trial at a time, so
    # this is the per-trial cost, not the amortized wall time.
    _trial_times = df.get("trial_time_s", pd.Series([], dtype=float)).dropna()
    mean_trial_time = float(_trial_times.mean()) if len(_trial_times) else None
    trial_time_sub = (
        f'<div style="font-size:12px;color:#6e6e73;margin-top:4px">'
        f'~{mean_trial_time:.1f} s/trial avg</div>'
        if mean_trial_time is not None else "")

    # Launch stats
    launch_max_q = df.get("launch_max_q_pa", pd.Series([])).dropna()
    launch_max_g = df.get("launch_max_g", pd.Series([])).dropna()

    # Data-driven Apollo cross-check rows + Known Limitations (shared sources,
    # computed from THIS run, so neither can drift from the actual results).
    crosscheck_rows = build_crosscheck_rows(nominal_results, df)

    # Phase-timing table: per-phase mission-duration min/avg/max across trials,
    # average compared to Apollo 11's actual timeline.
    pt_rows = phase_timing_stats(outdir)
    phase_timing_html = ""
    for r in pt_rows:
        if r["pct"] is None:
            pct_cell, color = "—", "#6e6e73"
        else:
            a = abs(r["pct"])
            color = "#2D5A3D" if a < 5 else ("#F18F01" if a < 30 else "#C73E1D")
            pct_cell = f"{r['pct']:+.0f}%"
        phase_timing_html += (
            f'<tr><td style="padding:6px">{r["phase"]}</td>'
            f'<td style="text-align:right">{_fmt_dur(r["min"])}</td>'
            f'<td style="text-align:right"><strong>{_fmt_dur(r["avg"])}</strong></td>'
            f'<td style="text-align:right">{_fmt_dur(r["max"])}</td>'
            f'<td style="text-align:right">{_fmt_dur(r["apollo"])}</td>'
            f'<td style="text-align:right;color:{color}">{pct_cell}</td></tr>\n')
    phase_timing_section = (
        f'''<h2>Phase Timing vs Apollo 11</h2>
<p style="color:#3c3c43">Mission-elapsed duration of each phase across all trials
(min / average / max), with the average compared to Apollo 11's actual flight
timeline. Per-trial phase timing is saved under <code>trials/</code>.</p>
<table style="border-collapse:collapse;width:100%;background:white;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
<thead><tr style="background:#f0f0f2;text-align:right">
<th style="padding:8px;text-align:left">Mission phase</th><th style="padding:8px">Min</th>
<th style="padding:8px">Avg</th><th style="padding:8px">Max</th>
<th style="padding:8px">Apollo&nbsp;11</th><th style="padding:8px">Avg vs A11</th></tr></thead>
<tbody>
{phase_timing_html}</tbody></table>''' if pt_rows else "")

    limitations_html = "\n".join(
        f'<div class="caveat"><strong>{_i}. {_title}.</strong> {_body}</div>'
        for _i, (_title, _body) in enumerate(known_limitations(nominal_results, df), 1)
    )

    # Source-file reference for the Files section: count lines live (no
    # drift) and avoid a broken in-dir link, since apollo11.py lives in the
    # project root, not the run directory.
    try:
        _src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "apollo11.py")
        with open(_src_path) as _sf:
            src_lines = sum(1 for _ in _sf)
    except Exception:
        src_lines = 0
    src_line = "<code>apollo11.py</code>"

    # Median guidance miss (vs aimed zone) for the splash-figure caption.
    disp = (float(df["splash_miss_km"].median())
            if "splash_miss_km" in df.columns and len(df["splash_miss_km"].dropna())
            else float("nan"))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Apollo 11 Physics Simulation — Results</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
                     sans-serif; margin: 0; padding: 24px;
                     background: #f5f5f7; color: #1d1d1f; }}
h1 {{ font-weight: 600; margin-bottom: 4px; font-size: 28px; }}
h2 {{ font-weight: 500; margin-top: 32px; border-bottom: 2px solid #d2d2d7;
       padding-bottom: 8px; font-size: 20px; }}
.summary-card {{ background: white; padding: 18px 24px; border-radius: 12px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin: 16px 0; }}
.stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
           margin: 16px 0; }}
.stat {{ background: white; padding: 14px; border-radius: 10px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.stat .label {{ color: #6e6e73; font-size: 12px; text-transform: uppercase;
                  letter-spacing: 0.5px; }}
.stat .value {{ font-size: 26px; font-weight: 600; margin-top: 4px; }}
table {{ border-collapse: collapse; width: 100%; background: white;
          border-radius: 10px; overflow: hidden;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
th, td {{ padding: 9px 14px; text-align: left;
            border-bottom: 1px solid #e8e8ed; vertical-align: top; }}
th {{ background: #f9f9fb; font-weight: 600; color: #6e6e73; font-size: 12px;
       text-transform: uppercase; letter-spacing: 0.4px; }}
tr:last-child td {{ border-bottom: none; }}
tr.ph-header td {{ background: #f0f0f4; color: #1d1d1f; font-size: 12px;
                    letter-spacing: 0.5px; }}
.plot {{ background: white; padding: 16px; border-radius: 10px; margin: 12px 0;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06); text-align: center; }}
.plot img {{ max-width: 100%; height: auto; }}
.caveat {{ background: #fffbeb; border-left: 4px solid #f59e0b;
            padding: 14px 18px; border-radius: 6px; margin: 14px 0;
            font-size: 14px; line-height: 1.6; }}
.context {{ background: #eff6ff; border-left: 4px solid #3b82f6;
            padding: 14px 18px; border-radius: 6px; margin: 14px 0;
            font-size: 14px; line-height: 1.6; }}
code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 3px;
          font-size: 13px; }}
</style></head>
<body>
<h1>Apollo 11 Physics-Integrated Monte Carlo</h1>
<p style="color:#6e6e73">{n} simulated Apollo 11 missions using 1969-era hardware
parameters and dispersions. Each trial runs full ODE integration of every
mission phase from Kennedy Space Center liftoff through Pacific splashdown.</p>

<div class="summary-card">
<div class="stats">
  <div class="stat"><div class="label">Trials</div><div class="value">{n}</div>{trial_time_sub}</div>
  <div class="stat"><div class="label">Mission Success</div><div class="value">{100*succ/max(1,n):.1f}%</div></div>
  <div class="stat"><div class="label">Full Crew Survived</div><div class="value">{100*crew_survived/max(1,n):.1f}%</div></div>
</div>
</div>

<div class="context">
<strong>What this simulation does.</strong> Each trial samples engine-out
events (F-1, J-2 reliability), engine performance dispersions (ISP, thrust),
insertion dispersions, and aerodynamic uncertainties, then runs the full
10-phase Apollo 11 mission with real-physics ODE integration (no Monte Carlo
shortcuts). When a mission fails, the cause is recorded as a
<code>mission_failure</code> value. The Failure Analysis below explains
each cause in plain language, and the Crew Survival section models the
Apollo abort architecture that saved the crew even when missions failed.
</div>

<h2>Crew Survival vs Mission Success</h2>
<div class="context">
<strong>Mission success rate ≠ crew survival rate.</strong> A <em>mission
success</em> here follows NASA's actual objective — <em>land the crew on the
Moon and return all three to Earth alive</em>. A trial counts as a success if it
achieved a Moon landing <strong>and</strong> all three astronauts splashed down
alive — <em>even if an in-flight anomaly occurred that the crew recovered from</em>
(for example a docking-mechanism failure resolved by a contingency EVA crew
transfer, or a service-module failure that struck late enough on the trans-Earth
coast for the crew to ride the remaining consumables home). <strong>{n_recovered}</strong>
missions succeeded despite such a recovered anomaly. A trial is a <em>failure</em>
only if the landing was never achieved or at least one crew member died. Apollo's
abort architecture (Launch Escape System, free-return trajectory, LM abort-stage
procedure, SPS contingency burns) was designed to save the crew even when the
landing could not be reached, so the crew comes home from many missions that
never landed — crew survival therefore exceeds mission success, and the gap
between them is essentially the set of missions aborted, with the crew recovered,
before a landing was achieved. {crew_deaths_summary}
</div>
<div class="plot">{embed_image("mission_vs_crew.png")}</div>
<div class="plot">{embed_image("crew_survival.png")}</div>
{crew_deaths_html}
<p style="color:#6e6e73;font-size:13px">Survival probabilities below are
estimates based on Apollo abort manuals, the Launch Escape System's
~98% reliability, free-return trajectory design (which saved Apollo 13's
crew), and the LM "abort stage" capability that allowed the ascent stage
to fire mid-descent. See <a href="crew_survival.txt">crew_survival.txt</a>
for the detailed mapping of each failure mode to its recovery path.</p>

<h2>Cumulative Phase Survival</h2>
<div class="context" style="font-size:13px">One bar per mission timeline phase
(the same ten phases as the Phase Timing table below). Each bar is the fraction
of trials still on-track at the END of that phase — i.e. whose mission was not
lost in it or any earlier phase — so the bars step down monotonically and the
final bar equals the mission success rate. (A recovered anomaly does not lose the
mission, so such a trial clears every phase.)</div>
<div class="plot">{embed_image("phase_survival.png")}</div>

<h2>Failure Analysis — what went wrong in the {n - succ} failed missions</h2>
<div class="plot">{embed_image("failure_modes.png")}</div>
<table>
<tr><th>Count</th><th>Failure Mode</th><th>Explanation</th></tr>
{failure_table}
</table>

<h2>Nominal Trajectory vs Apollo 11 Actuals</h2>
<table>
<tr><th>Metric</th><th>Nominal (mine)</th><th>Apollo 11 actual</th><th>Error</th></tr>
{nominal_table}
</table>

{phase_timing_section}

<h2>Detail plots</h2>
<div class="plot">{embed_image("descent_profile.png")}</div>
<div class="plot">{embed_image("trajectory_3d.png")}</div>
<div class="plot">{embed_image("fuel_margin.png")}</div>
<div class="plot">{embed_image("entry_fpa.png")}</div>
<div class="plot">{embed_image("splash_miss.png")}<div style="font-size:12px;color:#6e6e73;margin-top:4px">Guidance accuracy: distance from each trial's aimed per-opportunity recovery zone (median ~{disp:.1f} km; the rare tail is shallow-FPA overfly, not a targeting error).</div></div>

<h2>Apollo 11 Cross-Check</h2>
<div class="context">
Simulated nominal values compared with documented Apollo 11 mission
parameters (Wikipedia, NASA Mission Reports). Green = within 5% of actual,
yellow = within 30%, red = larger discrepancy. Discrepancies are explained
in the Known Limitations section below.
</div>
<table style="width:100%;max-width:900px;border-collapse:collapse;margin:20px auto;">
<thead>
<tr style="background:#f0f0f3">
<th style="padding:8px;text-align:left;border-bottom:2px solid #ccc">Parameter</th>
<th style="padding:8px;text-align:right;border-bottom:2px solid #ccc">Simulated</th>
<th style="padding:8px;text-align:right;border-bottom:2px solid #ccc">Apollo 11 Actual</th>
<th style="padding:8px;text-align:left;border-bottom:2px solid #ccc">Status</th>
</tr>
</thead>
<tbody style="font-size:14px">
{crosscheck_rows}</tbody>
</table>

<h2>Known Limitations</h2>
{limitations_html}
<h2>Files</h2>
<ul>
  <li><a href="results.csv">results.csv</a> — full per-trial data ({n} rows)</li>
  <li><a href="results_with_survival.csv">results_with_survival.csv</a> — per-trial data + crew-survival columns</li>
  <li><a href="summary.txt">summary.txt</a> — text summary</li>
  <li>{src_line} — simulation source ({src_lines} lines, in the project root, not this run directory)</li>
</ul>

<p style="color:#6e6e73;font-size:12px;margin-top:32px">
Generated by apollo11.py / generate_outputs.py.
References: NASA SP-2000-4029 "Apollo by the Numbers" (Orloff),
Smithsonian Apollo 11 (airandspace.si.edu),
Apollo 6/AS-501 post-flight analysis (thisdayinaviation.com).
</p>
</body></html>"""
    with open(os.path.join(outdir, "dashboard.html"), "w") as f:
        f.write(html)
    print("  ✓ dashboard.html")


def main(csv_path=None, traj_path=None, nominal_path=None, outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    # Prefer the survival-augmented CSV: it is a superset of results.csv (adds the
    # per-astronaut crew columns AND the mission_success headline metric). Fall
    # back to a bare results.csv if crew_survival.main() has not run yet.
    if csv_path is None:
        _ws = os.path.join(outdir, "results_with_survival.csv")
        csv_path = _ws if os.path.exists(_ws) else os.path.join(outdir, "results.csv")
    traj_path = traj_path or os.path.join(outdir, "nominal_traj.npz")
    nominal_path = nominal_path or os.path.join(outdir, "nominal_results.json")

    print("Generating output package...")
    print(f"  Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # MISSION SUCCESS (headline) = crew landed on the Moon AND all three returned
    # alive, even if a recovered in-flight anomaly occurred (see crew_survival.py).
    # Recompute defensively if we read a bare results.csv that lacks the column.
    if 'mission_success' not in df.columns:
        if 'crew_survived' in df.columns and 'land_lat_deg' in df.columns:
            df['mission_success'] = (df['land_lat_deg'].notna()
                                     & df['crew_survived'].fillna(False).astype(bool))
        else:
            df['mission_success'] = (df.get('full_success', pd.Series([False] * len(df)))
                                     .fillna(False).astype(bool))

    print(f"  Loading nominal trajectory...")
    npz = np.load(traj_path)
    nominal_traj = {}
    keys = set()
    for k in npz.files:
        if k.endswith("_t"):
            keys.add(k[:-2])
    for k in keys:
        nominal_traj[k] = (npz[k + "_t"], npz[k + "_y"])

    print(f"  Loading nominal results...")
    with open(nominal_path, "r") as f:
        nominal_results = json.load(f)

    # Add the TEI-targeting metric: splashdown DISPERSION = great-circle
    # distance from the nominal splashdown (the point the targeting drives
    # toward). This differs from splash_miss_km, which is absolute miss from
    # the fixed recovery target SPLASH_TARGET and is dominated by the
    # (entry-guidance-limited) systematic offset of the nominal from that
    # target. Computed here from coordinates already in results.csv.
    nlat = nominal_results.get("splash_lat")
    nlon = nominal_results.get("splash_lon")
    if nlat is not None and nlon is not None and "splash_lat" in df.columns:
        R_KM = 6371.0
        la0 = np.radians(nlat); lo0 = np.radians(nlon)
        la1 = np.radians(df["splash_lat"].astype(float))
        lo1 = np.radians(df["splash_lon"].astype(float))
        hav = (np.sin((la1 - la0) / 2) ** 2
               + np.cos(la0) * np.cos(la1) * np.sin((lo1 - lo0) / 2) ** 2)
        df["splash_dispersion_km"] = R_KM * 2 * np.arcsin(np.sqrt(hav.clip(upper=1.0)))

    print("\nGenerating outputs:")
    generate_summary(df, nominal_results, outdir)
    generate_plots(df, nominal_traj, nominal_results, outdir)
    generate_dashboard(df, nominal_results, outdir)

    # Copy source
    import shutil
    src = os.path.join(SIMDIR, "apollo11.py")
    dst = os.path.join(outdir, "apollo11.py")
    shutil.copy(src, dst)
    print(f"  ✓ apollo11.py (source copy)")

    print(f"\nAll outputs in {outdir}")


if __name__ == "__main__":
    main()
