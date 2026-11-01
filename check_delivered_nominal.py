"""ACCEPTANCE CHECK for a delivered Apollo 11 nominal package — run BEFORE spending compute.

    python3 check_delivered_nominal.py <package.json> [--no-physics]
    python3 check_delivered_nominal.py --self-test [<good_package.json>]     prove every check CAN fire

Tiers (the sim plan, section 4), cheapest first:
  T1 structural      check_artifact.check (any FAIL -> REJECT)
  T2 invariants      rocket equation per burn, durations vs thrust, ledgers, radii, speeds
  T3 sequence        event order, LOI-1 straddles perilune, burn durations plausible
  T4 ephemeris/frame liftoff UTC vs JD; with physics: the consumer's own DE440 Moon at the perilune epoch vs the
                     delivered perilune node, GMST0, the consumer's ground-track rotation vs the delivered EI subpoint
  T5 translunar      perilune altitude band, retrograde flyby, keep-out, v_inf band
  T6 lunar capture   SPS propellant for LOI-1 + LOI-2 + TEI with reserve, closed lunar orbit, LOI goals met
  T7 EI corridor     400 kft shell radius, entry speed, flight-path angle hard band and solve-to band
  T8 recovery        cross-track to the zone, prograde entry track, entry-track inclination; downrange SCREENED
  T9 coast parity    with physics: the delivered TLI cutoff state coasted in apollo11.gravity_earth_moon reaches the
                     delivered perilune; LOI-1 cutoff reaches the LOI-2 ignition state; TEI cutoff reaches the shell
                     at the delivered epoch and angle
Verdict: REJECT if any check rejects; ACCEPT-WITH-WARNINGS if only warnings; ACCEPT otherwise. The report lists
what it could not check. A check that has never fired is not evidence of health: --self-test injects one defect
per check and requires that check, and only that family, to REJECT.
"""
import argparse
import copy
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_artifact as CA        # noqa: E402

G0 = 9.80665
R_EARTH = 6_378_137.0
R_MOON = 1_737_400.0
EI_ALT_M = 400_000.0 * 0.3048
EI_RADIUS_M = R_EARTH + EI_ALT_M
SPS_USABLE_KG = 18_413.0
SPS_RESERVE_KG = 300.0
CSM_DRY_KG = 5_557.0 + 4_825.0
FPA_HARD = (-7.5, -5.5)
FPA_TARGET, FPA_SOLVE_TOL = -6.495, 0.05
EI_SPEED_BAND = (11_032.1 - 100.0, 11_032.1 + 100.0)
PERILUNE_BAND_M = (100_000.0, 185_000.0)
VINF_BAND = (800.0, 1500.0)
CROSSRANGE_HARD_KM = 150.0
DOWNRANGE_BAND_KM = (1500.0, 4000.0)
LOI1_LEAD_FRAC_BAND = (0.45, 0.80)        # ignition lead / burn duration; consumer 0.63 of the 825 m/s estimate -> ~0.60


class Report:
    def __init__(self, label):
        self.label = label
        self.rows = []                 # (tier, id, verdict, message)
        self.cannot = []

    def add(self, tier, cid, verdict, msg):
        self.rows.append((tier, cid, verdict, msg))

    def rejects(self):
        return [r for r in self.rows if r[2] == "REJECT"]

    def reject_ids(self):
        return {r[1] for r in self.rejects()}

    def verdict(self):
        if self.rejects():
            return "REJECT"
        if any(r[2] == "WARN" for r in self.rows):
            return "ACCEPT-WITH-WARNINGS"
        return "ACCEPT"

    def text(self):
        lines = [f"check_delivered_nominal: {self.label}"]
        for tier, cid, v, msg in self.rows:
            if v != "PASS":
                lines.append(f"  {tier} {v:6s} {cid:<22s} {msg}")
        lines.append(f"  {sum(1 for r in self.rows if r[2] == 'PASS')} PASS, {len(self.rejects())} REJECT, "
                     f"{sum(1 for r in self.rows if r[2] == 'WARN')} WARN")
        lines.append("  NOT CHECKED: " + ("; ".join(self.cannot) if self.cannot else "nothing left unchecked at this tier"))
        lines.append(f"VERDICT: {self.verdict()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# physics access (the consumer's own model)
# ---------------------------------------------------------------------------------------------
_PHYS = None


def physics():
    """Import apollo11 with its definitive ephemeris flag; None if unavailable."""
    global _PHYS
    if _PHYS is not None:
        return _PHYS or None
    os.environ.setdefault("APOLLO_JPL_EPHEMERIS", "1")
    try:
        import apollo11 as A
        import delivered as D
        _PHYS = (A, D)
    except Exception as e:
        print(f"  physics unavailable: {e}")
        _PHYS = False
    return _PHYS or None


def coast(A, y6, t0, t1, events=None, rtol=1e-10):
    from scipy.integrate import solve_ivp
    rhs = lambda t, y: np.concatenate((y[3:6], A.gravity_earth_moon(y[:3], t)))
    return solve_ivp(rhs, (t0, t1), np.asarray(y6, float), method="DOP853", rtol=rtol,
                     atol=np.array([1e-6] * 3 + [1e-9] * 3), max_step=1800.0, events=events)


def _state(node):
    return np.array(node["r_m"] + node["v_mps"], float)


# ---------------------------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------------------------
def check(doc, label="package", use_physics=True):
    r = Report(label)
    # T1
    art = CA.check(doc, label)
    for cid, lvl, msg in art.findings:
        if lvl == "FAIL":
            r.add("T1", cid, "REJECT", msg)
    if art.fails:
        r.add("T1", "T1.structural", "REJECT", f"{len(art.fails)} structural failures; physical tiers not run")
        r.cannot.append("tiers T2-T9 (structure failed)")
        return r
    r.add("T1", "T1.structural", "PASS", "check_artifact clean")
    burns = {b["name"]: b for b in doc["burn_table"]}
    nodes = doc["trajectory_nodes"]
    lb = doc["launch_block"]
    ei = nodes["ei"]

    # T2 invariants
    for n, b in burns.items():
        ve = float(b["isp_s"]) * G0
        m_pred = float(b["mass_before_kg"]) * math.exp(-float(b["dv_mag_mps"]) / ve)
        if abs(m_pred - float(b["mass_after_kg"])) > 0.005 * float(b["mass_before_kg"]):
            r.add("T2", "T2.rocket_eq", "REJECT", f"{n}: mass_after {b['mass_after_kg']:.1f} vs rocket equation {m_pred:.1f} kg")
        else:
            r.add("T2", "T2.rocket_eq", "PASS", f"{n}: rocket equation within 0.5 %")
        mdot = float(b["thrust_n"]) / ve
        dur_pred = (float(b["mass_before_kg"]) - float(b["mass_after_kg"])) / mdot
        if abs(dur_pred - float(b["duration_s"])) > 0.02 * max(float(b["duration_s"]), 1.0):
            r.add("T2", "T2.duration", "REJECT", f"{n}: duration {b['duration_s']:.1f} s vs propellant/mdot {dur_pred:.1f} s")
        else:
            r.add("T2", "T2.duration", "PASS", f"{n}: duration consistent with thrust and propellant")
    led = doc["ledgers"]
    sps = [float(led[k]) for k in ("sps_usable_kg", "sps_after_loi1_kg", "sps_after_loi2_kg", "sps_after_tei_kg")]
    if not (sps[0] > sps[1] > sps[2] > sps[3] >= 0.0):
        r.add("T2", "T2.sps_ledger", "REJECT", f"SPS ledger not monotone / non-negative: {sps}")
    else:
        r.add("T2", "T2.sps_ledger", "PASS", f"SPS ledger {sps[0]:.0f} -> {sps[3]:.0f} kg")
    if float(led["sivb_propellant_left_after_tli_kg"]) < 0.0:
        r.add("T2", "T2.sivb_ledger", "REJECT", "S-IVB propellant negative after TLI")
    for k, nd in nodes.items():
        if "r_m" in nd:
            rn = float(np.linalg.norm(nd["r_m"]))
            vn = float(np.linalg.norm(nd["v_mps"]))
            if rn < R_EARTH or vn > 12_000.0:
                r.add("T2", "T2.node_physical", "REJECT", f"node {k}: |r| {rn / 1e3:.0f} km, |v| {vn:.0f} m/s")
    if "T2.node_physical" not in r.reject_ids():
        r.add("T2", "T2.node_physical", "PASS", "node radii above the surface, speeds sub-escape")

    # T3 sequence
    peri_t = float(nodes["perilune"]["get_s"])
    l1 = burns["LOI-1"]
    # The consumer ignites LOI-1 0.63 of the *estimated* burn before perilune (actual lead fraction ~0.60): the burn
    # must straddle perilune (A9 already orders the events) and the lead fraction must sit in the consumer's band.
    lead_frac = (peri_t - float(l1["t_ignition_get_s"])) / float(l1["duration_s"])
    if not (float(l1["t_ignition_get_s"]) < peri_t < float(l1["t_cutoff_get_s"])):
        r.add("T3", "T3.loi1_straddle", "REJECT", "LOI-1 does not straddle the perilune epoch")
    elif not (LOI1_LEAD_FRAC_BAND[0] <= lead_frac <= LOI1_LEAD_FRAC_BAND[1]):
        r.add("T3", "T3.loi1_straddle", "REJECT",
              f"LOI-1 ignition lead {lead_frac:.2f} of the burn outside {LOI1_LEAD_FRAC_BAND} (consumer: 0.63 of the estimate)")
    else:
        r.add("T3", "T3.loi1_straddle", "PASS",
              f"LOI-1 ignites {peri_t - float(l1['t_ignition_get_s']):.0f} s ({lead_frac:.2f} of the burn) before perilune")
    tli = burns["TLI"]
    if not (200.0 < float(tli["duration_s"]) < 420.0):
        r.add("T3", "T3.tli_duration", "REJECT", f"TLI duration {tli['duration_s']} s outside 200-420 s")
    else:
        r.add("T3", "T3.tli_duration", "PASS", "TLI duration plausible")

    # T4 ephemeris / frame
    jd = float(lb["liftoff_jd"])
    phys = physics() if use_physics else None
    if phys:
        A, D = phys
        targets = D.adapt_delivered_package(doc)
        D.bind_delivered_targets(targets, A, strict=False)
        gm = A._gmst_rad(jd)
        if abs(gm - float(doc["epoch"]["gmst0_rad"])) > 1e-9:
            r.add("T4", "T4.gmst0", "REJECT", f"GMST0 {doc['epoch']['gmst0_rad']} vs the consumer's {gm}")
        else:
            r.add("T4", "T4.gmst0", "PASS", "GMST0 agrees with the consumer's formula")
        rm, _ = A.moon_state(peri_t)
        alt = float(np.linalg.norm(np.asarray(nodes["perilune"]["r_m"]) - rm) - R_MOON)
        if abs(alt - float(nodes["perilune"]["altitude_m"])) > 2000.0:
            r.add("T4", "T4.moon_ephemeris", "REJECT", f"perilune altitude vs the consumer's DE440 Moon: {alt / 1e3:.1f} km "
                                                       f"vs delivered {float(nodes['perilune']['altitude_m']) / 1e3:.1f} km")
        else:
            r.add("T4", "T4.moon_ephemeris", "PASS", f"delivered perilune sits {alt / 1e3:.2f} km above the consumer's Moon")
        la, lo = A.eci_to_latlon(np.asarray(ei["r_m"]), float(ei["get_s"]))
        dlon = ((float(lo) - float(ei["subpoint_lon_deg"]) + 180.0) % 360.0) - 180.0
        if abs(float(la) - float(ei["subpoint_lat_gc_deg"])) > 0.01 or abs(dlon) > 0.01:
            r.add("T4", "T4.ground_track", "REJECT", f"EI subpoint via the consumer's rotation ({la:.3f}, {lo:.3f}) vs delivered "
                                                     f"({ei['subpoint_lat_gc_deg']:.3f}, {ei['subpoint_lon_deg']:.3f})")
        else:
            r.add("T4", "T4.ground_track", "PASS", "EI subpoint agrees with the consumer's ground-track rotation")
    else:
        r.cannot.append("T4 ephemeris and ground-track parity (physics off)")

    # T5 translunar
    h = float(nodes["perilune"]["altitude_m"])
    if not (PERILUNE_BAND_M[0] <= h <= PERILUNE_BAND_M[1]):
        r.add("T5", "T5.perilune_band", "REJECT", f"perilune altitude {h / 1e3:.1f} km outside {PERILUNE_BAND_M[0] / 1e3:.0f}-{PERILUNE_BAND_M[1] / 1e3:.0f} km")
    else:
        r.add("T5", "T5.perilune_band", "PASS", f"perilune {h / 1e3:.1f} km")
    retro_screen = doc["free_return_screen"].get("retrograde_at_moon")
    incl = nodes["perilune"].get("lunar", {}).get("incl_lunar_equator_deg")
    if phys:
        A, _ = phys
        rm, vm = A.moon_state(peri_t)
        hz = float(np.cross(np.asarray(nodes["perilune"]["r_m"]) - rm, np.asarray(nodes["perilune"]["v_mps"]) - vm)[2])
        retro = hz < 0.0
        src = "consumer's Moon state"
    else:
        retro = bool(retro_screen) and (incl is None or float(incl) > 90.0)
        src = "screen + delivered inclination"
    if not retro or retro_screen is False or (incl is not None and float(incl) <= 90.0):
        r.add("T5", "T5.retrograde", "REJECT", f"lunar flyby not retrograde ({src}; incl {incl})")
    else:
        r.add("T5", "T5.retrograde", "PASS", f"retrograde at the Moon ({src})")
    if doc["free_return_screen"].get("keepout_violated"):
        r.add("T5", "T5.keepout", "REJECT", "100 km lunar keep-out violated")
    vinf = nodes["perilune"].get("v_inf_mps")
    if vinf is not None and not (VINF_BAND[0] <= float(vinf) <= VINF_BAND[1]):
        r.add("T5", "T5.vinf", "REJECT", f"arrival v_inf {vinf} m/s outside {VINF_BAND}")

    # T6 lunar capture
    prop = sum(float(burns[n]["mass_before_kg"]) - float(burns[n]["mass_after_kg"]) for n in ("LOI-1", "LOI-2", "TEI"))
    if prop > SPS_USABLE_KG - SPS_RESERVE_KG:
        r.add("T6", "T6.sps_margin", "REJECT", f"LOI-1 + LOI-2 + TEI propellant {prop:.0f} kg exceeds usable minus reserve")
    else:
        r.add("T6", "T6.sps_margin", "PASS", f"SPS propellant {prop:.0f} of {SPS_USABLE_KG:.0f} kg")
    lo_ = nodes["lunar_orbit"]
    rp, ra = R_MOON + float(lo_["perilune_alt_m"]), R_MOON + float(lo_["apolune_alt_m"])
    e = (ra - rp) / (ra + rp)
    if not (0.0 <= e < 0.2 and float(lo_["perilune_alt_m"]) > 50_000.0):
        r.add("T6", "T6.orbit_closed", "REJECT", f"lunar orbit {lo_['perilune_alt_m'] / 1e3:.0f} x {lo_['apolune_alt_m'] / 1e3:.0f} km, e {e:.3f}")
    else:
        r.add("T6", "T6.orbit_closed", "PASS", f"lunar parking orbit e {e:.4f}")
    for n in ("LOI-1", "LOI-2"):
        for g in burns[n]["goals"]:
            if abs(float(g["achieved"]) - float(g["target"])) > float(g["tolerance"]):
                r.add("T6", "T6.loi_goals", "REJECT", f"{n} goal {g['name']}: achieved {g['achieved']} vs target {g['target']} +/- {g['tolerance']}")
    if "T6.loi_goals" not in r.reject_ids():
        r.add("T6", "T6.loi_goals", "PASS", "LOI-1 and LOI-2 goals achieved within tolerance")

    # T7 EI corridor
    if abs(float(ei["radius_m"]) - EI_RADIUS_M) > 1.0:
        r.add("T7", "T7.shell", "REJECT", f"EI radius {ei['radius_m']} != 400 kft shell {EI_RADIUS_M}")
    else:
        r.add("T7", "T7.shell", "PASS", "EI on the 400 kft shell")
    v = float(ei["speed_mps"])
    if not (EI_SPEED_BAND[0] <= v <= EI_SPEED_BAND[1]):
        r.add("T7", "T7.speed", "REJECT", f"entry speed {v:.0f} m/s outside {EI_SPEED_BAND}")
    else:
        r.add("T7", "T7.speed", "PASS", f"entry speed {v:.1f} m/s")
    g = float(ei["fpa_deg"])
    if not (FPA_HARD[0] <= g <= FPA_HARD[1]):
        r.add("T7", "T7.fpa_hard", "REJECT", f"EI flight-path angle {g:.3f} outside the hard corridor {FPA_HARD}")
    elif abs(g - FPA_TARGET) > FPA_SOLVE_TOL:
        r.add("T7", "T7.fpa_solve", "WARN", f"EI flight-path angle {g:.3f} inside the corridor but {g - FPA_TARGET:+.3f} from {FPA_TARGET}")
    else:
        r.add("T7", "T7.fpa", "PASS", f"EI flight-path angle {g:.3f}")

    # T8 recovery
    c = float(ei["crossrange_to_zone_km"])
    d = float(ei["downrange_to_zone_km"])
    if abs(c) > CROSSRANGE_HARD_KM:
        r.add("T8", "T8.crossrange", "REJECT", f"|C| {abs(c):.0f} km to the zone exceeds {CROSSRANGE_HARD_KM:.0f} km")
    else:
        r.add("T8", "T8.crossrange", "PASS", f"cross-track to the zone {c:+.0f} km")
    if not (DOWNRANGE_BAND_KM[0] <= d <= DOWNRANGE_BAND_KM[1]):
        r.add("T8", "T8.downrange", "REJECT", f"downrange to the zone {d:.0f} km outside {DOWNRANGE_BAND_KM}")
    else:
        r.add("T8", "T8.downrange", "WARN", f"downrange {d:.0f} km SCREENED (the D target is a placeholder)")
    if not ei.get("earth_prograde", True) or float(ei.get("entry_track_inclination_deg", 0.0)) >= 90.0:
        r.add("T8", "T8.prograde", "REJECT", "entry track retrograde or inclination >= 90 deg")
    else:
        r.add("T8", "T8.prograde", "PASS", "prograde entry track")

    # T9 coast parity in the consumer's physics
    if phys:
        A, _ = phys
        y = _state(tli["state_post"])
        s = coast(A, y, float(tli["state_post"]["get_s"]), peri_t)
        dr = float(np.linalg.norm(s.y[:3, -1] - np.asarray(nodes["perilune"]["r_m"])))
        rm, _ = A.moon_state(peri_t)
        alt = float(np.linalg.norm(s.y[:3, -1] - rm) - R_MOON)
        if dr > 25_000.0 or abs(alt - h) > 5_000.0:
            r.add("T9", "T9.tli_coast", "REJECT", f"TLI cutoff coasted in apollo11 physics misses the delivered perilune by {dr / 1e3:.1f} km "
                                                   f"(altitude {alt / 1e3:.1f} vs {h / 1e3:.1f} km)")
        else:
            r.add("T9", "T9.tli_coast", "PASS", f"TLI cutoff -> perilune reproduced to {dr:.0f} m")
        l2 = burns["LOI-2"]
        s = coast(A, _state(l1["state_post"]), float(l1["state_post"]["get_s"]), float(l2["t_ignition_get_s"]))
        dr2 = float(np.linalg.norm(s.y[:3, -1] - np.asarray(l2["state_at_tig"]["r_m"])))
        if dr2 > 5_000.0:
            r.add("T9", "T9.loi_coast", "REJECT", f"LOI-1 cutoff coasted to LOI-2 ignition misses by {dr2 / 1e3:.1f} km")
        else:
            r.add("T9", "T9.loi_coast", "PASS", f"LOI-1 cutoff -> LOI-2 ignition reproduced to {dr2:.0f} m")
        tei = burns["TEI"]

        def shell(t, yv):
            return float(np.linalg.norm(yv[:3]) - EI_RADIUS_M)
        shell.terminal = True
        shell.direction = -1
        s = coast(A, _state(tei["state_post"]), float(tei["state_post"]["get_s"]), float(tei["state_post"]["get_s"]) + 5 * 86400.0, events=[shell])
        if not len(s.t_events[0]):
            r.add("T9", "T9.tei_coast", "REJECT", "TEI cutoff coasted in apollo11 physics never reaches the EI shell")
        else:
            te = float(s.t_events[0][0])
            ye = s.y_events[0][0]
            ge = math.degrees(math.asin(float(ye[:3] @ ye[3:6]) / (np.linalg.norm(ye[:3]) * np.linalg.norm(ye[3:6]))))
            if abs(te - float(ei["get_s"])) > 60.0 or abs(ge - g) > 0.05:
                r.add("T9", "T9.tei_coast", "REJECT", f"TEI cutoff -> shell at GET {te:.1f} (delivered {ei['get_s']:.1f}), fpa {ge:.3f} (delivered {g:.3f})")
            else:
                r.add("T9", "T9.tei_coast", "PASS", f"TEI cutoff -> EI reproduced: dt {te - float(ei['get_s']):+.1f} s, dfpa {ge - g:+.4f} deg")
    else:
        r.cannot.append("T9 coast parity in the consumer's force model (physics off)")
    r.cannot += ["the consumer's launch reproducing the insertion state (deterministic re-fly, refly_delivered_nominal.py)",
                 "descent, surface, rendezvous and entry guidance (the consumer's own phases)",
                 "the downrange target (placeholder; D is screened)"]
    return r


# ---------------------------------------------------------------------------------------------
# self-test: one defect per check
# ---------------------------------------------------------------------------------------------
def _resign(doc):
    doc["manifest"]["package_sha256"] = CA.package_sha256(doc)
    return doc


def injections():
    """[(name, mutate(doc) -> doc, expected reject id prefix, needs_physics)]"""
    def m_missing_block(d):
        del d["targets"]
        return _resign(d)

    def m_nan_node(d):
        d["trajectory_nodes"]["ei"]["r_m"][0] = None
        return _resign(d)

    def m_hash(d):
        d["burn_table"][0]["dv_mag_mps"] += 1.0
        return d

    def m_provenance(d):
        d["provenance"]["quantities"]["EI shell 400,000 ft"] = "[sourced] nowhere"
        return _resign(d)

    def m_epoch_convention(d):
        del d["burn_table"][1]["epoch_convention"]
        return _resign(d)

    def m_order(d):
        b = d["burn_table"][0]
        b["t_cutoff_get_s"] = b["t_ignition_get_s"] - 10.0
        b["duration_s"] = -10.0
        return _resign(d)

    def m_liftoff_jd(d):
        d["launch_block"]["liftoff_jd"] += 5.0 / 86400.0
        d["epoch"]["liftoff_jd"] = d["launch_block"]["liftoff_jd"]
        return _resign(d)

    def m_mass_gain(d):
        b = d["burn_table"][1]
        b["mass_after_kg"] = b["mass_before_kg"] + 10.0
        return _resign(d)

    def m_rocket_eq(d):
        d["burn_table"][1]["dv_mag_mps"] += 60.0
        return _resign(d)

    def m_sps_ledger(d):
        d["ledgers"]["sps_after_tei_kg"] = -100.0
        return _resign(d)

    def m_straddle(d):
        # perilune still inside the burn (A9 order holds) but 5 s before cutoff: lead fraction ~0.99
        d["trajectory_nodes"]["perilune"]["get_s"] = d["burn_table"][1]["t_cutoff_get_s"] - 5.0
        return _resign(d)

    def m_perilune_low(d):
        d["trajectory_nodes"]["perilune"]["altitude_m"] = 60_000.0
        return _resign(d)

    def m_prograde_flyby(d):
        d["free_return_screen"]["retrograde_at_moon"] = False
        d["trajectory_nodes"]["perilune"]["lunar"]["incl_lunar_equator_deg"] = 10.0
        return _resign(d)

    def m_orbit_open(d):
        d["trajectory_nodes"]["lunar_orbit"]["apolune_alt_m"] = 5_000_000.0
        return _resign(d)

    def m_loi_goal(d):
        d["burn_table"][1]["goals"][0]["achieved"] += 25.0
        return _resign(d)

    def m_sps_overdraw(d):
        d["burn_table"][3]["mass_before_kg"] += 15_000.0
        d["burn_table"][3]["dv_mag_mps"] = 2500.0
        return _resign(d)

    def m_ei_radius(d):
        d["trajectory_nodes"]["ei"]["radius_m"] += 5_000.0
        return _resign(d)

    def m_ei_steep(d):
        d["trajectory_nodes"]["ei"]["fpa_deg"] = -8.0
        return _resign(d)

    def m_ei_fast(d):
        d["trajectory_nodes"]["ei"]["speed_mps"] = 12_000.0
        return _resign(d)

    def m_crossrange(d):
        d["trajectory_nodes"]["ei"]["crossrange_to_zone_km"] = 500.0
        return _resign(d)

    def m_retrograde_entry(d):
        d["trajectory_nodes"]["ei"]["earth_prograde"] = False
        d["trajectory_nodes"]["ei"]["entry_track_inclination_deg"] = 120.0
        return _resign(d)

    def m_moon_ephem(d):
        d["trajectory_nodes"]["perilune"]["r_m"][0] += 50_000.0
        return _resign(d)

    def m_ground_track(d):
        d["trajectory_nodes"]["ei"]["subpoint_lon_deg"] += 1.0
        return _resign(d)

    def m_tli_coast(d):
        d["burn_table"][0]["state_post"]["v_mps"][0] += 20.0
        return _resign(d)

    def m_tei_coast(d):
        d["burn_table"][3]["state_post"]["v_mps"][1] += 10.0
        return _resign(d)

    return [
        ("missing_block", m_missing_block, "A1.blocks", False),
        ("nan_node", m_nan_node, "A5.node_state", False),
        ("hash_mismatch", m_hash, "A8.hash", False),
        ("bad_provenance", m_provenance, "A7.tokens", False),
        ("missing_epoch_convention", m_epoch_convention, "A4.burn_keys", False),
        ("burn_order", m_order, "A4.burn_order", False),
        ("liftoff_jd_off", m_liftoff_jd, "A2.liftoff_jd", False),
        ("mass_gain", m_mass_gain, "A4.mass_order", False),
        ("rocket_equation", m_rocket_eq, "T2.rocket_eq", False),
        ("sps_ledger", m_sps_ledger, "T2.sps_ledger", False),
        ("loi1_straddle", m_straddle, "T3.loi1_straddle", False),
        ("perilune_low", m_perilune_low, "T5.perilune_band", False),
        ("prograde_flyby", m_prograde_flyby, "T5.retrograde", False),
        ("lunar_orbit_open", m_orbit_open, "T6.orbit_closed", False),
        ("loi_goal_miss", m_loi_goal, "T6.loi_goals", False),
        ("sps_overdraw", m_sps_overdraw, "T6.sps_margin", False),
        ("ei_off_shell", m_ei_radius, "T7.shell", False),
        ("ei_steep", m_ei_steep, "T7.fpa_hard", False),
        ("ei_fast", m_ei_fast, "T7.speed", False),
        ("crossrange", m_crossrange, "T8.crossrange", False),
        ("retrograde_entry", m_retrograde_entry, "T8.prograde", False),
        ("moon_ephemeris", m_moon_ephem, "T4.moon_ephemeris", True),
        ("ground_track", m_ground_track, "T4.ground_track", True),
        ("tli_coast", m_tli_coast, "T9.tli_coast", True),
        ("tei_coast", m_tei_coast, "T9.tei_coast", True),
    ]


def self_test(good_path, use_physics=True):
    good = CA.load(good_path)
    base = check(good, "self-test base", use_physics=use_physics)
    if base.rejects():
        print(base.text())
        print("SELF-TEST: the base package must be clean before injecting defects")
        return False
    fired, total, skipped = 0, 0, 0
    for name, mutate, expect, needs_phys in injections():
        if needs_phys and not use_physics:
            skipped += 1
            continue
        total += 1
        doc = mutate(copy.deepcopy(good))
        rep = check(doc, f"inject {name}", use_physics=use_physics)
        hit = any(cid == expect or cid.startswith(expect) for cid in rep.reject_ids())
        print(f"  {'FIRED ' if hit else 'MISSED'} {name:<26s} expected {expect:<20s} rejects: {sorted(rep.reject_ids())[:4]}")
        fired += int(hit)
    print(f"SELF-TEST: {fired}/{total} checks fired" + (f" ({skipped} physics injections skipped)" if skipped else ""))
    return fired == total


def find_default_package():
    for pat in (os.path.join(HERE, "nominal_packages", "*.json"),
                os.path.join(os.path.dirname(HERE), "apollo_solver", "outputs", "apollo11_nominal_package.json")):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("packages", nargs="*")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-physics", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        path = args.packages[0] if args.packages else find_default_package()
        if not path:
            print("self-test needs a clean package (nominal_packages/*.json)")
            return 2
        return 0 if self_test(path, use_physics=not args.no_physics) else 1
    if not args.packages:
        print(__doc__)
        return 2
    worst = 0
    for path in args.packages:
        doc = CA.load(path)
        rep = check(doc, f"{path} ({doc.get('label')})", use_physics=not args.no_physics)
        print(rep.text())
        if rep.rejects():
            worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
