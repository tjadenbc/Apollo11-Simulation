#!/usr/bin/env python3
"""Deterministic re-fly of the delivered nominal in the consumer's own physics (sim plan S4).

Binds the package into apollo11, runs apollo11.run_mission(perturb=None) with the milestone recorder
(apollo11._REFLY_HOOK) on, and compares every recorded milestone with the package's authoritative
trajectory nodes and burn states: insertion (MECO), TLI cutoff, perilune, LOI-1/LOI-2 ignition and
cutoff, TEI ignition (the lunar-orbit coast through the surface stay), the delivered TEI burn flown
open-loop (cutoff and its ballistic EI), the TEI burn the consumer's 3-DOF corrector actually flew, the
entry interface after the trans-Earth MCC chain, and the splashdown against the recovery target.

Writes outputs/refly_report.json and prints the table. Exit 1 when a HARD tolerance is exceeded or
the mission does not reach splashdown; WARN rows are reported, not fatal.

    venv/bin/python refly_delivered_nominal.py [package.json] [--out outputs/refly_report.json] [--quiet]
"""
import argparse
import io
import json
import math
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check_artifact as CA          # noqa: E402
import delivered as D                # noqa: E402

R_EARTH_KM = 6371.0

# (id, hard?, limit) ; dr in m, dv in m/s, dt in s, angles in deg, km where named.
# HARD rows are what the package and the hooks control: the consumer's own launch and TLI at the bound
# controls, the perilune altitude, each delivered burn executed by the consumer's own burn integrator from
# the DELIVERED ignition state (the *_exec rows: pure burn-execution parity, independent of coast drift),
# the EI corridor after the consumer's own TEI corrector and MCC chain, and the entry guidance vs the zone
# it aims for. The *_burn rows (velocity change across the burn flown in the mission) include the gravity
# integral over the coast-drifted path (a 1.3 km radial offset at 113 km altitude is 0.7 m/s over LOI-1).
# WARN rows are the consumer's own coast drift (RK45 rtol 1e-9 vs the package's DOP853 1e-10 arcs:
# measured 1.6 km at perilune, 92 km / 55 s along-track at TEI after 28 lunar revs) and the screened
# recovery geometry (the package's EI sits 2,770 km / -22 km crossrange from the Pacific target; the
# consumer places its zone 2,784 km down its own track).
TOL = {
    "insertion.dr": (True, 50.0), "insertion.dv": (True, 0.1), "insertion.dt": (True, 0.01),
    "tli_cutoff.dr": (True, 100.0), "tli_cutoff.dv": (True, 0.1), "tli_cutoff.dt": (True, 0.1),
    "perilune.dh_km": (True, 1.0), "perilune.dr": (True, 200.0), "perilune.dt": (False, 15.0),
    "loi1_ignition.dr": (False, 2_000.0), "loi1_ignition.dv": (False, 1.0),
    "loi1_cutoff.dr": (False, 2_000.0), "loi1_cutoff.dv": (False, 1.0), "loi1_cutoff.dt": (True, 0.5),
    "loi1_burn.ddv": (False, 1.0), "loi1_burn.dangle": (False, 0.05),
    "loi1_exec.dr": (True, 1.0), "loi1_exec.dv": (True, 0.01), "loi1_exec.dt": (True, 0.01),
    "loi1_site.xtrack_km": (False, 1.0), "loi1_site.dplane_change_deg": (False, 0.05),
    "landing.site_downrange_km": (False, 1.0), "landing.site_crossrange_km": (False, 3.0),
    "loi2_ignition.dr": (False, 5_000.0), "loi2_ignition.dv": (False, 5.0),
    "loi2_cutoff.dr": (False, 5_000.0), "loi2_cutoff.dv": (False, 5.0), "loi2_cutoff.dt": (True, 0.5),
    "loi2_burn.ddv": (False, 1.0), "loi2_burn.dangle": (False, 0.05),
    "loi2_exec.dr": (True, 1.0), "loi2_exec.dv": (True, 0.01), "loi2_exec.dt": (True, 0.01),
    "tei_ignition.dr": (False, 10_000.0), "tei_ignition.dv": (False, 10.0), "tei_ignition.dmass_kg": (False, 10.0),
    "tei_cutoff_open_loop.dr": (False, 10_000.0), "tei_cutoff_open_loop.dv": (False, 10.0),
    "tei_burn_open_loop.ddv": (False, 20.0), "tei_burn_open_loop.dangle": (False, 0.2),
    "tei_exec.dr": (True, 1.0), "tei_exec.dv": (True, 0.01), "tei_exec.dt": (True, 0.01),
    "tei_exec.ei_dfpa": (True, 0.01), "tei_exec.ei_dt": (True, 5.0),
    "ei_open_loop.dfpa": (False, 0.05), "ei_open_loop.dt": (False, 120.0),
    "tei_corrector.ddv_mps": (False, 5.0),
    "ei.dfpa": (True, 0.01), "ei.dt": (True, 60.0), "ei.dr": (False, 100.0), "ei.dspeed": (True, 5.0),
    "ei.dlat": (True, 0.05), "ei.dlon": (True, 0.1),
    "splash.guidance_miss_km": (True, 25.0), "splash.target_displacement_km": (False, 25.0),
}


def _fpa_deg(y):
    r, v = np.asarray(y[:3], float), np.asarray(y[3:6], float)
    return math.degrees(math.asin(float(np.dot(r, v)) / (np.linalg.norm(r) * np.linalg.norm(v))))


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(a))


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


class Report:
    def __init__(self):
        self.rows = []        # (id, value, limit, hard, ok, note)

    def add(self, rid, value, note=""):
        hard, lim = TOL[rid]
        ok = bool(abs(value) <= lim)
        self.rows.append((rid, float(value), lim, hard, ok, note))
        return ok

    def hard_fails(self):
        return [r for r in self.rows if r[3] and not r[4]]

    def warns(self):
        return [r for r in self.rows if (not r[3]) and not r[4]]

    def text(self):
        out = [f"{'metric':<30s} {'value':>14s} {'limit':>10s}  verdict"]
        for rid, v, lim, hard, ok, note in self.rows:
            verdict = "PASS" if ok else ("FAIL" if hard else "WARN")
            out.append(f"{rid:<30s} {v:>14.4f} {lim:>10.4g}  {verdict}{('  ' + note) if note else ''}")
        return "\n".join(out)


def refly(package_path, verbose=True):
    pkg = CA.load(package_path)
    targets = D.adapt_delivered_package(pkg)
    os.environ.update(D.env_for_import(targets))
    import apollo11 as A                                   # import AFTER the env is set (import-time flags)
    manifest = D.bind_delivered_targets(targets, A)
    t = targets
    A._REFLY_HOOK = []
    A._BPLANE_TARGET = None                                # the nominal captures its own B-plane target
    t0 = time.time()
    if verbose:
        print(f"re-flying {os.path.basename(package_path)} (sha256 {t['package_sha256'][:16]}...) in apollo11 "
              f"{A.__file__}", flush=True)
    buf = io.StringIO()
    with redirect_stdout(buf if not verbose else sys.stdout):
        results, _traj = A.run_mission(perturb=None)
    wall = time.time() - t0
    rec = {}
    for ev, te, y in A._REFLY_HOOK:
        rec.setdefault(ev, (te, y))                        # first occurrence
    rep = Report()
    cmp = {}

    def state_cmp(ev, ref_state, ref_t, ids=("dr", "dv", "dt")):
        if ev not in rec:
            cmp[ev] = {"missing": True}
            return
        te, y = rec[ev]
        ref = np.asarray(ref_state, float)
        dr = float(np.linalg.norm(y[:3] - ref[:3]))
        dv = float(np.linalg.norm(y[3:6] - ref[3:6]))
        dt = float(te - ref_t)
        cmp[ev] = {"get_s": te, "ref_get_s": float(ref_t), "dr_m": dr, "dv_mps": dv, "dt_s": dt,
                   "state": [float(v) for v in y[:7]], "ref_state": [float(v) for v in ref[:6]]}
        if "dr" in ids and f"{ev}.dr" in TOL:
            rep.add(f"{ev}.dr", dr)
        if "dv" in ids and f"{ev}.dv" in TOL:
            rep.add(f"{ev}.dv", dv)
        if "dt" in ids and f"{ev}.dt" in TOL:
            rep.add(f"{ev}.dt", dt)
        return te, y

    def burn_cmp(name, ign_ev, cut_ev, ref_tig, ref_post):
        """The executed velocity change across a burn (cutoff - ignition) vs the delivered one: isolates the
        burn execution (dv, frozen direction, duration) from the coast drift of the state it starts from."""
        if ign_ev not in rec or cut_ev not in rec:
            return
        dv_c = rec[cut_ev][1][3:6] - rec[ign_ev][1][3:6]
        dv_d = np.asarray(ref_post[3:6], float) - np.asarray(ref_tig[3:6], float)
        ddv = float(np.linalg.norm(dv_c - dv_d))
        cosang = float(np.dot(dv_c, dv_d) / (np.linalg.norm(dv_c) * np.linalg.norm(dv_d)))
        dang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        cmp[name] = {"flown_dv_mps": float(np.linalg.norm(dv_c)), "delivered_dv_mps": float(np.linalg.norm(dv_d)),
                     "ddv_vector_mps": ddv, "dangle_deg": dang}
        rep.add(f"{name}.ddv", ddv, "flown-vs-delivered dv across the burn, includes the gravity integral over the drifted path")
        rep.add(f"{name}.dangle", dang)

    def exec_parity(key, burn):
        """The consumer's own burn integrator from the DELIVERED ignition state and mass vs the delivered cutoff."""
        y0 = np.r_[np.asarray(t[f"{key}_state_at_tig"], float), float(burn["mass_before_kg"])]
        t_ign = t[f"{key}_t_ignition_get_s"]
        ref = np.asarray(t[f"{key}_state_post"], float)
        t_ref = t_ign + t[f"{key}_duration_s"]
        if key == "tei":
            r = A._fly_fixed_tei(y0, t_ign, np.asarray(t["tei_dir_eci"], float), t["tei_dv_mps"], A.SPS_ISP, A.SPS_THRUST)
            if r is None:
                cmp["tei_exec"] = {"no_entry": True}
                rep.add("tei_exec.ei_dfpa", float("inf"), "delivered burn from the delivered state reaches no entry")
                return
            y, te = r["post_burn_state"], r["t_post_burn"]
            cmp["tei_exec"] = {"ei_fpa_deg": r["entry_fpa_deg"], "ei_get_s": r["entry_t"], "burn_time_s": r["burn_time_s"]}
            rep.add("tei_exec.ei_dfpa", r["entry_fpa_deg"] - t["ei_fpa_deg"], "ballistic coast of the executed burn to EI")
            rep.add("tei_exec.ei_dt", r["entry_t"] - t["ei_get_s"])
        else:
            y, te, _bt = A.phase_loi(y0, t_ign, t[f"{key}_dv_mps"], {}, tilt_deg=t[f"{key}_tilt_deg"],
                                     burn_dir_eci=t[f"{key}_dir_eci"])
            cmp[f"{key}_exec"] = {}
        cmp[f"{key}_exec"].update({"dr_m": float(np.linalg.norm(y[:3] - ref[:3])),
                                   "dv_mps": float(np.linalg.norm(y[3:6] - ref[3:6])), "dt_s": float(te - t_ref),
                                   "mass_after_kg": float(y[6]), "ref_mass_after_kg": float(burn["mass_after_kg"])})
        rep.add(f"{key}_exec.dr", cmp[f"{key}_exec"]["dr_m"], "consumer burn integrator from the delivered ignition state")
        rep.add(f"{key}_exec.dv", cmp[f"{key}_exec"]["dv_mps"])
        rep.add(f"{key}_exec.dt", cmp[f"{key}_exec"]["dt_s"])

    # MECO and TLI cutoff: the consumer's own IGM ascent and steered S-IVB burn at the bound controls
    state_cmp("insertion", t["insertion_state"], t["insertion_get_s"])
    state_cmp("tli_cutoff", t["tli_state_post"], t["tli_t_cutoff_get_s"])
    # perilune: the consumer's own closest approach (after its translunar MCC chain) vs the delivered node
    if "perilune" in rec:
        te, y = rec["perilune"]
        rm = A.moon_state(te)[0]
        h_km = (float(np.linalg.norm(y[:3] - rm)) - A.R_MOON) / 1e3
        cmp["perilune"] = {"get_s": te, "alt_km": h_km, "ref_alt_km": t["perilune_alt_m"] / 1e3,
                           "dt_s": te - t["perilune_get_s"],
                           "dr_m": float(np.linalg.norm(y[:3] - np.asarray(t["perilune_state"][:3])))}
        rep.add("perilune.dh_km", h_km - t["perilune_alt_m"] / 1e3, "the logged sample nearest perilune")
        if "perilune_at_delivered_epoch" in rec:
            # same-epoch comparison (the log sample is up to ~6 s from perilune, ~2.4 km/s)
            y_e = rec["perilune_at_delivered_epoch"][1]
            cmp["perilune"]["dr_same_epoch_m"] = float(np.linalg.norm(y_e[:3] - np.asarray(t["perilune_state"][:3])))
            cmp["perilune"]["dv_same_epoch_mps"] = float(np.linalg.norm(y_e[3:6] - np.asarray(t["perilune_state"][3:6])))
            rep.add("perilune.dr", cmp["perilune"]["dr_same_epoch_m"], "at the delivered perilune epoch")
        else:
            rep.add("perilune.dr", cmp["perilune"]["dr_m"], "logged sample vs delivered node (includes the sampling offset)")
        rep.add("perilune.dt", cmp["perilune"]["dt_s"])
    else:
        cmp["perilune"] = {"missing": True}
    # LOI-1 / LOI-2: ignition state (coast parity) and cutoff state (the flown burn)
    for b in ("loi1", "loi2"):
        state_cmp(f"{b}_ignition", t[f"{b}_state_at_tig"], t[f"{b}_t_ignition_get_s"], ids=("dr", "dv"))
        state_cmp(f"{b}_cutoff", t[f"{b}_state_post"], t[f"{b}_t_ignition_get_s"] + t[f"{b}_duration_s"])
        burn_cmp(f"{b}_burn", f"{b}_ignition", f"{b}_cutoff", t[f"{b}_state_at_tig"], t[f"{b}_state_post"])
        exec_parity(b, pkg["burn_table"][1 if b == "loi1" else 2])
    # LOI-1 Site 2 plane: the consumer's flown LOI-1 plane coasted to the PDI epoch (its own residual)
    def _finite(k):
        return results.get(k) is not None and math.isfinite(float(results[k]))
    # (a natural-landing package has no site: these rows are NaN and skipped)
    if _finite("loi1_site_xtrack_km"):
        rep.add("loi1_site.xtrack_km", float(results["loi1_site_xtrack_km"]),
                "flown LOI-1 plane coasted to PDI vs the package landing site (goal 0 +/- 1 km)")
    if results.get("loi1_plane_change_deg") is not None and t.get("loi1_plane_change_deg") is not None:
        rep.add("loi1_site.dplane_change_deg", float(results["loi1_plane_change_deg"]) - t["loi1_plane_change_deg"])
    # touchdown vs Site 2 (consumer-only: the package delivers no landing; PDI TIG from ENABLE_PDI_SITE_TARGETING)
    if _finite("land_site_downrange_km"):
        rep.add("landing.site_downrange_km", float(results["land_site_downrange_km"]),
                f"PDI {results.get('pdi_tig_offset_s', 0.0):+.1f} s from perilune (+ = long)")
        rep.add("landing.site_crossrange_km", float(results["land_site_crossrange_km"]))
    # TEI ignition: the consumer's lunar orbit coasted through descent / stay / ascent / rendezvous
    state_cmp("tei_ignition", t["tei_state_at_tig"], t["tei_t_ignition_get_s"], ids=("dr", "dv"))
    if "tei_ignition" in rec:
        cmp["tei_ignition"]["csm_mass_kg"] = float(rec["tei_ignition"][1][6])
        cmp["tei_ignition"]["ref_csm_mass_kg"] = float(pkg["burn_table"][3]["mass_before_kg"])
        rep.add("tei_ignition.dmass_kg", cmp["tei_ignition"]["csm_mass_kg"] - cmp["tei_ignition"]["ref_csm_mass_kg"],
                "consumer sheds consumables to the ascent-insertion GET (the solver's CSM_CONSUMABLES_GET_REF_S)")
    # delivered TEI flown open-loop on the consumer's orbit
    state_cmp("tei_cutoff_open_loop", t["tei_state_post"], t["tei_t_ignition_get_s"] + t["tei_duration_s"],
              ids=("dr", "dv"))
    burn_cmp("tei_burn_open_loop", "tei_ignition", "tei_cutoff_open_loop", t["tei_state_at_tig"], t["tei_state_post"])
    exec_parity("tei", pkg["burn_table"][3])
    if "ei_open_loop" in rec:
        te, y = rec["ei_open_loop"]
        cmp["ei_open_loop"] = {"get_s": te, "fpa_deg": _fpa_deg(y), "dt_s": te - t["ei_get_s"]}
        rep.add("ei_open_loop.dfpa", _fpa_deg(y) - t["ei_fpa_deg"])
        rep.add("ei_open_loop.dt", te - t["ei_get_s"])
    # the TEI burn the consumer's corrector flew vs the delivered vector
    if "tei_cutoff" in rec and results.get("tei_dv_ms") is not None:
        ddv = float(results["tei_dv_ms"]) - t["tei_dv_mps"]
        cmp["tei_corrector"] = {"flown_dv_mps": float(results["tei_dv_ms"]), "delivered_dv_mps": t["tei_dv_mps"],
                                "source": results.get("tei_source"), "ignition_source": results.get("tei_ignition_source"),
                                "post_burn_dr_m": float(np.linalg.norm(rec["tei_cutoff"][1][:3]
                                                                       - np.asarray(t["tei_state_post"][:3]))),
                                "post_burn_dv_mps": float(np.linalg.norm(rec["tei_cutoff"][1][3:6]
                                                                         - np.asarray(t["tei_state_post"][3:6])))}
        rep.add("tei_corrector.ddv_mps", ddv)
    # entry interface after the trans-Earth MCC chain
    if "ei" in rec:
        te, y = rec["ei"]
        la, lo = A.eci_to_latlon(y[:3], te)
        cmp["ei"] = {"get_s": te, "ref_get_s": t["ei_get_s"], "fpa_deg": _fpa_deg(y), "ref_fpa_deg": t["ei_fpa_deg"],
                     "speed_mps": float(np.linalg.norm(y[3:6])), "lat_gc_deg": float(la), "lon_deg": float(lo),
                     "radius_m": float(np.linalg.norm(y[:3])),
                     "dr_m": float(np.linalg.norm(y[:3] - np.asarray(t["ei_state"][:3]))),
                     "transearth_mcc_total_dv_mps": results.get("mcc_total_dv_ms"),
                     "transearth_mcc_executed": results.get("temcc_n_exec")}
        rep.add("ei.dfpa", _fpa_deg(y) - t["ei_fpa_deg"])
        rep.add("ei.dt", te - t["ei_get_s"])
        rep.add("ei.dr", cmp["ei"]["dr_m"], "ECI; Earth rotation over ei.dt at the same ground point")
        rep.add("ei.dspeed", float(np.linalg.norm(y[3:6])) - t["ei_speed_mps"])
        rep.add("ei.dlat", float(la) - t["ei_lat_gc_deg"])
        rep.add("ei.dlon", _wrap(float(lo) - t["ei_lon_deg"]))
    else:
        cmp["ei"] = {"missing": True}
    # splashdown vs the recovery target
    if results.get("splash_lat") is not None:
        disp = _haversine_km(results["splash_lat"], results["splash_lon"], t["splash_lat_deg"], t["splash_lon_deg"])
        cmp["splash"] = {"lat": float(results["splash_lat"]), "lon": float(results["splash_lon"]),
                         "target_lat": t["splash_lat_deg"], "target_lon": t["splash_lon_deg"],
                         "target_displacement_km": disp,
                         "guidance_miss_km": results.get("splash_miss_km"),
                         "zone_lat": results.get("recovery_zone_lat"), "zone_lon": results.get("recovery_zone_lon"),
                         "zone_displacement_km": results.get("recovery_zone_displacement_km"),
                         "package_downrange_km_screened": t.get("downrange_km_screened"),
                         "package_crossrange_km_screened": t.get("crossrange_km_screened")}
        if results.get("splash_miss_km") is not None:
            rep.add("splash.guidance_miss_km", float(results["splash_miss_km"]), "vs the zone the consumer aims for")
        rep.add("splash.target_displacement_km", disp, "vs the package's screened Pacific target")
    else:
        cmp["splash"] = {"missing": True}

    reached = bool(results.get("full_success", False)) and results.get("splash_lat") is not None
    verdict = "FAIL" if (rep.hard_fails() or not reached) else ("PASS-WITH-WARNINGS" if rep.warns() else "PASS")
    report = {
        "package": os.path.abspath(package_path), "package_sha256": t["package_sha256"], "label": t["label"],
        "apollo11_sha256": CA.file_sha256(A.__file__) if hasattr(CA, "file_sha256") else None,
        "wall_s": wall, "verdict": verdict, "reached_splashdown": reached,
        "tight_coast": bool(getattr(A, "ENABLE_TIGHT_COAST", False)),
        "coast_settings": dict(getattr(A, "COAST_TIGHT_KW", {})) if getattr(A, "ENABLE_TIGHT_COAST", False) else "original",
        "mission_failure": results.get("mission_failure"),
        "sources": {"loi1": results.get("loi1_source"), "loi2": results.get("loi2_source"),
                    "tei_ignition": results.get("tei_ignition_source"), "tei": results.get("tei_source")},
        "flown": {"tli_dv_mps": results.get("tli_dv_ms"), "translunar_mcc_dv_mps": results.get("mcc_dv_ms"),
                  "loi1_dv_mps": results.get("loi_dv_ms"), "loi1_burn_s": results.get("loi_burn_time_s"),
                  "loi1_yaw_deg": results.get("loi1_yaw_deg"), "loi1_plane_change_deg": results.get("loi1_plane_change_deg"),
                  "loi1_site_xtrack_km": results.get("loi1_site_xtrack_km"),
                  "pdi_get_s": results.get("pdi_get_s"), "pdi_tig_offset_s": results.get("pdi_tig_offset_s"),
                  "land_lat_deg": results.get("land_lat_deg"), "land_lon_deg": results.get("land_lon_deg"),
                  "fuel_margin_s": results.get("fuel_margin_s"),
                  "loi2_dv_mps": results.get("loi2_circ_dv_ms"), "loi2_burn_s": results.get("loi2_burn_time_s"),
                  "csm_mass_at_tei_kg": results.get("csm_mass_at_tei_kg"),
                  "tei_dv_mps": results.get("tei_dv_ms"), "tei_burn_s": results.get("tei_burn_time_s"),
                  "tei_ignition_get_s": results.get("tei_ignition_t_s"),
                  "tei_delivered_open_loop": results.get("tei_delivered_open_loop"),
                  "transearth_mcc_total_dv_mps": results.get("mcc_total_dv_ms"),
                  "ei_fpa_deg": results.get("fpa_at_entry_deg"), "ei_speed_mps": results.get("entry_speed_ms")},
        "delivered": {"tli_dv_consumer_mps": pkg["burn_table"][0].get("dv_consumer_definition_mps"),
                      "loi1_dv_mps": t["loi1_dv_mps"], "loi1_burn_s": t["loi1_duration_s"],
                      "loi1_yaw_deg": t.get("loi1_yaw_deg"), "loi1_plane_change_deg": t.get("loi1_plane_change_deg"),
                      "loi2_dv_mps": t["loi2_dv_mps"], "loi2_burn_s": t["loi2_duration_s"],
                      "tei_dv_mps": t["tei_dv_mps"], "tei_burn_s": t["tei_duration_s"],
                      "tei_mass_before_kg": pkg["burn_table"][3]["mass_before_kg"],
                      "ei_fpa_deg": t["ei_fpa_deg"], "ei_speed_mps": t["ei_speed_mps"]},
        "milestones": cmp,
        "rows": [{"id": r[0], "value": r[1], "limit": r[2], "hard": r[3], "ok": r[4]} for r in rep.rows],
        "phase_log": results.get("_phase_log"),
        "bind_manifest": {k: manifest[k] for k in ("consumed_by_phases", "not_consumed_yet")},
    }
    return report, rep


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("package", nargs="?")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "refly_report.json"))
    ap.add_argument("--quiet", action="store_true", help="suppress the consumer's own progress output")
    args = ap.parse_args(argv)
    path = args.package or os.path.join(HERE, "nominal_packages", "apollo11_nominal_package.json")
    report, rep = refly(path, verbose=not args.quiet)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=float)
    print()
    print(rep.text())
    print(f"\nsources: {report['sources']}")
    print(f"flown: {json.dumps(report['flown'], default=float)}")
    print(f"REFLY VERDICT: {report['verdict']}  (wall {report['wall_s']:.0f} s; report {args.out})")
    return 0 if report["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
