"""Ingestion of the solver's delivered nominal package into the Apollo 11 simulation (apollo11.py, untouched).

    adapt_delivered_package(pkg) -> flat dict keyed by DELIVERED_TARGET_KEYS
    env_for_import(targets)      -> the APOLLO_* environment a consumer process must set BEFORE `import apollo11`
                                    (the consumer evaluates those constants, and some function defaults, at import)
    bind_delivered_targets(targets, module, strict=True) -> manifest: what was bound into the module's globals, what
                                    the phases do not read yet (the S3 phase wiring), and what stays env-only

Ownership: the package gives the pre-flight design (launch azimuth, insertion plane, TLI/LOI-1/LOI-2/TEI burns,
the EI aim). The simulation keeps its in-flight guidance and correctors (PEG ascent, MCCs, LOI re-targeting from a
dispersed arrival, TEI re-aim, entry guidance).

What binds today without touching apollo11.py (module globals the code already reads at call time):
  LAUNCH_AZIMUTH_DEG                 the IGM ascent reads it at call time
  _CACHED_LAUNCH_TLI, _PRESET_REF    _solve_launch_tli() returns the cached (ign_angle, vcut, pitch, yaw) and the
                                     drift guard compares the nominal insertion with the delivered one
  JD_LAUNCH, _GMST0, _PREC_J2000_TO_DATE, _PRECESS_MEMO, _MOON_AXES_CACHE   the epoch (reset together)
  LUNAR_DOI_GET_S, TEI_PLAN_GET_S, SITE_TARGET_LAT/LON_DEG, SITE_PLANE_EPOCH_S
  _EI_TARGET                         {lat, lon (geocentric, the consumer's eci_to_latlon), fpa, t_ei}
  _DELIVERED                         the whole flat dict, for the phases that will read the LOI/TEI burns (S3)
  SPLASH_TARGET_LAT/LON_DEG          run_mission reads them at call time and passes them to phase_entry
env_for_import() gives the APOLLO_* environment that makes the import-time constants equal the delivered ones
for a fresh worker process (spawn workers re-import the module).
"""
import json
import math
import os

import numpy as np

CONSUMER_JD_LAUNCH_LITERAL = 2440419.0639          # apollo11.py JD_LAUNCH at APOLLO_LAUNCH_DELAY_H = 0

DELIVERED_TARGET_KEYS = frozenset({
    "label", "schema", "package_sha256",
    "liftoff_utc", "liftoff_jd", "launch_delay_h",
    "launch_azimuth_deg", "azimuth_convention",
    "insertion_get_s", "insertion_state", "insertion_mass_kg", "insertion_h_hat", "insertion_inclination_deg", "insertion_raan_deg",
    "tli_ign_angle_deg", "tli_vcut_mps", "tli_pitch_deg", "tli_yaw_deg", "tli_t_ignition_get_s", "tli_t_cutoff_get_s",
    "tli_duration_s", "tli_dv_mps", "tli_cutoff_rule", "tli_state_at_tig", "tli_state_post",
    "loi1_t_ignition_get_s", "loi1_duration_s", "loi1_dv_mps", "loi1_tilt_deg", "loi1_yaw_deg", "loi1_plane_change_deg",
    "loi1_dir_eci", "loi1_state_at_tig", "loi1_state_post",
    "tli_free_return_targeted", "tli_free_return_ei_fpa_deg", "tli_site_crosstrack_km",
    "loi2_t_ignition_get_s", "loi2_duration_s", "loi2_dv_mps", "loi2_tilt_deg", "loi2_dir_eci", "loi2_state_at_tig", "loi2_state_post",
    "tei_t_ignition_get_s", "tei_duration_s", "tei_dv_mps", "tei_dv_ric_mps", "tei_dir_eci", "tei_state_at_tig", "tei_state_post",
    "perilune_alt_m", "perilune_get_s", "perilune_state",
    "lunar_orbit_perilune_alt_m", "lunar_orbit_apolune_alt_m", "lunar_orbit_inclination_deg", "lunar_orbit_period_s",
    "site_lat_deg", "site_lon_deg", "site_plane_epoch_s", "doi_get_s", "tei_plan_get_s",
    "ei_get_s", "ei_fpa_deg", "ei_lat_gd_deg", "ei_lat_gc_deg", "ei_lon_deg", "ei_speed_mps", "ei_ground_azimuth_deg", "ei_state",
    "splash_lat_deg", "splash_lon_deg", "downrange_km_screened", "crossrange_km_screened",
})


def load_package(path):
    def bad_constant(c):
        raise ValueError(f"non-finite JSON constant {c!r} in {path}")
    with open(path) as f:
        return json.load(f, parse_constant=bad_constant)


def _state(node):
    return [float(v) for v in node["r_m"]] + [float(v) for v in node["v_mps"]]


def adapt_delivered_package(pkg):
    """Nested package -> flat targets. Raises KeyError on a missing block (the structural gate says which)."""
    b = {x["name"]: x for x in pkg["burn_table"]}
    n = pkg["trajectory_nodes"]
    lb = pkg["launch_block"]
    ip = pkg["insertion_plane"]
    tg = pkg["targets"]
    ei = n["ei"]
    t = {
        "label": pkg["label"], "schema": pkg.get("schema"), "package_sha256": pkg["manifest"]["package_sha256"],
        "liftoff_utc": lb["liftoff_utc"], "liftoff_jd": float(lb["liftoff_jd"]),
        "launch_delay_h": (float(lb["liftoff_jd"]) - CONSUMER_JD_LAUNCH_LITERAL) * 24.0,
        "launch_azimuth_deg": float(lb["azimuth_deg"]), "azimuth_convention": lb.get("azimuth_convention"),
        "insertion_get_s": float(n["insertion"]["get_s"]),
        "insertion_state": _state(n["insertion"]) + [float(n["insertion"].get("mass_kg", lb.get("insertion_mass_kg")))],
        "insertion_mass_kg": float(lb["insertion_mass_kg"]),
        "insertion_h_hat": [float(v) for v in ip["h_hat"]],
        "insertion_inclination_deg": float(ip["inclination_deg"]), "insertion_raan_deg": float(ip["raan_deg"]),
        "tli_ign_angle_deg": float(b["TLI"]["ignition_in_plane_angle_deg"]), "tli_vcut_mps": float(b["TLI"]["cutoff_value"]),
        "tli_pitch_deg": float(b["TLI"]["steer_pitch_deg"]), "tli_yaw_deg": float(b["TLI"]["steer_yaw_deg"]),
        "tli_t_ignition_get_s": float(b["TLI"]["t_ignition_get_s"]), "tli_t_cutoff_get_s": float(b["TLI"]["t_cutoff_get_s"]),
        "tli_duration_s": float(b["TLI"]["duration_s"]), "tli_dv_mps": float(b["TLI"]["dv_mag_mps"]),
        "tli_cutoff_rule": b["TLI"]["cutoff_rule"],
        "tli_state_at_tig": _state(b["TLI"]["state_at_tig"]), "tli_state_post": _state(b["TLI"]["state_post"]),
        "perilune_alt_m": float(n["perilune"]["altitude_m"]), "perilune_get_s": float(n["perilune"]["get_s"]),
        "perilune_state": _state(n["perilune"]),
        "lunar_orbit_perilune_alt_m": float(n["lunar_orbit"]["perilune_alt_m"]),
        "lunar_orbit_apolune_alt_m": float(n["lunar_orbit"]["apolune_alt_m"]),
        "lunar_orbit_inclination_deg": float(n["lunar_orbit"]["inclination_deg"]),
        "lunar_orbit_period_s": float(n["lunar_orbit"].get("period_s", float("nan"))),
        # None for a package that targets no site (natural landing, e.g. the Jul-23 stress test)
        "site_lat_deg": None if tg["landing_site"]["lat_deg"] is None else float(tg["landing_site"]["lat_deg"]),
        "site_lon_deg": None if tg["landing_site"]["lon_deg"] is None else float(tg["landing_site"]["lon_deg"]),
        "site_plane_epoch_s": float(tg["site_plane_epoch_get_s"]), "doi_get_s": float(tg["doi_get_s"]),
        "tei_plan_get_s": float(tg["tei_plan_get_s"]),
        "ei_get_s": float(ei["get_s"]), "ei_fpa_deg": float(ei["fpa_deg"]),
        "ei_lat_gd_deg": float(ei["subpoint_lat_gd_deg"]), "ei_lat_gc_deg": float(ei["subpoint_lat_gc_deg"]),
        "ei_lon_deg": float(ei["subpoint_lon_deg"]), "ei_speed_mps": float(ei["speed_mps"]),
        "ei_ground_azimuth_deg": float(ei["ground_azimuth_deg"]),
        "ei_state": [float(v) for v in ei["r_m"]] + [float(v) for v in ei["v_mps"]],
        "splash_lat_deg": float(tg["recovery_target"]["lat_deg"]), "splash_lon_deg": float(tg["recovery_target"]["lon_deg"]),
        "downrange_km_screened": float(ei["downrange_to_zone_km"]), "crossrange_km_screened": float(ei["crossrange_to_zone_km"]),
    }
    for name, key in (("LOI-1", "loi1"), ("LOI-2", "loi2")):
        x = b[name]
        t.update({f"{key}_t_ignition_get_s": float(x["t_ignition_get_s"]), f"{key}_duration_s": float(x["duration_s"]),
                  f"{key}_dv_mps": float(x["dv_mag_mps"]), f"{key}_tilt_deg": float(x["tilt_deg"]),
                  f"{key}_dir_eci": [float(v) for v in x["thrust_dir_eci"]],
                  f"{key}_state_at_tig": _state(x["state_at_tig"]), f"{key}_state_post": _state(x["state_post"])})
    # LOI-1 plane change (the Site 2 plane when the TLI is a free return); 0 / None for an in-plane package
    t["loi1_yaw_deg"] = float(b["LOI-1"].get("yaw_deg", 0.0))
    t["loi1_plane_change_deg"] = (None if b["LOI-1"].get("plane_change_deg") is None else float(b["LOI-1"]["plane_change_deg"]))
    scr = b["TLI"].get("screened", {})
    t["tli_free_return_targeted"] = bool(scr.get("free_return_targeted", False))
    t["tli_free_return_ei_fpa_deg"] = next((float(g["achieved"]) for g in b["TLI"]["goals"] if g["name"] == "free_return_ei_fpa_deg"), None)
    _xt = scr.get("site_crosstrack_at_site_epoch_km",
                  next((g["achieved"] for g in b["TLI"]["goals"] if g["name"] == "site_crosstrack_at_pdi_km"), 0.0))
    t["tli_site_crosstrack_km"] = float("nan") if _xt is None else float(_xt)   # None: no site (natural landing)
    x = b["TEI"]
    t.update({"tei_t_ignition_get_s": float(x["t_ignition_get_s"]), "tei_duration_s": float(x["duration_s"]),
              "tei_dv_mps": float(x["dv_mag_mps"]),
              "tei_dv_ric_mps": [float(x["dv_ric_mps"][k]) for k in ("along_track", "radial", "normal")],
              "tei_dir_eci": [float(v) for v in x["thrust_dir_eci"]],
              "tei_state_at_tig": _state(x["state_at_tig"]), "tei_state_post": _state(x["state_post"])})
    missing = DELIVERED_TARGET_KEYS - set(t)
    extra = set(t) - DELIVERED_TARGET_KEYS
    if missing or extra:
        raise KeyError(f"adapter key set drifted: missing {sorted(missing)}, extra {sorted(extra)}")
    return t


def env_for_import(targets):
    """The environment that makes apollo11.py's import-time constants equal the delivered ones. A package with no
    landing site switches off everything that aims at one (APOLLO_ASPLANNED_SITE and the LOI-1 site-plane, PDI site
    and DOI site re-targeting): the descent lands where the orbit passes at the DOI GET."""
    env = {
        "APOLLO_JPL_EPHEMERIS": "1",
        "APOLLO_LAUNCH_DELAY_H": repr(targets["launch_delay_h"]),
        "APOLLO_LAUNCH_AZ_DEG": repr(targets["launch_azimuth_deg"]),
        "APOLLO_DOI_GET_S": repr(targets["doi_get_s"]),
        "APOLLO_SPLASH_LAT": repr(targets["splash_lat_deg"]),
        "APOLLO_SPLASH_LON": repr(targets["splash_lon_deg"]),
    }
    if targets["site_lat_deg"] is None:
        env.update({"APOLLO_ASPLANNED_SITE": "0", "APOLLO_LOI1_SITE_PLANE": "0", "APOLLO_PDI_SITE_TARGET": "0",
                    "APOLLO_DOI_SITE_RETARGET": "0"})
    else:
        env.update({"APOLLO_SITE_LAT": repr(targets["site_lat_deg"]), "APOLLO_SITE_LON": repr(targets["site_lon_deg"])})
    return env


BOUND_AT_CALL_TIME = ("LAUNCH_AZIMUTH_DEG", "_CACHED_LAUNCH_TLI", "_PRESET_REF", "JD_LAUNCH", "_GMST0", "_PREC_J2000_TO_DATE",
                      "LUNAR_DOI_GET_S", "TEI_PLAN_GET_S", "SITE_TARGET_LAT_DEG", "SITE_TARGET_LON_DEG", "SITE_PLANE_EPOCH_S",
                      "SPLASH_TARGET_LAT_DEG", "SPLASH_TARGET_LON_DEG", "_EI_TARGET", "_DELIVERED")
IMPORT_TIME_ONLY = ()      # run_mission reads SPLASH_TARGET_* at call time and passes them to phase_entry explicitly
NOT_CONSUMED_YET = ("tli_duration_s (the consumer cuts on |v| = tli_vcut_mps)",
                    "insertion_h_hat/inclination/raan (the consumer's PEG flies the azimuth; the plane is checked, not flown)",
                    "tei_dv_ric_mps (diagnostic; the frozen tei_dir_eci and tei_dv_mps are flown)",
                    "*_state_at_tig / *_state_post / *_duration_s (parity references for refly_delivered_nominal.py)")
# Delivered burns consumed by apollo11.run_mission through _DELIVERED (sim plan S3):
#   loi1_*: nominal -> flown open-loop at the delivered epoch (dv, tilt, frozen dir_eci);
#           dispersed -> 2-DOF (dv, tilt) re-target from the delivered seed at the delivered lead before perilune
#   loi2_*: nominal -> coast to the delivered epoch, flown open-loop; dispersed -> second-perilune 2-DOF from the seed
#   tei_*:  ignition at the delivered epoch on the flown orbit, delivered direction and dv as the seed of the
#           consumer's 3-DOF EI refinement (block-data practice); scan fallback when the epoch is outside the window
CONSUMED_BY_PHASES = ("loi1_t_ignition_get_s", "loi1_dv_mps", "loi1_tilt_deg", "loi1_yaw_deg", "loi1_dir_eci",
                      "loi2_t_ignition_get_s", "loi2_dv_mps", "loi2_tilt_deg", "loi2_dir_eci",
                      "tei_t_ignition_get_s", "tei_dv_mps", "tei_dir_eci", "perilune_get_s")


def bind_delivered_targets(targets, module, strict=True):
    """Bind the flat targets into the imported apollo11 module. Returns the manifest. With strict, refuses an
    epoch the module cannot represent exactly (JD_LAUNCH must equal the delivered liftoff_jd after binding) and
    an unknown key set."""
    unknown = set(targets) - DELIVERED_TARGET_KEYS
    if unknown and strict:
        raise KeyError(f"unknown delivered keys: {sorted(unknown)}")
    if strict and not callable(getattr(module, "_delivered_burn", None)):
        raise AttributeError("module has no delivered-burn hooks (_delivered_burn); a verbatim apollo11.py "
                             "would ignore the delivered LOI/TEI burns")
    snapshot = {k: getattr(module, k, None) for k in BOUND_AT_CALL_TIME}
    # epoch, reset together (the consumer's _solve_launch_plane._set_epoch leaves _PREC_J2000_TO_DATE stale)
    module.JD_LAUNCH = float(targets["liftoff_jd"])
    module._GMST0 = module._gmst_rad(module.JD_LAUNCH) if module.ENABLE_REAL_EPHEMERIS else 0.0
    module._PREC_J2000_TO_DATE = (module._precession_j2000_to_date(module.JD_LAUNCH) if module.ENABLE_REAL_EPHEMERIS
                                  else np.eye(3))
    module._PRECESS_MEMO[0] = None
    module._MOON_AXES_CACHE[0] = None
    module.LAUNCH_AZIMUTH_DEG = float(targets["launch_azimuth_deg"])
    module._CACHED_LAUNCH_TLI = (float(targets["tli_ign_angle_deg"]), float(targets["tli_vcut_mps"]),
                                 float(targets["tli_pitch_deg"]), float(targets["tli_yaw_deg"]))
    module._PRESET_REF = {"state": [float(v) for v in targets["insertion_state"]], "t_ins": float(targets["insertion_get_s"])}
    module.LUNAR_DOI_GET_S = float(targets["doi_get_s"])
    module.TEI_PLAN_GET_S = float(targets["tei_plan_get_s"])
    if targets["site_lat_deg"] is not None:          # no site: the defaults stay, ENABLE_ASPLANNED_SITE is off
        module.SITE_TARGET_LAT_DEG = float(targets["site_lat_deg"])
        module.SITE_TARGET_LON_DEG = float(targets["site_lon_deg"])
    module.SITE_PLANE_EPOCH_S = float(targets["site_plane_epoch_s"])
    module.SPLASH_TARGET_LAT_DEG = float(targets["splash_lat_deg"])
    module.SPLASH_TARGET_LON_DEG = float(targets["splash_lon_deg"])
    module._EI_TARGET = {"lat": float(targets["ei_lat_gc_deg"]), "lon": float(targets["ei_lon_deg"]),
                         "fpa": float(targets["ei_fpa_deg"]), "t_ei": float(targets["ei_get_s"])}
    module._DELIVERED = dict(targets)
    env = env_for_import(targets)
    if strict and abs(module.JD_LAUNCH - targets["liftoff_jd"]) > 1e-12:
        raise ValueError("JD_LAUNCH did not bind exactly")
    manifest = {
        "label": targets["label"], "package_sha256": targets["package_sha256"],
        "bound_at_call_time": {k: ("set" if k != "_DELIVERED" else f"{len(targets)} keys") for k in BOUND_AT_CALL_TIME},
        "previous_values": {k: (None if v is None or k in ("_DELIVERED", "_PRESET_REF") else
                                (list(v) if isinstance(v, (tuple, list)) else (v.tolist() if hasattr(v, "tolist") else v)))
                            for k, v in snapshot.items()},
        "import_time_only": {k: getattr(module, k, None) for k in IMPORT_TIME_ONLY},
        "env_for_import": env,
        "not_consumed_yet": list(NOT_CONSUMED_YET),
        "consumed_by_phases": list(CONSUMED_BY_PHASES),
        "note": "TLI, azimuth, epoch, DOI GET, TEI plan GET, site and EI target are live globals; LOI-1/LOI-2/TEI "
                "burns are read by run_mission from apollo11._DELIVERED (sim plan S3).",
    }
    module._DELIVERED_MANIFEST = manifest
    return manifest


def bind_package(path, module, strict=True):
    pkg = load_package(path)
    targets = adapt_delivered_package(pkg)
    return targets, bind_delivered_targets(targets, module, strict=strict)
