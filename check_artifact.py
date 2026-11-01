"""STRUCTURAL check of a delivered nominal package (apollo11_nominal_package.json): JSON hygiene, blocks, keys,
types, unit vectors, event order, provenance tokens, hash manifest. Cheapest tier; nothing physical.

    python3 check_artifact.py <package.json> [more.json ...]     exit 0 = PASS (no FAIL), 1 = FAIL

Every check has an id (A1..A11) and can fail; check_delivered_nominal.py --self-test injects a defect for each.
The report ends with what this file CANNOT check.
"""
import hashlib
import json
import math
import sys
from datetime import datetime, timezone

REQUIRED_BLOCKS = ("schema", "label", "built_utc", "trajectory_span", "epoch", "launch_block", "insertion_plane", "burn_table",
                   "trajectory_nodes", "targets", "provenance", "disclosures", "manifest", "free_return_screen", "ei_screen", "ledgers")
LAUNCH_KEYS = ("liftoff_utc", "liftoff_jd", "pad_lat_deg", "pad_lon_deg", "azimuth_deg", "azimuth_convention", "meco_get_s",
               "meco_altitude_m", "parking_perigee_m", "parking_apogee_m", "insertion_mass_kg")
PLANE_KEYS = ("h_hat", "inclination_deg", "raan_deg", "frame", "defining_epoch_get_s", "defining_epoch_jd", "defining_model",
              "pad_datum", "pad_lat_deg", "pad_lon_deg", "azimuth_convention")
BURN_NAMES = ("TLI", "LOI-1", "LOI-2", "TEI")
BURN_KEYS = ("stage", "epoch_convention", "t_ignition_get_s", "t_cutoff_get_s", "duration_s", "dv_mag_mps", "thrust_n", "isp_s",
             "mass_before_kg", "mass_after_kg", "state_at_tig", "state_post", "goals")
EPOCH_CONVENTIONS = {"ignition", "mid_burn", "midburn", "centroid", "cutoff", "impulsive_equivalent", "impulsive"}
NODE_KEYS = ("insertion", "tli_ignition", "tli_cutoff", "perilune", "loi1_cutoff", "loi2_cutoff", "lunar_orbit", "pdi_epoch",
             "tei_ignition", "tei_cutoff", "ei")
EI_KEYS = ("get_s", "radius_m", "speed_mps", "fpa_deg", "subpoint_lat_deg", "subpoint_lon_deg", "ground_azimuth_deg",
           "downrange_to_zone_km", "crossrange_to_zone_km", "r_m", "v_mps", "earth_prograde", "entry_track_inclination_deg")
TARGET_KEYS = ("perilune_alt_m", "perilune_get_s", "landing_site", "site_plane_epoch_get_s", "doi_get_s", "tei_plan_get_s", "ei",
               "recovery_target")
SPEC_CLASSES = {"primary", "secondary", "derived", "measured", "estimate"}
NODE_TAGS = {"AUTHORITATIVE", "REFERENCE"}
EVENT_ORDER = (("launch_block", "meco_get_s"), ("burn", "TLI", "t_ignition_get_s"), ("burn", "TLI", "t_cutoff_get_s"),
               ("burn", "LOI-1", "t_ignition_get_s"), ("node", "perilune"), ("burn", "LOI-1", "t_cutoff_get_s"),
               ("burn", "LOI-2", "t_ignition_get_s"), ("burn", "LOI-2", "t_cutoff_get_s"), ("node", "pdi_epoch"),
               ("burn", "TEI", "t_ignition_get_s"), ("burn", "TEI", "t_cutoff_get_s"), ("node", "ei"))


class Report:
    def __init__(self, label):
        self.label = label
        self.findings = []            # (id, level, message)
        self.cannot = []

    def fail(self, cid, msg):
        self.findings.append((cid, "FAIL", msg))

    def warn(self, cid, msg):
        self.findings.append((cid, "WARN", msg))

    def ok(self, cid, msg):
        self.findings.append((cid, "PASS", msg))

    @property
    def fails(self):
        return [f for f in self.findings if f[1] == "FAIL"]

    @property
    def fail_ids(self):
        return {f[0] for f in self.fails}

    def text(self):
        lines = [f"check_artifact: {self.label}"]
        for cid, lvl, msg in self.findings:
            if lvl != "PASS":
                lines.append(f"  {lvl:4s} {cid:<14s} {msg}")
        n_pass = sum(1 for f in self.findings if f[1] == "PASS")
        lines.append(f"  {n_pass} checks passed, {len(self.fails)} FAIL, {sum(1 for f in self.findings if f[1] == 'WARN')} WARN")
        lines.append("  NOT CHECKED here: " + "; ".join(self.cannot))
        return "\n".join(lines)


def load(path):
    def bad(c):
        raise ValueError(f"non-finite JSON constant {c}")
    with open(path) as f:
        return json.load(f, parse_constant=bad)


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def package_sha256(pkg):
    clone = json.loads(json.dumps(pkg))
    clone["manifest"]["package_sha256"] = None
    return hashlib.sha256(canonical_json(clone).encode()).hexdigest()


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _vec3(x):
    return isinstance(x, list) and len(x) == 3 and all(_num(v) for v in x)


def _unit(x, tol=1e-9):
    return _vec3(x) and abs(math.sqrt(sum(v * v for v in x)) - 1.0) < tol


def _jd_from_iso(s):
    s = str(s).strip().replace("Z", "").replace(" ", "T")
    dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return 2440587.5 + (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0


def _prov_token(s):
    """'[primary] NTRS Table 2' -> 'primary'; the bracket is parsed, not stripped (the Artemis gate's lesson)."""
    s = str(s).strip()
    if not s.startswith("["):
        return None
    return s[1:s.index("]")].split("/")[0].strip().lower() if "]" in s else None


def check(doc, label="package"):
    r = Report(label)
    # A1 blocks
    missing = [b for b in REQUIRED_BLOCKS if b not in doc]
    if missing:
        r.fail("A1.blocks", f"missing top-level blocks: {missing}")
        r.cannot.append("everything below a missing block")
        return r
    r.ok("A1.blocks", "all required blocks present")
    if doc.get("trajectory_span") != "launch_to_ei":
        r.fail("A1.span", f"trajectory_span {doc.get('trajectory_span')!r} != 'launch_to_ei'")
    if not isinstance(doc["disclosures"], list) or not doc["disclosures"]:
        r.fail("A1.disclosures", "disclosures missing or empty")
    # A2 launch block
    lb = doc["launch_block"]
    bad = [k for k in LAUNCH_KEYS if k not in lb]
    if bad:
        r.fail("A2.launch_keys", f"launch_block missing {bad}")
    else:
        nums = [k for k in LAUNCH_KEYS if k not in ("liftoff_utc", "azimuth_convention") and not _num(lb[k])]
        if nums:
            r.fail("A2.launch_types", f"launch_block non-finite: {nums}")
        else:
            r.ok("A2.launch", "launch_block keys and types")
        try:
            dt_s = abs(_jd_from_iso(lb["liftoff_utc"]) - float(lb["liftoff_jd"])) * 86400.0
            if dt_s > 0.5:
                r.fail("A2.liftoff_jd", f"liftoff_utc vs liftoff_jd differ by {dt_s:.3f} s (> 0.5 s)")
            else:
                r.ok("A2.liftoff_jd", f"liftoff_utc and liftoff_jd agree to {dt_s * 1000:.1f} ms")
        except Exception as e:
            r.fail("A2.liftoff_utc", f"liftoff_utc unparsable: {e}")
    # A3 insertion plane
    ip = doc["insertion_plane"]
    bad = [k for k in PLANE_KEYS if k not in ip]
    if bad:
        r.fail("A3.plane_keys", f"insertion_plane missing {bad}")
    else:
        if not _unit(ip["h_hat"]):
            r.fail("A3.h_hat", f"insertion_plane.h_hat not a unit 3-vector: {ip['h_hat']}")
        elif abs(math.degrees(math.acos(max(-1.0, min(1.0, ip["h_hat"][2])))) - float(ip["inclination_deg"])) > 1e-6:
            r.fail("A3.inclination", "inclination_deg inconsistent with h_hat")
        else:
            r.ok("A3.plane", "insertion_plane unit normal, inclination consistent, frame and epoch stated")
    # A4 burn table
    bt = doc["burn_table"]
    names = [b.get("name") for b in bt] if isinstance(bt, list) else []
    if tuple(names) != BURN_NAMES:
        r.fail("A4.burn_names", f"burn_table names {names} != {list(BURN_NAMES)}")
    else:
        for b in bt:
            n = b["name"]
            bad = [k for k in BURN_KEYS if k not in b]
            if bad:
                r.fail("A4.burn_keys", f"{n}: missing {bad}")
                continue
            if b["epoch_convention"] not in EPOCH_CONVENTIONS:
                r.fail("A4.epoch_convention", f"{n}: epoch_convention {b['epoch_convention']!r} not in {sorted(EPOCH_CONVENTIONS)}")
            for k in ("t_ignition_get_s", "t_cutoff_get_s", "duration_s", "dv_mag_mps", "mass_before_kg", "mass_after_kg"):
                if not _num(b[k]):
                    r.fail("A4.burn_types", f"{n}: {k} non-finite")
            if _num(b["t_ignition_get_s"]) and _num(b["t_cutoff_get_s"]) and _num(b["duration_s"]):
                if b["t_cutoff_get_s"] <= b["t_ignition_get_s"]:
                    r.fail("A4.burn_order", f"{n}: cutoff {b['t_cutoff_get_s']} <= ignition {b['t_ignition_get_s']}")
                elif abs(b["t_cutoff_get_s"] - b["t_ignition_get_s"] - b["duration_s"]) > 1e-3:
                    r.fail("A4.burn_duration", f"{n}: duration {b['duration_s']} != cutoff - ignition")
            if _num(b["dv_mag_mps"]) and b["dv_mag_mps"] <= 0.0:
                r.fail("A4.dv_positive", f"{n}: dv_mag_mps {b['dv_mag_mps']} <= 0")
            if _num(b["mass_before_kg"]) and _num(b["mass_after_kg"]) and not (0.0 < b["mass_after_kg"] < b["mass_before_kg"]):
                r.fail("A4.mass_order", f"{n}: mass_after {b['mass_after_kg']} not below mass_before {b['mass_before_kg']}")
            for sk in ("state_at_tig", "state_post"):
                st = b[sk]
                if not (isinstance(st, dict) and _vec3(st.get("r_m")) and _vec3(st.get("v_mps")) and _num(st.get("get_s")) and st.get("frame")):
                    r.fail("A4.burn_state", f"{n}: {sk} lacks finite r_m/v_mps/get_s or a frame")
            if n != "TLI":
                if not _unit(b.get("thrust_dir_eci")):
                    r.fail("A4.thrust_dir", f"{n}: thrust_dir_eci not a unit 3-vector")
                if not b.get("thrust_dir_frame"):
                    r.fail("A4.thrust_frame", f"{n}: thrust_dir_frame missing")
            else:
                for k in ("cutoff_rule", "cutoff_value", "steer_pitch_deg", "steer_yaw_deg", "ignition_in_plane_angle_deg"):
                    if k not in b:
                        r.fail("A4.tli_keys", f"TLI: missing {k}")
            if not isinstance(b["goals"], list) or not b["goals"]:
                r.fail("A4.goals", f"{n}: goals missing")
            else:
                for g in b["goals"]:
                    if not all(_num(g.get(k)) for k in ("target", "tolerance", "achieved")):
                        r.fail("A4.goal_types", f"{n}: goal {g.get('name')} lacks finite target/tolerance/achieved")
        if not r.fail_ids & {c for c in r.fail_ids if c.startswith("A4")}:
            r.ok("A4.burns", "burn_table complete: conventions, states, directions, goals")
    # A5 nodes
    nodes = doc["trajectory_nodes"]
    bad = [k for k in NODE_KEYS if k not in nodes]
    if bad:
        r.fail("A5.node_keys", f"trajectory_nodes missing {bad}")
    else:
        for k, nd in nodes.items():
            if not isinstance(nd, dict):
                r.fail("A5.node_type", f"node {k} is not an object")
                continue
            if nd.get("tag") not in NODE_TAGS:
                r.fail("A5.node_tag", f"node {k}: tag {nd.get('tag')!r} not in {sorted(NODE_TAGS)}")
            if "r_m" in nd or "v_mps" in nd:
                if not (_vec3(nd.get("r_m")) and _vec3(nd.get("v_mps")) and _num(nd.get("get_s"))):
                    r.fail("A5.node_state", f"node {k}: r_m/v_mps/get_s not finite 3-vectors + epoch")
        ei = nodes["ei"]
        bad = [k for k in EI_KEYS if k not in ei]
        if bad:
            r.fail("A5.ei_keys", f"ei node missing {bad}")
        elif not all(_num(ei[k]) for k in ("get_s", "radius_m", "speed_mps", "fpa_deg", "subpoint_lat_deg", "subpoint_lon_deg")):
            r.fail("A5.ei_types", "ei node has non-finite numbers")
        if not r.fail_ids & {c for c in r.fail_ids if c.startswith("A5")}:
            r.ok("A5.nodes", "trajectory nodes tagged, finite, EI block complete")
    # A6 targets
    tg = doc["targets"]
    bad = [k for k in TARGET_KEYS if k not in tg]
    if bad:
        r.fail("A6.target_keys", f"targets missing {bad}")
    else:
        rt = tg["recovery_target"]
        if not all(k in rt for k in ("lat_deg", "lon_deg", "convention", "downrange_km", "crossrange_km")):
            r.fail("A6.recovery_keys", "recovery_target incomplete")
        elif rt["convention"] not in ("geodetic", "geocentric"):
            r.fail("A6.recovery_convention", f"recovery_target.convention {rt['convention']!r}")
        else:
            r.ok("A6.targets", "targets complete")
    # A7 provenance
    pv = doc["provenance"]
    classes = pv.get("classes", {})
    if not SPEC_CLASSES <= set(classes):
        r.fail("A7.classes", f"provenance.classes lacks the RFN v1.4 set {sorted(SPEC_CLASSES - set(classes))}")
    unknown = [(k, v) for k, v in pv.get("quantities", {}).items() if _prov_token(v) not in classes]
    if unknown:
        r.fail("A7.tokens", f"provenance tokens not in the declared classes: {[k for k, _ in unknown][:5]}")
    elif pv.get("quantities"):
        r.ok("A7.provenance", f"{len(pv['quantities'])} quantities carry declared classes")
    # A8 manifest
    man = doc["manifest"]
    if not isinstance(man.get("package_sha256"), str):
        r.fail("A8.hash_missing", "manifest.package_sha256 missing")
    else:
        got = package_sha256(doc)
        if got != man["package_sha256"]:
            r.fail("A8.hash", f"package_sha256 {man['package_sha256'][:16]} != recomputed {got[:16]} (edited after signing?)")
        else:
            r.ok("A8.hash", "package_sha256 verified over the canonical JSON")
    for k in ("inputs_sha256", "kernels_sha256"):
        if not isinstance(man.get(k), dict) or not man[k]:
            r.fail("A8.manifest_inputs", f"manifest.{k} missing or empty")
    # A9 event order
    try:
        burns = {b["name"]: b for b in doc["burn_table"]}
        seq = []
        for spec in EVENT_ORDER:
            if spec[0] == "launch_block":
                seq.append((spec[1], float(lb[spec[1]])))
            elif spec[0] == "burn":
                seq.append((f"{spec[1]}.{spec[2]}", float(burns[spec[1]][spec[2]])))
            else:
                seq.append((spec[1], float(nodes[spec[1]]["get_s"])))
        bad = [(a[0], b[0]) for a, b in zip(seq, seq[1:]) if not a[1] < b[1]]
        if bad:
            r.fail("A9.order", f"event order violated at {bad}")
        else:
            r.ok("A9.order", "MECO < TLI < LOI-1 ign < perilune < LOI-1 cut < LOI-2 < PDI epoch < TEI < EI")
    except Exception as e:
        r.fail("A9.order", f"could not evaluate the event order: {e}")
    # A10 epoch block
    ep = doc["epoch"]
    if not (_num(ep.get("liftoff_jd")) and ep.get("frame") and _num(ep.get("gmst0_rad"))):
        r.fail("A10.epoch", "epoch block lacks liftoff_jd / frame / gmst0_rad")
    elif abs(float(ep["liftoff_jd"]) - float(lb.get("liftoff_jd", float("nan")))) > 1e-9:
        r.fail("A10.epoch", "epoch.liftoff_jd != launch_block.liftoff_jd")
    else:
        r.ok("A10.epoch", "epoch block consistent with the launch block")
    # A11 screens present and boolean
    for blk, keys in (("free_return_screen", ("gamma_in_corridor", "retrograde_at_moon", "keepout_violated")),
                      ("ei_screen", ("fpa_in_corridor", "earth_prograde"))):
        for k in keys:
            if not isinstance(doc[blk].get(k), bool):
                r.fail("A11.screens", f"{blk}.{k} missing or not boolean")
    if "A11.screens" not in r.fail_ids:
        r.ok("A11.screens", "screens carry boolean verdicts")
    r.cannot += ["physics (ephemeris parity, rocket equation, corridor, coast consistency): check_delivered_nominal.py",
                 "whether the consumer's launch reproduces the insertion: the deterministic re-fly"]
    return r


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    worst = 0
    for path in argv:
        try:
            doc = load(path)
        except Exception as e:
            print(f"check_artifact: {path}: FAIL A0.json {e}")
            worst = 1
            continue
        r = check(doc, label=f"{path} ({doc.get('label')})")
        print(r.text())
        if r.fails:
            worst = 1
    print("VERDICT:", "FAIL" if worst else "PASS")
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
