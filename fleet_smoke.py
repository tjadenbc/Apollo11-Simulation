"""Pool smoke test (sim plan S5): fly trials of a pinned fleet outdir on spawned workers and check every worker flew
the bound package (delivered LOI-1 3-DOF seed, PDI site targeting, the package's EI target).

    APOLLO_DELIVERED_PACKAGE=<pkg> python fleet_smoke.py <pinned_outdir> <smoke_outdir> [n=25] [seed=42] [trials=0,1]

Copies the pinned nominal artifacts into <smoke_outdir> (so the nominal is not re-derived) and runs
main_parallel(indices=trials, workers=len(trials)). Exit 1 on any trial that did not fly the package."""
import os
import shutil
import sys

PINNED = ("ei_target.json", "bplane_target.json", "ca_target.json", "nominal_results.json", "od_cov.json",
          "run_config.json", "delivered_package.json")


def main():
    pinned, smoke = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    trials = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "0,1").split(",")]
    if not os.environ.get("APOLLO_DELIVERED_PACKAGE"):
        raise SystemExit("set APOLLO_DELIVERED_PACKAGE")
    os.makedirs(smoke, exist_ok=True)
    for f in PINNED:
        if os.path.exists(os.path.join(pinned, f)) and not os.path.exists(os.path.join(smoke, f)):
            shutil.copy2(os.path.join(pinned, f), os.path.join(smoke, f))
    import json
    import apollo11 as A
    d = A._DELIVERED
    assert d is not None, "apollo11 imported unbound"
    ei_pin = json.load(open(os.path.join(smoke, "ei_target.json")))
    ei_pkg = A._delivered_ei_target()
    bad = [k for k in ei_pkg if abs(float(ei_pin[k]) - float(ei_pkg[k])) > 1e-9]
    df, _, _ = A.main_parallel(n=n, outdir=smoke, seed=seed, workers=len(trials), indices=trials)
    fails = []
    for _, r in df.iterrows():
        # bound <=> the LOI-1 re-solve was seeded from the package (only _DELIVERED provides that source) and the
        # descent went through PDI site targeting; the TEI source may legitimately be the scan fallback, or absent
        # when the trial ended before TEI
        ok = (str(r.get("loi1_source")).startswith("delivered_seed")
              and r.get("pdi_tig_offset_s") == r.get("pdi_tig_offset_s"))           # not NaN
        print(f"trial {int(r['trial'])}: loi1 {r.get('loi1_source')}  tei_ign {r.get('tei_ignition_source')}  "
              f"pdi {r.get('pdi_tig_offset_s'):+.1f} s  landing {r.get('land_site_downrange_km'):+.2f} / "
              f"{r.get('land_site_crossrange_km'):+.2f} km  full_success {r.get('full_success')}  "
              f"{r.get('trial_time_s', float('nan')):.0f} s  {'OK' if ok else 'NOT BOUND'}")
        if not ok:
            fails.append(int(r["trial"]))
    print(f"package {d['label']} {d['package_sha256'][:16]}; pinned EI target "
          f"{'= package' if not bad else 'DIFFERS from package in ' + str(bad)}")
    sys.exit(1 if (fails or bad) else 0)


if __name__ == "__main__":
    main()
