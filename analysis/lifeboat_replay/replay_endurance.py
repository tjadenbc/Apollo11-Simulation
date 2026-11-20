"""Replay a fleet's translunar service-module failures at two lifeboat endurances.

For every trial the fleet labels sm_failure_translunar, fly the trial again at the fleet's own
endurance limit (the control, which must reproduce the fleet's noloi_* columns) and at an
alternate limit. The limit is set in the worker on EVERY job, so no job inherits another's
value. Each finished run is appended to a JSON-lines file at once; a restart skips runs
already there. No file of the fleet is read except results.csv and the four pinned targets,
and none is changed. replay_compare.py then compares the runs with the fleet.

Run from the repository root with the environment of the configuration to replay (the nine
flags of the definitive run; add APOLLO_DIRECT_ABORT=1 for the direct-abort family):
  python analysis/lifeboat_replay/replay_endurance.py <fleet_dir> <seed> <workers> <alt_hours> <shard> <n_shards> <out_jsonl> [--dry]
Shard k flies translunar service-module trials k, k + n_shards, ... of the list (trial ascending),
each at both limits (control, then alternate).
"""
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.getcwd())

DROP = ("_phase_log", "_temcc_burns", "phase_timeline")


def _work(args):
    idx, perturb, hours, targets = args
    import apollo11 as a
    a._parallel_worker_init(*targets)
    a.NOLOI_LIFEBOAT_MAX_H = float(hours)      # set on EVERY job; never inherit a prior job's value
    t0 = time.time()
    try:
        r, _ = a.run_mission(perturb=perturb, capture_trajectories=False)
    except Exception as e:  # report, do not hide
        return {"trial": idx, "hours": float(hours), "error": repr(e), "host": socket.gethostname()}
    out = {k: v for k, v in r.items() if k not in DROP}
    out.update({"trial": idx, "hours": float(hours), "limit_in_worker": float(a.NOLOI_LIFEBOAT_MAX_H),
                "direct_abort_flag": bool(getattr(a, "ENABLE_NOLOI_DIRECT_ABORT", False)),
                "host": socket.gethostname(), "replay_time_s": time.time() - t0})
    return out


def _plain(o):
    import numpy as np
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    argv = [x for x in sys.argv[1:] if x != "--dry"]
    dry = "--dry" in sys.argv
    fleet, seed, workers, alt = argv[0], int(argv[1]), int(argv[2]), float(argv[3])
    shard, n_shards, out = int(argv[4]), int(argv[5]), argv[6]
    import multiprocessing as mp
    import numpy as np
    import pandas as pd
    import apollo11 as a

    d = getattr(a, "_DELIVERED", None)
    if d is None:
        raise SystemExit("apollo11 imported without a bound package (APOLLO_DELIVERED_PACKAGE)")
    df = pd.read_csv(os.path.join(fleet, "results.csv"), low_memory=False)
    df = df.sort_values("trial").reset_index(drop=True)
    n = len(df)
    base_h = float(a.NOLOI_LIFEBOAT_MAX_H)
    tl = [int(t) for t in df.trial[df.mission_failure.fillna("") == "sm_failure_translunar"]]
    flags = {k: bool(getattr(a, k, False)) for k in
             ("ENABLE_NOLOI_RETURN", "ENABLE_LIFEBOAT_LIMIT", "ENABLE_NOLOI_DIRECT_ABORT", "_NL_ENV",
              "ENABLE_FREE_RETURN_PROFILE")}
    print(f"host {socket.gethostname()} | python {sys.version.split()[0]} | numpy {np.__version__} | "
          f"package {d['label']} sha256 {d['package_sha256'][:16]}", flush=True)
    print(f"fleet {fleet}: {n} trials; translunar service-module trials {len(tl)}; "
          f"limits {base_h} h (control) and {alt} h; flags {flags}", flush=True)

    def _load(name):
        with open(os.path.join(fleet, name)) as f:
            return json.load(f)

    targets = (_load("ei_target.json"), _load("bplane_target.json"),
               _load("ca_target.json"), _load("od_cov.json"))
    rng = np.random.default_rng(seed)
    perturbs = [a.sample_perturbation(rng) for _ in range(n)]
    jobs = [(i, perturbs[i], h, targets) for i in tl[shard::n_shards] for h in (base_h, alt)]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r:
                done.add((int(r["trial"]), float(r["hours"])))
    todo = [j for j in jobs if (j[0], float(j[2])) not in done]
    print(f"shard {shard}/{n_shards}: {len(jobs)} runs, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to fly on {workers} workers -> {out}", flush=True)
    if dry:
        print("dry run: first jobs", [(j[0], j[2]) for j in jobs[:6]], flush=True)
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    err = 0
    t0 = time.time()
    with open(out, "a") as f, mp.Pool(processes=workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, todo, chunksize=1), 1):
            err += "error" in r
            f.write(json.dumps(r, default=_plain) + "\n")
            f.flush()
            if k % 10 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} runs done, {err} errors, {time.time() - t0:.0f} s", flush=True)
    print(f"replay shard {shard}/{n_shards} complete: {len(todo)} runs flown, {err} errors", flush=True)


if __name__ == "__main__":
    main()
