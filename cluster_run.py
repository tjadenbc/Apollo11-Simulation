"""Multi-node Monte Carlo sharding driver for a SLURM compute cluster.

Three stages, chained by submit_mc.sh with sbatch dependencies:

  setup  — runs the nominal + captures the EI / B-plane targets into the
           master outdir (main_parallel with indices=[]: no trials).
  shard  — one per node: copies the master targets into a shard-local outdir
           (no cross-node writes to a shared results.csv) and runs the
           strided trial subset  shard_id, shard_id+n_shards, ...  with the
           same seed: trial i maps to the identical perturbation it would
           have in a single-node run, so a sharded run is trial-for-trial
           comparable to a local one.
  merge  — concatenates shard CSVs (sorted by trial), gathers per-trial
           debug JSONs into master/trials/, and leaves the master outdir in
           exactly the layout generate_outputs.py / crew_survival.py expect.

Resume: re-submitting the same pipeline is safe everywhere — setup skips a
captured nominal, shards fill only their missing trials (gap-safe), merge is
idempotent.

Usage (normally via submit_mc.sh, but callable by hand):
  python3 cluster_run.py setup  <outdir> <n> <seed>
  python3 cluster_run.py shard  <outdir> <n> <seed> <shard_id> <n_shards> <workers>
  python3 cluster_run.py merge  <outdir> <n_shards>
"""
import os
import shutil
import sys


def _shard_dir(outdir, k):
    return os.path.join(outdir, f"shard_{k:02d}")


def do_setup(outdir, n, seed):
    import apollo11
    apollo11.main_parallel(n=n, outdir=outdir, seed=seed, workers=1,
                           indices=[])
    for f in ("ei_target.json", "bplane_target.json",
              "nominal_results.json"):
        p = os.path.join(outdir, f)
        if not os.path.exists(p):
            print(f"WARNING: setup did not produce {f}")
    print("setup complete")


def do_shard(outdir, n, seed, shard_id, n_shards, workers):
    sd = _shard_dir(outdir, shard_id)
    os.makedirs(sd, exist_ok=True)
    # Seed the shard dir with the master's nominal artifacts so
    # main_parallel skips the nominal and loads the SAME targets everywhere.
    for f in ("ei_target.json", "bplane_target.json", "nominal_results.json"):
        src = os.path.join(outdir, f)
        dst = os.path.join(sd, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    if not os.path.exists(os.path.join(sd, "nominal_results.json")):
        raise SystemExit("shard: master nominal artifacts missing — "
                         "run setup first")
    import apollo11
    idx = list(range(shard_id, n, n_shards))
    print(f"shard {shard_id}/{n_shards}: {len(idx)} trials "
          f"({idx[:3]}...{idx[-1:]}) on {workers} workers")
    apollo11.main_parallel(n=n, outdir=sd, seed=seed, workers=workers,
                           indices=idx)
    print(f"shard {shard_id} complete")


def do_merge(outdir, n_shards):
    import pandas as pd
    frames = []
    missing = []
    os.makedirs(os.path.join(outdir, "trials"), exist_ok=True)
    for k in range(n_shards):
        sd = _shard_dir(outdir, k)
        csv = os.path.join(sd, "results.csv")
        if not os.path.exists(csv):
            missing.append(k)
            continue
        frames.append(pd.read_csv(csv))
        tdir = os.path.join(sd, "trials")
        if os.path.isdir(tdir):
            for fn in os.listdir(tdir):
                src = os.path.join(tdir, fn)
                dst = os.path.join(outdir, "trials", fn)
                # trial_nominal.json exists in every shard (copied master
                # nominal); identical content, first copy wins.
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
    if missing:
        print(f"WARNING: missing shard results: {missing}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="trial", keep="first")
    df = df.sort_values("trial").reset_index(drop=True)
    df.to_csv(os.path.join(outdir, "results.csv"), index=False)
    ok = df.get("full_success")
    n = len(df)
    print(f"merge complete: {n} trials, full_success "
          f"{int(ok.fillna(False).astype(bool).sum())}/{n}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "setup":
        do_setup(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "shard":
        do_shard(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                 int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
    elif mode == "merge":
        do_merge(sys.argv[2], int(sys.argv[3]))
    else:
        raise SystemExit(f"unknown mode {mode!r} (setup|shard|merge)")
