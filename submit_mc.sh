#!/bin/bash
# Submit a full sharded Monte Carlo to a SLURM compute cluster.
# SIM-REPO ADAPTATION (sim plan S5, 2026-10-25; copied from the read-only original):
# PROJ defaults to $HOME/apollo11 (the sim repo), APOLLO_DELIVERED_PACKAGE is
# required and injected into every job's env, and preflight runs under that env.
#
#   ./submit_mc.sh <outdir> <n_trials> [seed] [n_shards]
#
# Three chained jobs (per the cluster's SLURM documentation):
#   1. setup  — 1 node:  nominal + target capture into <outdir>
#   2. shard  — array of n_shards single-node jobs, 16 workers each,
#               strided disjoint trial subsets (gap-safe, resumable)
#   3. merge  — 1 node:  concatenate shard results into <outdir>/results.csv
#
# NOTE: the cluster DEFAULT job time limit is 1 hour (partition max is
# infinite) — every job must request --time explicitly or a >60-min shard
# gets killed mid-wave.
# Defaults: seed 42, 17 shards (all 9 CPU + 8 GPU nodes; the GPU itself is
# never requested — their 16 CPU cores each join the pool). Re-running the
# same command resumes: completed shards/trials are skipped.
set -euo pipefail

OUTDIR=${1:?usage: submit_mc.sh <outdir> <n_trials> [seed] [n_shards]}
NTRIALS=${2:?usage: submit_mc.sh <outdir> <n_trials> [seed] [n_shards]}
SEED=${3:-42}
NSHARDS=${4:-17}
WORKERS=16

# SHARD WALL-CLOCK LIMIT, scaled to the shard's actual load.
#
# A flat 8 h was too short for a 10,000-trial run and failed silently-ish: at 17
# shards each one carries ~588 trials, and at this project's measured ~816 s per
# trial on 16 workers that is 8.3 h — just past the limit. The Jul-23 shards died
# exactly this way and had to be resubmitted at 16 h. Gap-safe resume means no
# data is lost, but 8 h of cluster time is.
#
# Budget 2x the nominal estimate, because adverse-date epochs cost roughly double
# (long-tail TEI homotopy solves) and a too-generous limit costs nothing but
# backfill priority, whereas a too-tight one costs a whole run.
_TPS=$(( (NTRIALS + NSHARDS - 1) / NSHARDS ))          # trials per shard, rounded up
_EST_H=$(( (_TPS * 816 * 2) / (WORKERS * 3600) + 1 ))  # 2x margin, +1 h slack
[ "$_EST_H" -lt 8 ] && _EST_H=8                        # never below the old floor
[ "$_EST_H" -gt 48 ] && _EST_H=48                      # cluster sanity cap
SHARD_TIME=${APOLLO_SHARD_TIME:-${_EST_H}:00:00}
echo "shard wall-clock limit: $SHARD_TIME  (${_TPS} trials/shard on $WORKERS workers)"

PY=${APOLLO_PY:-$HOME/apollo-venv/bin/python}
PROJ=${APOLLO_PROJ:-$HOME/apollo11}              # the SIM repo (package-driven), not the original
LOGS=$PROJ/slurm_logs
mkdir -p "$LOGS"

# DELIVERED PACKAGE (sim plan S5): every job imports apollo11 with
# APOLLO_DELIVERED_PACKAGE set, so setup, every shard and every Pool worker fly
# the same package. Required; the path is resolved on the cluster filesystem.
# APOLLO_DELIVERED_PACKAGE_SHA256 (optional prefix) makes every job refuse any
# other package.
PKG=${APOLLO_DELIVERED_PACKAGE:?set APOLLO_DELIVERED_PACKAGE to the package json on the cluster (e.g. nominal_packages/apollo11_nominal_package.json)}
case "$PKG" in /*) ;; *) PKG="$PROJ/$PKG" ;; esac
[ -f "$PKG" ] || { echo "REFUSING: package $PKG not found"; exit 7; }
[ -f "$PROJ/ephemeris/de440.bsp" ] || { echo "REFUSING: $PROJ/ephemeris/de440.bsp missing or a dangling symlink (deploy with rsync -L)"; exit 7; }
PKG_ENV="APOLLO_DELIVERED_PACKAGE=$PKG"
[ -n "${APOLLO_DELIVERED_PACKAGE_SHA256:-}" ] && PKG_ENV="$PKG_ENV APOLLO_DELIVERED_PACKAGE_SHA256=$APOLLO_DELIVERED_PACKAGE_SHA256"
echo "package on every job: $PKG_ENV"

# Optional env prefix injected into every job's command (setup/shard/merge), so a run
# configuration carried in an environment variable reaches the compute nodes regardless
# of the cluster's sbatch --export policy AND regardless of fork-vs-spawn (apollo11 reads
# it at import). Default EMPTY => production runs are byte-for-byte unchanged. Set e.g.
# APOLLO_ENV_PREFIX="APOLLO_NO_LANDING=1" (no-landing comparison) or "APOLLO_HAZARDS=1"
# (landing + added 1969 hazards, the definitive run).
WRAP_ENV="$PKG_ENV ${APOLLO_ENV_PREFIX:-}"
if [ -n "${APOLLO_ENV_PREFIX:-}" ]; then
    echo "env prefix on every job: $WRAP_ENV"
else
    # Loud reminder, not a refusal: the definitive config carries
    # "APOLLO_HAZARDS=1 APOLLO_IGM_ROBUST=1" via APOLLO_ENV_PREFIX; a bare
    # submit silently runs with hazards + IGM-robust OFF (a forgotten prefix
    # once validated a run against the wrong config).
    # SIM-REPO CORRECTION (2026-10-26): that two-flag prefix is NOT the whole
    # definitive Jul-16 config. The original's CLAUDE.md also sets
    # APOLLO_SIVB_RESTART=1 (4.5% per TLI restart, 480 failures in its 10K) and
    # APOLLO_APS_IGNITION=1; the package supplies the ephemeris, azimuth, DOI GET,
    # site and splash, and APOLLO_USSA76_ENTRY defaults on. A 10K submitted with
    # only the two flags (2026-10-25) ran without both hazard overlays.
    echo "WARNING: APOLLO_ENV_PREFIX is EMPTY — env-gated flags (APOLLO_HAZARDS,"
    echo "         APOLLO_IGM_ROBUST, ...) will be OFF on every job. The definitive"
    echo "         config uses: APOLLO_ENV_PREFIX=\"APOLLO_HAZARDS=1 APOLLO_IGM_ROBUST=1 APOLLO_SIVB_RESTART=1 APOLLO_APS_IGNITION=1\""
fi

cd "$PROJ"

# NOMINAL-PIN GUARD: a production run (> 25 trials) must
# NOT silently re-derive the nominal — the cluster's scipy/numpy can converge a
# different TEI/return branch and shift the whole fleet. If the
# files needed to SKIP setup (the pinned nominal + flag-gated targets) are not already
# in OUTDIR, REFUSE to schedule and make the operator decide explicitly. Override with
# ALLOW_DERIVE=1 (the deliberate "yes, re-derive the nominal" choice). The 25-trial tier
# is exempt — that is where the reference nominal is derived in the first place.
if [ "$NTRIALS" -gt 25 ] && [ "${ALLOW_DERIVE:-0}" != "1" ]; then
    if ! env $WRAP_ENV "$PY" cluster_run.py preflight "$OUTDIR"; then
        echo ""
        echo "REFUSING TO SCHEDULE: '$OUTDIR' is missing the pinned nominal (files listed above)."
        echo "A >25-trial run would RE-DERIVE the nominal on the cluster (TEI-branch-flip risk)."
        echo "  -> pin the reference nominal into '$OUTDIR' and re-run, OR"
        echo "  -> re-run with ALLOW_DERIVE=1 to deliberately re-derive it."
        exit 3
    fi
fi

# SERIALIZE RUNS: never interleave with any job
# already queued or running under this account. The new chain's setup depends AFTERANY on every
# existing job, so it starts only when the queue has drained of prior work;
# the rest of the chain serializes behind setup as before. afterany (not
# afterok) so a failed/cancelled prior run cannot deadlock this one.
# Rationale: interleaved runs share nodes and stretch EACH run's completion
# latency, confusing per-run watchers and ETAs; serialized runs finish
# one-at-a-time at full width.
# %A = array-master id (one id covers a whole shard array; %i would emit
# invalid bracket tokens like 12345_[15-16] for pending array elements).
# APOLLO_NO_SERIALIZE=1 opts out: the chain then starts immediately on whatever
# nodes are idle instead of waiting for the queue to drain. Use it when the
# queue holds another project's long jobs and free nodes are sitting idle.
EXISTING=$(squeue -u "$USER" -h -o %A | sort -un | paste -sd: - || true)
SERIAL_DEP=""
if [ "${APOLLO_NO_SERIALIZE:-0}" = "1" ]; then
    echo "APOLLO_NO_SERIALIZE=1: not serializing behind existing jobs"
    EXISTING=""
fi
if [ -n "$EXISTING" ]; then
    SERIAL_DEP="--dependency=afterany:$EXISTING"
    echo "serializing behind existing job(s): $EXISTING"
fi

SETUP_ID=$(sbatch --parsable --time=2:00:00 $SERIAL_DEP \
    --job-name=a11-setup --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --output="$LOGS/setup-%j.out" \
    --wrap "$WRAP_ENV $PY cluster_run.py setup $OUTDIR $NTRIALS $SEED")
echo "setup job: $SETUP_ID"

SHARD_ID=$(sbatch --parsable --time=$SHARD_TIME --dependency=afterok:$SETUP_ID \
    --job-name=a11-shard --array=0-$((NSHARDS-1)) \
    --nodes=1 --ntasks=1 --cpus-per-task=$WORKERS \
    --output="$LOGS/shard-%A_%a.out" \
    --wrap "$WRAP_ENV $PY cluster_run.py shard $OUTDIR $NTRIALS $SEED \$SLURM_ARRAY_TASK_ID $NSHARDS $WORKERS")
echo "shard array: $SHARD_ID (0-$((NSHARDS-1)))"

MERGE_ID=$(sbatch --parsable --time=1:00:00 --dependency=afterok:$SHARD_ID \
    --job-name=a11-merge --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --output="$LOGS/merge-%j.out" \
    --wrap "$WRAP_ENV $PY cluster_run.py merge $OUTDIR $NSHARDS")
echo "merge job: $MERGE_ID"
echo "monitor:  squeue -u \$USER   |   logs in $LOGS"
