#!/bin/bash
# Submit a full sharded Monte Carlo to a SLURM compute cluster.
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
PY=${APOLLO_PY:-$HOME/apollo-venv/bin/python}
PROJ=${APOLLO_PROJ:-$HOME/apollo11_project}
LOGS=$PROJ/slurm_logs
mkdir -p "$LOGS"

# Optional env prefix injected into every job's command (setup/shard/merge), so a run
# configuration carried in an environment variable reaches the compute nodes regardless
# of the cluster's sbatch --export policy AND regardless of fork-vs-spawn (apollo11 reads
# it at import). Default EMPTY => production runs are byte-for-byte unchanged. Set e.g.
# APOLLO_ENV_PREFIX="APOLLO_NO_LANDING=1" (no-landing comparison) or "APOLLO_HAZARDS=1"
# (landing + added 1969 hazards, the definitive run).
WRAP_ENV="${APOLLO_ENV_PREFIX:-}"
[ -n "$WRAP_ENV" ] && echo "env prefix on every job: $WRAP_ENV"

cd "$PROJ"

# NOMINAL-PIN GUARD: a production run (> 25 trials) must
# NOT silently re-derive the nominal — the cluster's scipy/numpy can converge a
# different TEI/return branch and shift the whole fleet. If the
# files needed to SKIP setup (the pinned nominal + flag-gated targets) are not already
# in OUTDIR, REFUSE to schedule and make the operator decide explicitly. Override with
# ALLOW_DERIVE=1 (the deliberate "yes, re-derive the nominal" choice). The 25-trial tier
# is exempt — that is where the reference nominal is derived in the first place.
if [ "$NTRIALS" -gt 25 ] && [ "${ALLOW_DERIVE:-0}" != "1" ]; then
    if ! "$PY" cluster_run.py preflight "$OUTDIR"; then
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
EXISTING=$(squeue -u "$USER" -h -o %A | sort -un | paste -sd: - || true)
SERIAL_DEP=""
if [ -n "$EXISTING" ]; then
    SERIAL_DEP="--dependency=afterany:$EXISTING"
    echo "serializing behind existing job(s): $EXISTING"
fi

SETUP_ID=$(sbatch --parsable --time=2:00:00 $SERIAL_DEP \
    --job-name=a11-setup --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --output="$LOGS/setup-%j.out" \
    --wrap "$WRAP_ENV $PY cluster_run.py setup $OUTDIR $NTRIALS $SEED")
echo "setup job: $SETUP_ID"

SHARD_ID=$(sbatch --parsable --time=8:00:00 --dependency=afterok:$SETUP_ID \
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
