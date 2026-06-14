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
# gets killed mid-wave (bit us at 999/1000 on the first 1000-trial run).
# Defaults: seed 42, 17 shards (all 9 CPU + 8 GPU nodes; the GPU itself is
# never requested — their 16 CPU cores each join the pool). Re-running the
# same command resumes: completed shards/trials are skipped.
set -euo pipefail

OUTDIR=${1:?usage: submit_mc.sh <outdir> <n_trials> [seed] [n_shards]}
NTRIALS=${2:?usage: submit_mc.sh <outdir> <n_trials> [seed] [n_shards]}
SEED=${3:-42}
NSHARDS=${4:-17}
WORKERS=16
PY=$HOME/apollo-venv/bin/python
PROJ=$HOME/apollo11_project
LOGS=$PROJ/slurm_logs
mkdir -p "$LOGS"

cd "$PROJ"

SETUP_ID=$(sbatch --parsable --time=2:00:00 \
    --job-name=a11-setup --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --output="$LOGS/setup-%j.out" \
    --wrap "$PY cluster_run.py setup $OUTDIR $NTRIALS $SEED")
echo "setup job: $SETUP_ID"

SHARD_ID=$(sbatch --parsable --time=8:00:00 --dependency=afterok:$SETUP_ID \
    --job-name=a11-shard --array=0-$((NSHARDS-1)) \
    --nodes=1 --ntasks=1 --cpus-per-task=$WORKERS \
    --output="$LOGS/shard-%A_%a.out" \
    --wrap "$PY cluster_run.py shard $OUTDIR $NTRIALS $SEED \$SLURM_ARRAY_TASK_ID $NSHARDS $WORKERS")
echo "shard array: $SHARD_ID (0-$((NSHARDS-1)))"

MERGE_ID=$(sbatch --parsable --time=1:00:00 --dependency=afterok:$SHARD_ID \
    --job-name=a11-merge --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --output="$LOGS/merge-%j.out" \
    --wrap "$PY cluster_run.py merge $OUTDIR $NSHARDS")
echo "merge job: $MERGE_ID"
echo "monitor:  squeue -u \$USER   |   logs in $LOGS"
