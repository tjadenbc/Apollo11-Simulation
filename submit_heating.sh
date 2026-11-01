#!/bin/bash
# Fleet-wide Sutton-Graves entry-heating replay of one definitive 10K (sim repo, 2026-10-31).
#
#   bash submit_heating.sh HEAT_DIR PACKAGE PKG_SHA_PREFIX BASE_TRIAL [N_SHARDS] [WORKERS]
#
# HEAT_DIR holds the run's results_with_survival.csv and run_config.json (copied from the
# fetched run; the run itself is no longer on the cluster). Submits three jobs:
#   a11-heat        array 0..N_SHARDS-1: entry_heating_fleet.py, one stride shard each
#   a11-heat-merge  afterok on the array: entry_heating_fleet.py merge
#   a11-heat-bnd    entry_heating_boundary.py, FPA sweep off BASE_TRIAL (the shallowest
#                   captured entry), independent of the array
# Names start with a11- and the env prefix is inline in each --wrap, so queue_watcher.py
# can move them to idle compute nodes.
set -euo pipefail
HEAT_DIR=$1; PKG=$2; SHA=$3; BASE=$4; NSHARDS=${5:-17}; WORKERS=${6:-16}
PROJ=$HOME/apollo11_slots
PY=$HOME/apollo11-venv/bin/python
cd "$PROJ"
[ -f "$HEAT_DIR/results_with_survival.csv" ] && [ -f "$HEAT_DIR/run_config.json" ] \
    || { echo "REFUSING: $HEAT_DIR needs results_with_survival.csv and run_config.json"; exit 2; }
[ -f "$PROJ/ephemeris/de440.bsp" ] || { echo "REFUSING: ephemeris/de440.bsp missing"; exit 7; }
for f in entry_heating_validate.py entry_heating_fleet.py entry_heating_boundary.py; do
    [ -f "$PROJ/$f" ] || { echo "REFUSING: $f missing"; exit 2; }
done

ENV="APOLLO_HEAT_OUTDIR=$HEAT_DIR APOLLO_DELIVERED_PACKAGE=$PKG APOLLO_DELIVERED_PACKAGE_SHA256=$SHA \
APOLLO_HAZARDS=1 APOLLO_IGM_ROBUST=1 APOLLO_SIVB_RESTART=1 APOLLO_APS_IGNITION=1 \
APOLLO_TLC_IMPACT_RECOVERY=1 APOLLO_WATER_ZONE_REAIM=1 APOLLO_TEMCC_WIDE_TRIM=1 \
APOLLO_NOLOI_RETURN=1 APOLLO_LIFEBOAT_LIMIT=1"
LOG=$PROJ/slurm_logs

ARR=$(sbatch --parsable --time=8:00:00 --job-name=a11-heat --nodes=1 --ntasks=1 \
    --cpus-per-task=$WORKERS --array=0-$((NSHARDS - 1)) --chdir="$PROJ" \
    --output="$LOG/heat-%A_%a.out" \
    --wrap "$ENV $PY -W ignore entry_heating_fleet.py \$SLURM_ARRAY_TASK_ID $NSHARDS $WORKERS")
MRG=$(sbatch --parsable --time=1:00:00 --dependency=afterok:$ARR --job-name=a11-heat-merge \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --chdir="$PROJ" --output="$LOG/heat-merge-%j.out" \
    --wrap "$ENV $PY -W ignore entry_heating_fleet.py merge $NSHARDS")
BND=$(sbatch --parsable --time=2:00:00 --job-name=a11-heat-bnd --nodes=1 --ntasks=1 \
    --cpus-per-task=11 --chdir="$PROJ" --output="$LOG/heat-bnd-%j.out" \
    --wrap "$ENV $PY -W ignore entry_heating_boundary.py $BASE 11")
echo "HEAT $HEAT_DIR  package $PKG ($SHA)  $NSHARDS shards x $WORKERS workers  base trial $BASE"
echo "array: $ARR   merge: $MRG   boundary: $BND"
