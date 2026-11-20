# Lifeboat replay of the definitive run

Two results of the paper rest on flying the definitive run's translunar service-module
failures again:

- **Section 5.3, the direct abort.** The definitive run flies one return for a crew that loses
  its service module on the outbound coast, the return around the Moon. A direct abort, a burn
  that turns back short of the Moon, is a second family the run did not fly.
- **Section 7.3, the lifeboat limit.** The run gives the Lunar Module lifeboat 105 hours, set
  by Apollo 13's battery record. Water was Apollo 13's most restrictive consumable, so the
  replay asks what a shorter limit, 99.8 hours, costs.

## What was flown

All 243 trials of `outputs/cluster_fleet10k_jul16_s37_def9` labeled `sm_failure_translunar`
were flown again from the launch pad, each at two lifeboat limits, 105 h (the control) and
99.8 h, in two configurations:

| Prefix | Configuration |
|---|---|
| `asflown` | the definitive run's own: the nine flags of the README |
| `directabort` | the same with `APOLLO_DIRECT_ABORT=1` |

Each run was flown twice, on two kinds of cluster node (`_c_` and `_g_` files): 1,944 runs,
no errors. The two node types agree in every run and every recorded value.

```bash
# one shard of the replay (the definitive configuration set in the shell; see the README)
python analysis/lifeboat_replay/replay_endurance.py outputs/cluster_fleet10k_jul16_s37_def9 37 16 99.8 0 2 analysis/lifeboat_replay/out/asflown_c_0.jsonl
# the comparison with the fleet (per-trial JSONs from the release asset extracted to trials/)
python analysis/lifeboat_replay/replay_compare.py analysis/lifeboat_replay/out asflown outputs/cluster_fleet10k_jul16_s37_def9 analysis/lifeboat_replay/asflown_vs_fleet.csv --trials outputs/cluster_fleet10k_jul16_s37_def9/trials
python analysis/lifeboat_replay/replay_compare.py analysis/lifeboat_replay/out directabort outputs/cluster_fleet10k_jul16_s37_def9 analysis/lifeboat_replay/directabort_vs_fleet.csv --no-return-control --trials outputs/cluster_fleet10k_jul16_s37_def9/trials
```

## Results

**The control.** At 105 h in the definitive configuration every one of the 243 trials has the
fleet's failure label, strike time and return class. 184 reproduce the fleet's return time and
burn in every digit; 59 differ by at most 0.016 h and 1.1 m/s (median difference zero).
The difference is not the code: the replay flew release 4.1.0 and the fleet release 4.0.0, and the
two releases agree in every recorded value when flown side by side on one machine (README,
Reproducibility; four of the 59 trials are among the twelve flown there). It lies between the
two cluster runs, flown three weeks apart, and the return solver, which walks its arrival time
in to the limit, amplifies a last-digit difference in its input to this size.

**The 99.8-hour limit (Section 7.3).** In the definitive configuration 19 trials that return
at 105 h have no return at 99.8 h, each a descent-engine return around the Moon struck 23.6 to
28.5 h after launch. Each carries the lifeboat's 0.90 crew survival, so the shorter limit costs
19 x 0.90 = 17.1 crews in 10,000, 0.17 points of crew survival. No trial gains a return. Of the
43 descent-engine returns that survive at both limits, 38 must burn earlier and harder (mean
171 to 354 m/s).

**The direct abort (Section 5.3).** In the definitive run 70 translunar failures have no
return within 105 h. With the direct abort switched on, 47 of them return at 105 h, every one
on the direct family, with burns of 407 to 602 m/s against the docked stack's 605 m/s. Their
strikes fall 2.8 to 13.5 h after launch. Eleven of the 47 were struck before 4.28 h, the time
at which the Lunar Module was extracted from the S-IVB on Apollo 11; the simulation does not
model the extraction, and a crew without its Lunar Module has no descent engine to burn, so
the paper counts the remaining **36** (36 x 0.90 = 0.32 points of crew survival). The other 23
of the 70, struck 14.0 to 23.1 h after launch, have no return in either family.

## Files

| File | What it is |
|---|---|
| `replay_endurance.py` | The replay driver. |
| `replay_compare.py` | The comparison with the fleet; writes the per-trial tables. |
| `out/*.jsonl` | One line per run: the full result row, the limit asked and the limit in the worker, and the node. |
| `asflown_vs_fleet.csv`, `directabort_vs_fleet.csv` | One row per trial: the fleet's return, the control's and the 99.8 h run's. |
| `compare_asflown.txt`, `compare_directabort.txt` | The printed reports. |
