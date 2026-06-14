# Apollo 11 Monte Carlo Simulation — Project Guide

> This file orients Claude Code (and you) on the project. Claude Code reads it
> automatically at the start of a session in this directory.

## What this is

A physics-integrated Monte Carlo simulation answering: *"what would happen if
many thousands of Apollo 11 missions launched in July 1969 with then-current
technology?"* (the definitive run is 10,000 trials). It propagates full ODE
dynamics across every mission phase — launch through splashdown — under Earth
(+J2) and Moon gravity, with 1969-era engine reliabilities, Apollo guidance
laws, scheduled midcourse corrections, and a per-astronaut crew-survival model.

## Files

- `apollo11.py` — the simulation (~6,800 lines): physics, every mission phase,
  `run_mission()` (one mission), and `main()` / `main_parallel()` (the Monte
  Carlo drivers; `main_parallel` takes an optional `indices=` subset for
  cluster sharding).
- `crew_survival.py` — per-astronaut survival model; reads the MC results and
  writes `results_with_survival.csv` plus crew statistics.
- `generate_outputs.py` — builds `dashboard.html`, `summary.txt`, and the PNG
  figures from a run directory. Data-driven: the Apollo cross-check table, the
  **Phase-Timing table** (per-phase mission-duration min/avg/max across trials
  vs Apollo 11, aggregated from the `trials/` debug files), and the
  Known-Limitations text are all computed from the run, not hard-coded.
- **Per-trial debug output** — each trial writes `outputs/<run>/trials/trial_<i>.json`
  (and `trial_nominal.json`): a full per-trial overview with `phase_timeline`
  (each mission phase's GET start, mission-elapsed `duration_s`, and compute
  `compute_s`) plus every outcome field — for reviewing what happened in any
  single trial. `run_mission` records the timeline via `_mark()` boundaries;
  `build_phase_timeline()` + `PHASE_SEGMENTS`/`APOLLO_PHASE_DUR_S` define the
  phases and Apollo reference durations. The `_phase_log` is dropped from the CSV
  (scalar-only) and saved per-trial instead.
- `cluster_run.py` + `submit_mc.sh` — the SLURM-cluster pipeline
  (setup → sharded trials → merge) that produced the definitive run; see
  **How to run** and the cluster memory note.
- `outputs/apollo11_final10000/` — **the definitive run** (10,000 trials,
  seed 37; dir name is legacy from a 2,500-trial base later extended):
  `results.csv`, `results_with_survival.csv`, `nominal_results.json`,
  `nominal_traj.npz`, `ei_target.json`, `bplane_target.json`, the
  dashboard/summary, and figures. Earlier run dirs remain as historical
  references.

## Environment

```
pip install -r requirements.txt
```
Python 3.12; numpy / scipy / pandas / matplotlib. Plotting is headless — the
scripts select the non-interactive `Agg` backend.

## How to run

Full Monte Carlo (writes into the output dir; checkpoints `results.csv` every
trial):
```
python3 -c "import apollo11; apollo11.main(n=200, outdir='outputs/apollo11_tei200', seed=42)"
```
Parallel Monte Carlo (`main_parallel`, recommended — same results, bit-identical
to serial; ~10× faster on a multicore machine):
```
python3 -c "import apollo11; apollo11.main_parallel(n=100, outdir='outputs/apollo11_bplane100', seed=42, workers=10)"
```
**Must** be launched via `python3 -c` or a `__main__`-guarded `.py` file, never a
stdin heredoc — macOS multiprocessing uses the `spawn` start method, which
re-imports `__main__` in each worker. Resume is gap-safe and the CSV is written
sorted by `trial`. Regression test: `python3 test_resume_gapfix.py`.
Resume an interrupted run (idempotent — re-run until `results.csv` has n+1 rows,
the +1 being the nominal):
```
python3 -c "import apollo11; apollo11.main(n=200, outdir='outputs/apollo11_tei200', seed=42, resume=True)"
```
Regenerate the dashboard / summary / figures from an existing run:
```
python3 -c "import matplotlib; matplotlib.use('Agg'); import crew_survival;   crew_survival.main()"
python3 -c "import matplotlib; matplotlib.use('Agg'); import generate_outputs; generate_outputs.main()"
```
The `OUTDIR` constant near the top of `crew_survival.py` and
`generate_outputs.py` selects which run directory they read/write.

**Cluster (large runs):** the definitive 10,000-trial run was produced on the
SLURM cluster via `./submit_mc.sh <outdir> <n_trials> [seed] [n_shards]`
(default seed 42, 17 shards across all nodes). It chains three SLURM jobs —
setup (nominal + target capture), a shard array (disjoint strided trial
subsets via `main_parallel(indices=...)`), and merge (concatenate, sorted,
deduped). Resume is gap-safe. Before submitting, PIN the laptop-derived
`ei_target.json` / `bplane_target.json` / `nominal_results.json` and
`launch_tli_preset.json` into the run dir so cross-machine scipy/numpy version
skew can't flip the marginal nominal-TEI branches (see the cluster memory
note). The cluster's `~/apollo-venv` must be built on Python 3.11 (compute
nodes lack 3.12). 10,000 trials ≈ 2 h on 272 cores.

## Compute notes

- Each trial is ~350 s on an M-series laptop core, ~625 s on the cluster's
  EPYC 7252 cores (~1.8× slower per core; long-tail TEI-homotopy trials reach
  ~29 min). `main()` runs trials serially; `main_parallel()` fans them across
  worker processes (`workers` defaults to cores−1) and is the recommended
  local path. The serial and parallel drivers are bit-identical (per-trial
  perturbations are pre-generated in the main process and dispatched by trial
  index); cluster runs are NOT bit-identical to the laptop (different
  scipy/numpy), so treat a cluster run as its own population.
- Both drivers checkpoint after every trial, so an interrupted run loses nothing.
- Nominal-run targets are persisted so resumes reuse them:
  `ei_target.json` (entry interface) and `bplane_target.json` (lunar-approach
  B-plane). `main()` resets and re-captures them if missing.

## Key feature flags (top of `apollo11.py`)

| flag | default | effect |
|------|---------|--------|
| `ENABLE_IGM_ASCENT` | True | Iterative-guidance-mode S-IVB steering to a 185×185 km parking orbit |
| `ENABLE_TEI_TARGETING` | True | Solve the 3-DOF TEI burn vector so the return passes through the nominal entry interface |
| `ENABLE_TRANS_EARTH_MCC` | True | Trans-Earth FPA-targeting trims (MCC-5/6/7) |
| `ENABLE_TRANS_LUNAR_MCC` | True | Outbound MCC-1..4 correction chain |
| `ENABLE_BPLANE_TLMCC` | True | **(latest)** trans-lunar MCC uses a 2-DOF B-plane solve (perilune altitude **and** approach-plane orientation), not perilune-altitude-only |
| `ENABLE_SKIP_ENTRY_GUIDANCE` | **True** (default) | **(guided entry — latest)** Closed-loop entry guidance: a g-AWARE numerical predictor-corrector (`_predict_landing`/`_solve_bank` — predictions include the skip and peak-g; candidates scored miss + penalty above 7 g, hard-rejected above 9.5 g) picks the bank, with a predictive lift-up HARD GUARD at 9.5 g (Apollo's guidance limit was 10 g per TN D-6725; structural bound 12 g) and crossrange via deadband bank reversals (sign matches the proven unguided P64 logic — the inherited inverted sign parked trials ~580 km off-axis). Recovery target = the guided nominal's landing point at the SHORT (direct-range) corridor end — Apollo's own design choice (EI-to-splash ~2,780 km, dug-in early), which makes range intrinsically dispersion-insensitive. **Validated: ~1–2 km miss across the entry-FPA corridor; end-to-end nominal ~1 km at ~8.6 g.** Residual: nominal peak g ~8.6 vs Apollo's as-flown 6.5 (closing it needs an FPA-indexed HUNTEST reference-profile family + closed-loop drag tracking — a 6-variant corpus was tried and closed, see `ENABLE_HUNTEST_PROFILE`/`ENABLE_REF_PROFILE_ENTRY`). The structural-failure threshold is 12 g (steep-tail successes can reach 9.5–12 g — the 9.5 g lift-up guard is predictive, not a hard cap). OFF = legacy unguided fixed profile. Regression: `test_entry_guidance.py`. |
| `ENABLE_DESCENT_FAILURE_MODES` | True | Apollo 11-specific descent + rendezvous/docking failure modeling |
| `ENABLE_DOI` | True | **(latest)** Realistic descent chain: CSM parked in a near-circular ~100 km lunar orbit, a discrete DOI burn (~19 m/s, charged to the DPS) lowers only the LM perilune to ~15 km, PDI fires from there (vs braking directly off the old unphysical ~5×413 km capture orbit), an efficient P63-like braking law flies that profile, and a fat-tailed manual-flying fuel penalty (hazard-avoidance overfly) keeps descent-fuel exhaustion a genuine rare-tail risk. When OFF, the legacy descent path is bit-identical to before. |
| `ENABLE_REAL_EPHEMERIS` | **True** (default) | **(fidelity-first config)** Replaces the idealized circular Moon (fixed 28.4° incl, node +X) with a real July-1969 lunar ephemeris (validated Meeus series, <1″ vs the worked example) + real launch-epoch GMST anchoring, so the return plane and splashdown reflect the true sky. Three pieces: the ephemeris/GMST, a TEI **near-miss refinement** (3-DOF `trf` burn-vector solve driving the integrated return perigee into the entry corridor — prograde-only TEI left it ~400 km high), and the **faithful Apollo GET timeline**: 26.7 h LOI→PDI stay (`LUNAR_PARK_COAST_S` = 23.55 h loiter + ~3.15 h irreducible two-burn-LOI/DOI coast) **and** ~10.9 h ascent→TEI leg (`POST_RENDEZVOUS_COAST_S` = 9.7 h docking/jettison/crew-prep coast, then the TEI alignment scan prefers FIRST-REV candidates with a **rev-slip fallback**: later revs in the 10 h scan are used only when rev-1 yields no valid return — a hard one-rev window caused `tei_no_earth_return_found` on perturbed trials, and a score-based gate let later revs win and moved the splash ~6,600 km; the gate must be "any valid rev-1 solution wins"). Total mission **~8.18 d vs Apollo's 8.16 d**. With launch continuity + Apollo return-timing the guided nominal splashes in the **western Pacific near (13.5°N, 146°E)** — latitude within ~0.2° of Apollo 11's 13.3°N, with a ~13° (~4,800 km) longitude offset from Apollo's 169°W (the residual region/geometry gap). The **definitive headline is the 10,000-trial cluster run `apollo11_final10000` (85.5% mission success; 84.0% flawless)** — see Status. When OFF, bit-identical to the idealized model. |
| `ENABLE_INTEGRATED_TLI` | **True** (default) | **(fixes known-limitation #1 — latest)** The S-IVB TLI burn is physically FLOWN. The outbound transfer was re-architected from the legacy unflyable −45° near-radial lob (FPA 66° at 185 km, transfer perigee ~5,300 km *inside* the Earth) to a realistic **−170° near-tangential transfer** (`TLI_TRANSFER_ANGLE_DEG`, departure FPA +1.3°, perigee at departure altitude; same transfer PLANE, so the TEI-favorable lunar-arrival geometry is preserved). The ignition state is back-solved on the parking orbit (~298 km at circular speed); each trial integrates the ~274 s burn with its own J-2 dispersions, pointing misalignment (consumes `tli_pointing_rad`) and a velocity-cutoff guidance law (consumes `tli_dv_bias_ms`) — RNG stream untouched. New honest failure mode: `tli_propellant_depleted` (~130 m/s reserve). Validated: flown nominal matches the transfer target to 0.000 m/s; full mission end-to-end. (Superseded for the splash location by `ENABLE_LAUNCH_CONTINUITY` below — the nominal now splashes in the western Pacific, not the 37°N/55°W this flag alone produced.) When OFF, the legacy lob + synthetic post-TLI state are bit-identical (needed to reproduce pre-fix runs). |
| `ENABLE_LAUNCH_CONTINUITY` | **True** (default) | **(fully resolves known-limitation #1 — latest)** TLI ignites from THE TRIAL'S OWN LAUNCHED PARKING ORBIT — the pad-to-splashdown trajectory is one continuous flown path. Pieces: (1) **azimuth-capable ascent** (target-plane steering through the gravity turn + IGM); (2) the **solved launch window** `LAUNCH_AZIMUTH_DEG = 72.48°` vs Apollo's actual 72.058° — solved so the **J2-coasted** plane AT TLI IGNITION (Earth J2 regresses the node ~0.5° over the 2.7 h parking coast) contains the Moon-arrival direction — an independent ~0.4° validation of the ephemeris+GMST+ascent stack; (3) an **IGM-style STEERED cutoff** (`_solve_launch_tli`, disk-cached preset `launch_tli_preset.json`): constant pitch/yaw thrust tilts (−2.35°/+2.54°) + cutoff speed, solved by damped 2×2 Newton on (arrival-time, periselene) + brentq on out-of-plane — open-loop arrival periselene **109 km**, out-of-plane −15 km, arrival slip −0.01 h, cost only cosine losses (~0.1%, no SPS charge); (4) the capture is **RETROGRADE — Apollo's actual handedness** (the pitch root choice picks the near-side periselene root; the far root is prograde with a wildly different TOF); (5) TEI candidate selection is **handedness-agnostic**: score by the predicted two-body departure ASYMPTOTE against the perigee-nulling INBOUND departure direction (≈43° off anti-v_moon; the OUTBOUND crossing also nulls perigee but via a ~550,000 km apogee, 10+ days — filtered by requiring Earth-ward departure velocity). Validated nominal: full success end-to-end. NOTE: the original (−26.72°, −59.51°) splash fixed-point was a backwards-long-skip artifact and was later corrected — combined with the as-flown mass calibration and Apollo return-timing, the nominal now splashes in the **western Pacific near (13.5°N, 146°E)**, TEI ΔV ~996 m/s nominal / ~953 median (Apollo 1,008), entry FPA −6.5°, splash miss ~1 km. When OFF, the integrated-TLI (or legacy) path is used — but note SPLASH_TARGET is bound at import; reproducing legacy configs requires editing the flag in source. |
| `ENABLE_LUNAR_SH_FIELD` | **True** (default) | **(#3 — latest)** Real degree-8 lunar gravity: GRAIL **GRGM1200A** coefficients (degrees 2–12 embedded in `lunar_gravity_coeffs.py`, downloaded from NASA PDS with conventions verified; implied J2/C22 match the code's sourced constants to 0.003%, enforced by import-time asserts), evaluated with a singularity-free Cunningham V/W recursion (~49 µs/eval, tier-gated to <3,500 km from the Moon; deg-2 closed form in the mid zone). The parking orbit now **evolves physically (~+5.7 km/day incremental drift; total in Apollo's documented 5–20 km/day band** — Apollo left LOI-2 elliptical *because* of this drift). Deliberately NOT named a "mascon field": maria mascons live at degree ≥50, so the calibrated `MASCON_DOWNRANGE_*` landing proxy stays. Moon-fixed frame is the existing synchronous-lock approximation (~6.7° pole error, ~7° librations ignored — fine for zonal-driven drift, a prerequisite-to-fix for mascon-grade work). Regression suite: `test_sh_field.py`. When OFF, the legacy degree-2 closed form is bit-identical. |
| `ENABLE_LUNAR_HARMONICS` | **True** (default) | Legacy degree-2 (C20/C22) lunar field + the calibrated `MASCON_DOWNRANGE_*` landing-dispersion proxy. Coexists with `ENABLE_LUNAR_SH_FIELD`: the degree-8 GRAIL field supplies orbital evolution, while the mascon proxy still injects the landing-point dispersion (true mascons are degree ≥50, not in the degree-8 field). |
| `ENABLE_SM_SYSTEMS_FAILURES` | **True** (default) | **Service-module catastrophic systems failure (the Apollo 13 mode).** Per-mission draw at `PROB_SM_CATASTROPHIC = 1/15` (empirical: 1 mission-ending SM anomaly in 15 crewed CSM flights), struck at a uniform timeline fraction; the consequence label and crew survival depend on phase (translunar/lunar-orbit/surface/trans-Earth) — LM-attached phases get the lifeboat abort, post-jettison trans-Earth has none. Largest crew-risk family in the run (~6%). When OFF, no SM systems mode. |
| `ENABLE_ENTRY_TARGETING` | False | Experimental 3-DOF differential corrector at MCC-6 for entry targeting; not used in the production config (entry targeting is handled within TEI + the entry guidance). |
| `ENABLE_HUNTEST_PROFILE` | False | Experimental HUNTEST-style multi-regime entry bank profile aimed at Apollo's 6.5 g nominal. Validated NOT-READY (a 5-variant corpus: two-segment, fixed/range-aware drag reference — all lost range control or mis-scored on coarse predictions). Kept OFF; the ~8.6 g residual stands. See the flag's in-code comment for the full corpus. |
| `ENABLE_REF_PROFILE_ENTRY` | False | Experimental OFFLINE-optimized open-loop entry reference profile (6th HUNTEST variant). Validated NOT-READY (band mis-centered; skip-out at the shallow corridor edge). Kept OFF. |

## Status (validated)

- **CODE REVIEW PASSED (2026-06-13):** all 8 modules verified correct against
  code + the 10k run — RNG/parallel determinism (shard k reproduces serial
  trial i exactly), gravity/ephemeris magnitudes, launch/TLI incl. the
  wrap-fix crossing math, lunar-ops propellant ledgers, run_mission failure
  gating (strict causal order, one failure/trial, no double-count),
  TEI/entry/splash recovery-zone geometry (great-circle 2,784 km exact),
  crew survival (reproducible, Collins≥LM invariant), stats decomposition
  (under the 2026-06-13 land+return success definition: 8545 success + 1455
  failure = 10000, of which 8404 are flawless; the pre-reclassification strict
  split was 8404+1596). The 46 high-g(>9.5) successes (steep-FPA tail under the
  12 g limit) and the below-PROB failure rates (upstream attrition) were
  investigated and confirmed correct. One latent bug found & fixed:
  `apply_survival_model` index-robustness (`reset_index`, no-op on production).
- Full physics across all phases; IGM ascent; physical finite-thrust TEI
  targeting; trans-Earth and trans-lunar MCC chains; per-astronaut crew
  survival; data-driven dashboard.
- **LATEST — launch-state continuity landed** (`ENABLE_LAUNCH_CONTINUITY`,
  default ON; see flag table): the nominal flies pad-to-splashdown as ONE
  continuous trajectory (solved 72.48° launch window, steered TLI cutoff,
  retrograde Apollo-handed capture, handedness-agnostic TEI targeting,
  splash fixed-point 2.1 km). First validation MC `outputs/apollo11_continuity48`
  (48 trials, seed 42): 77.1% — exposed that continuity's honest arrival-TIME
  dispersion (translunar coast 72.7–74.7 h) breaks frozen-epoch navigation
  stand-ins: 4 trials missed-SOI-after-MCC because **MCC-4b ran one open-loop
  pass** (the 0.15 m/s execution residual re-scatters a converged trim by
  ~200 km at MCC-4's perilune sensitivity). FIXED: MCC-4b now ITERATES
  verify→trim→re-verify (≤5 passes, in-band/no-improvement early stop;
  rng_exec is phase-local so in-band trials are bit-identical — verified on
  trials 0–2). All 4 failures rescued (perilunes 91/93/96/111 km, full
  success). Re-validation MC `outputs/apollo11_continuity48b`: **87.5%
  (Wilson 95% CI 75.3–94.1), ZERO navigation-mode failures** (remaining: 2
  entry-g steep tails, 2 descent-fuel, 1 launch, 1 TLI-starve) —
  statistically overlaps production500b's 92.1%. **The dashboard/`OUTDIR`
  now points at continuity48b** (the current-model run; production500b stays
  on disk as the pre-continuity reference). Dashboard known-limitation #1
  rewritten: architecture resolved; remaining limitation = sourced
  engine-reliability ESTIMATES + the 0.4° ascent-window residual.
  SPLASH SCATTER — FIXED in two steps (final MC `apollo11_continuity48d`:
  **87.5% [75.3–94.1], splash miss vs aimed zone median 1.3 km / fleet MAX
  2.3 km** — Apollo 11's actual was ~3 km from its aim point; dashboard
  points here). Step 1: the 1,500 km median scatter was a BACKWARDS
  LONG-SKIP artifact — the continuity fixed point had placed SPLASH_TARGET
  at bearing ~242° from the EI (behind the entry point, azimuth 54.7°), so
  the nominal flew a near-full-circumference skip (~30,000 km) to land "2 km
  accurate". Re-derived at the SHORT corridor end ~2,784 km downrange of the
  captured nominal EI: **(15.449°N, 33.712°E)**. LESSON: sanity-check any
  splash fixed point against the EI ground track (bearing ≈ entry azimuth,
  range ≈ 2,780 km). Step 2: the remaining bimodal tail (p75 3,454 km in MC
  48c) was REV-SLIPPED returns whose EI rotates up to ~110° east —
  unreachable from a fixed zone. **PER-OPPORTUNITY RECOVERY ZONES** (RTCC
  practice): each trial's zone = 2,784 km along ITS OWN entry ground track;
  `splash_miss_km` = guidance accuracy vs the zone aimed for;
  `recovery_zone_displacement_km` = the slip's operational cost (~0–200 km
  on-time, thousands for slipped revs). Worst trials went 7,264/6,710 km →
  1.5/1.7 km. Per-trial EI recorded (ei_lat/ei_lon/ei_t_s).
  HUNTEST 6.5-g — attempted, NOT ready (`ENABLE_HUNTEST_PROFILE=False`
  stays): a fixed 5-g drag-reference regime achieved peak 6.50 g but lost
  range control (500–2,500 km); real HUNTEST solves the drag reference
  against range-to-go (3-knob coupled solve, future work). Honest residual:
  nominal peak ~8.2 g vs Apollo's 6.5. Steep-tail entry-g failures (2) and
  splash region (NE Africa — timeline residual) remain documented.
- **Most recent work:** a realistic descent chain (`ENABLE_DOI`, on by default).
  The LM no longer brakes directly off the unphysical ~5×413 km capture orbit;
  instead the CSM is parked in a near-circular ~100 km orbit, a discrete DOI
  burn (~19 m/s, charged to the DPS) lowers the LM perilune to ~15 km, and an
  efficient P63-like braking law flies PDI from there. A fat-tailed manual-flying
  fuel penalty (hazard-avoidance overfly, à la Apollo 11's boulder field) makes
  descent-fuel exhaustion a genuine rare-tail risk again. Nominal lands with a
  ~95 s reserve (≈ Apollo's planned ~2-min hover budget); the as-flown Apollo 11
  ~25 s marginal landing now lives in the MC tail (min landed margin ~10 s).
  Built on the earlier B-plane TLMCC upgrade (`ENABLE_BPLANE_TLMCC`): 2-DOF
  (along-track + cross-track) solve in `phase_translunar_mcc`, approach-plane
  error ~0.04° (~18× tighter than altitude-only).
- **DEFINITIVE HEADLINE — 10,000-trial cluster production, fully-sourced
  failure model** (`apollo11_final10000`, **seed 37**, extended 2,500->10,000 — intentionally NOT
  trial-aligned with the seed-42 lineage; doubles as a seed-robustness
  check): **mission success 85.5%** (Wilson 95% CI **84.7–86.1%**, half-width
  ±0.69) under the 2026-06-13 success definition — landed on the Moon AND all
  three returned to Earth alive (Kennedy's objective), even with a recovered
  in-flight anomaly; **141 recovered-anomaly trials** (77 rendezvous-docking via
  EVA crew transfer, 21 SM-surface, 20 SM-lunar-orbit, 20 SM-trans-Earth, 3 radar)
  reclassified from failure to success. Of those successes, **84.0% flawless**
  (no anomaly; Wilson 83.3–84.7). **Crew survival 93.4%** (unchanged by the
  reclassification) (corrected from 93.2% by the dashboard audit — two
  failure modes, missed_lunar_soi_after_mcc + descent_radar_dropout_marginal,
  were falling through crew_survival.py's 0.5 default; now aliased to their
  proper survival classes). Wall clock: 2,500-trial base 1 h 57 m + 7,500-trial
  extension ~5 h 10 m (272 cores). The 10k tightens the CI ~4x vs the 2,500
  and confirms the estimate (85.4%->84.0%, within sampling). The model now carries
  ALL sourced failure modes: SM systems (Apollo-13 class, 1/15 sourced;
  5.4% total split translunar 2.0 / trans-Earth 1.5 / lunar orbit 1.2 /
  surface 0.6), two-docking decomposition (T&D 0.76 + rendezvous 0.68,
  sourced 2-of-21 capture anomalies x workaround), surface ops (LM
  electrical 1.0, tip-over 0.2, EVA suit 0.12, all sourced/anchored),
  launch family 4.0%, descent-fuel 1.2%, entry-g 1.0%, navigation
  vestigial 0.16%, **TLI modes ZERO** (the 4.8-5.9% "starve" of earlier
  runs was a wrap-boundary bug in the ignition-crossing detector — fixed +
  label-separated, see _coast_to_ignition). Splash vs aimed zone median
  1.49 km / p95 2.21 (one 174-km shallow outlier); zones on-time 68%; FPA
  [-7.37, -6.30]; TEI dv 953; peak g median 8.44 (the 6.5-g profile
  residual stands after a 6-variant corpus — see ENABLE_HUNTEST_PROFILE /
  ENABLE_REF_PROFILE_ENTRY); burn times LOI 369.3 / TEI 155.6 vs Apollo
  357.5 / 152. Seed-robustness: consistent with the seed-42 lineage after
  accounting for the model deltas (masscal 86.5% had +5-6 pts of fake
  starve and lacked the -7 pts of new sourced modes — net wash, observed).
- **Prior — 2,500-trial mass-calibrated run (pre-sourced-modes)** (`apollo11_masscal200`, seed 42, the definitive estimate): 
  full-success **86.5%** (Wilson 95% CI **85.1–87.8%**), **crew survival
  92.8%**. Run as a 200-trial validation + 2,300-trial extension on all 17
  nodes; the extension took **1 h 54 m wall** (~10 days serial-equivalent).
  Includes the as-flown mass corrections (CSM_SM_DRY 4,825 kg, SPS_ISP
  314.5): fleet burn-time medians now **LOI 367.7 s / TEI 155.3 s vs
  Apollo's 357.5 / 152** (~3%). Failure decomposition: TLI-starve 5.9%
  (watch item — engine dispersion vs the honest S-IVB reserve), launch
  family 4.3% (max-q 1.7, parking decay 2.2, S-IVB ignition 0.4), entry-g
  1.1%, descent-fuel 1.0%, docking 1.0%, navigation vestigial 0.24%.
  Splash vs aimed zone: median 1.49 km, p95 2.22 (one 87-km shallow-tail
  outlier in 2,162); zones on-time 65%. Peak g median 8.48 (HUNTEST
  five-variant corpus closed, flag stays OFF — see ENABLE_HUNTEST_PROFILE).
  Statistically consistent with production1000 (88.0% [85.8–89.9],
  pre-mass-correction).
- **Prior — 1,000-trial cluster production, pre-mass-correction**
  (`apollo11_production1000`, seed 42, run on the SLURM cluster via the
  sharding pipeline with PINNED laptop-derived targets — see submit_mc.sh /
  cluster_run.py): full-success **88.0%** (Wilson 95% CI **85.8–89.9%**),
  **crew survival 92.6%**. Failure decomposition at 1,000-trial resolution:
  TLI propellant starve 4.9% (dominant single mode; engine-dispersion
  physics, watch item), launch family 4.0% (max-q 1.8, parking decay 1.7,
  S-IVB ignition 0.5), entry-g steep tails 1.0%, docking 0.9%, descent-fuel
  0.5%, navigation vestigial 0.7%. Splash vs aimed per-opportunity zone:
  **median 1.46 km, p95 2.18 km, max 3.2 km**; zones on-time 62%, slipped
  median ~6,200 km (recorded). TEI dv median 974 (Apollo 1,008), FPA median
  −6.48, peak g median 8.46. Wall clock: 999/1000 trials in ~62 min on 272
  cores (17 nodes × 16); full completion 96 min incl. a one-time recovery
  (the cluster's 1-h DEFAULT job time limit killed one shard at 58/59 —
  submit_mc.sh now pins --time on every stage). Long-tail trials exist
  (max 29 min: heavy high-dv TEI homotopy solves). Statistically overlaps
  the laptop production500c below.
- **Prior headline — laptop 500-trial run (pre-Apollo-timing)**
  (`apollo11_production500c`, 500 trials, seed 42, the COMPLETE continuity
  model: pad-to-splashdown continuous trajectory + iterated MCC-4b +
  per-opportunity recovery zones): full-success **85.6%** (Wilson 95% CI
  **82.3–88.4%**), crew survival **91.6%**. Every failure mode is
  historically-grounded or honest-dispersion: TLI propellant starve 4.8%
  (S-IVB Isp/thrust draws against the honest ~75–175 m/s reserve — starved
  trials' launch stats are identical to survivors', so it is engine
  dispersion, not a guidance artifact; WATCH ITEM: ~1.7σ margin vs Apollo's
  ~3σ design intent), launch family 4.8% (max-q 2.2, parking decay 2.0,
  S-IVB ignition 0.6), entry-g steep tails 2.2%, descent-fuel 1.4%, docking
  0.6%, navigation vestigial (missed-SOI 0.4%, TEI-no-return 0.2%). Splash
  vs aimed recovery zone: **median 1.4 km, p95 2.2 km**, max 76 km (1 trial
  >20 km in 428 successes — a −7.2° steep delivery at 10.7 g). Zone
  on-time 67%; rev-slipped 33% (displacement median ~5,000 km, recorded).
  Peak g median 8.47 (vs Apollo's 6.5 — documented profile residual). The
  ~6.5-pt drop vs pre-continuity production500b (92.1%, 76 trials, CIs
  marginally overlap) is the cost of honestly modeling TLI propellant + the
  launched parking orbit — physics the constructed post-TLI state could not
  fail. ~354 s/trial, 4.9 h wall on 10 workers.
- Prior full-fidelity run (pre-navigation-fix, complete 500 trials,
  flown two-burn LOI + Apollo GET timeline + degree-8 GRGM1200A + guided entry,
  in `outputs/apollo11_production500/`): full-success **63.8%** (Wilson 95% CI
  **59.5–67.9%**), full-crew survival **76.4%**. Phase timing matches Apollo at
  scale (worst leg ~2 h off over 8.0 d). READ THE NUMBER WITH ITS DECOMPOSITION:
  - **Historically-grounded modes ~7.4%**: launch 4.6%, descent-fuel 2.2%,
    docking 0.6% — anchored to sourced reliabilities.
  - **Deep-space NAVIGATION/solver modes ~28.8%**: missed-SOI 14.0%,
    entry-g 10.0% (delivered entry-FPA σ≈2.0°, tails to −25.7°!), TEI-no-return
    4.8%. Real Apollo flew 9 lunar missions with ZERO occurrences of any of
    these — its ground-tracking navigation continuously re-solved trajectories,
    while the sim's open-loop solver chain (B-plane target captured once from
    the nominal; brittle TEI refinement; deadband-gated MCCs) is a coarse
    stand-in. A large share of these ~29 points is SOLVER fragility, not 1969
    risk. The NEXT fidelity frontier is navigation robustness; correcting it
    toward Apollo's actual zero-rate record implies a true estimate in the
    ~high-80s/low-90s.
  Splash: median 145 km from the recovery target, bimodal (p25 = 8 km —
  guidance-grade when delivery is good; p75 ~4,900 km on bad deliveries).
  Mean ~218 s/trial, 30 h compute. Long runs launch DETACHED (nohup+disown;
  harness background tasks get reaped by app lifecycle events).
- Prior reference runs: idealized 1000-trial `apollo11_doi1000/` **94.4%**
  (CI 92.8–95.7%, fast model: idealized Moon, unguided entry — reproducible
  with ENABLE_REAL_EPHEMERIS/ENABLE_LUNAR_SH_FIELD/ENABLE_SKIP_ENTRY_GUIDANCE
  all False); pre-B-plane baseline (`apollo11_tei200/`): ~88% with the
  artifact-driven descent-fuel mode dominant.

## Splashdown metrics

- **Dispersion / absolute miss** (~200 km median in the 1000-trial run) — spread
  around the sim's own nominal splashdown; the TEI-targeting accuracy result.
  `SPLASH_TARGET` is set to the nominal landing point, so `splash_miss_km` ≈ this
  dispersion (they coincide). **Re-derive `SPLASH_TARGET` whenever the return
  geometry changes** — a stale value inflates the "miss" to a meaningless ~8000 km
  (this happened after the B-plane/DOI work moved the nominal; fixed by repointing
  the constant at the current nominal).
- **Splashdown ACCURACY is guidance-grade; recovery REGION is the residual.**
  In the definitive config (all defaults ON, Apollo return-timing) the guided
  nominal splashes in the **western Pacific near (13.5°N, 146°E)** with ~1 km
  miss, and the 10k MC splash-vs-aimed-zone is **median 1.5 km / p95 ~2.2 km**
  (Apollo 11 itself splashed ~3 km from its aim point). Recovery uses
  per-opportunity zones: each trial aims at a zone ~2,784 km down its own
  entry ground track (`recovery_zone_displacement_km` records the operational
  cost of a rev slip). The region's **latitude matches Apollo 11's 13.3°N to
  ~0.2°**; the ~13° (~4,800 km) longitude offset from Apollo's 169.15°W is the
  residual, from the mission timeline (~8.18 d vs 8.16 d) plus the sim's
  TLI/TEI plane geometry. Shallow-tail entries (~+2σ FPA) can overfly yet
  survive. `SPLASH_TARGET` (bound at import, continuity branch) is the primary
  zone (13.48°N, 146.22°E); **re-derive it whenever the return geometry
  changes** — a stale value turns `splash_miss_km` into a meaningless thousands-
  of-km "miss". With `ENABLE_REAL_EPHEMERIS=False` the idealized Moon splashes
  in the south Pacific (~26.6°S/104°W), the documented idealized-Moon artifact.

## Known limitations (audited against the code)

> The AUTHORITATIVE, data-driven Known-Limitations list is generated from the
> actual run by `known_limitations()` in `generate_outputs.py` and rendered on
> the dashboard. The summary below is the current state; see the
> improvement-backlog memory note for effort/impact and the deferred #5 work.

1. **Saturn V ascent — reliability & guidance are estimates.** Engine-out
   probabilities (F-1 98.5%, J-2 99%, S-IVB 99.5%) are sourced estimates; the
   solved launch window (72.48°) sits ~0.4° from Apollo's as-flown 72.058°.
2. **Navigation is open-loop, not ground-tracked.** TEI/MCC targets solved
   once from the nominal and flown open-loop (not continuously re-solved from
   tracking) → larger trans-Earth trims than Apollo and rev-slips that inflate
   the rendezvous→TEI leg. The biggest fidelity frontier (#2 in the backlog).
3. **Rendezvous burns lumped, not flown.** Modeled as a propellant-budget
   check; individual CSI/CDH/terminal burns and phasing dispersions absent.
   (Docking failure itself is SOURCED — ~0.95%/docking from 2 anomalies in
   ~21 program dockings, applied at both docking events.)
4. **Surface operations** — three SOURCED failure modes (LM electrical
   ~0.85%, tip-over ~0.5%, EVA suit ~0.1%); thermal/power/dust dispersions
   not modeled.
5. **Service-module systems failure (Apollo-13 mode)** — occurrence sourced
   (1/15), but per-phase survival probabilities, the catastrophic-vs-
   recoverable split, and the uniform timing are estimates (the deferred #5
   sourcing task).
6. **Midcourse corrections use a simplified basis** — along-track +
   cross-track only (radial omitted); deadband and execution residual are
   estimates.
7. **Moon model & lunar gravity** — real July-1969 ephemeris + degree-8
   GRAIL field (default ON), but the Moon-fixed frame is a synchronous-lock
   approximation (~6.7° pole error, librations ignored) and true mascons
   (degree ≥50) are a calibrated landing-dispersion proxy. Idealized circular
   Moon reproducible with the flags OFF.
8. **Entry & splashdown** — closed-loop guided entry (default ON); nominal
   peak g ~8.6 vs Apollo's 6.5 (the 6-variant HUNTEST corpus is closed,
   `ENABLE_HUNTEST_PROFILE`/`ENABLE_REF_PROFILE_ENTRY` OFF); recovery region
   ~13° longitude off Apollo's mid-Pacific (timeline residual); no winds/
   weather, static atmosphere.
