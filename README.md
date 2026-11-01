# Apollo 11 Monte Carlo Simulation

A physics-integrated Monte Carlo that estimates Apollo 11's **probability of
mission success and crew survival** by flying the entire mission — launch pad to
splashdown — as one continuous, numerically-integrated trajectory, thousands of
times.

Every powered manoeuvre is a finite-thrust integration and every coast an n-body
integration under Earth (+J2–J6 near-field), Moon and Sun gravity, using the JPL
**DE440** planetary/lunar ephemeris and a degree-8 GRAIL gravity field, with faithful
renderings of Apollo's own guidance laws (iterative-guidance ascent, B-plane
midcourse corrections, a two-burn lunar-orbit insertion that steers the orbit plane
over the landing site, MSFN/RTCC-style ground navigation, constant-drag entry
guidance), 1969-era engine reliabilities, and a suite of failure modes sourced to the
historical record. The aborts those failures admit — the free return and Apollo 13's
Lunar Module lifeboat among them — are flown rather than assumed. A per-astronaut
survival model then maps each mission outcome through Apollo's abort architecture to
a crew-survival result.

## Headline result

Across **10,000 trials** (seed 37, distributed on a compute cluster):

| Metric | Estimate |
|---|---|
| **Full-mission success** | **83.8 %** |
| **Crew survival** | **93.7 %** |
| **Flawless (no anomaly at all)** | **82.3 %** |

Two different uncertainties attach to that number, and they are not the same size.
The 95 % Wilson interval, **83.1–84.5 %**, is a *Monte Carlo sampling* interval: it says
how well 10,000 trials have converged on the model's answer **at its input failure
rates**. How well those rates are themselves known is a separate and much larger
question — propagating binomial posteriors on the three rates that rest on a countable
historical record (one service-module loss in 15 flights, no S-IVB restart failure in 10,
about two docking anomalies in 21) gives a 95 % credible interval of **63–92 %** on
mission success and **85–96 %** on crew survival.

"Mission success" follows NASA's actual objective: the crew reached the lunar
surface **and** all three returned to Earth alive — even if a recovered in-flight
anomaly occurred (e.g. a docking failure resolved by a contingency EVA crew
transfer). Of the 8,380 successful missions, **8,225 were flawless** (no anomaly
at all); 155 succeeded despite a recovered anomaly. The 9.9-point gap between crew
survival and mission success is Apollo's abort architecture at work: many
missions never complete the objective but still bring the crew home — most often a
S-IVB that fails to relight for translunar injection, which strands the mission in a
safe Earth parking orbit.

The mission is **re-flown, not replayed**. A separate mission-design solver solves the
launch azimuth and the translunar-injection, lunar-orbit-insertion and trans-Earth-injection
burns from the launch window, the landing site, the mission rules and the mission's
targets, and delivers them as a sealed, SHA-256-fingerprinted package
([`nominal_packages/apollo11_nominal_package.json`](nominal_packages/apollo11_nominal_package.json)).
The simulation binds that package at import and re-solves every burn from the state its own
integration reaches; it reads no recorded state of the Apollo 11 flight. Where the flight
record does enter, it enters as vehicle data, a rule or a target — the as-flown masses and
SPS thrust calibration, the Mission Report's burn-execution error model, the liftoff time,
the lunar-orbit-insertion ignition lead, and, as targets, the achieved pericynthion, the
trans-Earth-injection and entry-interface times, the −6.495° entry angle and the USS Hornet
recovery zone.

The nominal reproduces Apollo 11's flown record end to end — the lunar-orbit-insertion and
trans-Earth burn times to within two and three seconds, an entry flight-path angle within
0.02° and an entry speed within 0.1 m/s of the flown values, a peak entry deceleration of
6.42 g against the flown 6.5, the eight-day mission duration to within 0.1 %, a lunar
touchdown **0.7 km from the planned landing site**, and a splashdown **3.3 km from its aimed
recovery zone** (Apollo 11 missed its own by about 3 km). The solved launch azimuth,
72.254°, lands within 0.2° of the flown 72.058° without that value ever entering the
solution.

The definitive run lives in
[`outputs/cluster_fleet10k_jul16_s37_def9/`](outputs/cluster_fleet10k_jul16_s37_def9/).
Open [`dashboard.html`](outputs/cluster_fleet10k_jul16_s37_def9/dashboard.html) in a browser
for the full breakdown — failure-mode decomposition, per-astronaut death attribution, phase
timing vs. Apollo 11's flown values, and known limitations.

## Repository layout

| Path | What it is |
|---|---|
| `apollo11.py` | The simulation (~12,200 lines): physics, every mission phase, `run_mission()`, and the `main()` / `main_parallel()` Monte Carlo drivers. Feature flags are documented in the constants block near the top. |
| `delivered.py` | Binds the delivered nominal package at import (named by `APOLLO_DELIVERED_PACKAGE`, checked against `APOLLO_DELIVERED_PACKAGE_SHA256`). |
| `nominal_packages/apollo11_nominal_package.json` | The mission-design solver's sealed package for the 16 July 1969 opportunity (`APOLLO11_NOMINAL_JUL16_BUILD1`, SHA-256 prefix `3a3a4b0e`): every target and control with its provenance class. |
| `check_artifact.py`, `check_delivered_nominal.py` | Structural and acceptance checks of a package, run before spending compute. |
| `refly_delivered_nominal.py` | Deterministic re-fly of the package's nominal in the simulation's own physics. |
| `crew_survival.py` | Per-astronaut survival model; reads the MC results and writes `results_with_survival.csv` plus crew statistics. |
| `generate_outputs.py` | Builds `dashboard.html`, `summary.txt`, and the figures from a run directory (data-driven). |
| `lunar_gravity_coeffs.py` | Embedded GRAIL GRGM1200A spherical-harmonic coefficients (see data note below). |
| `od_filter.py` | STM-LinCov orbit-determination covariance primitives for the MSFN ground-navigation model. |
| `epistemic_uncertainty.py` | Parameter (epistemic) uncertainty of the headline: reweights the fleet over the three failure rates that rest on a countable record and propagates their Beta posteriors to a credible interval. |
| `entry_heating_validate.py`, `entry_heating_fleet.py`, `entry_heating_boundary.py` | Sutton–Graves stagnation-point entry heating, recovered by re-flying each trial's entry from its recorded interface state (a 25-trial gate, the sharded fleet driver, and a corridor-boundary sweep). |
| `test_verification.py` | The unit-level verification suite: 38 checks in four tiers (invariants and symmetries, independent derivations, external standards, recorded-trajectory replay). |
| `tests/`, `pytest.ini`, `Makefile` | 76 tests of the package boundary: binding, the nominal re-fly, trial re-fly, and flag-off bit-identity of every switchable mechanism. |
| `cluster_run.py`, `submit_mc.sh`, `submit_heating.sh`, `queue_watcher.py` | The SLURM sharding pipeline used to produce the definitive run and its heating replay on a compute cluster. |
| `fleet_smoke.py` | Pool smoke test used by `make fleet-smoke`. |
| `analysis/` | The midcourse-correction comparison across dispersion scales (`mcc_scale/`) and the Sun elevation at touchdown (`sun_elevation/`). |
| `launch_tli_preset.json` | Legacy launch/TLI targeting preset, read only by runs that bind no package; a bound run takes its TLI controls from the package. |
| `outputs/cluster_fleet10k_jul16_s37_def9/` | **The definitive run**: per-trial results CSVs, the nominal trajectory, captured targeting products, the bound package, the run configuration, the dashboard, the crew-survival report, the parameter-uncertainty tables, the entry-heating replay and boundary sweep, and figures. |

## Quick start

```bash
pip install -r requirements.txt   # numpy, scipy, pandas, matplotlib, jplephem, global-land-mask (Python 3.11+)
```

The definitive configuration binds the package and sets nine environment flags. Set them in
the shell that launches the run (spawned workers inherit them):

```bash
export APOLLO_DELIVERED_PACKAGE=nominal_packages/apollo11_nominal_package.json
export APOLLO_DELIVERED_PACKAGE_SHA256=3a3a4b0e1d4ac3b9
export APOLLO_HAZARDS=1 APOLLO_IGM_ROBUST=1 APOLLO_SIVB_RESTART=1 APOLLO_APS_IGNITION=1 \
       APOLLO_TLC_IMPACT_RECOVERY=1 APOLLO_WATER_ZONE_REAIM=1 APOLLO_TEMCC_WIDE_TRIM=1 \
       APOLLO_NOLOI_RETURN=1 APOLLO_LIFEBOAT_LIMIT=1
```

Run the Monte Carlo (resumable — checkpoints `results.csv` every trial):

```bash
# parallel (recommended — bit-identical to serial)
python3 -c "import apollo11; apollo11.main_parallel(n=100, outdir='outputs/myrun', seed=42, workers=10)"

# serial
python3 -c "import apollo11; apollo11.main(n=200, outdir='outputs/myrun', seed=42)"
```

> **Note:** runs **must** be launched via `python3 -c` or a `__main__`-guarded
> script, never a stdin heredoc — macOS multiprocessing uses the `spawn` start
> method, which re-imports `__main__` in each worker.

Regenerate the crew-survival results, dashboard, summary, and figures for a run directory:

```bash
python3 -c "import matplotlib; matplotlib.use('Agg'); import crew_survival as c; c.OUTDIR='outputs/myrun'; c.main()"
python3 generate_outputs.py outputs/myrun
```

The two released analyses **must be given the run directory explicitly**; their defaults
still name `outputs/final`, the run directory of v3.0.0 and earlier. For the definitive run:

```bash
python3 epistemic_uncertainty.py outputs/cluster_fleet10k_jul16_s37_def9 37        # run dir, seed
APOLLO_HEAT_OUTDIR=outputs/cluster_fleet10k_jul16_s37_def9 python3 entry_heating_validate.py 25 10   # trials, workers
```

`entry_heating_fleet.py` and `entry_heating_boundary.py` read the same `APOLLO_HEAT_OUTDIR`.

Verify the implementation:

```bash
python3 test_verification.py    # tiers 1-3 run here; tier 4 looks for outputs/final, so replay the definitive run with:
python3 -c "import test_verification as t; p='outputs/cluster_fleet10k_jul16_s37_def9/nominal_traj.npz'; t.t4_recorded_trajectory(p); t.t4_phase_handoff(p)"
make test PY=python3            # the 76 package-boundary tests (~19 min, mostly the full re-fly)
```

A trial costs a median of 862 s on one core of the cluster that produced the definitive
run. The definitive 10,000-trial run was produced on a SLURM cluster via `./submit_mc.sh`.

## Configuration

The model's fidelity features are controlled by flags in the constants block near
the top of `apollo11.py`, each documented inline; turning one OFF is bit-identical to the
pre-feature behaviour. Nine of them are read from the environment and are OFF unless set,
which is why the definitive configuration above sets them:

| Flag | What it turns on |
|---|---|
| `APOLLO_HAZARDS` | The 1969-environment hazards (SPS ignition, micrometeoroid, solar particle event, guidance-platform loss, pyrotechnic separation, heatshield, parachutes) with exposure scaling. |
| `APOLLO_IGM_ROBUST` | Accepts the ascent guidance cutoff on the achieved orbit's quality rather than on the solver's return code. |
| `APOLLO_SIVB_RESTART` | The S-IVB restart failure at translunar injection (4.5 % per restart). |
| `APOLLO_APS_IGNITION` | The ascent-engine ignition failure at lunar liftoff. |
| `APOLLO_TLC_IMPACT_RECOVERY` | A translunar trajectory tracked onto a lunar-impact course is corrected at the next opportunity rather than flown into the Moon. |
| `APOLLO_WATER_ZONE_REAIM` | A recovery zone on or near land is moved along the entry track into water before entry. |
| `APOLLO_TEMCC_WIDE_TRIM` | The trans-Earth flight-path-angle trim widens from ±8 m/s to ±30 m/s when the narrow window has no solution. |
| `APOLLO_NOLOI_RETURN` | A crew that loses its service module on the outbound coast, or whose SPS fails to light for lunar-orbit insertion, is returned on the free return or a descent-engine burn, flown. |
| `APOLLO_LIFEBOAT_LIMIT` | That return must reach entry within Apollo 13's 105-hour Lunar Module endurance. |

`APOLLO_DISPERSION_SCALE` (default 1) multiplies the navigation and execution dispersions
together; the paper's stress test sets it to 2 and 4. A no-landing, cislunar-only profile is enabled with the
`APOLLO_NO_LANDING=1` environment variable.

Binding the package turns on the JPL **DE440** ephemeris, whose kernel is **not
redistributed** here (~114 MB; NASA/JPL publishes it directly). Point `APOLLO_JPL_KERNEL`
at your own copy, or drop one at `ephemeris/de440.bsp` beside `apollo11.py`, which is the
default location. With the kernel unreadable the simulation **stops rather than silently
falling back** to an analytic ephemeris.

## Reproducibility

Serial, parallel and sharded cluster runs are **bit-identical on the same platform**:
per-trial perturbations are generated up front from a single seeded stream and dispatched
by trial index, so a trial's outcome depends only on its index. The first 250 trials of the
definitive fleet, flown again in a separate run, matched it in every result both runs
record. On a different operating system a last-digit difference in the floating-point
library changes the navigation and hazard draws seeded from each trial's own state, and two
runs then agree statistically rather than trial for trial. Every job binds the package by
path and SHA-256 and refuses to start on a fingerprint mismatch; the nominal targeting
products (`ei_target.json`, `bplane_target.json`, `ca_target.json`, `od_cov.json`,
`nominal_results.json`, `delivered_package.json`) are pinned into the run directory, and a
preflight check compares each job's effective configuration with the pinned
`run_config.json` and refuses to start on any mismatch.

## Related project

A sibling simulation applies the same fidelity-first methodology to NASA's
**Artemis I** mission:
[ArtemisI-Simulation](https://github.com/tjadenbc/ArtemisI-Simulation). The
matched 1969-vs-2026 comparison is maintained with that project.

## Definitive-run trial data

The per-trial debug JSONs for the definitive run (`trial_0.json` …
`trial_9999.json`, plus `trial_nominal.json`) are distributed as the release asset
`cluster_fleet10k_jul16_s37_def9_trials.tar.gz` rather than committed to the repository.
Extract it inside `outputs/cluster_fleet10k_jul16_s37_def9/`; it unpacks to `trials/`. Each
file is a full per-trial overview — the phase timeline (per-phase GET, mission-elapsed
duration) and every outcome field — complementing the per-trial summary rows in the
committed CSVs.

## Data provenance

The lunar gravity field (`lunar_gravity_coeffs.py`) is derived from NASA's
**GRAIL GRGM1200A** model; the definitive run's ephemeris is JPL **DE440** (read from the
kernel named above, not redistributed here), with an analytic Meeus series retained as the
flag-off fallback. The nominal package is the output of a separate mission-design solver;
each quantity in it carries its provenance class (a named primary or secondary source, a
derivation, a value of this simulation's own model, or a solve result). Mission constants
are public NASA values. NASA data products are in the public domain in the U.S. The MIT
license below covers the original source code only — see `LICENSE` for the third-party
data notice.

## Use of generative AI

This work was produced in a sustained collaboration between the author and Claude
(Anthropic), a large language model used across successive versions over the project's
development, and the division of labor was consequential enough to warrant a fuller statement
than the customary disclosure line. The conception is the author's: the question, the
decision to answer it with a physics-integrated Monte Carlo rather than a reliability fault
tree, and the design doctrines that define the simulation and its validation. Claude performed
essentially all of the implementation. Design and refinement of the simulation proceeded in stages and were a genuine dialogue between the two, in which Claude's artifacts were often constrained, redirected, and sometimes rejected by the author's
judgment and/or validation procedure. The honest summary is that the author served as principal investigator —
conception, judgment, quality control, and accountability — while Claude served as the
research staff: construction, diagnosis, analysis, and drafting at a speed and volume no
individual could match. The author could not have completed this project without Claude,
and Claude could not have completed this project without the author.

## License

MIT — see [`LICENSE`](LICENSE). The license covers the simulation source code;
embedded NASA scientific data carries its own (public-domain) terms, noted in
the license file.
