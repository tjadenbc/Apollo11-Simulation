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
over the landing site, the lunar module's own powered-descent guidance flown on a
navigated state under the 1969 mission rules, MSFN/RTCC-style ground navigation,
constant-drag entry guidance), 1969-era engine reliabilities, and a suite of failure modes sourced to the
historical record. The aborts those failures admit — the free return and Apollo 13's
Lunar Module lifeboat among them — are flown rather than assumed. A per-astronaut
survival model then maps each mission outcome through Apollo's abort architecture to
a crew-survival result.

## Headline result

Across **10,000 trials** (seed 37, distributed on a compute cluster):

| Metric | Estimate |
|---|---|
| **Full-mission success** | **83.4 %** |
| **Crew survival** | **94.2 %** |
| **Flawless (no anomaly at all)** | **81.9 %** |

Two different uncertainties attach to that number, and they are not the same size.
The 95 % Wilson interval, **82.7–84.1 %**, is a *Monte Carlo sampling* interval: it says
how well 10,000 trials have converged on the model's answer **at its input failure
rates**. How well those rates are themselves known is a separate and much larger
question — propagating binomial posteriors on the three rates that rest on a countable
historical record (one service-module loss in 15 flights, no S-IVB restart failure in 10,
about two docking anomalies in 21) gives a 95 % credible interval of **63–91 %** on
mission success and **85–97 %** on crew survival.

"Mission success" follows NASA's actual objective: the crew reached the lunar
surface **and** all three returned to Earth alive — even if a recovered in-flight
anomaly occurred (e.g. a docking failure resolved by a contingency EVA crew
transfer). Of the 8,341 successful missions, **8,190 were flawless** (no anomaly
at all); 151 succeeded despite a recovered anomaly. A touchdown reached from an orbit
other than the planned one is not counted as a landing, and a crew whose service module
fails on the outbound coast before the Lunar Module has been extracted from the S-IVB has
no lifeboat. The 10.8-point gap between crew survival and mission success is Apollo's abort architecture at work: many
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
0.02° and an entry speed within 0.3 m/s of the flown values, a peak entry deceleration of
6.42 g against the flown 6.5, the eight-day mission duration to within 0.1 %, a powered
descent that reaches high gate 506 s after ignition (flown: 507 s) and touches down **0.2 km
from its aim point at Landing Site 2**, and a splashdown **3.3 km from its aimed recovery
zone** (Apollo 11 missed its own by about 3 km). The solved launch azimuth,
72.254°, lands within 0.2° of the flown 72.058° without that value ever entering the
solution.

The definitive run lives in
[`outputs/cluster_fleet10k_jul16_s37_fin/`](outputs/cluster_fleet10k_jul16_s37_fin/). It was flown on a
SLURM cluster with flight code whose `apollo11.py` has SHA-256 prefix `9a415490`; release 5.0.0
reproduces it (see [Reproducibility](#reproducibility)), and the `apollo11.py` in that directory is
release 5.0.0's. The previous definitive run (83.8 % / 93.7 %, legacy descent, 105-hour lifeboat, no
direct abort) remains available in release 4.1.0.
Open [`dashboard.html`](outputs/cluster_fleet10k_jul16_s37_fin/dashboard.html) in a browser
for the full breakdown — failure-mode decomposition, per-astronaut death attribution, phase
timing vs. Apollo 11's flown values, and known limitations.

## Repository layout

| Path | What it is |
|---|---|
| `apollo11.py` | The simulation (~13,500 lines): physics, every mission phase, `run_mission()`, and the `main()` / `main_parallel()` Monte Carlo drivers. Feature flags are documented in the constants block near the top. |
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
| `fast_kernels.py`, `fast_kernels.c` | Optional compiled speed-ups (ephemeris evaluation, lunar gravity field, vector arithmetic), each behind its own switch and bit-identical to the Python path; build once per machine with `python3 fast_kernels.py build`. |
| `test_verification.py` | The unit-level verification suite: 38 checks in four tiers (invariants and symmetries, independent derivations, external standards, recorded-trajectory replay). |
| `tests/`, `pytest.ini`, `Makefile` | 86 tests of the package boundary: binding, the nominal re-fly, trial re-fly, flag-off bit-identity of every switchable mechanism, and bit-identity of the compiled kernels. |
| `cluster_run.py`, `submit_mc.sh`, `submit_heating.sh`, `queue_watcher.py` | The SLURM sharding pipeline used to produce the definitive run and its heating replay on a compute cluster. The merge keeps every shard's cells byte for byte. |
| `env_fingerprint.py` | Records each job's software environment and a numerical fingerprint of it (exact bits of a fixed battery of libm, numpy, BLAS, LAPACK and scipy computations), so two environments can be compared without re-flying trials. |
| `fleet_smoke.py` | Pool smoke test used by `make fleet-smoke`. |
| `analysis/` | The Sun elevation at touchdown for the definitive run (`sun_elevation/`). Two analyses of the release-4.x definitive run are kept for reference: the midcourse-correction comparison across dispersion scales (`mcc_scale/`) and the lifeboat replay (`lifeboat_replay/`), its translunar service-module failures flown again with a 99.8-hour lifeboat limit and with the direct-abort return, which led to the configuration of the present definitive run. |
| `launch_tli_preset.json` | Legacy launch/TLI targeting preset, read only by runs that bind no package; a bound run takes its TLI controls from the package. |
| `outputs/cluster_fleet10k_jul16_s37_fin/` | **The definitive run**: per-trial results CSVs, the nominal trajectory, captured targeting products, the bound package, the run configuration, the dashboard, the crew-survival report, the parameter-uncertainty tables and sensitivities, the entry-heating replay, and figures. |

## Quick start

```bash
pip install -r requirements.txt   # numpy, scipy, pandas, matplotlib, jplephem, global-land-mask (Python 3.11+)
```

The definitive configuration binds the package and sets twelve environment flags. Set them in
the shell that launches the run (spawned workers inherit them):

```bash
export APOLLO_DELIVERED_PACKAGE=nominal_packages/apollo11_nominal_package.json
export APOLLO_DELIVERED_PACKAGE_SHA256=3a3a4b0e1d4ac3b9
export APOLLO_HAZARDS=1 APOLLO_IGM_ROBUST=1 APOLLO_SIVB_RESTART=1 APOLLO_APS_IGNITION=1 \
       APOLLO_TLC_IMPACT_RECOVERY=1 APOLLO_WATER_ZONE_REAIM=1 APOLLO_TEMCC_WIDE_TRIM=1 \
       APOLLO_NOLOI_RETURN=1 APOLLO_LIFEBOAT_LIMIT=1 \
       APOLLO_DESCENT_NAV=1 APOLLO_DIRECT_ABORT=1 APOLLO_LIFEBOAT_MAX_H=99.8
```

The compiled speed-ups change no result and cut a trial's cost about elevenfold. Build the
library once per machine (a C compiler is needed), then switch them on:

```bash
python3 fast_kernels.py build
export APOLLO_EPHEM_CACHE=1 APOLLO_COMPILED_LUNAR_FIELD=1 APOLLO_VECTOR_KERNELS=1
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

Regenerate the crew-survival results, dashboard, summary, and figures for a run directory. The
headline is scored with three settings of the survival model: crew risk by recovery access
time, a touchdown from an orbit other than the planned one counted as no landing, and no
lifeboat before the Lunar Module is extracted from the S-IVB (4.28 h):

```bash
export APOLLO_RECOVERY_MODEL=access APOLLO_OFFSITE_NO_LANDING=1 APOLLO_LM_FREE_GET_H=4.28
python3 -c "import matplotlib; matplotlib.use('Agg'); import crew_survival as c; c.OUTDIR='outputs/myrun'; c.main()"
python3 generate_outputs.py outputs/myrun
```

The two released analyses **must be given the run directory explicitly**; their defaults
still name `outputs/final`, the run directory of v3.0.0 and earlier. For the definitive run, with
the definitive configuration and the scoring settings above exported:

```bash
python3 epistemic_uncertainty.py outputs/cluster_fleet10k_jul16_s37_fin 37        # run dir, seed
APOLLO_HEAT_OUTDIR=outputs/cluster_fleet10k_jul16_s37_fin python3 entry_heating_validate.py 25 10   # trials, workers
```

`entry_heating_fleet.py` and `entry_heating_boundary.py` read the same `APOLLO_HEAT_OUTDIR`.

Verify the implementation:

```bash
python3 test_verification.py    # tiers 1-3 run here; tier 4 looks for outputs/final, so replay the definitive run with:
python3 -c "import test_verification as t; p='outputs/cluster_fleet10k_jul16_s37_fin/nominal_traj.npz'; t.t4_recorded_trajectory(p); t.t4_phase_handoff(p)"
make test PY=python3            # the 86 package-boundary tests (~12 min, mostly the full re-fly)
```

With the compiled speed-ups a trial costs a median of 80 s on one core of the cluster that
produced the definitive run, and a 10,000-trial fleet about 210 core-hours. The definitive run
was produced on a SLURM cluster via `./submit_mc.sh`.

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
| `APOLLO_LIFEBOAT_LIMIT` | That return must reach entry within the Lunar Module's endurance: 105 hours (Apollo 13's batteries) unless `APOLLO_LIFEBOAT_MAX_H` sets another limit. |

`APOLLO_DISPERSION_SCALE` (default 1) multiplies the navigation and execution dispersions
together, the powered descent's navigation error and descent-orbit execution error among them;
the paper's stress test sets it to 2 and 4. A no-landing, cislunar-only profile is enabled with the
`APOLLO_NO_LANDING=1` environment variable.

Release 4.1.0 adds three further switches, read from the environment and OFF unless set:

| Flag | What it turns on |
|---|---|
| `APOLLO_TEI_DELIVERED_RETARGET` | The trans-Earth ignition-time update: the delivered trans-Earth burn is carried forward on the tracked orbit, its ignition moved to where the trial's orbital phase matches the delivered one and its magnitude trimmed to a perigee in the entry corridor. The paper's 21 July mission, whose return leaves the Moon out of the orbit plane, needs it. |
| `APOLLO_DIRECT_ABORT` | A second return for an early service-module failure on the outbound coast: a direct abort that turns back short of the Moon, solved on the same descent-engine budget and lifeboat limit as the return around the Moon. Used by `analysis/lifeboat_replay/`. |
| `APOLLO_FREE_RETURN` | A circumlunar free-return profile with no lunar-orbit insertion: the stack rounds the Moon with the Lunar Module attached, and a corridor burn two hours after pericynthion stands where the trans-Earth injection stands in the landing mission. |

Release 5.0.0 adds the switches below, all OFF unless set. The definitive run sets the first two
and `APOLLO_DIRECT_ABORT`; the speed-ups are optional.

| Flag | What it turns on |
|---|---|
| `APOLLO_DESCENT_NAV` | The powered descent steered by the lunar module guidance computer's own equations: Klumpp's quartic guidance with the flight program's coefficients and thrust limiting and Apollo 11's pre-launch targets, flown on the true state plus a navigation error drawn at ignition and sized to Apollo 11's premission 99 % landing ellipse, with landing-radar updates of altitude and velocity, a crew phase of drawn duration, and the mission rules: a go/no-go before ignition (one slip, then a wave-off) and aborts for no radar data by 10,000 ft, no throttle-down 15 s into the approach phase, or the propellant land-or-abort call above 100 ft. The time after the descent keeps the flown schedule. `APOLLO_WAIVEOFF_KM` (default 10) and `APOLLO_LR_NO_ACQ_PROB` (default 0.01) set the go/no-go reach and the radar's acquisition-failure rate. |
| `APOLLO_LIFEBOAT_MAX_H` | The lifeboat endurance in hours (default 105). The definitive run uses 99.8, Mission Control's projection for Apollo 13's water. |
| `APOLLO_EPHEM_CACHE`, `APOLLO_COMPILED_LUNAR_FIELD`, `APOLLO_VECTOR_KERNELS` | The compiled speed-ups of `fast_kernels.c`: the DE440 evaluation, the lunar spherical-harmonic field, and the 3-vector arithmetic and Earth gravity. Each reproduces the Python path bit for bit (`tests/test_fast_kernels.py`). |

The survival model (`crew_survival.py`) reads three scoring settings, also OFF unless set; the
headline sets all three:

| Setting | Effect |
|---|---|
| `APOLLO_RECOVERY_MODEL=access` | A full success that splashes outside the primary recovery area is charged 0.5 % crew risk inside the recovery plan's contingency band and 1.5 % outside it (`distance` is a sensitivity rule; `none`, the default, charges nothing). |
| `APOLLO_OFFSITE_NO_LANDING=1` | A touchdown flown from an orbit other than the planned descent orbit is not counted as a landing. |
| `APOLLO_LM_FREE_GET_H=4.28` | A service-module failure after translunar injection and before the Lunar Module is extracted from the S-IVB has no lifeboat. |

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

**Release 5.0.0 against the definitive run's flight code.** The definitive run was flown with an
`apollo11.py` whose SHA-256 prefix is `9a415490`. Release 5.0.0's differs from it in comments and in
one line that applies `APOLLO_DISPERSION_SCALE` to the descent's navigation and descent-orbit
errors, a multiplication by exactly 1 at the definitive scale. Twelve trials of the definitive
configuration (seed 37, the twelve trial indices below) were flown with both on one machine: all
141 result columns agree in every trial, and only the wall-clock timing fields differ. The 86
package-boundary tests and the 38 verification checks pass on release 5.0.0.

**Release 4.1.0 against 4.0.0.** With its three new switches off, release 4.1.0 reproduces
release 4.0.0. Twelve trials of the definitive configuration (seed 37, trial indices 0, 1, 3, 60,
85, 123, 177, 337, 352, 706, 1117 and 1428: clean flights, an S-IVB restart failure, a rendezvous
docking failure, and service-module failures in lunar orbit and on both coasts, with a free return,
a descent-engine return and no return among them) were flown on both releases on one machine, and
the two agree in every one of the 117 result columns of every trial.

## Related projects

Sibling simulations apply the same fidelity-first methodology to NASA's
**Artemis I** mission,
[ArtemisI-Simulation](https://github.com/tjadenbc/ArtemisI-Simulation), and to the crewed
**Artemis II** lunar flyby,
[ArtemisII-Simulation](https://github.com/tjadenbc/ArtemisII-Simulation). The
matched 1969-vs-2026 comparison is maintained with the Artemis I project.

## Definitive-run trial data

The per-trial debug JSONs for the definitive run (`trial_0.json` …
`trial_9999.json`) are distributed as the release asset
`cluster_fleet10k_jul16_s37_fin_trials.tar.gz` rather than committed to the repository.
Extract it inside `outputs/cluster_fleet10k_jul16_s37_fin/`; it unpacks to `trials/`. Each
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

The statement below is the Acknowledgements section of the paper, verbatim.

This work was produced in a sustained collaboration between the author and Claude (Anthropic), a large language model. The conception is the author's: the question, the decision to answer it with a physics-integrated Monte Carlo rather than a reliability fault tree, and the design doctrines that define the simulation and its validation. Claude performed essentially all of the implementation. Specifically: (i) text: every section, from the abstract to the appendices, was drafted by Claude at the author's direction and then reviewed and edited by the author; (ii) figures and tables: every figure was produced from the simulation's output by code that Claude wrote, every table was compiled by Claude from the run records and the cited sources, and no image was made by an image-generation model; (iii) code: the simulator, the mission-design solver and the analysis, cluster and typesetting scripts were written by Claude under the author's direction and validation. Design and refinement of the simulation proceeded in stages and were a genuine dialogue between the two, in which Claude's artifacts were often constrained, redirected and sometimes rejected by the author's judgment and validation procedure.

## License

MIT — see [`LICENSE`](LICENSE). The license covers the simulation source code;
embedded NASA scientific data carries its own (public-domain) terms, noted in
the license file.
