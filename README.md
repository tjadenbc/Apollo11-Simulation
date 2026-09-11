# Apollo 11 Monte Carlo Simulation

A physics-integrated Monte Carlo that estimates Apollo 11's **probability of
mission success and crew survival** by flying the entire mission — launch pad to
splashdown — as one continuous, numerically-integrated trajectory, thousands of
times.

Every powered manoeuvre is a finite-thrust integration and every coast a
three-body integration under Earth (+J2–J6 near-field) and Moon gravity, using the
JPL **DE440** planetary/lunar ephemeris and a degree-8 GRAIL gravity field, with faithful
renderings of Apollo's own guidance laws (iterative-guidance ascent, B-plane
midcourse corrections, a two-burn lunar-orbit insertion, MSFN/RTCC-style ground
navigation, constant-drag entry guidance), 1969-era engine reliabilities, and a
suite of failure modes sourced to the historical record. A per-astronaut
survival model then maps each mission outcome through Apollo's abort
architecture to a crew-survival result.

## Headline result

Across **10,000 trials** (seed 37, distributed on a compute cluster):

| Metric | Estimate |
|---|---|
| **Full-mission success** | **83.6 %** |
| **Crew survival** | **94.2 %** |
| **Flawless (no anomaly at all)** | **82.1 %** |

Two different uncertainties attach to that number, and they are not the same size.
The 95 % Wilson interval, **82.9–84.3 %**, is a *Monte Carlo sampling* interval: it says
how well 10,000 trials have converged on the model's answer **at its input failure
rates**. How well those rates are themselves known is a separate and much larger
question — propagating binomial posteriors on the three rates that rest on a countable
historical record (one service-module loss in 15 flights, no S-IVB restart failure in 10,
about two docking anomalies in 21) gives a 95 % credible interval of **63–92 %** on
mission success and **87–96 %** on crew survival.

"Mission success" follows NASA's actual objective: the crew reached the lunar
surface **and** all three returned to Earth alive — even if a recovered in-flight
anomaly occurred (e.g. a docking failure resolved by a contingency EVA crew
transfer). Of the 8,359 successful missions, **8,208 were flawless** (no anomaly
at all); 151 succeeded despite a recovered anomaly. The 10.6-point gap between crew
survival and mission success is Apollo's abort architecture at work: many
missions never complete the objective but still bring the crew home — most often a
S-IVB that fails to relight for translunar injection, which strands the mission in a
safe Earth parking orbit.

The nominal trajectory reproduces Apollo 11's flown record end to end — burn times to
within about a second, trans-Earth injection ΔV to within **0.1 %**, an eight-day mission
duration to within 0.1 %, a peak entry deceleration of 6.38 g against the flown 6.5, a lunar
touchdown **8.0 km from the aim point**, and a splashdown **14.5 km from Apollo 11's actual
recovery position**. Crucially it is derived without using any of Apollo's flown trajectory
values: the launch azimuth, the injection-guidance preset and the descent-orbit timing are
all solved from physics and geometry rather than commanded, so the model predicts the flight
rather than replaying it. Re-deriving it from scratch at the program's other 1969 launch
opportunities and running a matched 10,000-trial fleet at each gives **83.5–83.8 %** — the
estimate is a property of the vehicle and mission design, not of the day it flew.

The definitive run lives in [`outputs/final/`](outputs/final/).
Open [`outputs/final/dashboard.html`](outputs/final/dashboard.html) in a browser
for the full breakdown — failure-mode decomposition, per-astronaut death
attribution, phase timing vs. Apollo 11's flown values, and known limitations.

## Repository layout

| Path | What it is |
|---|---|
| `apollo11.py` | The simulation (~10,100 lines): physics, every mission phase, `run_mission()`, and the `main()` / `main_parallel()` Monte Carlo drivers. Feature flags are documented in the constants block near the top. |
| `crew_survival.py` | Per-astronaut survival model; reads the MC results and writes `results_with_survival.csv` plus crew statistics. |
| `generate_outputs.py` | Builds `dashboard.html`, `summary.txt`, and the figures from a run directory (data-driven). |
| `lunar_gravity_coeffs.py` | Embedded GRAIL GRGM1200A spherical-harmonic coefficients (see data note below). |
| `od_filter.py` | STM-LinCov orbit-determination covariance primitives for the MSFN ground-navigation model. |
| `epistemic_uncertainty.py` | Parameter (epistemic) uncertainty of the headline: reweights the fleet over the three failure rates that rest on a countable record and propagates their Beta posteriors to a credible interval. |
| `entry_heating_validate.py`, `entry_heating_fleet.py`, `entry_heating_boundary.py` | Sutton–Graves stagnation-point entry heating, recovered by re-flying each trial's entry from its recorded interface state (a 25-trial gate, the sharded fleet driver, and a corridor-boundary sweep). |
| `cluster_run.py`, `submit_mc.sh` | The SLURM sharding pipeline used to produce the definitive run on a compute cluster. |
| `launch_tli_preset.json` | Cached launch/TLI targeting preset, pinned for cross-machine reproducibility. |
| `outputs/final/` | **The definitive run**: per-trial results CSVs, the nominal trajectory, captured targeting products, the run configuration, the dashboard, a stage-by-stage realism audit, the parameter-uncertainty tables, the per-entry heating, and figures. |

## Quick start

```bash
pip install -r requirements.txt   # numpy, scipy, pandas, matplotlib (Python 3.11+)
```

Run the Monte Carlo (resumable — checkpoints `results.csv` every trial):

```bash
# parallel (recommended — ~10× faster, bit-identical to serial)
python3 -c "import apollo11; apollo11.main_parallel(n=100, outdir='outputs/myrun', seed=42, workers=10)"

# serial
python3 -c "import apollo11; apollo11.main(n=200, outdir='outputs/myrun', seed=42)"
```

> **Note:** runs **must** be launched via `python3 -c` or a `__main__`-guarded
> script, never a stdin heredoc — macOS multiprocessing uses the `spawn` start
> method, which re-imports `__main__` in each worker.

Regenerate the dashboard, summary, and figures from a run directory (the
`OUTDIR` constant at the top of each script selects which run it reads/writes):

```bash
python3 -c "import matplotlib; matplotlib.use('Agg'); import crew_survival;    crew_survival.main()"
python3 -c "import matplotlib; matplotlib.use('Agg'); import generate_outputs;  generate_outputs.main()"
```

Each trial costs ~350 s on a modern laptop core. The definitive 10,000-trial run
was produced on a SLURM cluster via `./submit_mc.sh`.

## Configuration

The model's fidelity features are controlled by flags in the constants block near
the top of `apollo11.py` — all default ON for the fidelity-first configuration,
each documented inline, and turning one OFF is bit-identical to the pre-feature
behaviour. A no-landing, cislunar-only profile is enabled with the
`APOLLO_NO_LANDING=1` environment variable.

The definitive configuration reads the JPL **DE440** kernel, which is **not redistributed**
here (~114 MB; NASA/JPL publishes it directly). Set `APOLLO_JPL_EPHEMERIS=1` and point
`APOLLO_JPL_KERNEL` at your own copy, or drop one at `ephemeris/de440.bsp` beside
`apollo11.py`, which is the default location. With the flag set and the kernel unreadable the
simulation **stops rather than silently falling back** to an analytic ephemeris.

## Reproducibility

Local serial and parallel drivers are **bit-identical** (per-trial perturbations
are pre-generated in the main process and dispatched by trial index). Cluster
runs are *not* bit-identical to a laptop (different CPU/BLAS numerics), so each
cluster run is treated as its own population. The captured targeting products
(`ei_target.json`, `bplane_target.json`, `ca_target.json`, `od_cov.json`,
`nominal_results.json`, `launch_tli_preset.json`) are pinned into the run
directory so cross-machine numerical skew cannot flip the marginal
nominal-trajectory branches — but pinning is a convenience, not a requirement:
the nominal derivation carries a robust multi-seed fallback and a
`check_nominal()` plausibility gate, and an independent cluster derivation of
the nominal reproduced the pinned fleet's success count exactly in a
2,000-trial matched-seed test.

## Related project

A sibling simulation applies the same fidelity-first methodology to NASA's
**Artemis I** mission:
[ArtemisI-Simulation](https://github.com/tjadenbc/ArtemisI-Simulation). The
matched 1969-vs-2026 comparison is maintained with that project.

## Definitive-run trial data

The per-trial debug JSONs for the definitive run (`trial_0.json` …
`trial_9999.json`) are distributed as the release asset
`apollo11_final_trials.tar.gz` rather than committed to the repository. Extract
into `outputs/final/trials/`. Each file is a full per-trial overview — the phase
timeline (per-phase GET, mission-elapsed duration) and every outcome field —
complementing the per-trial summary rows in the committed CSVs.

## Data provenance

The lunar gravity field (`lunar_gravity_coeffs.py`) is derived from NASA's
**GRAIL GRGM1200A** model; the definitive run's ephemeris is JPL **DE440** (read from the
kernel named above, not redistributed here), with an analytic Meeus series retained as the
flag-off fallback. Mission constants are public NASA values. NASA data products are in the public domain in
the U.S. The MIT license below covers the original source code only — see
`LICENSE` for the third-party data notice.

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
