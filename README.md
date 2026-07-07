# Apollo 11 Monte Carlo Simulation

A physics-integrated Monte Carlo that estimates Apollo 11's **probability of
mission success and crew survival** by flying the entire mission — launch pad to
splashdown — as one continuous, numerically-integrated trajectory, thousands of
times.

Every powered manoeuvre is a finite-thrust integration and every coast a
three-body integration under Earth (+J2–J6 near-field) and Moon gravity, using a
real July-1969 lunar ephemeris and a degree-8 GRAIL gravity field, with faithful
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
| **Full-mission success** | **88.1 %**  (95 % CI 87.4–88.7 %) |
| **Crew survival** | **94.1 %** |

"Mission success" follows NASA's actual objective: the crew reached the lunar
surface **and** all three returned to Earth alive — even if a recovered in-flight
anomaly occurred (e.g. a docking failure resolved by a contingency EVA crew
transfer). Of the 8,808 successful missions, **8,643 were flawless** (no anomaly
at all); 165 succeeded despite a recovered anomaly. The ~6-point gap between crew
survival and mission success is Apollo's abort architecture at work: many
missions never complete the objective but still bring the crew home.

The definitive run lives in [`outputs/final/`](outputs/final/).
Open [`outputs/final/dashboard.html`](outputs/final/dashboard.html) in a browser
for the full breakdown — failure-mode decomposition, per-astronaut death
attribution, phase timing vs. Apollo 11's flown values, and known limitations.
A companion [`outputs/final/Apollo11_Realism_Audit.html`](outputs/final/Apollo11_Realism_Audit.html)
documents the stage-by-stage fidelity review behind the model.

## Repository layout

| Path | What it is |
|---|---|
| `apollo11.py` | The simulation (~10,100 lines): physics, every mission phase, `run_mission()`, and the `main()` / `main_parallel()` Monte Carlo drivers. Feature flags are documented in the constants block near the top. |
| `crew_survival.py` | Per-astronaut survival model; reads the MC results and writes `results_with_survival.csv` plus crew statistics. |
| `generate_outputs.py` | Builds `dashboard.html`, `summary.txt`, and the figures from a run directory (data-driven). |
| `lunar_gravity_coeffs.py` | Embedded GRAIL GRGM1200A spherical-harmonic coefficients (see data note below). |
| `od_filter.py` | STM-LinCov orbit-determination covariance primitives for the MSFN ground-navigation model. |
| `cluster_run.py`, `submit_mc.sh` | The SLURM sharding pipeline used to produce the definitive run on a compute cluster. |
| `launch_tli_preset.json` | Cached launch/TLI targeting preset, pinned for cross-machine reproducibility. |
| `outputs/final/` | **The definitive run**: per-trial results CSVs, the nominal trajectory, captured targeting products, the dashboard, the realism-audit report, and figures. |

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

## Reproducibility

Local serial and parallel drivers are **bit-identical** (per-trial perturbations
are pre-generated in the main process and dispatched by trial index). Cluster
runs are *not* bit-identical to a laptop (different scipy/numpy builds), so each
cluster run is treated as its own population. The captured targeting products
(`ei_target.json`, `bplane_target.json`, `ca_target.json`, `od_cov.json`,
`nominal_results.json`, `launch_tli_preset.json`) are pinned into the run
directory so cross-machine numerical skew cannot flip the marginal
nominal-trajectory branches.

## Definitive-run trial data

The per-trial debug JSONs for the definitive run (`trial_0.json` …
`trial_9999.json`) are distributed as the release asset
`apollo11_final_trials.tar.gz` rather than committed to the repository. Extract
into `outputs/final/trials/`. Each file is a full per-trial overview — the phase
timeline (per-phase GET, mission-elapsed duration) and every outcome field —
complementing the per-trial summary rows in the committed CSVs.

## Data provenance

The lunar gravity field (`lunar_gravity_coeffs.py`) is derived from NASA's
**GRAIL GRGM1200A** model; the ephemeris uses the Meeus analytic series and
public NASA mission constants. NASA data products are in the public domain in
the U.S. The MIT license below covers the original source code only — see
`LICENSE` for the third-party data notice.

## Use of generative AI

This project was developed with substantial assistance from Claude (Anthropic),
under the author's direction and review.

## License

MIT — see [`LICENSE`](LICENSE). The license covers the simulation source code;
embedded NASA scientific data carries its own (public-domain) terms, noted in
the license file.
