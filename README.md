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
| **Full-mission success** | **88.0 %**  (95 % CI 87.3–88.6 %) |
| **Crew survival** | **94.0 %** |

"Mission success" follows NASA's actual objective: the crew reached the lunar
surface **and** all three returned to Earth alive — even if a recovered in-flight
anomaly occurred (e.g. a docking failure resolved by a contingency EVA crew
transfer). Of the 8,800 successful missions, **8,646 were flawless** (no anomaly
at all); 154 succeeded despite a recovered anomaly. The ~6-point gap between crew
survival and mission success is Apollo's abort architecture at work: many
missions never complete the objective but still bring the crew home.

The nominal trajectory reproduces Apollo 11's flown record end to end — burn
times to within seconds, the major ΔV budgets to better than one percent, and a
lunar touchdown **6.8 km from the targeted landing site** (Apollo 11's own
as-flown miss of its aim point was about 6 km).

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
| `cluster_run.py`, `submit_mc.sh` | The SLURM sharding pipeline used to produce the definitive run on a compute cluster. |
| `launch_tli_preset.json` | Cached launch/TLI targeting preset, pinned for cross-machine reproducibility. |
| `outputs/final/` | **The definitive run**: per-trial results CSVs, the nominal trajectory, captured targeting products, the dashboard, and figures. |

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
**GRAIL GRGM1200A** model; the ephemeris uses the Meeus analytic series and
public NASA mission constants. NASA data products are in the public domain in
the U.S. The MIT license below covers the original source code only — see
`LICENSE` for the third-party data notice.

## Use of generative AI

This project was produced in a sustained collaboration between the author and
Claude (Anthropic), a large language model used across successive versions over
the project's development, and the division of labor was consequential enough to
warrant a fuller statement than the customary disclosure line. The conception is
the author's: the question, the decision to answer it with a physics-integrated
Monte Carlo rather than a reliability fault tree, and the design doctrines that
define the model — fidelity first; as-planned targeting; the mission-success
definition; and the requirement that every stage be validated against the
historical record before being built upon. Claude performed essentially all of
the implementation: the simulation code, the guidance and targeting solvers, the
failure and crew-survival models, the cluster pipeline, the excavation of the
historical sources behind the calibrated constants, the diagnostic
investigations, the statistical analysis of the Monte Carlo campaigns, and the
drafting of the project's documentation and manuscript. Design and refinement
were a genuine dialogue between the two: the highest-leverage corrections
typically began as the author's questions or catches — among them the as-planned
landing-aim doctrine and the demand for the cross-machine reproducibility tests
described above — and were then diagnosed and engineered by Claude, while
Claude's technical designs were in turn constrained, redirected, and sometimes
rejected by the author's judgment of what faithfulness required.

The process safeguards matter as much as the division of labor. The author
directed the work throughout; set the validation discipline under which no model
change was adopted without tiered Monte Carlo revalidation; reviewed, verified,
and edited all code, numerical results, physical assumptions, and text; and
takes full responsibility for the content of this project. Errors surfaced late
in development — in the model's landing-site targeting and in the project's own
reported numbers — were caught by exactly that review structure, some by the
author's reading and some by adversarial verification passes the author
required. The honest summary is that the author served as principal
investigator — conception, judgment, quality control, and accountability — while
Claude served as the research staff: construction, diagnosis, analysis, and
drafting at a speed and volume no individual could match. The author could not
have completed this project without Claude, and Claude could not have completed
this project without the author.

## License

MIT — see [`LICENSE`](LICENSE). The license covers the simulation source code;
embedded NASA scientific data carries its own (public-domain) terms, noted in
the license file.
