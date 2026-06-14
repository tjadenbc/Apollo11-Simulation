# Apollo 11 Monte Carlo Simulation

A physics-integrated Monte Carlo that estimates Apollo 11's **probability of
mission success and crew survival** by flying the entire mission — launch pad to
splashdown — as one continuous, numerically-integrated trajectory, thousands of
times.

Every powered manoeuvre is a finite-thrust integration and every coast a
three-body integration under Earth (+J2) and Moon gravity, using a real
July-1969 lunar ephemeris and a degree-8 GRAIL gravity field, with faithful
renderings of Apollo's own guidance laws (iterative-guidance ascent, B-plane
midcourse corrections, a two-burn lunar-orbit insertion, a predictor-corrector
entry), 1969-era engine reliabilities, and a suite of failure modes sourced to
the historical record. A per-astronaut survival model then maps each mission
outcome through Apollo's abort architecture to a crew-survival result.

## Headline result

Across **10,000 trials** (seed 37, distributed over a 272-core cluster):

| Metric | Estimate |
|---|---|
| **Full-mission success** | **85.5 %**  (95 % CI 84.7–86.1 %) |
| **Crew survival** | **93.4 %** |

"Mission success" follows NASA's actual objective: the crew reached the lunar
surface **and** all three returned to Earth alive — even if a recovered in-flight
anomaly occurred (e.g. a docking failure resolved by a contingency EVA crew
transfer). 141 of the 10,000 missions succeeded despite such an anomaly; 8,404
were flawless (no anomaly at all). The ~8-point gap between mission success and
crew survival is Apollo's abort architecture at work: many missions never land
but still bring the crew home.

The definitive run lives in [`outputs/apollo11_final10000/`](outputs/apollo11_final10000/).
Open [`outputs/apollo11_final10000/dashboard.html`](outputs/apollo11_final10000/dashboard.html)
in a browser for the full breakdown — failure-mode decomposition, per-astronaut
death attribution, phase timing vs. Apollo 11's flown values, and known
limitations.

## Repository layout

| Path | What it is |
|---|---|
| `apollo11.py` | The simulation (~6,800 lines): physics, every mission phase, `run_mission()`, and the `main()` / `main_parallel()` Monte Carlo drivers. |
| `crew_survival.py` | Per-astronaut survival model; reads the MC results and writes `results_with_survival.csv` plus crew statistics. |
| `generate_outputs.py` | Builds `dashboard.html`, `summary.txt`, and the figures from a run directory (data-driven). |
| `lunar_gravity_coeffs.py` | Embedded GRAIL GRGM1200A spherical-harmonic coefficients (see data note below). |
| `cluster_run.py`, `submit_mc.sh` | The SLURM sharding pipeline used to produce the definitive run on a compute cluster. |
| `launch_tli_preset.json` | Cached launch/TLI targeting preset, pinned for cross-machine reproducibility. |
| `test_*.py` | Regression tests (ephemeris, entry guidance, gravity field, resume gap-fix). |
| `outputs/apollo11_final10000/` | **The definitive run**: per-trial results CSVs, the nominal trajectory, captured targeting products, the generated dashboard, and figures. |
| `CLAUDE.md` | Detailed architecture / status / flags reference. |
| `paper.md` | The accompanying manuscript draft. |
| `make_pdf.py` | Builds a review `paper.pdf` from `paper.md` (Markdown → PDF via WeasyPrint; figures embedded). |

## Quick start

```bash
pip install -r requirements.txt   # numpy, scipy, pandas, matplotlib (Python 3.12)
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
was produced on a SLURM cluster via `./submit_mc.sh`; see `CLAUDE.md` for the
cluster pipeline.

## Tests

```bash
python3 test_resume_gapfix.py     # checkpoint/resume gap-safety
python3 test_ephemeris.py         # lunar/solar ephemeris vs. reference
python3 test_entry_guidance.py    # predictor-corrector entry
python3 test_sh_field.py          # spherical-harmonic gravity field
```

## Reproducibility

Local serial and parallel drivers are **bit-identical** (per-trial perturbations
are pre-generated in the main process and dispatched by trial index). Cluster
runs are *not* bit-identical to a laptop (different scipy/numpy builds), so each
cluster run is treated as its own population. The captured targeting products
(`ei_target.json`, `bplane_target.json`, `nominal_results.json`,
`launch_tli_preset.json`) are pinned into the run directory so cross-machine
numerical skew cannot flip the marginal nominal-trajectory branches.

## Data provenance

The lunar gravity field (`lunar_gravity_coeffs.py`) is derived from NASA's
**GRAIL GRGM1200A** model; the ephemeris uses the Meeus analytic series and
public NASA mission constants. NASA data products are in the public domain in
the U.S. The MIT license below covers the original source code only — see
`LICENSE` for the third-party data notice.

## Paper

This code accompanies the manuscript *"Estimating Apollo 11's Probability of
Success: A Physics-Integrated Monte Carlo Reassessment"* (see `paper.md`). Every
result in the paper can be regenerated from this repository and the definitive
run directory.

## Use of generative AI

This project was developed with substantial assistance from Claude (Anthropic),
under the author's direction and review. The AI is **not** credited as an author;
its role is disclosed in the paper. See the manuscript's "Use of Generative AI"
declaration for details.

## License

MIT — see [`LICENSE`](LICENSE). The license covers the simulation source code;
embedded NASA scientific data carries its own (public-domain) terms, noted in
the license file.
