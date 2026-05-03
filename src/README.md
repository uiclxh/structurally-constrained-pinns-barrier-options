# Source Code Map

This folder contains the Python scripts used to generate or support the chapter-level outputs in the accompanying working paper:

**Structurally Constrained PINNs for Barrier Option Pricing: Benchmarking Against High-Precision Implicit Finite Differences**

The scripts are organized by manuscript chapter. They are research scripts rather than a packaged Python library.

## Files

| File | Chapter | Purpose | Primary Output Folder |
| --- | --- | --- | --- |
| `chapter3_fdm_benchmark_only.py` | Chapter 3 | Builds the high-precision implicit finite-difference benchmark and convergence table. | `results_chapter3_only/` |
| `chapter4_barrier_surrogate_framework.py` | Chapter 4 | Defines the barrier-aware neural surrogate framework, architecture tables, loss terms, and initial artifact. | `results_chapter4_only/` |
| `chapter5_validation_protocol_framework.py` | Chapter 5 | Generates validation panels, stress panels, validation metrics, and acceptance-rule assets. | `results_chapter5_only/` |
| `chapter6_experimental_design_framework.py` | Chapter 6 | Generates scenario-family tables, baseline-family tables, and comparison-design figures. | `results_chapter6_only/` |
<<<<<<< HEAD
| `chapter7_ablation_failure_diagnostics_real.py` | Chapter 7 | Runs the formal ablation and failure-diagnostics workflow used for the final Chapter 7 evidence package. | `results_chapter7_only/` |
=======
| `chapter7_ablation_failure_diagnostics_framework.py` | Chapter 7 helper | Legacy plotting/helper module used by the formal Chapter 7 script. | Helper only |
| `chapter7_ablation_failure_diagnostics_real.py` | Chapter 7 | Runs the formal ablation and failure-diagnostics workflow used for the final Chapter 7 evidence package. | `results_chapter7_real_formal/` |
>>>>>>> 4bc5f19c08c73e9d02bc2d1cdd83bfe632fc14c2
| `chapter8_results_accuracy_real.py` | Chapter 8 | Evaluates pricing accuracy, Greeks, boundary consistency, residual diagnostics, and validation scorecards. | `results_chapter8_only/` |
| `chapter9_results_runtime_real.py` | Chapter 9 | Measures runtime, throughput, break-even behavior, and repeated-query deployment economics. | `results_chapter9_only/` |
| `chapter10_discussion_roadmap_framework.py` | Chapter 10 | Generates the solver-selection decision map, roadmap figure, and establishes-versus-not-establishes table. | `results_chapter10_only/` |

## Important Notes

The current `src/` folder reflects the scripts available in this repository snapshot. The Chapter 7 script is the formal real-run version and is the source-code counterpart to the final uploaded Chapter 7 result package.

The Chapter 8 and Chapter 9 scripts depend on trained artifacts and result folders from earlier chapters. In particular, Chapter 8 expects the formal Chapter 7 barrier-aware PINN artifact produced under:

```text
results_chapter7_only/full_baac_guard_probe/
```

The repository stores generated chapter outputs separately under `results/`, `figures/`, `tables/`, and `models/`. These outputs are the primary reproducibility evidence package.

## Running the Scripts

From the repository root, install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The target environment is Python 3.11 with CPU-only PyTorch. For stricter reproducibility, use `requirements-lock.txt` or `environment.yml` from the repository root.

Then run an individual chapter script:

```powershell
python src/chapter5_validation_protocol_framework.py
```

To run the full workflow, use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce_all.ps1
```

The full workflow can take substantial time, especially Chapter 7 and Chapter 8. It is intended for research reproduction rather than quick smoke testing.

## Known Script-Level Caveat

The repository is primarily a curated reproducibility package. The Chapter 7 formal script uses `chapter7_ablation_failure_diagnostics_framework.py` as a legacy helper for the failure-taxonomy figure. The curated Chapter 7 outputs and trained artifacts are already included under `results/results_chapter7_only/`, and these are the recommended source for inspecting the final reported evidence.

## Output Policy

The scripts may write to chapter-specific folders in the working directory, such as:

```text
results_chapter3_only/
results_chapter4_only/
results_chapter5_only/
results_chapter6_only/
results_chapter7_only/
results_chapter8_only/
results_chapter9_only/
results_chapter10_only/
```

The curated GitHub-ready outputs are stored under the repository-level folders:

```text
results/
figures/
tables/
models/
```

If you regenerate results, review the outputs before replacing the curated result package.
