# Source Code Map

This folder contains the Python scripts used to generate or support the chapter-level outputs in the accompanying working paper:

**Structurally Constrained PINNs for Barrier Option Pricing under a High-precision Implicit Finite-difference Benchmark**

The scripts are organized by manuscript chapter. They are research scripts rather than a packaged Python library.

## Files

| Script | Chapter | Main role | Output |
| --- | --- | --- | --- |
| `chapter3_fdm_benchmark_only.py` | 3 | FDM benchmark and convergence table | `results_chapter3_only/` |
| `chapter4_barrier_surrogate_framework.py` | 4 | Surrogate framework, architecture, and loss assets | `results_chapter4_only/` |
| `chapter5_validation_protocol_framework.py` | 5 | Validation panels, stress panels, and metric assets | `results_chapter5_only/` |
| `chapter6_experimental_design_framework.py` | 6 | Scenario families and comparison design | `results_chapter6_only/` |
| `chapter7_ablation_failure_diagnostics_framework.py` | 7 helper | Legacy plotting and helper utilities | Helper only |
| `chapter7_ablation_failure_diagnostics_real.py` | 7 | Formal ablation and failure diagnostics | `results_chapter7_only/` |
| `chapter8_results_accuracy_real.py` | 8 | Accuracy, Greeks, residuals, and scorecards | `results_chapter8_only/` |
| `chapter9_results_runtime_real.py` | 9 | Runtime, throughput, and break-even analysis | `results_chapter9_only/` |
| `chapter10_discussion_roadmap_framework.py` | 10 | Decision map, roadmap, and claim-status table | `results_chapter10_only/` |

## Important Notes

The current `src/` folder reflects the scripts available in this repository snapshot. The Chapter 7 real-run script is the source code counterpart to the final uploaded Chapter 7 result package.

Chapter 8 and Chapter 9 depend on trained artifacts and result folders from earlier chapters. In particular, Chapter 8 expects the formal Chapter 7 barrier-aware PINN artifact under:

```text
results_chapter7_only/full_baac_guard_probe/
```

The repository stores generated chapter outputs separately under `results/`, `figures/`, `tables/`, and `models/`. These outputs form the primary reproducibility evidence package.

## Running the Scripts

From the repository root, install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The target environment is Python 3.11 with CPU-only PyTorch. For stricter reproducibility, use `requirements-lock.txt` or `environment.yml` from the repository root.

Then run an individual chapter script, for example:

```powershell
python src/chapter5_validation_protocol_framework.py
```

To run the full workflow, use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce_all.ps1
```

The full workflow can take substantial time, especially Chapter 7 and Chapter 8. It is intended for research reproduction rather than quick smoke testing.

## Known Script-Level Caveat

The repository is primarily a curated reproducibility package. The Chapter 7 formal script uses `chapter7_ablation_failure_diagnostics_framework.py` as a legacy helper for the failure-taxonomy figure. The curated Chapter 7 outputs and trained artifacts are already included under `results/results_chapter7_only/`; these are the recommended source for inspecting the final reported evidence.

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
