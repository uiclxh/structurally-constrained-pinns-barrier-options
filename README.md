# Structurally Constrained PINNs for Barrier Option Pricing

[![Smoke test](https://github.com/uiclxh/structurally-constrained-pinns-barrier-options/actions/workflows/smoke.yml/badge.svg?branch=main)](https://github.com/uiclxh/structurally-constrained-pinns-barrier-options/actions/workflows/smoke.yml?query=branch%3Amain)

This repository is the reproducibility package for the working paper:

**Structurally Constrained PINNs for Barrier Option Pricing under a High-precision Implicit Finite-difference Benchmark**

SSRN working paper: <https://ssrn.com/abstract=6628159>

DOI: <https://doi.org/10.2139/ssrn.6628159>

GitHub repository: <https://github.com/uiclxh/structurally-constrained-pinns-barrier-options>

## Overview

This project studies whether structurally constrained physics-informed neural networks can be made credible for continuously monitored down-and-out European call option pricing under the Black-Scholes framework.

Barrier options are a demanding test case for neural PDE solvers because the absorbing boundary creates a localized high-curvature region near the knockout boundary. A model can look acceptable under average pricing error while still failing near the barrier, especially in Delta, Gamma, and boundary consistency.

The project compares a strong implicit finite-difference benchmark with barrier-aware neural surrogates. The retained neural framework combines transformed coordinates, hard barrier enforcement, barrier-aware adaptive collocation, hybrid optimization, and protocol-based validation.

## Reproducibility Status

This is a curated reproducibility package, not a clean-room rebuild.

The repository provides the final working paper, an extended technical appendix with full derivations, generated figures, generated tables, trained model artifacts, validation scorecards, residual diagnostics, and chapter-level summary files used by the manuscript. Some workflows can be rerun from `src/`, but the repository is not yet a fully packaged one-command software system that rebuilds every result from a fresh environment.

For details, see [docs/reproducibility.md](docs/reproducibility.md).

## Core Results

- The implicit finite-difference benchmark remains the strongest method for local pricing accuracy and Greek-sensitive tasks.
- Naive PINNs fail systematically in near-barrier high-curvature regimes.
- Hard barrier enforcement and barrier-aware adaptive collocation materially improve learned behavior.
- The barrier-aware PINN is the strongest learned model in near-barrier Gamma control and boundary consistency.
- No learned surrogate passes the full validation protocol in the current workflow.
- Learned surrogates become economically relevant only under sufficiently large repeated-query workloads and acceptable task-specific error tolerance.

## Main Result Snapshot

The table below condenses the validation scorecard. Lower values are better for all reported error and residual metrics.

| Model | Price q95 | Gamma q95 | Barrier max | Residual q95 | Takeaway |
| --- | ---: | ---: | ---: | ---: | --- |
| FDM benchmark | **0.018%** | **0.0007** | **4.84e-14** | 0.509 | Best local-precision reference |
| Barrier-aware PINN | 21.322% | **0.106** | **8.67e-06** | 9.957 | Best learned Gamma and barrier behavior |
| Supervised surrogate | 18.978% | 0.747 | **6.44e-06** | 132.134 | Fast, but weak near-barrier Gamma |
| Differential surrogate | 18.605% | 0.739 | **6.44e-06** | 133.814 | Delta-aware, still weak near barrier |

![Gamma comparison heatmaps and representative slice](figures/results_chapter8_only/figure25_gamma_heatmaps_and_slices.png)

Runtime economics are conditional on workload size and validation status:

| Method | Latency (s) | Throughput | Break-even N* | Status |
| --- | ---: | ---: | ---: | --- |
| FDM | 0.003451 | - | 0 | Benchmark |
| PINN | 0.000153 | 1,893,293/s | 58,195 | Pass Gamma + barrier |
| Supervised surrogate | 0.000175 | 1,376,967/s | 36,765 | Pass barrier only |

Full tables are available in `results/results_chapter8_only/table14_validation_scorecard.csv` and `results/results_chapter9_only/table15_runtime_inputs_break_even_summary.csv`.

## Repository Structure

```text
paper/
  Final SSRN working paper PDF and extended technical appendix.

src/
  Chapter-level Python research scripts.

scripts/
  Reproduction helper scripts.

docs/
  SSRN notes, project summary, and reproducibility statement.

results/
  Curated chapter-level result packages from Chapter 3 through Chapter 10.

figures/
  Selected figure and table-image exports for quick browsing.

tables/
  Selected CSV and LaTeX table exports for quick inspection.

models/
  Trained surrogate model artifacts and related metadata.
```

## Chapter Map

| Chapter | Evidence folder | What it contains |
| --- | --- | --- |
| 3 | `results/results_chapter3_only/` | FDM benchmark and convergence checks |
| 4 | `results/results_chapter4_only/` | Surrogate architecture, loss terms, and initial artifact |
| 5 | `results/results_chapter5_only/` | Validation panels, metrics, and acceptance rule |
| 6 | `results/results_chapter6_only/` | Scenario families and comparison design |
| 7 | `results/results_chapter7_only/` | Ablation and failure diagnostics |
| 8 | `results/results_chapter8_only/` | Accuracy, Greeks, residuals, scorecard, and models |
| 9 | `results/results_chapter9_only/` | Runtime, throughput, and break-even analysis |
| 10 | `results/results_chapter10_only/` | Solver-selection map and research roadmap |

## Quickstart

Clone the repository:

```powershell
git clone https://github.com/uiclxh/structurally-constrained-pinns-barrier-options.git
cd structurally-constrained-pinns-barrier-options
```

Create and activate a Python environment. The target environment is Python 3.11 with CPU-only PyTorch:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a stricter pip environment, use:

```powershell
python -m pip install -r requirements-lock.txt
```

For Conda or Mamba, use:

```powershell
conda env create -f environment.yml
conda activate barrier-pinn-repro
```

The runtime results in Chapter 9 are hardware-dependent. The reported package is CPU-oriented; GPU runs may change latency, throughput, and break-even thresholds.

Run a lightweight subset of the workflow:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce_all.ps1 -SkipHeavy
```

Run the full scripted workflow:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce_all.ps1
```

The full workflow can take substantial time because Chapter 7 and Chapter 8 include neural training and evaluation. The curated outputs already included in `results/`, `figures/`, `tables/`, and `models/` are the primary reproducibility package.

## Reproduction Boundary

`scripts/reproduce_all.ps1` is a convenience runner for chapter-level scripts. It is useful for rerunning selected workflows and checking how the generated artifacts are organized.

It should not be read as a clean-room rebuild guarantee. The curated release already includes the reported figures, tables, trained artifacts, and result summaries. If you rerun scripts, inspect the regenerated chapter output folders before replacing the curated files under `results/`, `figures/`, `tables/`, or `models/`.

Use `-SkipHeavy` for a faster pass that skips the neural-training and runtime-heavy chapters:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce_all.ps1 -SkipHeavy
```

## Suggested Reading Order

1. Read the final SSRN paper in `paper/`.
2. Use `paper/Structurally_Constrained_PINNs_for_Barrier_Option_Pricing_technical_appendix.pdf` for full derivations and extended technical details.
3. Review the validation scorecard in `results/results_chapter8_only/table14_validation_scorecard.csv`.
4. Inspect the runtime break-even table in `results/results_chapter9_only/table15_runtime_inputs_break_even_summary.csv`.
5. Check the decision map and roadmap in `results/results_chapter10_only/`.
6. Use `src/README.md` if you want to trace which script generated each chapter-level output.

## Licensing

This repository uses split licensing because it contains code, manuscript material, generated research outputs, and trained model artifacts.

- Code, scripts, and repository documentation are licensed under the MIT License. See [LICENSE](LICENSE).
- Paper, figures, tables, and generated result files are released for non-commercial academic use under CC BY-NC 4.0. See [docs/content-license.md](docs/content-license.md).
- Trained model weights are research artifacts with non-commercial academic-use terms. See [docs/content-license.md](docs/content-license.md).

The models and numerical outputs are research artifacts. They are not production trading systems and should not be used for live pricing, risk management, or investment decisions without independent validation.

## Citation

If you use this repository, please cite the working paper and repository. GitHub can read the citation metadata from [CITATION.cff](CITATION.cff).

Suggested paper citation:

Lin, Tom. "Structurally Constrained PINNs for Barrier Option Pricing under a High-precision Implicit Finite-difference Benchmark." SSRN working paper, April 22, 2026. <https://doi.org/10.2139/ssrn.6628159>.
