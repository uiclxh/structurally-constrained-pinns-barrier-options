# Reproducibility Statement

This repository accompanies the working paper:

**Structurally Constrained PINNs for Barrier Option Pricing under a High-precision Implicit Finite-difference Benchmark**

The repository is intended to provide a transparent research record for the generated figures, tables, model artifacts, validation outputs, and chapter-level summaries used in the manuscript.

## Reproducibility Scope

The current release is a reproducibility package for reported results. It is not yet a fully packaged software library with a single end-to-end command for rebuilding every artifact from scratch.

The repository includes:

- Final SSRN working paper PDF.
- Extended technical appendix with full derivations.
- Chapter-level result folders from Chapter 3 through Chapter 10.
- Generated figures in PNG format.
- Generated tables in CSV and LaTeX format.
- Summary JSON files recording major workflow outputs and configuration choices.
- Trained model artifacts where available.
- Validation scorecards, residual diagnostics, runtime tables, and deployment-economics outputs.

## Environment

The target environment for this release is:

```text
Python 3.11
CPU-only PyTorch execution
Windows / PowerShell-oriented helper scripts
```

Primary dependency files:

- `requirements.txt`: concise pinned dependency list.
- `requirements-lock.txt`: stricter pip-oriented lock-style list for the core dependency tree.
- `environment.yml`: Conda/Mamba environment specification.

The core Python dependencies are:

```text
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
scipy==1.13.1
torch==2.3.1
```

The project was organized as a CPU-oriented reproducibility package. GPU execution is possible in principle, but runtime, throughput, and break-even values in Chapter 9 should be remeasured if the hardware changes.

Install with pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install with a stricter dependency list:

```powershell
python -m pip install -r requirements-lock.txt
```

Install with Conda:

```powershell
conda env create -f environment.yml
conda activate barrier-pinn-repro
```

If `torch==2.3.1` cannot be resolved from the default pip index for your platform, install the CPU wheel from the official PyTorch CPU index:

```powershell
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

## Hardware and Runtime Caveat

The runtime and deployment-economics outputs are not hardware-invariant constants. Chapter 9 measures latency, throughput, and break-even behavior for the active runtime host used in the reported workflow. A different CPU, GPU, batch size, BLAS backend, PyTorch build, or memory hierarchy can change the absolute timing values.

For this reason, Chapter 9 should be read as evidence of the amortization structure rather than a universal latency benchmark.

## Source Material

The repository integrates the following local materials:

```text
Structurally Constrained PINNs for Barrier Option Pricing.pdf
Structurally_Constrained_PINNs_for_Barrier_Option_Pricing_technical_appendix.pdf
results_chapter3_only
results_chapter4_only
results_chapter5_only
results_chapter6_only
results_chapter7_only
results_chapter8_only
results_chapter9_only
results_chapter10_only
```

## Chapter-Level Reproducibility Map

### Chapter 3: High-Precision Implicit FDM Benchmark

Folder:

```text
results/results_chapter3_only/
```

Key files:

- `fdm_convergence.csv`
- `fdm_convergence_detail.csv`
- `fdm_convergence.png`
- `fdm_convergence_frontier.png`
- `fdm_convergence.tex`

Purpose:

Verifies the finite-difference benchmark through convergence diagnostics, price errors, Greek errors, and runtime per contract.

### Chapter 4: Barrier-Aware Neural Surrogate Framework

Folder:

```text
results/results_chapter4_only/
```

Key files:

- `chapter4_summary.json`
- `chapter4_dry_run_history.csv`
- `table6_architecture_hyperparameters.csv`
- `table7_loss_terms.csv`
- `artifact/barrier_surrogate.pt`
- `artifact/config.json`

Purpose:

Records the neural surrogate scaffold, architecture settings, loss construction, and initial model artifact.

### Chapter 5: Validation Protocol and Acceptance Rules

Folder:

```text
results/results_chapter5_only/
```

Key files:

- `chapter5_summary.json`
- `scenario_panel_summary.csv`
- `train_panel.csv`
- `validation_panel.csv`
- `test_panel.csv`
- `stress_barrier_panel.csv`
- `stress_short_lowvol_panel.csv`
- `stress_wide_panel.csv`
- `table8_validation_metrics_dictionary.csv`
- `table9_acceptance_rule.csv`
- `figure13_data_split_and_scenario_family_map.png`
- `figure14_regional_validation_zones.png`

Purpose:

Defines the validation design, data panels, stress panels, metrics dictionary, and threshold-based acceptance rule.

### Chapter 6: Experimental Design and Baseline Family

Folder:

```text
results/results_chapter6_only/
```

Key files:

- `chapter6_summary.json`
- `table10_scenario_families.csv`
- `table11_baseline_family.csv`
- `figure15_scenario_matrix.png`
- `figure16_comparison_design_map.png`

Purpose:

Documents the scenario families, parameter ranges, baseline models, and comparison design.

### Chapter 7: Ablation and Failure Diagnostics

Folder:

```text
results/results_chapter7_only/
```

Key files:

- `chapter7_summary.json`
- `chapter7_protocol_notes.json`
- `table12_ablation_summary_matrix.csv`
- `figure17_failure_taxonomy.png`
- `figure18_training_pathology_naive_pinn.png`
- `figure19_effect_coordinate_choice.png`
- `figure20_effect_hard_constrained_ansatz.png`
- `figure21_effect_baac.png`

Purpose:

Shows why naive PINNs fail and how coordinate transforms, hard barrier enforcement, and barrier-aware adaptive collocation improve learned behavior.

### Chapter 8: Accuracy, Greeks, Boundary Consistency, and Error Control

Folder:

```text
results/results_chapter8_only/
```

Key files:

- `chapter8_summary.json`
- `table13_main_pricing_comparison.csv`
- `table14_validation_scorecard.csv`
- `regional_residuals.csv`
- `residual_quantiles.csv`
- `panel_sizes.csv`
- `figure22_pricing_error_heatmaps.png`
- `figure23_near_barrier_zoom_price_error_map.png`
- `figure24_delta_comparison_heatmaps.png`
- `figure25_gamma_heatmaps_and_slices.png`
- `figure26_boundary_positivity_diagnostics.png`
- `figure27_residual_diagnostics.png`
- `models/supervised_surrogate/best_model.pt`
- `models/differential_surrogate/best_model.pt`

Purpose:

Reports pricing accuracy, Greek diagnostics, boundary residuals, positivity checks, PDE residual diagnostics, and validation scorecards.

### Chapter 9: Runtime and Deployment Economics

Folder:

```text
results/results_chapter9_only/
```

Key files:

- `chapter9_summary.json`
- `runtime_curve.csv`
- `average_cost_curve.csv`
- `batch_throughput.csv`
- `use_case_risk_surface.csv`
- `table15_runtime_inputs_break_even_summary.csv`
- `table16_use_case_economics.csv`
- `figure28_total_runtime_vs_evaluations.png`
- `figure29_average_cost_per_evaluation.png`
- `figure30_batch_throughput_comparison.png`
- `figure31_use_case_barrier_risk_surface_generation.png`

Purpose:

Compares single-query and repeated-query workloads, break-even behavior, throughput, and deployment use cases.

### Chapter 10: Discussion, Decision Map, and Research Roadmap

Folder:

```text
results/results_chapter10_only/
```

Key files:

- `chapter10_summary.json`
- `table17_establishes_vs_not_establishes.csv`
- `figure32_decision_map_when_to_use_which_solver.png`
- `figure33_research_roadmap.png`

Purpose:

Summarizes what the study establishes, what it does not establish, solver-selection logic, and future research directions.

## Known Limitations

- The current workflow does not provide certified worst-case error bounds.
- PDE residual diagnostics are informative but overly severe in the near-barrier layer.
- The current evidence is restricted to a one-dimensional Black-Scholes barrier-option setting.
- Runtime thresholds are hardware-dependent and should not be treated as universal constants.
- Some folders contain generated outputs rather than all scripts needed for full reconstruction.

## Recommended Repository Hygiene

The following local files should not be uploaded:

- `.idea/`
- `venv/`
- `.venv/`
- Python cache folders
- local logs or scratch outputs

Model weight files such as `.pt` may be included if they are part of the reproducibility package, but Git LFS is recommended.

## Reproducibility Claim

Recommended wording:

This repository provides a structured reproducibility package for the reported working-paper results, including generated tables, figures, validation outputs, trained model artifacts, and chapter-level summary files.

Avoid claiming:

This repository fully reproduces all results from a clean environment with one command.

That stronger claim should only be made after adding a complete environment specification, source scripts, data-generation pipeline, and automated reproduction workflow.
