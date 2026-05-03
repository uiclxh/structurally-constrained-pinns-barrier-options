# Release Notes: v1.0.0

Release date: 2026-05-02

## Summary

`v1.0.0` is the first public research-asset release of the repository accompanying:

**Structurally Constrained PINNs for Barrier Option Pricing: Benchmarking Against High-Precision Implicit Finite Differences**

This release is intended as a curated reproducibility package for academic inspection, master's application review, and future research extension.

## Included

- Final working paper PDF under `paper/`.
- Chapter-level result packages under `results/`.
- Selected figure exports under `figures/`.
- Selected CSV and LaTeX table exports under `tables/`.
- Trained surrogate model artifacts under `models/`.
- Chapter-level research scripts under `src/`.
- Reproduction helper script under `scripts/`.
- SSRN, project-summary, and reproducibility documentation under `docs/`.

## Reproducibility Status

This release is a curated reproducibility package, not a clean-room rebuild.

The included result folders, tables, figures, model artifacts, and summaries are the primary evidence package. The scripts can be used to rerun selected workflows, but regenerated outputs should be reviewed before replacing the curated release artifacts.

## Environment

Target environment:

- Python 3.11
- CPU-only PyTorch execution
- Windows / PowerShell-oriented helper scripts

Environment files:

- `requirements.txt`
- `requirements-lock.txt`
- `environment.yml`

## Known Limitations

- The workflow does not yet provide certified worst-case error bounds.
- Residual diagnostics are informative but remain severe in the near-barrier layer.
- Runtime results are hardware-dependent and should be remeasured on new machines.
- The project is organized as research scripts and curated artifacts rather than a packaged Python library.

## Suggested Citation

Use `CITATION.cff` for repository citation metadata. Once the SSRN page is available, cite both the SSRN working paper and this GitHub release.
