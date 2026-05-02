# Project Summary

## Title

Structurally Constrained PINNs for Barrier Option Pricing: Benchmarking Against High-Precision Implicit Finite Differences

## Short Description

This project studies structurally constrained physics-informed neural networks for barrier option pricing. It compares barrier-aware neural surrogates against a high-precision implicit finite-difference benchmark and evaluates prices, Greeks, boundary consistency, PDE residual diagnostics, and repeated-query deployment economics.

## One-Sentence Application Description

A reproducible computational finance research project combining an SSRN working paper, a GitHub reproducibility package, high-precision finite-difference benchmarking, and structurally constrained PINNs for barrier option pricing.

## Research Area

- Computational finance
- Quantitative finance
- Financial mathematics
- Derivative pricing
- Physics-informed neural networks
- Numerical PDE methods
- Neural surrogate modeling

## Problem Motivation

Barrier options are more difficult than vanilla options for neural PDE solvers because the absorbing barrier creates a thin high-curvature region near the knockout boundary. A model can achieve acceptable average pricing error while still failing in the near-barrier region or producing unstable Greeks. This project treats barrier option pricing as a local-precision stress test for neural financial PDE solvers.

## Research Questions

1. Why do naive PINNs fail in near-barrier high-curvature regimes?
2. Which structural constraints and sampling mechanisms stabilize neural surrogates for barrier PDEs?
3. How should prices, Delta, Gamma, boundary consistency, positivity, and residual diagnostics be jointly validated?
4. Under what workload regime does offline-online amortization make learned surrogates economically attractive relative to a strong finite-difference benchmark?

## Methodology

The project combines:

- A high-precision implicit finite-difference benchmark.
- Log-domain transformation and exact barrier alignment.
- Rannacher-smoothed Crank-Nicolson time stepping.
- Structurally constrained barrier-aware neural ansatz.
- Barrier-aware adaptive collocation.
- Supervised and differential neural surrogate baselines.
- Validation panels, stress panels, and threshold-based acceptance rules.
- Region-wise diagnostics for near-barrier, near-strike, smooth, and far-field regimes.
- Runtime and break-even analysis for repeated-query deployment.

## Main Findings

- The implicit finite-difference benchmark remains the strongest method for local precision and Greek-sensitive tasks.
- Naive PINNs fail systematically in near-barrier high-curvature regimes.
- Hard barrier enforcement and barrier-aware sampling materially improve neural surrogate behavior.
- The barrier-aware PINN is the strongest learned model in near-barrier Gamma control and boundary consistency.
- Supervised surrogates become economically attractive only under sufficiently large repeated-query workloads and conditional error tolerance.
- No learned surrogate passes the full validation protocol in the current workflow.

## Key Numerical Results

From the Chapter 8 validation scorecard:

- FDM passes price accuracy, near-barrier Gamma, barrier condition, and positivity, but fails the current residual criterion.
- PINN passes near-barrier Gamma, barrier condition, and positivity, but fails price q95 and residual q95.
- Supervised and differential surrogates pass the barrier condition and positivity, but fail price q95, near-barrier Gamma, and residual q95.

From the Chapter 9 runtime analysis:

- FDM has no training cost but pays recurring solve cost.
- PINN has higher offline training cost but lower inference latency.
- Supervised surrogates become economically attractive under large repeated-query workloads, subject to validation constraints.

## What This Project Establishes

- A strong benchmark institution rather than a weak classical baseline.
- A barrier-aware surrogate framework combining transformed coordinates, hard barrier structure, BAAC, and protocol-based validation.
- Evidence that hard-constrained barrier encoding and barrier-aware sampling improve learned models.
- A structured solver-selection view where local precision and query volume jointly determine the preferred method.
- A nontrivial one-dimensional barrier-option test bed for evaluating neural PDE solvers.

## What This Project Does Not Establish

- It does not prove that neural surrogates universally replace classical solvers.
- It does not provide certified worst-case error bounds.
- It does not establish production readiness for all barrier or Greek-sensitive workloads.
- It does not provide hardware-independent runtime thresholds.
- It does not automatically generalize to higher-dimensional dynamics, stochastic volatility, jump processes, or broader exotic-product families.

## Repository Assets

The project integrates:

- Final manuscript PDF from `D:\桌面\Structurally Constrained PINNs for Barrier Option Pricing.pdf`.
- Chapter 3 results from `E:\results_chapter3_only`.
- Chapter 4 results from `E:\results_chapter4_only`.
- Chapter 5 results from `E:\results_chapter5_only`.
- Chapter 6 results from `E:\results_chapter6_only`.
- Chapter 7 results from `E:\results_chapter7_only`.
- Chapter 8 results from `E:\results_chapter8_only`.
- Chapter 9 results from `E:\results_chapter9_only`.
- Chapter 10 results from `E:\results_chapter10_only`.

## Suggested GitHub Tagline

Reproducible computational finance research on structurally constrained PINNs for barrier option pricing.

## Suggested CV Entry

Working Paper and Research Project: Structurally Constrained PINNs for Barrier Option Pricing. Developed a reproducible computational finance project comparing high-precision implicit finite differences with barrier-aware physics-informed neural surrogates, including Greek diagnostics, validation scorecards, residual analysis, and deployment-economics evaluation.

## Suggested SOP Wording

In my research project on barrier option pricing, I studied structurally constrained physics-informed neural networks against a high-precision implicit finite-difference benchmark. The work required combining financial PDE modeling, numerical methods, neural surrogate design, validation protocols, and runtime analysis. I also organized the project as a public reproducibility package with an accompanying working paper, which strengthened my interest in computational finance and research-oriented quantitative modeling.

## Recommended Public Positioning

Use:

- Working paper and reproducibility package.
- Computational finance research project.
- SSRN preprint with GitHub repository.

Avoid:

- Published paper, unless the work is formally accepted by a peer-reviewed venue.
- Production-ready pricing engine.
- Certified neural solver, unless a formal certification layer is added.
