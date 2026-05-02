# SSRN Working Paper Notes

This document summarizes the intended SSRN positioning for the project:

**Structurally Constrained PINNs for Barrier Option Pricing: Benchmarking Against High-Precision Implicit Finite Differences**

The SSRN upload should be presented as a working paper or preprint, not as a peer-reviewed publication.

## Recommended SSRN Title

Structurally Constrained PINNs for Barrier Option Pricing: Benchmarking Against High-Precision Implicit Finite Differences

## Recommended SSRN Abstract

This paper studies structurally constrained physics-informed neural networks for continuously monitored down-and-out European call option pricing under the Black-Scholes framework. Barrier options provide a demanding test case for neural PDE solvers because the absorbing boundary creates localized high-curvature behavior near the knockout region, where pricing accuracy alone may fail to reveal instability in Greeks and boundary consistency. The study constructs a high-precision implicit finite-difference benchmark using log-domain transformation, exact barrier alignment, Rannacher-smoothed Crank-Nicolson time stepping, and sparse linear solves. Against this benchmark, it evaluates a barrier-aware neural surrogate framework combining transformed coordinates, hard barrier enforcement, barrier-aware adaptive collocation, hybrid optimization, and protocol-based validation. The empirical analysis jointly examines price errors, Delta and Gamma behavior, boundary residuals, positivity, PDE residual diagnostics, and repeated-query deployment economics. The results show that the finite-difference method remains the strongest local-precision benchmark, while the barrier-aware PINN is the strongest learned model in near-barrier Gamma control and boundary consistency. Supervised surrogates become economically attractive only under sufficiently large repeated-query workloads and conditional error tolerance. The paper positions barrier-option surrogate selection as a structured decision problem in which accuracy determines admissibility and amortization determines economic preference.

## Recommended Keywords

barrier option pricing; physics-informed neural networks; computational finance; finite differences; Greeks; PDE-constrained learning; neural surrogates; validation protocol

## Suggested JEL Codes

G13; C63; C45

## Recommended SSRN Status Wording

Use one of the following phrases in applications and public profiles:

- Working paper available on SSRN.
- Preprint available on SSRN.
- Research manuscript and reproducibility package available on SSRN and GitHub.
- Manuscript prepared for submission to a computational finance journal; preprint available on SSRN.

Avoid wording such as:

- Published on SSRN.
- SSRN publication.
- Peer-reviewed SSRN paper.

## Repository Linkage

The SSRN page should link to the GitHub repository as a reproducibility package. The GitHub repository should link back to the SSRN page once the SSRN paper URL is available.

Suggested repository wording:

This repository provides the reproducibility package for the SSRN working paper, including generated figures, tables, validation outputs, model artifacts, and chapter-level result summaries.

## Source Material Integrated

The working paper and reproducibility package are built from the following local materials:

- Final manuscript: `Structurally Constrained PINNs for Barrier Option Pricing.pdf`
- Chapter 3 results: `results_chapter3_only`
- Chapter 4 results: `results_chapter4_only`
- Chapter 5 results: `results_chapter5_only`
- Chapter 6 results: `results_chapter6_only`
- Chapter 7 results: `results_chapter7_only`
- Chapter 8 results: `results_chapter8_only`
- Chapter 9 results: `results_chapter9_only`
- Chapter 10 results: `results_chapter10_only`

## AI Disclosure Template

If generative AI tools were used for language polishing, formatting assistance, or code review, the paper should include a disclosure statement such as:

Generative AI tools were used only for language polishing, formatting assistance, and limited drafting support. All modeling choices, numerical experiments, analysis, interpretation, and conclusions were produced, checked, and approved by the author.

If no AI tools were used, omit this section from the paper.

## Application Positioning

For master's applications in financial engineering, quantitative finance, financial mathematics, computational finance, or applied mathematics, the strongest positioning is:

This is a reproducible computational finance research project with an accompanying SSRN working paper and GitHub repository.

This positioning emphasizes research maturity, numerical modeling, machine learning implementation, and reproducibility rather than claiming formal journal publication.
