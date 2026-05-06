# SSRN Working Paper

This document records the SSRN listing and citation information for the working paper:

**Structurally Constrained PINNs for Barrier Option Pricing under a High-precision Implicit Finite-difference Benchmark**

The paper is publicly available as an SSRN working paper. SSRN availability should be described as a preprint or working paper, not as peer-reviewed journal publication.

## Public Links

- SSRN page: <https://ssrn.com/abstract=6628159>
- DOI: <https://doi.org/10.2139/ssrn.6628159>

## SSRN Metadata

- Author: Tom Lin
- Affiliation: Beijing Normal-Hong Kong Baptist University (BNBU)
- Date written: April 22, 2026
- Posted: May 4, 2026
- Length: 19 pages

## SSRN Abstract

Continuously monitored barrier options are a local-precision stress test rather than a generic smooth pricing task. The absorbing barrier creates a narrow region in which both the option value and local sensitivities vary sharply. We study down-and-out European calls in a one-dimensional Black-Scholes setting and evaluate neural surrogates against a deliberately strong classical benchmark. The benchmark uses a transformed logdomain formulation, exact barrier alignment, and Rannacher-smoothed Crank-Nicolson time stepping. The proposed neural framework combines transformed coordinates, a barrier-preserving hard-constrained ansatz, barrier-aware adaptive collocation, and hybrid optimization. Evaluation is protocol-based: prices, Delta, Gamma, barrier consistency, positivity, residual diagnostics, and deployment economics are assessed jointly rather than through pricing error alone. The benchmark FDM remains strongest for absolute local accuracy and Greek-sensitive tasks. Among learned models, the barrier-aware PINN is strongest in near-barrier Gamma control and boundary consistency, while supervised surrogates become economically attractive only under sufficiently large repeated-query workloads and conditional error tolerance. The main conclusion is therefore not neural replacement, but structured solver selection in which accuracy determines admissibility and amortization determines economic preference.

## Keywords

Barrier Option Pricing; Physics-informed Neural Networks; Implicit Finite Differences; Validation Protocol; Deployment Economics

## Suggested Citation

Lin, Tom. "Structurally Constrained PINNs for Barrier Option Pricing under a High-precision Implicit Finite-difference Benchmark." April 22, 2026. Available at SSRN: https://ssrn.com/abstract=6628159 or http://dx.doi.org/10.2139/ssrn.6628159.

## Recommended Status Wording

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

Suggested repository wording:

This repository provides the reproducibility package for the SSRN working paper, including generated figures, tables, validation outputs, model artifacts, and chapter-level result summaries.

## Source Material Integrated

The working paper and reproducibility package are built from the following local materials:

- Final SSRN manuscript: `Structurally Constrained PINNs for Barrier Option Pricing.pdf`
- Extended technical appendix with full derivations: `Structurally_Constrained_PINNs_for_Barrier_Option_Pricing_technical_appendix.pdf`
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
