# Structurally Constrained PINNs for Barrier Option Pricing



This repository accompanies the working paper:



**Structurally Constrained PINNs for Barrier Option Pricing: Benchmarking Against High-Precision Implicit Finite Difference**



The project studies whether physics-informed neural surrogates can be made credible for continuously monitored down-and-out European call options under the Black-Scholes framework. The central emphasis is not only pricing accuracy, but also barrier consistency, Greek stability, residual diagnostics, benchmark strength, and deployment economics.



## Project Overview



Barrier options are a demanding test case for neural PDE solvers because the absorbing barrier creates a localized high-curvature region near the knockout boundary. A model that performs well on average pricing error may still fail in the near-barrier region, especially for Delta and Gamma.



This project compares a strong implicit finite-difference benchmark with structurally constrained neural surrogates. The retained neural framework combines transformed coordinates, hard barrier enforcement, barrier-aware adaptive collocation, hybrid optimization, and protocol-based validation.



## Research Questions



1. Why do naive PINNs fail in near-barrier high-curvature regimes?

2. Which structural constraints and sampling mechanisms stabilize neural surrogates for barrier PDEs?

3. How should prices, Greeks, boundary consistency, and residual diagnostics be jointly validated?

4. When does offline-online amortization make neural surrogates economically attractive relative to a strong finite-difference benchmark?



## Main Contributions



- A high-precision implicit finite-difference benchmark using log-domain transformation, exact barrier alignment, Rannacher-smoothed Crank-Nicolson stepping, and sparse linear solves.

- A barrier-aware neural surrogate framework with hard structural enforcement of the absorbing boundary.

- A validation protocol that jointly evaluates price error, Delta, Gamma, barrier residuals, positivity, and PDE residual diagnostics.

- Ablation evidence showing that transformed coordinates, hard barrier constraints, and barrier-aware sampling materially improve learned behavior.

- A deployment-economics analysis comparing one-off solves with repeated-query workloads.



## Repository Structure



```text

paper/

&#x20; Final working paper PDF.



results/results\_chapter3\_only/

&#x20; High-precision implicit finite-difference benchmark and convergence verification.



results/results\_chapter4\_only/

&#x20; Barrier-aware neural surrogate framework, architecture tables, loss terms, and initial model artifact.



results/results\_chapter5\_only/

&#x20; Validation protocol, train/validation/test/stress panels, metric dictionary, and acceptance rule.



results/results\_chapter6\_only/

&#x20; Scenario family construction, baseline family, and comparison design.



results/results\_chapter7\_only/

&#x20; Ablation and failure diagnostics for naive PINNs, coordinate transforms, hard barrier ansatz, and BAAC.



results/results\_chapter8\_only/

&#x20; Accuracy, Greek diagnostics, boundary consistency, residual diagnostics, validation scorecard, and trained surrogate models.



results/results\_chapter9\_only/

&#x20; Runtime measurements, break-even analysis, throughput comparison, and repeated-query use cases.



results/results\_chapter10\_only/

&#x20; Solver-selection decision map, research roadmap, and summary of established versus non-established claims.



