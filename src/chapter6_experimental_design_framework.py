import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

import chapter4_barrier_surrogate_framework as ch4


@dataclass
class ScenarioFamilySpec:
    family: str
    sigma_range: Tuple[float, float]
    rho_d_range: Tuple[float, float]
    maturity_range: Tuple[float, float]
    n_contracts: int
    purpose: str


@dataclass
class BaselineSpec:
    model: str
    labels_needed: str
    structural_constraints: str
    greek_quality_expectation: str
    deployment_role: str
    local_precision_score: float
    many_query_efficiency_score: float


@dataclass
class ExperimentalDesignConfig:
    scenario_families: List[ScenarioFamilySpec] = field(default_factory=list)
    baselines: List[BaselineSpec] = field(default_factory=list)


def default_config() -> ExperimentalDesignConfig:
    scenario_families = [
        ScenarioFamilySpec(
            family="Core scenarios",
            sigma_range=(0.18, 0.35),
            rho_d_range=(0.03, 0.18),
            maturity_range=(0.50, 1.50),
            n_contracts=260,
            purpose="Main operating regime used for structured training and core comparison."
        ),
        ScenarioFamilySpec(
            family="Near-barrier stress",
            sigma_range=(0.15, 0.35),
            rho_d_range=(0.002, 0.03),
            maturity_range=(0.30, 1.50),
            n_contracts=80,
            purpose="Tests whether local pricing and Gamma remain stable in immediate knockout proximity."
        ),
        ScenarioFamilySpec(
            family="Short-maturity / low-vol stress",
            sigma_range=(0.12, 0.20),
            rho_d_range=(0.01, 0.10),
            maturity_range=(0.05, 0.35),
            n_contracts=70,
            purpose="Targets steep local geometry caused by short time horizons and low volatility."
        ),
        ScenarioFamilySpec(
            family="Wide random panel",
            sigma_range=(0.12, 0.40),
            rho_d_range=(0.002, 0.20),
            maturity_range=(0.05, 2.00),
            n_contracts=210,
            purpose="Broad held-out coverage for post-acceptance generalization and deployment reporting."
        ),
    ]

    baselines = [
        BaselineSpec(
            model="High-precision implicit FDM",
            labels_needed="No",
            structural_constraints="Exact PDE discretization, aligned barrier, implicit stability",
            greek_quality_expectation="Strong local benchmark",
            deployment_role="Single-query benchmark institution",
            local_precision_score=9.7,
            many_query_efficiency_score=2.8,
        ),
        BaselineSpec(
            model="Fixed-parameter PINN",
            labels_needed="No",
            structural_constraints="PDE residual; often weak unless barrier-aware structure is added",
            greek_quality_expectation="Moderate and regime-sensitive",
            deployment_role="One-network-per-instance neural PDE solver",
            local_precision_score=5.2,
            many_query_efficiency_score=5.3,
        ),
        BaselineSpec(
            model="Parametric PINN surrogate",
            labels_needed="No",
            structural_constraints="PDE-informed with shared parametric representation",
            greek_quality_expectation="Potentially stronger if validated regionally",
            deployment_role="One-network-many-contracts physics-informed surrogate",
            local_precision_score=6.3,
            many_query_efficiency_score=7.6,
        ),
        BaselineSpec(
            model="Supervised surrogate",
            labels_needed="Yes",
            structural_constraints="Depends on architecture; not automatic",
            greek_quality_expectation="Good in price, variable in Greeks unless designed explicitly",
            deployment_role="Fast batch pricing when labels are available",
            local_precision_score=7.0,
            many_query_efficiency_score=9.1,
        ),
        BaselineSpec(
            model="Differential surrogate",
            labels_needed="Yes (price + Greek labels)",
            structural_constraints="Derivative-aware supervision, but not inherently PDE-consistent",
            greek_quality_expectation="Typically strongest among label-based neural baselines",
            deployment_role="Fast deployment when Greek quality matters",
            local_precision_score=8.0,
            many_query_efficiency_score=8.4,
        ),
    ]
    return ExperimentalDesignConfig(
        scenario_families=scenario_families,
        baselines=baselines,
    )


def scenario_df(cfg: ExperimentalDesignConfig) -> pd.DataFrame:
    rows = []
    for s in cfg.scenario_families:
        rows.append({
            "Family": s.family,
            "Sigma range": f"[{s.sigma_range[0]:.3f}, {s.sigma_range[1]:.3f}]",
            "Rho_d range": f"[{s.rho_d_range[0]:.3f}, {s.rho_d_range[1]:.3f}]",
            "T range": f"[{s.maturity_range[0]:.2f}, {s.maturity_range[1]:.2f}]",
            "Number of contracts": s.n_contracts,
            "Purpose": s.purpose,
        })
    return pd.DataFrame(rows)


def baseline_df(cfg: ExperimentalDesignConfig) -> pd.DataFrame:
    rows = []
    for b in cfg.baselines:
        rows.append({
            "Model": b.model,
            "Labels needed": b.labels_needed,
            "Structural constraints": b.structural_constraints,
            "Greek quality expectation": b.greek_quality_expectation,
            "Deployment role": b.deployment_role,
        })
    return pd.DataFrame(rows)


def export_tables(cfg: ExperimentalDesignConfig, output_dir: Path) -> None:
    t10 = scenario_df(cfg)
    t11 = baseline_df(cfg)

    t10.to_csv(output_dir / "table10_scenario_families.csv", index=False)
    t11.to_csv(output_dir / "table11_baseline_family.csv", index=False)

    with open(output_dir / "table10_scenario_families.tex", "w", encoding="utf-8") as f:
        f.write(
            t10.to_latex(
                index=False,
                escape=False,
                caption="Scenario families and parameter ranges used in the experimental design.",
                label="tab:scenario_families",
            )
        )

    with open(output_dir / "table11_baseline_family.tex", "w", encoding="utf-8") as f:
        f.write(
            t11.to_latex(
                index=False,
                escape=False,
                caption="Baseline family used in the comparison design.",
                label="tab:baseline_family",
            )
        )

    ch4.save_dataframe_as_png(
        t10,
        output_dir / "table10_scenario_families.png",
        "Table 10. Scenario families and parameter ranges",
    )
    ch4.save_dataframe_as_png(
        t11,
        output_dir / "table11_baseline_family.png",
        "Table 11. Baseline family",
    )


def plot_scenario_matrix(cfg: ExperimentalDesignConfig, output_path: Path) -> None:
    colors = {
        "Core scenarios": "#1f77b4",
        "Near-barrier stress": "#d62728",
        "Short-maturity / low-vol stress": "#9467bd",
        "Wide random panel": "#2ca02c",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    axes = axes.flatten()

    for ax, spec in zip(axes, cfg.scenario_families):
        rho0, rho1 = spec.rho_d_range
        sig0, sig1 = spec.sigma_range
        rect = patches.Rectangle(
            (rho0, sig0),
            rho1 - rho0,
            sig1 - sig0,
            facecolor=colors[spec.family],
            edgecolor=colors[spec.family],
            alpha=0.18,
            linewidth=2.0,
        )
        ax.add_patch(rect)
        ax.set_xlim(0.0, 0.22)
        ax.set_ylim(0.10, 0.42)
        ax.set_xlabel(r"Barrier proximity $\rho_d$")
        ax.set_ylabel(r"Volatility $\sigma$")
        ax.set_title(spec.family)
        ax.grid(alpha=0.28, linestyle="--")

        text = (
            rf"$\sigma \in [{sig0:.2f},{sig1:.2f}]$" "\n"
            rf"$\rho_d \in [{rho0:.3f},{rho1:.3f}]$" "\n"
            rf"$T \in [{spec.maturity_range[0]:.2f},{spec.maturity_range[1]:.2f}]$" "\n"
            rf"$N={spec.n_contracts}$"
        )
        ax.text(
            0.97, 0.95, text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.92),
        )

        # Put a short purpose line near bottom
        ax.text(
            0.03, 0.06, spec.purpose,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=9.6,
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.85", alpha=0.88),
        )

    fig.suptitle("Figure 15. Scenario matrix", fontsize=16, y=0.98)
    fig.text(
        0.5, 0.02,
        "Experimental design is organized around four scenario families rather than isolated case studies.",
        ha="center", fontsize=11
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison_design_map(cfg: ExperimentalDesignConfig, output_path: Path) -> None:
    color_map = {
        "No": "#1f77b4",
        "Yes": "#2ca02c",
        "Yes (price + Greek labels)": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(11.5, 7.2))

    for b in cfg.baselines:
        ax.scatter(
            b.local_precision_score,
            b.many_query_efficiency_score,
            s=250,
            color=color_map.get(b.labels_needed, "#7f7f7f"),
            alpha=0.82,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.text(
            b.local_precision_score + 0.12,
            b.many_query_efficiency_score + 0.10,
            b.model,
            fontsize=10.5,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.85", alpha=0.92),
        )

    ax.set_xlim(0.0, 10.5)
    ax.set_ylim(0.0, 10.5)
    ax.set_xlabel("Local precision", fontsize=12)
    ax.set_ylabel("Many-query efficiency", fontsize=12)
    ax.set_title("Figure 16. Comparison design map", fontsize=15)
    ax.grid(alpha=0.28, linestyle="--")

    ax.annotate(
        "Benchmark region",
        xy=(9.7, 2.8),
        xytext=(8.2, 1.4),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9),
    )
    ax.annotate(
        "Deployment-efficient region",
        xy=(9.1, 9.1),
        xytext=(6.2, 9.7),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9),
    )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Label-free baseline", markerfacecolor="#1f77b4",
                   markeredgecolor="black", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="Label-based baseline", markerfacecolor="#2ca02c",
                   markeredgecolor="black", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="Price+Greek supervised", markerfacecolor="#9467bd",
                   markeredgecolor="black", markersize=10),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def export_summary(cfg: ExperimentalDesignConfig, output_dir: Path) -> None:
    summary = {
        "status": "chapter6 experimental design assets generated",
        "scenario_families": [asdict(s) for s in cfg.scenario_families],
        "baselines": [asdict(b) for b in cfg.baselines],
        "table10_csv": (output_dir / "table10_scenario_families.csv").as_posix(),
        "table11_csv": (output_dir / "table11_baseline_family.csv").as_posix(),
        "figure15": (output_dir / "figure15_scenario_matrix.png").as_posix(),
        "figure16": (output_dir / "figure16_comparison_design_map.png").as_posix(),
    }
    with open(output_dir / "chapter6_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    cfg = default_config()
    output_dir = Path("results/results_chapter6_only")
    ch4.ensure_dir(output_dir)

    export_tables(cfg, output_dir)
    plot_scenario_matrix(cfg, output_dir / "figure15_scenario_matrix.png")
    plot_comparison_design_map(cfg, output_dir / "figure16_comparison_design_map.png")
    export_summary(cfg, output_dir)

    print("=" * 72)
    print("Chapter 6 experimental design and baseline family")
    print("=" * 72)
    print("Exported:")
    print("  - Figure 15: scenario matrix")
    print("  - Table 10: scenario families and parameter ranges")
    print("  - Table 11: baseline family")
    print("  - Figure 16: comparison design map")
    print(f"Output directory: {output_dir.as_posix()}")


if __name__ == "__main__":
    main()
