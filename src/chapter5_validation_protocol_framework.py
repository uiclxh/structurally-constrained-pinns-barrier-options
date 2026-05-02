import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import chapter4_barrier_surrogate_framework as ch4


@dataclass
class ScenarioDomainConfig:
    sigma_train: Tuple[float, float] = (0.18, 0.35)
    sigma_valid: Tuple[float, float] = (0.18, 0.35)
    sigma_test: Tuple[float, float] = (0.15, 0.40)
    sigma_stress: Tuple[float, float] = (0.12, 0.40)

    rho_train: Tuple[float, float] = (0.03, 0.18)
    rho_valid: Tuple[float, float] = (0.03, 0.18)
    rho_test: Tuple[float, float] = (0.01, 0.20)
    rho_stress: Tuple[float, float] = (0.002, 0.15)

    maturity_train: Tuple[float, float] = (0.50, 1.50)
    maturity_valid: Tuple[float, float] = (0.50, 1.50)
    maturity_test: Tuple[float, float] = (0.25, 2.00)
    maturity_stress: Tuple[float, float] = (0.05, 1.00)

    n_train: int = 260
    n_valid: int = 80
    n_test: int = 120
    n_stress_barrier: int = 80
    n_stress_short_lowvol: int = 70
    n_stress_wide: int = 90


@dataclass
class RegionalZoneConfig:
    near_barrier_band: float = 0.06
    near_strike_halfwidth: float = 0.08
    far_field_start: float = 0.45
    tau_max: float = 1.0


@dataclass
class ValidationProtocolConfig:
    scenario: ScenarioDomainConfig = field(default_factory=ScenarioDomainConfig)
    zones: RegionalZoneConfig = field(default_factory=RegionalZoneConfig)
    full: ch4.FullConfig = field(default_factory=ch4.FullConfig)


def latin_uniform(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n + 1)
    u = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(u)
    return low + (high - low) * u


def build_scenario_panels(cfg: ValidationProtocolConfig, seed: int = 123) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    s = cfg.scenario

    train = pd.DataFrame({
        "split": "Train",
        "family": "Core train",
        "sigma": latin_uniform(s.n_train, *s.sigma_train, rng),
        "rho_d": latin_uniform(s.n_train, *s.rho_train, rng),
        "T": latin_uniform(s.n_train, *s.maturity_train, rng),
    })

    valid = pd.DataFrame({
        "split": "Validation",
        "family": "Core validation",
        "sigma": latin_uniform(s.n_valid, *s.sigma_valid, rng),
        "rho_d": latin_uniform(s.n_valid, *s.rho_valid, rng),
        "T": latin_uniform(s.n_valid, *s.maturity_valid, rng),
    })

    test = pd.DataFrame({
        "split": "Test",
        "family": "Wide random test",
        "sigma": latin_uniform(s.n_test, *s.sigma_test, rng),
        "rho_d": latin_uniform(s.n_test, *s.rho_test, rng),
        "T": latin_uniform(s.n_test, *s.maturity_test, rng),
    })

    stress_barrier = pd.DataFrame({
        "split": "Stress",
        "family": "Near-barrier stress",
        "sigma": latin_uniform(s.n_stress_barrier, 0.15, 0.35, rng),
        "rho_d": latin_uniform(s.n_stress_barrier, 0.002, 0.03, rng),
        "T": latin_uniform(s.n_stress_barrier, 0.30, 1.50, rng),
    })

    stress_short_lowvol = pd.DataFrame({
        "split": "Stress",
        "family": "Short-maturity / low-vol stress",
        "sigma": latin_uniform(s.n_stress_short_lowvol, 0.12, 0.20, rng),
        "rho_d": latin_uniform(s.n_stress_short_lowvol, 0.01, 0.10, rng),
        "T": latin_uniform(s.n_stress_short_lowvol, 0.05, 0.35, rng),
    })

    stress_wide = pd.DataFrame({
        "split": "Stress",
        "family": "Wide stress panel",
        "sigma": latin_uniform(s.n_stress_wide, *s.sigma_stress, rng),
        "rho_d": latin_uniform(s.n_stress_wide, *s.rho_stress, rng),
        "T": latin_uniform(s.n_stress_wide, *s.maturity_stress, rng),
    })

    return {
        "train": train,
        "validation": valid,
        "test": test,
        "stress_barrier": stress_barrier,
        "stress_short_lowvol": stress_short_lowvol,
        "stress_wide": stress_wide,
    }


def plot_data_split_and_scenario_map(panels: Dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    split_styles = {
        "Train": dict(color="#1f77b4", marker="o", alpha=0.60, s=24, label="Train"),
        "Validation": dict(color="#ff7f0e", marker="s", alpha=0.75, s=36, label="Validation"),
        "Test": dict(color="#2ca02c", marker="^", alpha=0.70, s=36, label="Test"),
        "Stress": dict(color="#d62728", marker="D", alpha=0.72, s=30, label="Stress"),
    }

    all_df = pd.concat(panels.values(), ignore_index=True)

    # Left panel: sigma vs rho_d
    ax = axes[0]
    for split, style in split_styles.items():
        df = all_df[all_df["split"] == split]
        ax.scatter(df["rho_d"], df["sigma"], **style)
    ax.set_xlabel(r"Barrier proximity $\rho_d$")
    ax.set_ylabel(r"Volatility $\sigma$")
    ax.set_title("Parameter coverage by split")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=True, fontsize=10)

    # Right panel: maturity vs rho_d, distinguish stress families
    ax = axes[1]
    family_styles = {
        "Core train": dict(color="#1f77b4", marker="o", alpha=0.55, s=20),
        "Core validation": dict(color="#ff7f0e", marker="s", alpha=0.75, s=32),
        "Wide random test": dict(color="#2ca02c", marker="^", alpha=0.70, s=34),
        "Near-barrier stress": dict(color="#d62728", marker="D", alpha=0.72, s=34),
        "Short-maturity / low-vol stress": dict(color="#9467bd", marker="P", alpha=0.72, s=36),
        "Wide stress panel": dict(color="#8c564b", marker="X", alpha=0.68, s=34),
    }
    for family, style in family_styles.items():
        df = all_df[all_df["family"] == family]
        ax.scatter(df["rho_d"], df["T"], label=family, **style)
    ax.set_xlabel(r"Barrier proximity $\rho_d$")
    ax.set_ylabel(r"Maturity $T$")
    ax.set_title("Scenario family map")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=True, fontsize=8.8, loc="upper right")

    fig.suptitle("Figure 13. Data split and scenario family map", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_regional_validation_zones(cfg: ValidationProtocolConfig, output_path: Path) -> None:
    p = cfg.full.problem
    z = cfg.zones

    y_min = p.y_barrier
    y_max = p.y_max
    tau_max = z.tau_max

    fig, ax = plt.subplots(figsize=(13.5, 5.7))

    # Near barrier
    ax.axvspan(y_min, y_min + z.near_barrier_band, color="#1f77b4", alpha=0.16, label="Near-barrier band")
    # Near strike
    ax.axvspan(-z.near_strike_halfwidth, z.near_strike_halfwidth, color="#ff7f0e", alpha=0.12, label="Near-strike band")
    # Far field
    ax.axvspan(z.far_field_start, y_max, color="#2ca02c", alpha=0.10, label="Far-field band")

    # Smooth interior
    ax.axvspan(y_min + z.near_barrier_band, z.far_field_start, color="#7f7f7f", alpha=0.05, label="Smooth interior")

    # Reference lines
    ax.axvline(y_min, color="#1f77b4", linestyle="--", linewidth=2.0)
    ax.axvline(0.0, color="#ff7f0e", linestyle="--", linewidth=1.8)
    ax.axvline(z.far_field_start, color="#2ca02c", linestyle=":", linewidth=2.0)

    ax.text(y_min - 0.005, 0.94 * tau_max, r"Barrier $\ln \beta$", ha="right", va="center", fontsize=11, backgroundcolor="white")
    ax.text(0.01, 0.94 * tau_max, r"Strike $y=0$", ha="left", va="center", fontsize=11, backgroundcolor="white")
    ax.text(z.far_field_start + 0.01, 0.94 * tau_max, "Far-field start", ha="left", va="center", fontsize=11, backgroundcolor="white")

    ax.text(y_min + 0.5 * z.near_barrier_band, 0.52 * tau_max, "Near-barrier\nvalidation zone",
            ha="center", va="center", fontsize=11, backgroundcolor="white")
    ax.text(0.0, 0.52 * tau_max, "Near-strike\ntransition zone",
            ha="center", va="center", fontsize=11, backgroundcolor="white")
    ax.text(0.23, 0.52 * tau_max, "Smooth interior", ha="center", va="center", fontsize=11, backgroundcolor="white")
    ax.text((z.far_field_start + y_max) / 2.0, 0.52 * tau_max, "Far-field\nconsistency zone",
            ha="center", va="center", fontsize=11, backgroundcolor="white")

    ax.annotate(
        "Worst-case control matters here",
        xy=(y_min + 0.03, 0.18),
        xytext=(y_min + 0.18, 0.10),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
    )

    ax.set_xlim(y_min - 0.02, y_max + 0.03)
    ax.set_ylim(0.0, tau_max)
    ax.set_xlabel(r"Transformed state $y = \ln(S/K)$")
    ax.set_ylabel(r"Time to maturity $\tau$")
    ax.set_title("Figure 14. Regional validation zones")
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(frameon=True, fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_table8_metrics_dictionary(cfg: ValidationProtocolConfig) -> pd.DataFrame:
    a = cfg.full.acceptance
    rows = [
        ("Pricing", "Pointwise price relative error", "mean / median / q95 / q99", "Global test set", "Report only"),
        ("Pricing", "Pointwise price relative error", "q95", "Validation set", f"<= {a.max_price_q95_pct:.2f}%"),
        ("Pricing", "Worst-case price error", "max", "Near-barrier stress band", "Report only"),
        ("Greeks", "Delta absolute error", "median / q95", "Validation + stress", "Report only"),
        ("Greeks", "Gamma absolute error", "q95", "Near-barrier band", f"<= {a.max_gamma_q95_abs:.3f}"),
        ("Greeks", "Gamma absolute error", "max", "Stress set", "Report only"),
        ("Admissibility", "Positivity violation rate", "fraction", "Validation + test", f"<= {a.max_positivity_violation_rate:.3f}"),
        ("Admissibility", "Monotonicity violation rate", "fraction", "Validation + test", "Report only"),
        ("Boundary", r"$|V(B_d,t)|$", "max / q95", "Barrier line", f"max <= {a.max_barrier_abs:.1e}"),
        ("Boundary", "Terminal payoff mismatch", "q95", r"$\tau=0$", "Report only"),
        ("Residual", "PDE residual norm", "q95", "Validation interior", f"<= {a.max_residual_q95:.2e}"),
        ("Residual", "Regional residual map", "heatmap", "Near-barrier / strike / smooth / far-field", "Visual diagnostic"),
        ("Runtime", "Training time", "wall-clock", "Offline", "Report only"),
        ("Runtime", "Inference latency", "per evaluation", "Online", "Report only"),
        ("Runtime", "Break-even workload", r"$N^\*$", "Deployment analysis", "Report only"),
    ]
    return pd.DataFrame(rows, columns=["Category", "Metric", "Statistic", "Region", "Acceptance threshold"])


def build_table9_acceptance_rule(cfg: ValidationProtocolConfig) -> pd.DataFrame:
    a = cfg.full.acceptance
    rows = [
        ("Validation price q95 relative error", f"<= {a.max_price_q95_pct:.2f}%", "Pass if upper-tail pricing error is acceptable on held-out validation points."),
        ("Near-barrier Gamma q95 absolute error", f"<= {a.max_gamma_q95_abs:.3f}", "Pass if local curvature remains stable in the most fragile region."),
        ("Barrier residual max", f"<= {a.max_barrier_abs:.1e}", "Pass if the hard barrier is respected up to numerical tolerance."),
        ("Validation PDE residual q95", f"<= {a.max_residual_q95:.2e}", "Pass if operator consistency is acceptable out of sample."),
        ("Positivity violation rate", f"<= {a.max_positivity_violation_rate:.3f}", "Pass if no economically inadmissible negative values appear."),
        ("Any criterion fails", "Refine or restart", "Reject checkpoint; revise architecture, sampling, or optimization and retrain."),
        ("All criteria pass", "Accept", "Promote checkpoint to deployable surrogate candidate and proceed to test / stress reporting."),
    ]
    return pd.DataFrame(rows, columns=["Criterion", "Threshold", "Pass-fail meaning"])


def export_tables(cfg: ValidationProtocolConfig, output_dir: Path) -> None:
    ch4.ensure_dir(output_dir)

    t8 = build_table8_metrics_dictionary(cfg)
    t9 = build_table9_acceptance_rule(cfg)

    t8.to_csv(output_dir / "table8_validation_metrics_dictionary.csv", index=False)
    t9.to_csv(output_dir / "table9_acceptance_rule.csv", index=False)

    with open(output_dir / "table8_validation_metrics_dictionary.tex", "w", encoding="utf-8") as f:
        f.write(
            t8.to_latex(
                index=False,
                escape=False,
                caption="Validation metrics dictionary for the barrier surrogate protocol.",
                label="tab:validation_metrics_dictionary",
            )
        )

    with open(output_dir / "table9_acceptance_rule.tex", "w", encoding="utf-8") as f:
        f.write(
            t9.to_latex(
                index=False,
                escape=False,
                caption="Acceptance rule for checkpoint approval in the validation protocol.",
                label="tab:acceptance_rule",
            )
        )

    ch4.save_dataframe_as_png(
        t8,
        output_dir / "table8_validation_metrics_dictionary.png",
        "Table 8. Validation metrics dictionary",
    )
    ch4.save_dataframe_as_png(
        t9,
        output_dir / "table9_acceptance_rule.png",
        "Table 9. Acceptance rule",
    )


def export_scenario_data(panels: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    for name, df in panels.items():
        df.to_csv(output_dir / f"{name}_panel.csv", index=False)

    summary_rows = []
    for name, df in panels.items():
        summary_rows.append({
            "panel": name,
            "n": len(df),
            "sigma_min": df["sigma"].min(),
            "sigma_max": df["sigma"].max(),
            "rho_min": df["rho_d"].min(),
            "rho_max": df["rho_d"].max(),
            "T_min": df["T"].min(),
            "T_max": df["T"].max(),
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "scenario_panel_summary.csv", index=False)


def main():
    cfg = ValidationProtocolConfig()
    output_dir = Path("results_chapter5_only")
    ch4.ensure_dir(output_dir)

    panels = build_scenario_panels(cfg)
    export_scenario_data(panels, output_dir)
    export_tables(cfg, output_dir)
    plot_data_split_and_scenario_map(panels, output_dir / "figure13_data_split_and_scenario_family_map.png")
    plot_regional_validation_zones(cfg, output_dir / "figure14_regional_validation_zones.png")

    summary = {
        "status": "chapter5 validation protocol assets generated",
        "table8_csv": str((output_dir / "table8_validation_metrics_dictionary.csv").resolve()),
        "table9_csv": str((output_dir / "table9_acceptance_rule.csv").resolve()),
        "figure13": str((output_dir / "figure13_data_split_and_scenario_family_map.png").resolve()),
        "figure14": str((output_dir / "figure14_regional_validation_zones.png").resolve()),
        "scenario_summary_csv": str((output_dir / "scenario_panel_summary.csv").resolve()),
        "acceptance_defaults": asdict(cfg.full.acceptance),
    }
    with open(output_dir / "chapter5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Chapter 5 validation protocol framework")
    print("=" * 72)
    print("Exported:")
    print("  - Figure 13: data split and scenario family map")
    print("  - Figure 14: regional validation zones")
    print("  - Table 8: validation metrics dictionary")
    print("  - Table 9: acceptance rule")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
