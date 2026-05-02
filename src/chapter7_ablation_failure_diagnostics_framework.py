import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import chapter4_barrier_surrogate_framework as ch4


# ============================================================
# Configuration and data containers
# ============================================================
@dataclass
class AblationVariant:
    name: str
    group: str
    median_re: float
    q95_re: float
    worst_re: float
    delta_err: float
    gamma_err: float
    barrier_residual: float
    positivity_violation: float
    success_rate: float
    training_time: float


@dataclass
class Chapter7Config:
    seed: int = 42
    output_dir: str = "results_chapter7_only"
    use_demo_data: bool = True
    variants: List[AblationVariant] = field(default_factory=list)


def default_config() -> Chapter7Config:
    variants = [
        AblationVariant("Naive PINN", "Failure baseline", 6.80, 15.20, 26.50, 0.110, 0.720, 3.4e-2, 0.060, 0.20, 310.0),
        AblationVariant("Raw S-space", "Coordinate choice", 5.10, 12.80, 23.20, 0.084, 0.560, 2.2e-2, 0.030, 0.34, 325.0),
        AblationVariant("x = S/K", "Coordinate choice", 3.00, 7.20, 14.40, 0.051, 0.310, 7.0e-3, 0.010, 0.58, 333.0),
        AblationVariant("y = ln(S/K)", "Coordinate choice", 1.65, 3.80, 8.40, 0.028, 0.180, 1.2e-3, 0.000, 0.84, 340.0),
        AblationVariant("Soft BC", "Ansatz", 2.85, 6.40, 12.60, 0.049, 0.290, 5.5e-3, 0.008, 0.55, 348.0),
        AblationVariant("Hard barrier only", "Ansatz", 1.95, 4.55, 9.20, 0.032, 0.205, 9.0e-5, 0.004, 0.77, 355.0),
        AblationVariant("Hard barrier + positivity", "Ansatz", 1.58, 3.75, 7.95, 0.027, 0.165, 7.0e-5, 0.000, 0.86, 360.0),
        AblationVariant("No refinement", "BAAC", 2.45, 5.85, 11.20, 0.041, 0.260, 1.5e-4, 0.002, 0.64, 350.0),
        AblationVariant("Static oversampling", "BAAC", 1.92, 4.45, 8.85, 0.032, 0.205, 1.2e-4, 0.001, 0.78, 360.0),
        AblationVariant("Residual refinement", "BAAC", 1.66, 3.92, 8.10, 0.028, 0.173, 1.0e-4, 0.000, 0.84, 372.0),
        AblationVariant("Full BAAC", "BAAC", 1.49, 3.48, 7.45, 0.024, 0.151, 8.0e-5, 0.000, 0.91, 384.0),
    ]
    return Chapter7Config(variants=variants)


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def save_df_png(df: pd.DataFrame, output_path: Path, title: str) -> None:
    ch4.save_dataframe_as_png(df, output_path, title)


def variants_df(cfg: Chapter7Config) -> pd.DataFrame:
    rows = []
    for v in cfg.variants:
        rows.append({
            "Variant": v.name,
            "Group": v.group,
            "Median RE (%)": v.median_re,
            "95th RE (%)": v.q95_re,
            "Worst-case RE (%)": v.worst_re,
            "Delta error": v.delta_err,
            "Gamma error": v.gamma_err,
            "Barrier residual": v.barrier_residual,
            "Positivity violation": v.positivity_violation,
            "Success rate": v.success_rate,
            "Training time (s)": v.training_time,
        })
    return pd.DataFrame(rows)


# ============================================================
# Figure 17: Failure taxonomy
# ============================================================
def plot_failure_taxonomy(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    ax.axis("off")

    root_xy = (0.50, 0.88)
    branches = {
        "Scaling failure": (0.12, 0.60),
        "Boundary failure": (0.32, 0.60),
        "Sampling failure": (0.52, 0.60),
        "Optimization failure": (0.72, 0.60),
        "Greek failure": (0.90, 0.60),
    }
    children = {
        "Scaling failure": ["Raw S-space imbalance", "Slow convergence", "Poor operator conditioning"],
        "Boundary failure": ["Soft BC conflict", "Barrier residual leakage", "Local knockout misfit"],
        "Sampling failure": ["Too few near-barrier points", "Missed strike transition", "Hidden residual hotspots"],
        "Optimization failure": ["Adam plateau", "L-BFGS sensitivity", "Checkpoint instability"],
        "Greek failure": ["Noisy Gamma", "Delta drift", "Hedging instability"],
    }

    def box(x, y, w, h, text, fc="#ffffff", ec="#444444", fontsize=11, weight="normal"):
        rect = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2, edgecolor=ec, facecolor=fc
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)

    box(root_xy[0], root_xy[1], 0.26, 0.09, "Naive barrier-PINN underperforms", fc="#f7f7f7", ec="#222222", fontsize=13, weight="bold")

    branch_colors = {
        "Scaling failure": "#1f77b4",
        "Boundary failure": "#ff7f0e",
        "Sampling failure": "#2ca02c",
        "Optimization failure": "#9467bd",
        "Greek failure": "#d62728",
    }

    for name, (x, y) in branches.items():
        ax.annotate("", xy=(x, y + 0.05), xytext=(root_xy[0], root_xy[1] - 0.06),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#555555"))
        box(x, y, 0.18, 0.08, name, fc=branch_colors[name], ec=branch_colors[name], fontsize=11, weight="bold")
        ys = [0.36, 0.24, 0.12]
        for child, cy in zip(children[name], ys):
            ax.annotate("", xy=(x, cy + 0.035), xytext=(x, y - 0.05),
                        arrowprops=dict(arrowstyle="->", lw=1.1, color="#777777"))
            box(x, cy, 0.18, 0.065, child, fc="#ffffff", ec=branch_colors[name], fontsize=9.8)

    ax.text(
        0.50, 0.02,
        "Figure 17. Failure taxonomy. The ablation chapter is organized as a causal chain rather than as a list of isolated tricks.",
        ha="center", va="bottom", fontsize=11
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 18: Training pathology of naive PINN
# ============================================================
def plot_training_pathology(output_path: Path, seed: int = 42) -> None:
    rg = rng(seed)
    epochs = np.arange(1, 3001)

    # Extreme regime
    pde_extreme = 0.35 * np.exp(-epochs / 900) + 0.022 + 0.004 * rg.normal(size=len(epochs))
    bc_extreme = 3.5e3 * (1 - np.exp(-epochs / 350)) + 300 * np.exp(-epochs / 1600) + 60 * rg.normal(size=len(epochs))
    bc_extreme = np.clip(bc_extreme, 2.0, None)

    # Smooth regime
    pde_smooth = 0.28 * np.exp(-epochs / 800) + 0.014 + 0.003 * rg.normal(size=len(epochs))
    bc_smooth = 2.2e3 * np.exp(-epochs / 700) + 850 + 45 * rg.normal(size=len(epochs))
    bc_smooth = np.clip(bc_smooth, 1.5, None)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), sharey=True)

    for ax, pde, bc, title in [
        (axes[0], pde_extreme, bc_extreme, "Extreme curvature regime"),
        (axes[1], pde_smooth, bc_smooth, "Smoother regime"),
    ]:
        ax.plot(epochs, pde, color="#1f77b4", linewidth=2.0, label="PDE loss")
        ax.plot(epochs, bc, color="#d62728", linewidth=2.0, linestyle="--", label="Boundary loss")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(alpha=0.28, linestyle="--")
        ax.legend(frameon=True)

    axes[0].set_ylabel("Loss scale (log)")
    fig.suptitle("Figure 18. Training pathology of naive PINN", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 19: Effect of coordinate choice
# ============================================================
def plot_coordinate_choice(output_path: Path, seed: int = 42) -> None:
    rg = rng(seed + 1)
    epochs = np.arange(1, 2501)

    curves = {
        "Raw S": 0.62 * np.exp(-epochs / 1250) + 0.15 + 0.010 * rg.normal(size=len(epochs)),
        "x = S/K": 0.45 * np.exp(-epochs / 900) + 0.07 + 0.006 * rg.normal(size=len(epochs)),
        "y = ln(S/K)": 0.35 * np.exp(-epochs / 700) + 0.03 + 0.004 * rg.normal(size=len(epochs)),
    }
    final_err = {"Raw S": 5.10, "x = S/K": 3.00, "y = ln(S/K)": 1.65}
    grad_norm = {"Raw S": 8.2, "x = S/K": 4.8, "y = ln(S/K)": 2.7}

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))

    # Convergence curves
    ax = axes[0]
    colors = {"Raw S": "#d62728", "x = S/K": "#ff7f0e", "y = ln(S/K)": "#1f77b4"}
    for k, v in curves.items():
        ax.plot(epochs, np.clip(v, 1e-4, None), linewidth=2.0, label=k, color=colors[k])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (log)")
    ax.set_title("Convergence")
    ax.grid(alpha=0.28, linestyle="--")
    ax.legend(frameon=True)

    # Final error
    ax = axes[1]
    labels = list(final_err.keys())
    ax.bar(labels, [final_err[k] for k in labels], color=[colors[k] for k in labels], alpha=0.82)
    ax.set_ylabel("Median RE (%)")
    ax.set_title("Final error")
    ax.grid(axis="y", alpha=0.28, linestyle="--")

    # Gradient norm
    ax = axes[2]
    ax.bar(labels, [grad_norm[k] for k in labels], color=[colors[k] for k in labels], alpha=0.82)
    ax.set_ylabel("Median gradient norm")
    ax.set_title("Gradient scale")
    ax.grid(axis="y", alpha=0.28, linestyle="--")

    fig.suptitle("Figure 19. Effect of coordinate choice", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 20: Effect of hard-constrained ansatz
# ============================================================
def plot_ansatz_effect(output_path: Path) -> None:
    labels = ["Soft BC", "Hard barrier only", "Hard barrier + positivity"]
    price_err = [2.85, 1.95, 1.58]
    barrier_err = [5.5e-3, 9.0e-5, 7.0e-5]
    positivity = [0.008, 0.004, 0.000]

    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))

    # Price + barrier
    ax = axes[0]
    ax.bar(x - width/2, price_err, width=width, color="#1f77b4", alpha=0.84, label="Median RE (%)")
    ax2 = ax.twinx()
    ax2.bar(x + width/2, barrier_err, width=width, color="#d62728", alpha=0.76, label="Barrier residual")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median RE (%)")
    ax2.set_ylabel("Barrier residual")
    ax2.set_yscale("log")
    ax.set_title("Boundary enforcement vs price quality")
    ax.grid(axis="y", alpha=0.28, linestyle="--")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    # Positivity
    ax = axes[1]
    ax.bar(labels, positivity, color=["#ff7f0e", "#2ca02c", "#9467bd"], alpha=0.82)
    ax.set_ylabel("Positivity violation rate")
    ax.set_title("Economic admissibility")
    ax.grid(axis="y", alpha=0.28, linestyle="--")

    fig.suptitle("Figure 20. Effect of hard-constrained ansatz", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 21: Effect of BAAC
# ============================================================
def synthetic_error_map(scale: float, bias: float) -> np.ndarray:
    sigma_vals = np.linspace(0.15, 0.40, 6)
    rho_vals = np.linspace(0.002, 0.15, 6)
    Z = np.zeros((len(sigma_vals), len(rho_vals)))
    for i, sig in enumerate(sigma_vals):
        for j, rho in enumerate(rho_vals):
            barrier_term = 1.0 / (rho + 0.01)
            vol_term = 0.4 / sig
            Z[i, j] = bias + scale * (0.65 * barrier_term + 0.35 * vol_term)
    return Z


def plot_baac_effect(output_path: Path) -> None:
    panels = {
        "No refinement": synthetic_error_map(0.060, 0.55),
        "Static oversampling": synthetic_error_map(0.044, 0.38),
        "Residual refinement": synthetic_error_map(0.036, 0.28),
        "Full BAAC": synthetic_error_map(0.030, 0.20),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.8))
    axes = axes.flatten()
    vmax = max(np.max(v) for v in panels.values())

    for ax, (title, Z) in zip(axes, panels.items()):
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r"Barrier proximity index $\rho_d$")
        ax.set_ylabel(r"Volatility index $\sigma$")
        ax.set_xticks(range(Z.shape[1]))
        ax.set_yticks(range(Z.shape[0]))
        ax.set_xticklabels([f"{v:.3f}" for v in np.linspace(0.002, 0.15, 6)], rotation=35, ha="right")
        ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(0.15, 0.40, 6)])

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.86)
    cbar.set_label("Synthetic regional error level")
    fig.suptitle("Figure 21. Effect of BAAC", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Table 12: Ablation summary matrix
# ============================================================
def build_table12(cfg: Chapter7Config) -> pd.DataFrame:
    df = variants_df(cfg)
    cols = [
        "Variant",
        "Group",
        "Median RE (%)",
        "95th RE (%)",
        "Worst-case RE (%)",
        "Delta error",
        "Gamma error",
        "Barrier residual",
        "Positivity violation",
        "Success rate",
        "Training time (s)",
    ]
    return df[cols]


def export_table12(cfg: Chapter7Config, output_dir: Path) -> None:
    t12 = build_table12(cfg)
    t12.to_csv(output_dir / "table12_ablation_summary_matrix.csv", index=False)

    with open(output_dir / "table12_ablation_summary_matrix.tex", "w", encoding="utf-8") as f:
        f.write(
            t12.to_latex(
                index=False,
                escape=False,
                caption="Ablation summary matrix for failure diagnostics and structural design choices.",
                label="tab:ablation_summary_matrix",
            )
        )

    save_df_png(
        t12,
        output_dir / "table12_ablation_summary_matrix.png",
        "Table 12. Ablation summary matrix",
    )


# ============================================================
# Experiment scaffold / hooks
# ============================================================
def export_protocol_notes(output_dir: Path) -> None:
    notes = {
        "purpose": "Chapter 7 ablation and failure diagnostics scaffold",
        "mode": "demo-ready with replaceable metrics",
        "how_to_upgrade": [
            "Replace synthetic trajectory generators with actual training logs from each ablation variant.",
            "Replace synthetic BAAC heatmaps with real regional error surfaces from validation outputs.",
            "Keep Table 12 schema unchanged so the chapter text remains stable."
        ],
        "variant_groups": [
            "Failure baseline",
            "Coordinate choice",
            "Ansatz",
            "BAAC"
        ]
    }
    with open(output_dir / "chapter7_protocol_notes.json", "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)


def export_summary(cfg: Chapter7Config, output_dir: Path) -> None:
    summary = {
        "status": "chapter7 ablation assets generated",
        "use_demo_data": cfg.use_demo_data,
        "num_variants": len(cfg.variants),
        "figure17": str((output_dir / "figure17_failure_taxonomy.png").resolve()),
        "figure18": str((output_dir / "figure18_training_pathology_naive_pinn.png").resolve()),
        "figure19": str((output_dir / "figure19_effect_coordinate_choice.png").resolve()),
        "figure20": str((output_dir / "figure20_effect_hard_constrained_ansatz.png").resolve()),
        "figure21": str((output_dir / "figure21_effect_baac.png").resolve()),
        "table12_csv": str((output_dir / "table12_ablation_summary_matrix.csv").resolve()),
        "variants": [asdict(v) for v in cfg.variants],
    }
    with open(output_dir / "chapter7_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    cfg = default_config()
    out = Path(cfg.output_dir)
    ensure_dir(out)

    plot_failure_taxonomy(out / "figure17_failure_taxonomy.png")
    plot_training_pathology(out / "figure18_training_pathology_naive_pinn.png", seed=cfg.seed)
    plot_coordinate_choice(out / "figure19_effect_coordinate_choice.png", seed=cfg.seed)
    plot_ansatz_effect(out / "figure20_effect_hard_constrained_ansatz.png")
    plot_baac_effect(out / "figure21_effect_baac.png")
    export_table12(cfg, out)
    export_protocol_notes(out)
    export_summary(cfg, out)

    print("=" * 72)
    print("Chapter 7 ablation and failure diagnostics")
    print("=" * 72)
    print("Exported:")
    print("  - Figure 17: failure taxonomy")
    print("  - Figure 18: training pathology of naive PINN")
    print("  - Figure 19: effect of coordinate choice")
    print("  - Figure 20: effect of hard-constrained ansatz")
    print("  - Figure 21: effect of BAAC")
    print("  - Table 12: ablation summary matrix")
    print(f"Output directory: {out.resolve()}")


if __name__ == "__main__":
    main()
