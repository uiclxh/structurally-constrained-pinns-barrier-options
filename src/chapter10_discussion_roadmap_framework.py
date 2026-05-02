import json
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

import chapter4_barrier_surrogate_framework as ch4


@dataclass
class Chapter10Config:
    output_dir: str = "results_chapter10_only"
    chapter8_scorecard_csv: str = r"E:\results_chapter8_only\table14_validation_scorecard.csv"
    chapter9_runtime_table_csv: str = r"E:\results_chapter9_only\table15_runtime_inputs_break_even_summary.csv"
    query_volume_ticks: tuple = (1, 10, 100, 1000, 10000, 100000)
    precision_ticks: tuple = (0, 1, 2, 3)
    precision_labels: tuple = (
        "Low\nprecision need",
        "Moderate",
        "High",
        "Audit-grade /\nGreek-sensitive",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_runtime_inputs(cfg: Chapter10Config) -> pd.DataFrame:
    path = Path(cfg.chapter9_runtime_table_csv)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_scorecard(cfg: Chapter10Config) -> pd.DataFrame:
    path = Path(cfg.chapter8_scorecard_csv)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_establishes_table() -> pd.DataFrame:
    rows = [
        {
            "Establishes": "A strong benchmark institution based on an implicit finite-difference solver rather than a weak classical baseline.",
            "Does not establish": "A universal replacement theorem under which neural surrogates dominate the benchmark across all tasks.",
        },
        {
            "Establishes": "A barrier-aware surrogate framework combining transformed coordinates, hard barrier structure, BAAC, and protocol-based validation.",
            "Does not establish": "Full certified worst-case error bounds or formal verification of every reported surrogate output.",
        },
        {
            "Establishes": "Real evidence that hard-constrained barrier encoding and barrier-aware sampling materially improve learned models.",
            "Does not establish": "That the current supervised or differential surrogates are production-ready for all near-barrier or Greek-sensitive workloads.",
        },
        {
            "Establishes": "That solver preference changes with workload because amortization matters once repeated-query volume is large enough.",
            "Does not establish": "Hardware-independent break-even constants or universally transferable runtime thresholds.",
        },
        {
            "Establishes": "A one-dimensional barrier-option test bed that is numerically nontrivial and diagnostic-rich.",
            "Does not establish": "Automatic generalization of the same conclusions to higher dimensions, richer dynamics, or broader exotic-product families.",
        },
    ]
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, tex_path: Path, csv_path: Path, png_path: Path, caption: str, label: str) -> None:
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False, caption=caption, label=label))
    try:
        ch4.save_dataframe_as_png(df, png_path, caption)
    except Exception:
        pass


def plot_decision_map(cfg: Chapter10Config, runtime_df: pd.DataFrame, scorecard_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(12.8, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[5.0, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    ax_note = fig.add_subplot(gs[1, 0])
    ax_note.axis("off")

    ax.set_xscale("log")
    ax.set_xlim(1, 140000)
    ax.set_ylim(0, 3.25)
    ax.set_xticks(cfg.query_volume_ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks(cfg.precision_ticks)
    ax.set_yticklabels(cfg.precision_labels)
    ax.set_xlabel("Query volume")
    ax.set_ylabel("Local precision requirement")
    ax.set_title("Figure 32. Decision map: when to use which solver")
    ax.grid(alpha=0.25, linestyle="--")

    # Broad regions
    regions = [
        (1, 0.0, 1500, 3.25, "#d9eaf7", "FDM-dominant zone"),
        (1500, 1.2, 50000, 2.1, "#fde7c7", "PINN conditional zone"),
        (25000, 0.0, 115000, 1.4, "#dff1df", "Supervised conditional zone"),
    ]
    for x, y, w, h, color, label in regions:
        patch = Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=0.85)
        ax.add_patch(patch)
        if "Supervised" in label:
            tx = 18500
            ty = 1.24
            label = "Supervised\nconditional zone"
        elif "PINN" in label:
            tx = 1900
            ty = 2.93
        else:
            tx = 1.3
            ty = 2.68
        text_kwargs = {"fontsize": 11, "weight": "bold"}
        if "Supervised" in label:
            text_kwargs["bbox"] = dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.6)
        ax.text(tx, ty, label, **text_kwargs)

    # Break-even guide lines from Chapter 9 if present
    if not runtime_df.empty:
        try:
            pinn_n = float(runtime_df.loc[runtime_df["Method"] == "PINN", "Break-even N*"].iloc[0])
            sup_n = float(runtime_df.loc[runtime_df["Method"] == "Supervised surrogate", "Break-even N*"].iloc[0])
            ax.axvline(pinn_n, color="#c77000", linestyle="--", linewidth=1.8)
            ax.axvline(sup_n, color="#1f7a1f", linestyle="--", linewidth=1.8)
            ax.text(
                pinn_n * 1.012, 2.98, "PINN break-even", color="#c77000", rotation=90, va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8)
            )
            ax.text(
                sup_n * 1.012, 1.52, "Supervised break-even", color="#1f7a1f", rotation=90, va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8)
            )
        except Exception:
            pass

    # Model placements
    points = [
        ("FDM", 60, 2.9, "#1f77b4"),
        ("PINN", 22000, 2.05, "#ff7f0e"),
        ("Supervised surrogate", 60000, 0.95, "#2ca02c"),
    ]
    for label, x, y, color in points:
        ax.scatter([x], [y], s=180, color=color, edgecolors="black", linewidths=0.8, zorder=3)
        if label == "PINN":
            ax.text(x * 1.04, y + 0.06, label, fontsize=10.5, weight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
        elif label == "Supervised surrogate":
            ax.text(82000, y + 0.02, label, fontsize=10.2, weight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
        else:
            ax.text(x * 1.10, y + 0.04, label, fontsize=10.5, weight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))

    note = (
        "Interpretation:\n"
        "FDM remains preferred when audit-grade precision or Greek reliability dominates.\n"
        "PINN becomes relevant when repeated-query volume grows but barrier-aware structure still matters.\n"
        "Supervised surrogates become economically attractive only when workload is very large and tolerated error is wider."
    )
    ax_note.text(
        0.5,
        0.52,
        note,
        ha="center",
        va="center",
        fontsize=10.1,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.96),
        transform=ax_note.transAxes,
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _road_box(ax, x, y, w, h, title, subtitle, facecolor):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="black",
        facecolor=facecolor,
        alpha=0.95,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=12.5, weight="bold")
    ax.text(
        x + w / 2,
        y + h * 0.30,
        textwrap.fill(subtitle, width=38),
        ha="center",
        va="center",
        fontsize=9.6,
        wrap=True,
    )


def plot_research_roadmap(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.4, 9.0), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Figure 33. Research roadmap", ha="center", va="center", fontsize=16, weight="bold")
    ax.text(
        0.5,
        0.91,
        "Three forward paths: stronger validation, broader surrogate families, and richer market realism.",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Main lanes
    _road_box(
        ax, 0.04, 0.64, 0.28, 0.18,
        "Roadmap A",
        "Stronger validation and certification. Residual estimators, tighter Greek diagnostics, localized certification, and out-of-distribution stress control.",
        "#d9eaf7",
    )
    _road_box(
        ax, 0.36, 0.64, 0.28, 0.18,
        "Roadmap B",
        "Parametric surrogate and operator learning. One-network-many-contracts scaling and PINO/FNO/DeepONet-style extensions.",
        "#fde7c7",
    )
    _road_box(
        ax, 0.68, 0.64, 0.28, 0.18,
        "Roadmap C",
        "Richer dynamics and products. Stochastic volatility, jumps, double barriers, discrete monitoring, rebates, and portfolio tasks.",
        "#dff1df",
    )

    # Supporting milestones
    milestones = [
        (0.06, 0.36, 0.24, 0.14, "Short term", "Validation scorecards that survive stronger residual diagnostics."),
        (0.38, 0.36, 0.24, 0.14, "Medium term", "Parametric and operator surrogates with stronger many-query scaling."),
        (0.70, 0.36, 0.24, 0.14, "Long term", "Market-realistic product families under richer dynamics."),
    ]
    for x, y, w, h, t, s in milestones:
        _road_box(ax, x, y, w, h, t, s, "#f6f6f6")

    # Connector arrows
    arrowprops = dict(arrowstyle="-|>", lw=1.6, color="black")
    ax.annotate("", xy=(0.18, 0.62), xytext=(0.18, 0.50), arrowprops=arrowprops)
    ax.annotate("", xy=(0.50, 0.62), xytext=(0.50, 0.50), arrowprops=arrowprops)
    ax.annotate("", xy=(0.82, 0.62), xytext=(0.82, 0.50), arrowprops=arrowprops)

    # Bottom interpretation
    bottom = FancyBboxPatch(
        (0.07, 0.09), 0.86, 0.15,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2, edgecolor="black", facecolor="#f3f3f3"
    )
    ax.add_patch(bottom)
    ax.text(
        0.50, 0.175,
        "Overall interpretation:\n"
        "The next stage is not wider claiming, but stronger evidence, broader operators, and more realistic products.",
        ha="center", va="center", fontsize=11, weight="bold"
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    cfg = Chapter10Config()
    out = Path(cfg.output_dir)
    ensure_dir(out)

    runtime_df = load_runtime_inputs(cfg)
    scorecard_df = load_scorecard(cfg)

    table17 = build_establishes_table()
    save_table(
        table17,
        out / "table17_establishes_vs_not_establishes.tex",
        out / "table17_establishes_vs_not_establishes.csv",
        out / "table17_establishes_vs_not_establishes.png",
        "What this paper establishes and does not establish.",
        "tab:establishes_vs_not_establishes",
    )

    plot_decision_map(cfg, runtime_df, scorecard_df, out / "figure32_decision_map_when_to_use_which_solver.png")
    plot_research_roadmap(out / "figure33_research_roadmap.png")

    summary = {
        "status": "chapter10 discussion and roadmap assets prepared",
        "config": asdict(cfg),
        "inputs": {
            "chapter8_scorecard_csv": cfg.chapter8_scorecard_csv,
            "chapter9_runtime_table_csv": cfg.chapter9_runtime_table_csv,
        },
        "outputs": {
            "figure32": str((out / "figure32_decision_map_when_to_use_which_solver.png").resolve()),
            "figure33": str((out / "figure33_research_roadmap.png").resolve()),
            "table17": str((out / "table17_establishes_vs_not_establishes.csv").resolve()),
        },
    }
    with open(out / "chapter10_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Chapter 10 discussion and roadmap workflow")
    print("=" * 72)
    print("Exported:")
    print("  - Figure 32: decision map")
    print("  - Figure 33: research roadmap")
    print("  - Table 17: establishes vs does not establish")
    print(f"Output directory: {out.resolve()}")


if __name__ == "__main__":
    main()
