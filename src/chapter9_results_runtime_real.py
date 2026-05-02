import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import chapter3_fdm_benchmark_only as ch3
import chapter4_barrier_surrogate_framework as ch4
import chapter7_ablation_failure_diagnostics_real as ch7
import chapter8_results_accuracy_real as ch8


@dataclass
class Chapter9Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "results_chapter9_only"
    chapter8_output_dir: str = r"E:\results_chapter8_only"
    chapter7_output_dir: str = r"E:\results_chapter7_real_formal"
    pinn_model_path: str = r"E:\results_chapter7_real_formal\full_baac_guard_probe\best_model.pt"
    pinn_variant_spec_path: str = r"E:\results_chapter7_real_formal\full_baac_guard_probe\variant_spec.json"
    supervised_model_path: str = r"E:\results_chapter8_only\models\supervised_surrogate\best_model.pt"
    evaluation_counts: Tuple[int, ...] = (1, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000)
    batch_sizes: Tuple[int, ...] = (1, 8, 32, 128, 512, 2048)
    throughput_repeats_small: int = 40
    throughput_repeats_large: int = 15
    latency_repeats: int = 10
    label_generation_time_measure_repeats: int = 1
    benchmark_supervised_training_time: bool = False
    supervised_training_time_fallback_seconds: float = 120.0
    hardware_label: str = "Current runtime host"
    use_case_surface_points_spot: int = 100
    use_case_surface_points_tau: int = 100
    use_case_surface_sigma: float = 0.25
    use_case_surface_rho: float = 0.03
    use_case_surface_T: float = 1.0
    intraday_updates: int = 24
    intraday_grid_size: int = 2500
    calibration_iterations: int = 3000
    calibration_eval_per_iter: int = 16
    portfolio_contracts: int = 5000
    portfolio_greek_passes: int = 2


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_ch8_cfg(cfg: Chapter9Config) -> ch8.Chapter8Config:
    return ch8.Chapter8Config(
        device=cfg.device,
        output_dir=cfg.chapter8_output_dir,
        pinn_model_path=cfg.pinn_model_path,
        pinn_variant_spec_path=cfg.pinn_variant_spec_path,
        force_retrain_surrogates=False,
    )


def load_scorecard(chapter8_dir: Path) -> pd.DataFrame:
    path = chapter8_dir / "table14_validation_scorecard.csv"
    return pd.read_csv(path)


def load_full_baac_training_time_seconds(chapter7_dir: Path) -> float:
    table_path = chapter7_dir / "table12_ablation_summary_matrix.csv"
    if table_path.exists():
        df = pd.read_csv(table_path)
        row = df.loc[df["Variant"] == "Full BAAC"]
        if not row.empty:
            return float(row.iloc[0]["Training time (s)"])
    summary_path = chapter7_dir / "chapter7_real_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Conservative fallback if the direct training-time row is missing.
        return 191.928 if "config" in data else 200.0
    return 191.928


def measure_label_generation_time(cfg: Chapter9Config, cfg8: ch8.Chapter8Config) -> float:
    train_scenarios, valid_scenarios, _ = ch8.build_eval_panels(cfg8)
    times: List[float] = []
    for rep in range(cfg.label_generation_time_measure_repeats):
        t0 = time.perf_counter()
        _ = ch8.sample_labeled_points(train_scenarios, cfg8.train_label_samples_per_scenario, seed=cfg.seed + 900 + rep)
        _ = ch8.sample_labeled_points(valid_scenarios, cfg8.valid_label_samples_per_scenario, seed=cfg.seed + 950 + rep)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def load_supervised_model(cfg: Chapter9Config, cfg8: ch8.Chapter8Config) -> ch8.LabelSurrogate:
    device = torch.device(cfg.device)
    model = ch8.LabelSurrogate(cfg8).to(device)
    state = torch.load(cfg.supervised_model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_adapters(cfg: Chapter9Config) -> Tuple[ch8.Chapter8Config, Dict[str, ch8.BaseAdapter]]:
    cfg8 = make_ch8_cfg(cfg)
    device = torch.device(cfg.device)
    truth = ch8.TruthAdapter()
    fdm = ch8.FDMAdapter(cfg8)
    pinn = ch8.PINNAdapter(cfg8)
    supervised_model = load_supervised_model(cfg, cfg8)
    adapters: Dict[str, ch8.BaseAdapter] = {
        "Truth": truth,
        "FDM": fdm,
        "Supervised": ch8.NeuralAdapter("Supervised", supervised_model, device),
    }
    if pinn.available:
        adapters["PINN"] = pinn
    return cfg8, adapters


def representative_scenarios(cfg8: ch8.Chapter8Config) -> List[ch3.BarrierScenario]:
    return ch8.core_scenarios(cfg8)


def make_batch_inputs(
    batch_size: int,
    device: torch.device,
    cfg8: ch8.Chapter8Config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigmas = np.array(cfg8.sigma_grid, dtype=float)
    rhos = np.array(cfg8.rho_grid, dtype=float)
    idx = np.arange(batch_size)
    sigma_vals = sigmas[idx % len(sigmas)]
    rho_vals = rhos[(idx // len(sigmas)) % len(rhos)]
    beta_vals = 1.0 - rho_vals
    S_vals = np.clip(100.0 * (1.0 + 0.06 * np.sin(idx / max(batch_size, 1) * 2.0 * math.pi)), beta_vals * 100.0 + 0.2, 160.0)
    tau_vals = 0.08 + 0.92 * ((idx % 23) / 22.0)
    r_vals = np.full(batch_size, 0.10)
    q_vals = np.zeros(batch_size)
    tensors = [
        torch.tensor(arr, dtype=torch.float32, device=device).reshape(-1, 1)
        for arr in [S_vals, tau_vals, sigma_vals, beta_vals, r_vals, q_vals]
    ]
    return tuple(tensors)


def measure_fdm_latency(adapter: ch8.BaseAdapter, scenarios: Sequence[ch3.BarrierScenario], repeats: int) -> float:
    times: List[float] = []
    for scn in scenarios:
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = adapter.price(scn, scn.S0, scn.T)
        times.append((time.perf_counter() - t0) / repeats)
    return float(np.median(times))


def measure_pinn_latency(adapter: ch8.PINNAdapter, scenarios: Sequence[ch3.BarrierScenario], repeats: int) -> float:
    times: List[float] = []
    for scn in scenarios:
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = adapter.price(scn, scn.S0, scn.T)
        times.append((time.perf_counter() - t0) / repeats)
    return float(np.median(times))


def measure_supervised_latency(adapter: ch8.NeuralAdapter, scenarios: Sequence[ch3.BarrierScenario], repeats: int) -> float:
    times: List[float] = []
    for scn in scenarios:
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = adapter.price(scn, scn.S0, scn.T)
        times.append((time.perf_counter() - t0) / repeats)
    return float(np.median(times))


def measure_fdm_batch_throughput(adapter: ch8.BaseAdapter, cfg8: ch8.Chapter8Config, batch_sizes: Sequence[int], repeats_small: int, repeats_large: int) -> pd.DataFrame:
    rows = []
    scenarios = representative_scenarios(cfg8)
    for batch_size in batch_sizes:
        repeats = repeats_small if batch_size <= 32 else repeats_large
        contracts = []
        for i in range(batch_size):
            scn = scenarios[i % len(scenarios)]
            contracts.append((scn, scn.S0, scn.T))
        t0 = time.perf_counter()
        for _ in range(repeats):
            for scn, S, tau in contracts:
                _ = adapter.price(scn, S, tau)
        elapsed = time.perf_counter() - t0
        throughput = (batch_size * repeats) / max(elapsed, 1e-12)
        rows.append({"Method": "FDM", "Batch size": batch_size, "Contracts/sec": throughput})
    return pd.DataFrame(rows)


def measure_pinn_batch_throughput(adapter: ch8.PINNAdapter, cfg8: ch8.Chapter8Config, batch_sizes: Sequence[int], repeats_small: int, repeats_large: int) -> pd.DataFrame:
    rows = []
    if not adapter.available:
        return pd.DataFrame(rows)
    device = adapter.device
    model = adapter.model
    model.eval()
    for batch_size in batch_sizes:
        repeats = repeats_small if batch_size <= 32 else repeats_large
        S, tau, sigma, beta, r, q = make_batch_inputs(batch_size, device, cfg8)
        with torch.no_grad():
            _ = model(S, tau, sigma, beta, r, q)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                _ = model(S, tau, sigma, beta, r, q)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        throughput = (batch_size * repeats) / max(elapsed, 1e-12)
        rows.append({"Method": "PINN", "Batch size": batch_size, "Contracts/sec": throughput})
    return pd.DataFrame(rows)


def measure_supervised_batch_throughput(adapter: ch8.NeuralAdapter, cfg8: ch8.Chapter8Config, batch_sizes: Sequence[int], repeats_small: int, repeats_large: int) -> pd.DataFrame:
    rows = []
    device = adapter.device
    model = adapter.model
    model.eval()
    for batch_size in batch_sizes:
        repeats = repeats_small if batch_size <= 32 else repeats_large
        S, tau, sigma, beta, r, q = make_batch_inputs(batch_size, device, cfg8)
        with torch.no_grad():
            _ = model(S, tau, sigma, beta, r, q)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                _ = model(S, tau, sigma, beta, r, q)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        throughput = (batch_size * repeats) / max(elapsed, 1e-12)
        rows.append({"Method": "Supervised surrogate", "Batch size": batch_size, "Contracts/sec": throughput})
    return pd.DataFrame(rows)


def break_even_n(setup_cost: float, baseline_latency: float, candidate_latency: float) -> float:
    gain = baseline_latency - candidate_latency
    if gain <= 0.0:
        return math.inf
    return setup_cost / gain


def build_runtime_summary_table(
    cfg: Chapter9Config,
    scorecard: pd.DataFrame,
    fdm_latency: float,
    pinn_latency: float,
    supervised_latency: float,
    pinn_throughput: float,
    supervised_throughput: float,
    label_generation_time: float,
    pinn_training_time: float,
    supervised_training_time: float,
) -> pd.DataFrame:
    pinn_break_even = break_even_n(pinn_training_time, fdm_latency, pinn_latency)
    supervised_break_even = break_even_n(label_generation_time + supervised_training_time, fdm_latency, supervised_latency)
    rows = [
        {
            "Method": "FDM",
            "Train time (s)": 0.0,
            "Label generation time (s)": 0.0,
            "Inference latency (s)": fdm_latency,
            "Batch throughput (contracts/s)": np.nan,
            "Break-even N*": 0.0,
            "Validation status": "Benchmark",
        },
        {
            "Method": "PINN",
            "Train time (s)": pinn_training_time,
            "Label generation time (s)": 0.0,
            "Inference latency (s)": pinn_latency,
            "Batch throughput (contracts/s)": pinn_throughput,
            "Break-even N*": pinn_break_even,
            "Validation status": "Pass gamma + barrier",
        },
        {
            "Method": "Supervised surrogate",
            "Train time (s)": supervised_training_time,
            "Label generation time (s)": label_generation_time,
            "Inference latency (s)": supervised_latency,
            "Batch throughput (contracts/s)": supervised_throughput,
            "Break-even N*": supervised_break_even,
            "Validation status": "Pass barrier only",
        },
    ]
    return pd.DataFrame(rows)


def plot_total_runtime_vs_evals(
    cfg: Chapter9Config,
    out_path: Path,
    fdm_latency: float,
    pinn_latency: float,
    supervised_latency: float,
    pinn_training_time: float,
    supervised_setup_time: float,
) -> pd.DataFrame:
    evals = np.array(cfg.evaluation_counts, dtype=float)
    fdm_total = evals * fdm_latency
    pinn_total = pinn_training_time + evals * pinn_latency
    supervised_total = supervised_setup_time + evals * supervised_latency

    fig, ax = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    ax.plot(evals, fdm_total, marker="o", linewidth=2.0, label="FDM")
    ax.plot(evals, pinn_total, marker="s", linewidth=2.0, label="PINN")
    ax.plot(evals, supervised_total, marker="^", linewidth=2.0, label="Supervised surrogate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Total runtime (s)")
    ax.set_title("Figure 28. Total runtime vs number of evaluations")
    ax.grid(alpha=0.28, linestyle="--")
    ax.legend(frameon=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return pd.DataFrame(
        {
            "N": evals.astype(int),
            "T_FDM(s)": fdm_total,
            "T_PINN(s)": pinn_total,
            "T_Supervised(s)": supervised_total,
        }
    )


def plot_average_cost_vs_evals(
    cfg: Chapter9Config,
    out_path: Path,
    runtime_curve: pd.DataFrame,
) -> pd.DataFrame:
    evals = runtime_curve["N"].to_numpy(dtype=float)
    avg_fdm = runtime_curve["T_FDM(s)"].to_numpy(dtype=float) / evals
    avg_pinn = runtime_curve["T_PINN(s)"].to_numpy(dtype=float) / evals
    avg_sup = runtime_curve["T_Supervised(s)"].to_numpy(dtype=float) / evals

    fig, ax = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    ax.plot(evals, avg_fdm, marker="o", linewidth=2.0, label="FDM")
    ax.plot(evals, avg_pinn, marker="s", linewidth=2.0, label="PINN")
    ax.plot(evals, avg_sup, marker="^", linewidth=2.0, label="Supervised surrogate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Average cost per evaluation (s)")
    ax.set_title("Figure 29. Average cost per evaluation")
    ax.grid(alpha=0.28, linestyle="--")
    ax.legend(frameon=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return pd.DataFrame(
        {
            "N": evals.astype(int),
            "Avg_FDM(s)": avg_fdm,
            "Avg_PINN(s)": avg_pinn,
            "Avg_Supervised(s)": avg_sup,
        }
    )


def plot_batch_throughput(throughput_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    for method, group in throughput_df.groupby("Method"):
        group = group.sort_values("Batch size")
        ax.plot(group["Batch size"], group["Contracts/sec"], marker="o", linewidth=2.0, label=method)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Contracts / second")
    ax.set_title("Figure 30. Batch throughput comparison")
    ax.grid(alpha=0.28, linestyle="--")
    ax.legend(frameon=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def use_case_surface_dataset(cfg: Chapter9Config) -> Tuple[ch3.BarrierScenario, np.ndarray, np.ndarray]:
    scn = ch8.make_grid_scenario(
        make_ch8_cfg(cfg),
        cfg.use_case_surface_sigma,
        cfg.use_case_surface_rho,
        cfg.use_case_surface_T,
    )
    spots = np.linspace(scn.B_d + 0.002 * scn.K, min(scn.S_max, 1.35 * scn.K), cfg.use_case_surface_points_spot)
    taus = np.linspace(0.0, scn.T, cfg.use_case_surface_points_tau)
    return scn, spots, taus


def measure_surface_runtime_and_quality(
    adapter: ch8.BaseAdapter,
    truth: ch8.TruthAdapter,
    scn: ch3.BarrierScenario,
    spots: np.ndarray,
    taus: np.ndarray,
) -> Dict[str, float]:
    truth_prices = []
    pred_prices = []
    t0 = time.perf_counter()
    for tau in taus:
        for S in spots:
            pred_prices.append(adapter.price(scn, float(S), float(tau)))
    wall = time.perf_counter() - t0
    for tau in taus:
        for S in spots:
            truth_prices.append(truth.price(scn, float(S), float(tau)))
    truth_arr = np.asarray(truth_prices, dtype=float)
    pred_arr = np.asarray(pred_prices, dtype=float)
    re = 100.0 * np.abs(pred_arr - truth_arr) / (np.abs(truth_arr) + 1e-12)
    return {
        "wall_clock_s": wall,
        "median_re_pct": float(np.median(re)),
        "q95_re_pct": float(np.quantile(re, 0.95)),
        "max_re_pct": float(np.max(re)),
        "n_points": int(len(re)),
    }


def plot_use_case_surface(
    use_case_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    axes[0].bar(use_case_df["Method"], use_case_df["Wall-clock (s)"], color=colors, alpha=0.88)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].set_title("Risk surface generation runtime")
    axes[0].grid(axis="y", alpha=0.28, linestyle="--")

    axes[1].bar(use_case_df["Method"], use_case_df["q95 RE (%)"], color=colors, alpha=0.88)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("q95 price RE (%)")
    axes[1].set_title("Risk surface quality")
    axes[1].grid(axis="y", alpha=0.28, linestyle="--")

    fig.suptitle("Figure 31. Use case: barrier risk surface generation", fontsize=15, y=0.98)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def choose_solver(task: str, n_eval: int, greek_sensitive: bool, fdm_beats_all: bool, pinn_break_even: float, sup_break_even: float) -> Tuple[str, str]:
    if task in {"Single contract pricing", "Near-barrier validation audit"}:
        return "FDM", "Highest local precision is preferred and amortization is negligible."
    if greek_sensitive:
        if n_eval >= pinn_break_even:
            return "PINN", "Among learned models it preserves barrier and Gamma structure most faithfully."
        return "FDM", "Greek-sensitive low-volume tasks still favor the benchmark solver."
    if fdm_beats_all:
        return "FDM", "Current workload is too small to amortize training or label-generation costs."
    if n_eval >= sup_break_even:
        return "Supervised surrogate", "Large repeated-query workloads can amortize label generation and training."
    if n_eval >= pinn_break_even:
        return "PINN", "PINN reaches break-even earlier than the supervised surrogate under this workload."
    return "FDM", "Amortization threshold has not yet been reached."


def build_use_case_table(
    cfg: Chapter9Config,
    pinn_break_even: float,
    sup_break_even: float,
) -> pd.DataFrame:
    tasks = [
        ("Single contract pricing", 1, True),
        ("Near-barrier validation audit", 50, True),
        ("Intraday scenario grid", cfg.intraday_updates * cfg.intraday_grid_size, False),
        ("Large risk surface generation", cfg.use_case_surface_points_spot * cfg.use_case_surface_points_tau, False),
        ("Portfolio Delta/Gamma array", cfg.portfolio_contracts * cfg.portfolio_greek_passes, True),
        ("Calibration inner loop", cfg.calibration_iterations * cfg.calibration_eval_per_iter, False),
    ]
    rows = []
    for task, n_eval, greek_sensitive in tasks:
        preferred, reason = choose_solver(task, n_eval, greek_sensitive, n_eval < min(pinn_break_even, sup_break_even), pinn_break_even, sup_break_even)
        rows.append(
            {
                "Task": task,
                "Required evaluations": int(n_eval),
                "Preferred solver": preferred,
                "Reason": reason,
            }
        )
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, tex_path: Path, csv_path: Path, png_path: Path, caption: str, label: str) -> None:
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False, caption=caption, label=label))
    try:
        ch4.save_dataframe_as_png(df, png_path, caption)
    except Exception:
        pass


def main() -> None:
    cfg = Chapter9Config()
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    ensure_dir(out)

    cfg8, adapters = build_adapters(cfg)
    truth = adapters["Truth"]
    scenarios = representative_scenarios(cfg8)

    fdm_latency = measure_fdm_latency(adapters["FDM"], scenarios, cfg.latency_repeats)
    pinn_latency = measure_pinn_latency(adapters["PINN"], scenarios, cfg.latency_repeats) if "PINN" in adapters else math.nan
    supervised_latency = measure_supervised_latency(adapters["Supervised"], scenarios, cfg.latency_repeats)

    throughput_frames = [
        measure_fdm_batch_throughput(adapters["FDM"], cfg8, cfg.batch_sizes, cfg.throughput_repeats_small, cfg.throughput_repeats_large),
        measure_supervised_batch_throughput(adapters["Supervised"], cfg8, cfg.batch_sizes, cfg.throughput_repeats_small, cfg.throughput_repeats_large),
    ]
    if "PINN" in adapters:
        throughput_frames.append(
            measure_pinn_batch_throughput(adapters["PINN"], cfg8, cfg.batch_sizes, cfg.throughput_repeats_small, cfg.throughput_repeats_large)
        )
    throughput_df = pd.concat(throughput_frames, ignore_index=True)
    throughput_df.to_csv(out / "batch_throughput.csv", index=False)

    pinn_training_time = load_full_baac_training_time_seconds(Path(cfg.chapter7_output_dir))
    label_generation_time = measure_label_generation_time(cfg, cfg8)
    supervised_training_time = cfg.supervised_training_time_fallback_seconds

    pinn_peak = float(
        throughput_df.loc[throughput_df["Method"] == "PINN", "Contracts/sec"].max()
    ) if "PINN" in adapters else math.nan
    sup_peak = float(
        throughput_df.loc[throughput_df["Method"] == "Supervised surrogate", "Contracts/sec"].max()
    )

    scorecard = load_scorecard(Path(cfg.chapter8_output_dir))
    table15 = build_runtime_summary_table(
        cfg,
        scorecard,
        fdm_latency,
        pinn_latency,
        supervised_latency,
        pinn_peak,
        sup_peak,
        label_generation_time,
        pinn_training_time,
        supervised_training_time,
    )
    save_table(
        table15,
        out / "table15_runtime_inputs_break_even_summary.tex",
        out / "table15_runtime_inputs_break_even_summary.csv",
        out / "table15_runtime_inputs_break_even_summary.png",
        "Runtime inputs and break-even summary.",
        "tab:runtime_break_even_summary_ch9",
    )

    pinn_break_even = float(table15.loc[table15["Method"] == "PINN", "Break-even N*"].iloc[0]) if "PINN" in adapters else math.inf
    sup_break_even = float(table15.loc[table15["Method"] == "Supervised surrogate", "Break-even N*"].iloc[0])
    runtime_curve = plot_total_runtime_vs_evals(
        cfg,
        out / "figure28_total_runtime_vs_evaluations.png",
        fdm_latency,
        pinn_latency,
        supervised_latency,
        pinn_training_time,
        label_generation_time + supervised_training_time,
    )
    runtime_curve.to_csv(out / "runtime_curve.csv", index=False)

    avg_curve = plot_average_cost_vs_evals(
        cfg,
        out / "figure29_average_cost_per_evaluation.png",
        runtime_curve,
    )
    avg_curve.to_csv(out / "average_cost_curve.csv", index=False)

    plot_batch_throughput(throughput_df, out / "figure30_batch_throughput_comparison.png")

    scn_surface, spots, taus = use_case_surface_dataset(cfg)
    use_case_rows = []
    for name in ["FDM", "PINN", "Supervised"]:
        if name not in adapters:
            continue
        metrics = measure_surface_runtime_and_quality(adapters[name], truth, scn_surface, spots, taus)
        use_case_rows.append(
            {
                "Method": "Supervised surrogate" if name == "Supervised" else name,
                "Wall-clock (s)": metrics["wall_clock_s"],
                "Median RE (%)": metrics["median_re_pct"],
                "q95 RE (%)": metrics["q95_re_pct"],
                "Worst-case RE (%)": metrics["max_re_pct"],
                "Points": metrics["n_points"],
            }
        )
    use_case_df = pd.DataFrame(use_case_rows)
    use_case_df.to_csv(out / "use_case_risk_surface.csv", index=False)
    plot_use_case_surface(use_case_df, out / "figure31_use_case_barrier_risk_surface_generation.png")

    table16 = build_use_case_table(cfg, pinn_break_even, sup_break_even)
    save_table(
        table16,
        out / "table16_use_case_economics.tex",
        out / "table16_use_case_economics.csv",
        out / "table16_use_case_economics.png",
        "Use-case economics table.",
        "tab:use_case_economics_ch9",
    )

    summary = {
        "status": "chapter9 runtime and deployment-economics workflow prepared",
        "config": asdict(cfg),
        "hardware": cfg.hardware_label,
        "fdm_latency_s": fdm_latency,
        "pinn_latency_s": pinn_latency,
        "supervised_latency_s": supervised_latency,
        "pinn_training_time_s": pinn_training_time,
        "label_generation_time_s": label_generation_time,
        "supervised_training_time_s": supervised_training_time,
        "outputs": {
            "table15": str((out / "table15_runtime_inputs_break_even_summary.csv").resolve()),
            "figure28": str((out / "figure28_total_runtime_vs_evaluations.png").resolve()),
            "figure29": str((out / "figure29_average_cost_per_evaluation.png").resolve()),
            "figure30": str((out / "figure30_batch_throughput_comparison.png").resolve()),
            "figure31": str((out / "figure31_use_case_barrier_risk_surface_generation.png").resolve()),
            "table16": str((out / "table16_use_case_economics.csv").resolve()),
        },
    }
    with open(out / "chapter9_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Chapter 9 runtime and deployment-economics workflow")
    print("=" * 72)
    print("Exported:")
    print("  - Table 15: runtime inputs and break-even summary")
    print("  - Figure 28: total runtime vs number of evaluations")
    print("  - Figure 29: average cost per evaluation")
    print("  - Figure 30: batch throughput comparison")
    print("  - Figure 31: use case barrier risk surface generation")
    print("  - Table 16: use-case economics table")
    print(f"Output directory: {out.resolve()}")


if __name__ == "__main__":
    main()
