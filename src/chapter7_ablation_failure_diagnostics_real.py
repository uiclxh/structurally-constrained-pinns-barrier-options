import json
import math
import time
import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import chapter3_fdm_benchmark_only as ch3
import chapter4_barrier_surrogate_framework as ch4
import chapter7_ablation_failure_diagnostics_framework as demo


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "results_chapter7_real_formal"
    quick_mode: bool = False
    K: float = 100.0
    S0: float = 100.0
    T: float = 1.0
    r: float = 0.10
    q: float = 0.0
    S_max: float = 200.0
    hidden_layers: int = 4
    width: int = 128
    adam_epochs: int = 520
    adam_lr: float = 6e-4
    adam_gamma: float = 0.9985
    weight_decay: float = 1e-6
    grad_clip: float = 5.0
    eval_every: int = 26
    min_epochs_before_early_stop: int = 156
    early_stop_patience: int = 5
    refinement_warmup_epochs: int = 96
    train_panel_size: int = 20
    valid_panel_size: int = 8
    report_panel_size: int = 8
    lbfgs_rounds: int = 3
    lbfgs_chunk_steps: int = 8
    lbfgs_lr: float = 0.35
    lbfgs_history_size: int = 80
    w_pde: float = 1.0
    w_terminal: float = 35.0
    w_farfield: float = 8.0
    w_boundary: float = 12.0
    w_anchor: float = 18.0
    w_positivity: float = 0.08
    w_monotonicity: float = 0.03
    w_gamma_smooth: float = 0.003
    adaptive_loss_ema_beta: float = 0.92
    adaptive_weight_eps: float = 1e-8
    adaptive_weight_floor: float = 0.30
    adaptive_weight_cap: float = 3.50
    low_rho_threshold: float = 0.03
    low_rho_sampling_boost: float = 2.5
    short_T_threshold: float = 0.65
    short_T_sampling_boost: float = 1.6
    high_sigma_threshold: float = 0.33
    high_sigma_sampling_boost: float = 1.4
    train_low_rho_fraction: float = 0.55
    valid_low_rho_fraction: float = 0.50
    shape_warmup_frac: float = 0.45
    positivity_warmup_frac: float = 0.65
    constraint_ramp_frac: float = 0.20
    full_baac_uniform_share: float = 0.40
    full_baac_barrier_share: float = 0.25
    full_baac_strike_share: float = 0.20
    full_baac_refine_share: float = 0.15
    static_uniform_share: float = 0.55
    static_barrier_share: float = 0.25
    static_strike_share: float = 0.20
    residual_uniform_share: float = 0.60
    residual_refine_share: float = 0.40
    n_anchor: int = 180
    anchor_cache_per_scenario: int = 320
    anchor_rel_floor: float = 0.02
    anchor_rel_weight: float = 0.0
    pde_tail_fraction: float = 0.20
    pde_tail_weight: float = 0.0
    residual_polish_enabled: bool = True
    residual_polish_epochs: int = 72
    residual_polish_lr: float = 1.5e-4
    residual_polish_resample_every: int = 8
    residual_polish_patience: int = 4
    residual_polish_pde_boost: float = 2.4
    residual_polish_anchor_boost: float = 1.1
    residual_polish_terminal_scale: float = 0.50
    residual_polish_farfield_scale: float = 0.65
    residual_polish_hotspot_weight: float = 1.0
    residual_polish_barrier_hotspot_weight: float = 0.35
    residual_polish_top_fraction: float = 0.25
    residual_polish_score_scale: float = 80.0
    n_interior: int = 1400
    n_terminal: int = 500
    n_farfield: int = 360
    n_boundary: int = 360
    n_barrier: int = 700
    n_strike: int = 420
    n_refine: int = 420
    candidate_refine_pool: int = 1800
    sigma_grid: Tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
    rho_grid: Tuple[float, ...] = (0.002, 0.010, 0.030, 0.060, 0.100, 0.150)

    def __post_init__(self):
        if self.quick_mode:
            self.output_dir = "results_chapter7_real"
            self.adam_epochs = 360
            self.adam_lr = 8e-4
            self.adam_gamma = 0.997
            self.eval_every = 20
            self.min_epochs_before_early_stop = 180
            self.early_stop_patience = 5
            self.refinement_warmup_epochs = 80
            self.train_panel_size = 12
            self.valid_panel_size = 6
            self.report_panel_size = 8
            self.lbfgs_rounds = 3
            self.lbfgs_chunk_steps = 6
            self.lbfgs_lr = 0.30
            self.lbfgs_history_size = 40
            self.w_terminal = 25.0
            self.w_farfield = 5.0
            self.w_boundary = 25.0
            self.w_anchor = 12.0
            self.w_positivity = 0.05
            self.w_monotonicity = 0.10
            self.w_gamma_smooth = 0.02
            self.n_interior = 1200
            self.n_terminal = 420
            self.n_farfield = 300
            self.n_boundary = 300
            self.n_barrier = 600
            self.n_strike = 300
            self.n_refine = 400
            self.candidate_refine_pool = 1600
            self.n_anchor = 120
            self.anchor_cache_per_scenario = 220
            self.anchor_rel_floor = 0.02
            self.anchor_rel_weight = 0.0
            self.pde_tail_fraction = 0.20
            self.pde_tail_weight = 0.0
            self.residual_polish_epochs = 36
            self.residual_polish_lr = 2.0e-4
            self.residual_polish_resample_every = 6
            self.residual_polish_patience = 3


@dataclass
class VariantSpec:
    name: str
    group: str
    coordinate_mode: str
    ansatz_mode: str
    sampling_mode: str
    optimizer_mode: str
    use_monotonicity: bool = False
    use_gamma_smooth: bool = False
    barrier_kappa: float = 14.0


def feature_dim_for_mode(mode: str) -> int:
    return 10 if mode == "xy" else 9


def is_residual_polish_target(spec: VariantSpec) -> bool:
    return spec.name == "Full BAAC"


def default_variants() -> List[VariantSpec]:
    return [
        VariantSpec("Naive PINN", "Failure baseline", "raw_s", "soft_bc", "uniform", "adam_only"),
        VariantSpec("Raw S-space", "Coordinate choice", "raw_s", "hard_barrier_positivity", "full_baac", "hybrid", True, True),
        VariantSpec("x = S/K", "Coordinate choice", "x", "hard_barrier_positivity", "full_baac", "hybrid", True, True),
        VariantSpec("y = ln(S/K)", "Coordinate choice", "y", "hard_barrier_positivity", "full_baac", "hybrid", True, True),
        VariantSpec("Soft BC", "Ansatz", "xy", "soft_bc", "full_baac", "hybrid"),
        VariantSpec("Hard barrier only", "Ansatz", "xy", "hard_barrier", "full_baac", "hybrid"),
        VariantSpec("Hard barrier + positivity", "Ansatz", "xy", "hard_barrier_positivity", "full_baac", "hybrid", True, True),
        VariantSpec("No refinement", "BAAC", "xy", "hard_barrier_positivity", "uniform", "hybrid", True, True),
        VariantSpec("Static oversampling", "BAAC", "xy", "hard_barrier_positivity", "static_oversampling", "hybrid", True, True),
        VariantSpec("Residual refinement", "BAAC", "xy", "hard_barrier_positivity", "residual_refinement", "hybrid", True, True),
        VariantSpec("Full BAAC", "BAAC", "xy", "hard_barrier_positivity", "full_baac", "hybrid", True, True),
    ]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_text_seed(text: str, modulo: int = 10000) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def scenario(cfg: ExperimentConfig, sigma: float, rho_d: float) -> ch3.BarrierScenario:
    return ch3.BarrierScenario(sigma=sigma, rho_d=rho_d, S0=cfg.S0, K=cfg.K, T=cfg.T, r=cfg.r, delta=cfg.q, S_max=cfg.S_max)


def core_scenarios(cfg: ExperimentConfig) -> List[ch3.BarrierScenario]:
    return [scenario(cfg, sig, rho) for sig in (0.15, 0.25, 0.40) for rho in (0.002, 0.150)]


def scenario_panel(
    cfg: ExperimentConfig,
    n: int,
    sigma_range: Tuple[float, float],
    rho_range: Tuple[float, float],
    T_range: Tuple[float, float],
    seed_offset: int,
) -> List[ch3.BarrierScenario]:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    scenarios: List[ch3.BarrierScenario] = []
    for _ in range(n):
        sigma = float(rng.uniform(*sigma_range))
        rho_d = float(rng.uniform(*rho_range))
        T = float(rng.uniform(*T_range))
        scenarios.append(
            ch3.BarrierScenario(
                sigma=sigma,
                rho_d=rho_d,
                S0=cfg.S0,
                K=cfg.K,
                T=T,
                r=cfg.r,
                delta=cfg.q,
                S_max=cfg.S_max,
            )
        )
    return scenarios


def _stratified_samples(low: float, high: float, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    edges = np.linspace(low, high, n + 1)
    vals = [float(rng.uniform(left, right)) for left, right in zip(edges[:-1], edges[1:])]
    vals = np.array(vals, dtype=float)
    rng.shuffle(vals)
    return vals


def focused_panel(
    cfg: ExperimentConfig,
    n: int,
    sigma_range: Tuple[float, float],
    rho_range: Tuple[float, float],
    T_range: Tuple[float, float],
    seed_offset: int,
    low_rho_fraction: float,
) -> List[ch3.BarrierScenario]:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    sigma_mid = 0.5 * (sigma_range[0] + sigma_range[1])
    rho_mid = 0.5 * (rho_range[0] + rho_range[1])
    T_mid = 0.5 * (T_range[0] + T_range[1])
    low_rho_cap = min(cfg.low_rho_threshold, rho_range[1])
    anchors = [
        (sigma_range[0], rho_range[0], T_range[0]),
        (sigma_range[0], rho_range[0], T_range[1]),
        (sigma_range[1], rho_range[0], T_range[0]),
        (sigma_range[1], rho_range[0], T_range[1]),
        (sigma_mid, rho_range[0], T_mid),
        (sigma_mid, rho_mid, T_mid),
    ]
    scenarios = [
        ch3.BarrierScenario(sigma=sig, rho_d=rho, S0=cfg.S0, K=cfg.K, T=T, r=cfg.r, delta=cfg.q, S_max=cfg.S_max)
        for sig, rho, T in anchors[: min(len(anchors), n)]
    ]
    remaining = max(0, n - len(scenarios))
    if remaining == 0:
        return scenarios
    low_count = int(math.ceil(remaining * low_rho_fraction)) if low_rho_cap > rho_range[0] else 0
    sigma_vals = _stratified_samples(sigma_range[0], sigma_range[1], remaining, rng)
    T_vals = _stratified_samples(T_range[0], T_range[1], remaining, rng)
    rho_low = _stratified_samples(rho_range[0], low_rho_cap, low_count, rng)
    rho_hi = _stratified_samples(max(low_rho_cap, rho_range[0]), rho_range[1], remaining - low_count, rng)
    rho_vals = np.concatenate([rho_low, rho_hi])
    if rho_vals.size == 0:
        rho_vals = _stratified_samples(rho_range[0], rho_range[1], remaining, rng)
    rng.shuffle(rho_vals)
    for sig, rho, T in zip(sigma_vals, rho_vals, T_vals):
        scenarios.append(
            ch3.BarrierScenario(
                sigma=float(sig),
                rho_d=float(rho),
                S0=cfg.S0,
                K=cfg.K,
                T=float(T),
                r=cfg.r,
                delta=cfg.q,
                S_max=cfg.S_max,
            )
        )
    return scenarios


def build_panels(cfg: ExperimentConfig) -> Tuple[List[ch3.BarrierScenario], List[ch3.BarrierScenario], List[ch3.BarrierScenario]]:
    report_sigma = (0.15, 0.40)
    report_rho = (0.002, 0.15)
    report_T = (0.35, 1.75)
    train = focused_panel(cfg, cfg.train_panel_size, report_sigma, report_rho, report_T, seed_offset=101, low_rho_fraction=cfg.train_low_rho_fraction)
    valid = focused_panel(cfg, cfg.valid_panel_size, report_sigma, report_rho, report_T, seed_offset=202, low_rho_fraction=cfg.valid_low_rho_fraction)
    report = scenario_panel(cfg, cfg.report_panel_size, report_sigma, report_rho, report_T, seed_offset=303)
    report.extend(core_scenarios(cfg))
    return train, valid, report


def save_panel_summary(panels: Dict[str, List[ch3.BarrierScenario]], output_path: Path) -> None:
    rows = []
    for name, scenarios in panels.items():
        sigmas = [scn.sigma for scn in scenarios]
        rhos = [scn.rho_d for scn in scenarios]
        maturities = [scn.T for scn in scenarios]
        rows.append(
            {
                "panel": name,
                "n": len(scenarios),
                "sigma_min": min(sigmas),
                "sigma_max": max(sigmas),
                "rho_min": min(rhos),
                "rho_max": max(rhos),
                "T_min": min(maturities),
                "T_max": max(maturities),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def coord_from_S(S: torch.Tensor, mode: str, K: float) -> torch.Tensor:
    if mode == "raw_s":
        return S
    if mode == "x":
        return S / K
    if mode in {"y", "xy"}:
        return torch.log(S / K)
    raise ValueError(mode)


def barrier_coord(beta: torch.Tensor, mode: str, K: float) -> torch.Tensor:
    if mode == "raw_s":
        return beta * K
    if mode == "x":
        return beta
    if mode in {"y", "xy"}:
        return torch.log(beta)
    raise ValueError(mode)


class SmoothMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: int, width: int):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RealBarrierPINN(nn.Module):
    def __init__(self, cfg: ExperimentConfig, spec: VariantSpec):
        super().__init__()
        self.cfg = cfg
        self.spec = spec
        self.core = SmoothMLP(feature_dim_for_mode(spec.coordinate_mode), cfg.hidden_layers, cfg.width)

    def features(self, S, tau, sigma, beta, r, q):
        x = S / self.cfg.K
        y = torch.log(x)
        z = coord_from_S(S, self.spec.coordinate_mode, self.cfg.K)
        b = barrier_coord(beta, self.spec.coordinate_mode, self.cfg.K)
        d = torch.clamp(z - b, min=0.0)
        d_fast = 1.0 - torch.exp(-12.0 * d)
        d_slow = 1.0 - torch.exp(-1.2 * d)
        p = torch.cat([
            (sigma - 0.25) / 0.15,
            (beta - 0.90) / 0.10,
            (r - 0.05) / 0.10,
            (q - 0.00) / 0.05,
        ], dim=1)
        if self.spec.coordinate_mode == "xy":
            return torch.cat([y, x, tau, d, d_fast, d_slow, p], dim=1), z, b
        return torch.cat([z, tau, d, d_fast, d_slow, p], dim=1), z, b

    def forward(self, S, tau, sigma, beta, r, q):
        x, z, b = self.features(S, tau, sigma, beta, r, q)
        raw = self.core(x)
        if self.spec.ansatz_mode == "hard_barrier_positivity":
            raw = F.softplus(raw)
        if self.spec.ansatz_mode == "soft_bc":
            return raw
        psi = 1.0 - torch.exp(-self.spec.barrier_kappa * torch.clamp(z - b, min=0.0))
        return psi * raw


def payoff_u(S: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    return torch.clamp(S / cfg.K - 1.0, min=0.0)


def farfield_u(S: torch.Tensor, tau: torch.Tensor, r: torch.Tensor, q: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    x = S / cfg.K
    return x * torch.exp(-q * tau) - torch.exp(-r * tau)


def pde_residual(model: RealBarrierPINN, spec: VariantSpec, cfg: ExperimentConfig, S, tau, sigma, beta, r, q):
    S.requires_grad_(True)
    tau.requires_grad_(True)
    u = model(S, tau, sigma, beta, r, q)
    ones = torch.ones_like(u)
    u_tau = torch.autograd.grad(u, tau, grad_outputs=ones, create_graph=True)[0]

    z = coord_from_S(S, spec.coordinate_mode, cfg.K)
    z.requires_grad_(True)
    if spec.coordinate_mode == "raw_s":
        uz = torch.autograd.grad(u, S, grad_outputs=ones, create_graph=True)[0]
        uzz = torch.autograd.grad(uz, S, grad_outputs=torch.ones_like(uz), create_graph=True)[0]
        rhs = 0.5 * sigma**2 * S**2 * uzz + (r - q) * S * uz - r * u
    elif spec.coordinate_mode == "x":
        x = S / cfg.K
        ux = torch.autograd.grad(u, S, grad_outputs=ones, create_graph=True)[0] * cfg.K
        uxx = torch.autograd.grad(ux, S, grad_outputs=torch.ones_like(ux), create_graph=True)[0] * cfg.K
        rhs = 0.5 * sigma**2 * x**2 * uxx + (r - q) * x * ux - r * u
    else:
        uy = torch.autograd.grad(u, S, grad_outputs=ones, create_graph=True)[0] * S
        uyy = torch.autograd.grad(uy, S, grad_outputs=torch.ones_like(uy), create_graph=True)[0] * S
        rhs = 0.5 * sigma**2 * uyy + (r - q - 0.5 * sigma**2) * uy - r * u
    return u_tau - rhs


def _scenario_probabilities(cfg: ExperimentConfig, scenarios: List[ch3.BarrierScenario]) -> np.ndarray:
    if not scenarios:
        raise ValueError("scenarios must be non-empty")
    rhos = np.array([scn.rho_d for scn in scenarios], dtype=float)
    sigmas = np.array([scn.sigma for scn in scenarios], dtype=float)
    maturities = np.array([scn.T for scn in scenarios], dtype=float)
    low_ratio = np.clip((cfg.low_rho_threshold - rhos) / max(cfg.low_rho_threshold, 1e-8), 0.0, 1.0)
    short_ratio = np.clip((cfg.short_T_threshold - maturities) / max(cfg.short_T_threshold, 1e-8), 0.0, 1.0)
    high_sigma_ratio = np.clip((sigmas - cfg.high_sigma_threshold) / max(1.0 - cfg.high_sigma_threshold, 1e-8), 0.0, 1.0)
    weights = (
        1.0
        + cfg.low_rho_sampling_boost * low_ratio
        + cfg.short_T_sampling_boost * short_ratio
        + cfg.high_sigma_sampling_boost * high_sigma_ratio
    )
    return weights / np.sum(weights)


def _sample_points(cfg: ExperimentConfig, spec: VariantSpec, scenarios: List[ch3.BarrierScenario], n: int, mode: str, device: torch.device, rng: np.random.Generator):
    S_list, tau_list, sigma_list, beta_list, r_list, q_list = [], [], [], [], [], []
    strike = cfg.K
    scenario_probs = _scenario_probabilities(cfg, scenarios)
    for idx in rng.choice(len(scenarios), size=n, replace=True, p=scenario_probs):
        scn = scenarios[int(idx)]
        if mode == "interior":
            S = rng.uniform(scn.B_d + 1e-4, scn.S_max)
        elif mode == "barrier":
            S = rng.uniform(scn.B_d + 1e-4, min(scn.S_max, scn.B_d + 0.06 * cfg.K))
        elif mode == "strike":
            S = rng.uniform(strike - 0.08 * cfg.K, strike + 0.08 * cfg.K)
            S = min(max(S, scn.B_d + 1e-4), scn.S_max)
        elif mode == "terminal":
            S = rng.uniform(scn.B_d + 1e-4, scn.S_max)
        elif mode == "boundary":
            S = scn.B_d
        elif mode == "farfield":
            S = scn.S_max
        else:
            raise ValueError(mode)
        tau = 0.0 if mode == "terminal" else rng.uniform(0.0, scn.T)
        S_list.append(S)
        tau_list.append(tau)
        sigma_list.append(scn.sigma)
        beta_list.append(scn.B_d / scn.K)
        r_list.append(scn.r)
        q_list.append(scn.delta)
    tensors = [
        torch.tensor(arr, dtype=torch.float32, device=device).reshape(-1, 1)
        for arr in [S_list, tau_list, sigma_list, beta_list, r_list, q_list]
    ]
    return tuple(tensors)


def _sample_residual_refinement(cfg: ExperimentConfig, spec: VariantSpec, model: RealBarrierPINN, scenarios: List[ch3.BarrierScenario], device: torch.device, rng: np.random.Generator):
    per_scenario_cand = max(120, cfg.candidate_refine_pool // max(1, len(scenarios)))
    per_scenario_keep = max(1, cfg.n_refine // max(1, len(scenarios)))
    chunks: List[Tuple[torch.Tensor, ...]] = []
    for scn in scenarios:
        cand = _sample_points(cfg, spec, [scn], per_scenario_cand, "interior", device, rng)
        S, tau, sigma, beta, r, q = [t.clone().detach().requires_grad_(True) for t in cand]
        res = pde_residual(model, spec, cfg, S, tau, sigma, beta, r, q).detach().abs().flatten()
        scale = torch.quantile(res, 0.75).clamp_min(1e-8)
        score = res / scale
        topk = torch.topk(score, k=min(per_scenario_keep, score.numel()), largest=True).indices
        chunks.append(tuple(t.detach()[topk] for t in cand))

    merged = [torch.cat([chunk[i] for chunk in chunks], dim=0) for i in range(6)]
    if merged[0].shape[0] < cfg.n_refine:
        filler = _sample_points(cfg, spec, scenarios, cfg.n_refine - merged[0].shape[0], "interior", device, rng)
        merged = [torch.cat([merged[i], filler[i]], dim=0) for i in range(6)]
    elif merged[0].shape[0] > cfg.n_refine:
        keep = torch.randperm(merged[0].shape[0], device=device)[:cfg.n_refine]
        merged = [tensor[keep] for tensor in merged]
    return tuple(merged)


def build_anchor_cache(cfg: ExperimentConfig, scenarios: List[ch3.BarrierScenario], seed_offset: int = 404) -> List[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    cache: List[Dict[str, np.ndarray]] = []
    for scn in scenarios:
        S_vals = []
        tau_vals = []
        target_vals = []
        for _ in range(cfg.anchor_cache_per_scenario):
            u = rng.random()
            if u < 0.40:
                S = rng.uniform(max(scn.B_d + 1e-4, cfg.K - 0.08 * cfg.K), min(scn.S_max, cfg.K + 0.08 * cfg.K))
            elif u < 0.75:
                S = rng.uniform(scn.B_d + 1e-4, min(scn.S_max, scn.B_d + 0.08 * cfg.K))
            else:
                S = rng.uniform(scn.B_d + 1e-4, scn.S_max)
            if rng.random() < 0.50:
                tau = scn.T
            else:
                tau = scn.T * float(rng.beta(0.85, 1.15))
            target = ch3.rr_price_scalar(scn, S=float(S), tau=float(tau)) / cfg.K
            S_vals.append(float(S))
            tau_vals.append(float(tau))
            target_vals.append(float(target))
        cache.append(
            {
                "S": np.array(S_vals, dtype=np.float32),
                "tau": np.array(tau_vals, dtype=np.float32),
                "target": np.array(target_vals, dtype=np.float32),
                "sigma": np.full(cfg.anchor_cache_per_scenario, scn.sigma, dtype=np.float32),
                "beta": np.full(cfg.anchor_cache_per_scenario, scn.B_d / scn.K, dtype=np.float32),
                "r": np.full(cfg.anchor_cache_per_scenario, scn.r, dtype=np.float32),
                "q": np.full(cfg.anchor_cache_per_scenario, scn.delta, dtype=np.float32),
            }
        )
    return cache


def sample_anchor_batch(
    cfg: ExperimentConfig,
    scenarios: List[ch3.BarrierScenario],
    anchor_cache: List[Dict[str, np.ndarray]],
    device: torch.device,
    rng: np.random.Generator,
):
    scenario_probs = _scenario_probabilities(cfg, scenarios)
    S_list, tau_list, sigma_list, beta_list, r_list, q_list, target_list = [], [], [], [], [], [], []
    for idx in rng.choice(len(scenarios), size=cfg.n_anchor, replace=True, p=scenario_probs):
        idx = int(idx)
        cached = anchor_cache[idx]
        j = int(rng.integers(0, cached["S"].shape[0]))
        S_list.append(cached["S"][j])
        tau_list.append(cached["tau"][j])
        sigma_list.append(cached["sigma"][j])
        beta_list.append(cached["beta"][j])
        r_list.append(cached["r"][j])
        q_list.append(cached["q"][j])
        target_list.append(cached["target"][j])
    tensors = [
        torch.tensor(arr, dtype=torch.float32, device=device).reshape(-1, 1)
        for arr in [S_list, tau_list, sigma_list, beta_list, r_list, q_list, target_list]
    ]
    return tuple(tensors)


def active_sampling_mode(cfg: ExperimentConfig, spec: VariantSpec, epoch: int) -> str:
    if spec.sampling_mode == "full_baac" and epoch < cfg.refinement_warmup_epochs:
        return "static_oversampling"
    if spec.sampling_mode == "residual_refinement" and epoch < cfg.refinement_warmup_epochs:
        return "uniform"
    return spec.sampling_mode


def sample_batch(
    cfg: ExperimentConfig,
    spec: VariantSpec,
    model: Optional[RealBarrierPINN],
    scenarios: List[ch3.BarrierScenario],
    anchor_cache: Optional[List[Dict[str, np.ndarray]]],
    device: torch.device,
    rng: np.random.Generator,
    epoch: int,
):
    sampling_mode = active_sampling_mode(cfg, spec, epoch)
    batch = {
        "interior": _sample_points(cfg, spec, scenarios, cfg.n_interior, "interior", device, rng),
        "terminal": _sample_points(cfg, spec, scenarios, cfg.n_terminal, "terminal", device, rng),
        "farfield": _sample_points(cfg, spec, scenarios, cfg.n_farfield, "farfield", device, rng),
        "boundary": _sample_points(cfg, spec, scenarios, cfg.n_boundary, "boundary", device, rng),
        "barrier_band": _sample_points(cfg, spec, scenarios, cfg.n_barrier, "barrier", device, rng),
        "strike_band": _sample_points(cfg, spec, scenarios, cfg.n_strike, "strike", device, rng),
        "interior_aux": _sample_points(cfg, spec, scenarios, max(400, cfg.n_interior // 2), "interior", device, rng),
    }
    if sampling_mode in {"residual_refinement", "full_baac"} and model is not None:
        batch["refine"] = _sample_residual_refinement(cfg, spec, model, scenarios, device, rng)
    else:
        batch["refine"] = _sample_points(cfg, spec, scenarios, cfg.n_refine, "interior", device, rng)
    if anchor_cache is not None:
        batch["anchor"] = sample_anchor_batch(cfg, scenarios, anchor_cache, device, rng)
    return batch, sampling_mode


def stage_progress(cfg: ExperimentConfig, epoch: int) -> float:
    return float(min(max(epoch / max(cfg.adam_epochs, 1), 0.0), 1.0))


def ramp_factor(progress: float, start: float, width: float) -> float:
    if progress <= start:
        return 0.0
    if progress >= start + width:
        return 1.0
    return float((progress - start) / max(width, 1e-8))


def stage_multipliers(cfg: ExperimentConfig, spec: VariantSpec, epoch: int) -> Dict[str, float]:
    progress = stage_progress(cfg, epoch)
    shape_ramp = ramp_factor(progress, cfg.shape_warmup_frac, cfg.constraint_ramp_frac)
    positivity_ramp = ramp_factor(progress, cfg.positivity_warmup_frac, cfg.constraint_ramp_frac)
    price_focus = max(0.35, 1.20 - 0.85 * progress)
    anchor_focus = max(0.80, 1.45 - 0.55 * progress)
    pde_focus = 0.80 + 2.40 * (progress ** 1.20)
    return {
        "pde": pde_focus,
        "terminal": price_focus,
        "anchor": anchor_focus,
        "farfield": max(0.70, 1.20 - 0.40 * progress),
        "boundary": 1.0 if spec.ansatz_mode == "soft_bc" else 0.0,
        "positivity": positivity_ramp if spec.ansatz_mode == "hard_barrier_positivity" else 0.0,
        "monotonicity": shape_ramp if spec.use_monotonicity else 0.0,
        "gamma_smooth": shape_ramp if spec.use_gamma_smooth else 0.0,
    }


def base_term_weights(cfg: ExperimentConfig) -> Dict[str, float]:
    return {
        "pde": cfg.w_pde,
        "terminal": cfg.w_terminal,
        "anchor": cfg.w_anchor,
        "farfield": cfg.w_farfield,
        "boundary": cfg.w_boundary,
        "positivity": cfg.w_positivity,
        "monotonicity": cfg.w_monotonicity,
        "gamma_smooth": cfg.w_gamma_smooth,
    }


def adaptive_term_weights(cfg: ExperimentConfig, spec: VariantSpec, ema_state: Dict[str, float], epoch: int) -> Dict[str, float]:
    multipliers = stage_multipliers(cfg, spec, epoch)
    base = base_term_weights(cfg)
    active = {k: base[k] * multipliers.get(k, 0.0) for k in base if base[k] > 0.0 and multipliers.get(k, 0.0) > 0.0}
    if not active:
        return {k: 0.0 for k in base}
    raw = {}
    for term, target in active.items():
        denom = math.sqrt(max(ema_state.get(term, 1.0), cfg.adaptive_weight_eps))
        raw_val = target / denom
        raw[term] = min(max(raw_val, target * cfg.adaptive_weight_floor), target * cfg.adaptive_weight_cap)
    total_target = sum(active.values())
    total_raw = max(sum(raw.values()), cfg.adaptive_weight_eps)
    scale = total_target / total_raw
    weights = {k: 0.0 for k in base}
    for term, val in raw.items():
        weights[term] = float(val * scale)
    return weights


def update_loss_ema(cfg: ExperimentConfig, ema_state: Dict[str, float], losses: Dict[str, torch.Tensor]) -> None:
    beta = cfg.adaptive_loss_ema_beta
    for term, loss in losses.items():
        value = float(loss.detach().item())
        if term not in ema_state:
            ema_state[term] = value
        else:
            ema_state[term] = beta * ema_state[term] + (1.0 - beta) * value


def pde_mix(cfg: ExperimentConfig, sampling_mode: str, epoch: int) -> Dict[str, float]:
    progress = stage_progress(cfg, epoch)
    if sampling_mode == "uniform":
        return {"interior": 1.0}
    if sampling_mode == "static_oversampling":
        late = ramp_factor(progress, 0.45, 0.30)
        interior = cfg.static_uniform_share - 0.15 * late
        barrier = cfg.static_barrier_share + 0.08 * late
        strike = 1.0 - interior - barrier
        return {
            "interior": interior,
            "barrier_band": barrier,
            "strike_band": strike,
        }
    if sampling_mode == "residual_refinement":
        late = ramp_factor(progress, 0.35, 0.35)
        refine = min(0.70, cfg.residual_refine_share + 0.20 * late)
        return {
            "interior": 1.0 - refine,
            "refine": refine,
        }
    late = ramp_factor(progress, 0.35, 0.35)
    refine = min(0.40, cfg.full_baac_refine_share + 0.20 * late)
    barrier = cfg.full_baac_barrier_share + 0.08 * late
    strike = max(0.10, cfg.full_baac_strike_share - 0.06 * late)
    interior = 1.0 - refine - barrier - strike
    return {
        "interior": interior,
        "barrier_band": barrier,
        "strike_band": strike,
        "refine": refine,
    }


def compute_loss_terms(
    model: RealBarrierPINN,
    spec: VariantSpec,
    cfg: ExperimentConfig,
    batch: Dict[str, Tuple[torch.Tensor, ...]],
    sampling_mode: str,
    epoch: int,
):
    losses: Dict[str, torch.Tensor] = {}
    device = batch["interior"][0].device
    losses["pde"] = torch.zeros(1, device=device).squeeze()
    for name, weight in pde_mix(cfg, sampling_mode, epoch).items():
        S, tau, sigma, beta, r, q = batch[name]
        res_sq = pde_residual(model, spec, cfg, S, tau, sigma, beta, r, q).pow(2).flatten()
        k = max(1, int(math.ceil(res_sq.numel() * cfg.pde_tail_fraction)))
        tail = torch.topk(res_sq, k=k, largest=True).values.mean()
        losses["pde"] = losses["pde"] + weight * (res_sq.mean() + cfg.pde_tail_weight * tail)

    S, tau, sigma, beta, r, q = batch["terminal"]
    losses["terminal"] = (model(S, tau, sigma, beta, r, q) - payoff_u(S, cfg)).pow(2).mean()

    if "anchor" in batch:
        S, tau, sigma, beta, r, q, target_u = batch["anchor"]
        pred_u = model(S, tau, sigma, beta, r, q)
        abs_term = (pred_u - target_u).pow(2).mean()
        rel_scale = target_u.abs() + cfg.anchor_rel_floor
        rel_term = ((pred_u - target_u) / rel_scale).pow(2).mean()
        losses["anchor"] = abs_term + cfg.anchor_rel_weight * rel_term
    else:
        losses["anchor"] = torch.zeros(1, device=device).squeeze()

    S, tau, sigma, beta, r, q = batch["farfield"]
    losses["farfield"] = (model(S, tau, sigma, beta, r, q) - farfield_u(S, tau, r, q, cfg)).pow(2).mean()

    if spec.ansatz_mode == "soft_bc":
        S, tau, sigma, beta, r, q = batch["boundary"]
        losses["boundary"] = model(S, tau, sigma, beta, r, q).pow(2).mean()
    else:
        losses["boundary"] = torch.zeros(1, device=device).squeeze()

    if spec.ansatz_mode == "hard_barrier_positivity":
        S, tau, sigma, beta, r, q = batch["interior_aux"]
        losses["positivity"] = F.relu(-model(S, tau, sigma, beta, r, q)).pow(2).mean()
    else:
        losses["positivity"] = torch.zeros(1, device=device).squeeze()

    if spec.use_monotonicity:
        S, tau, sigma, beta, r, q = batch["interior_aux"]
        S.requires_grad_(True)
        u = model(S, tau, sigma, beta, r, q)
        uS = torch.autograd.grad(u, S, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        losses["monotonicity"] = F.relu(-uS).pow(2).mean()
    else:
        losses["monotonicity"] = torch.zeros(1, device=device).squeeze()

    if spec.use_gamma_smooth:
        S, tau, sigma, beta, r, q = batch["barrier_band"]
        S.requires_grad_(True)
        u = model(S, tau, sigma, beta, r, q)
        uS = torch.autograd.grad(u, S, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        uSS = torch.autograd.grad(uS, S, grad_outputs=torch.ones_like(uS), create_graph=True)[0]
        losses["gamma_smooth"] = uSS.pow(2).mean()
    else:
        losses["gamma_smooth"] = torch.zeros(1, device=device).squeeze()
    return losses


def combine_losses(losses: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    total = torch.zeros_like(next(iter(losses.values())))
    for term, loss in losses.items():
        total = total + weights.get(term, 0.0) * loss
    return total


def top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = values.flatten()
    k = max(1, int(math.ceil(flat.numel() * fraction)))
    return torch.topk(flat, k=k, largest=True).values.mean()


def residual_hotspot_penalty(
    model: RealBarrierPINN,
    spec: VariantSpec,
    cfg: ExperimentConfig,
    batch: Dict[str, Tuple[torch.Tensor, ...]],
) -> torch.Tensor:
    penalties = []
    for name, scale in (("refine", 1.0), ("barrier_band", cfg.residual_polish_barrier_hotspot_weight)):
        if name not in batch:
            continue
        S, tau, sigma, beta, r, q = batch[name]
        abs_res = pde_residual(model, spec, cfg, S, tau, sigma, beta, r, q).abs()
        penalties.append(scale * top_fraction_mean(abs_res, cfg.residual_polish_top_fraction))
    if penalties:
        return torch.stack(penalties).sum()
    return torch.zeros(1, device=batch["interior"][0].device).squeeze()


def residual_polish_weights(cfg: ExperimentConfig, weights: Dict[str, float]) -> Dict[str, float]:
    tuned = dict(weights)
    tuned["pde"] = tuned.get("pde", 0.0) * cfg.residual_polish_pde_boost
    tuned["anchor"] = tuned.get("anchor", 0.0) * cfg.residual_polish_anchor_boost
    tuned["terminal"] = tuned.get("terminal", 0.0) * cfg.residual_polish_terminal_scale
    tuned["farfield"] = tuned.get("farfield", 0.0) * cfg.residual_polish_farfield_scale
    return tuned


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(torch.sum(p.grad.detach() ** 2).item())
    return math.sqrt(max(total, 0.0))


def price_scalar(model: RealBarrierPINN, cfg: ExperimentConfig, scn: ch3.BarrierScenario, S: float, tau: float) -> float:
    device = next(model.parameters()).device
    beta = scn.B_d / scn.K
    with torch.no_grad():
        u = model(
            torch.tensor([[S]], dtype=torch.float32, device=device),
            torch.tensor([[tau]], dtype=torch.float32, device=device),
            torch.tensor([[scn.sigma]], dtype=torch.float32, device=device),
            torch.tensor([[beta]], dtype=torch.float32, device=device),
            torch.tensor([[scn.r]], dtype=torch.float32, device=device),
            torch.tensor([[scn.delta]], dtype=torch.float32, device=device),
        ).item()
    return float(cfg.K * u)


def delta_gamma_fd(model: RealBarrierPINN, cfg: ExperimentConfig, scn: ch3.BarrierScenario, S: float, tau: float, h: float = 1e-2):
    fp = price_scalar(model, cfg, scn, S + h, tau)
    fm = price_scalar(model, cfg, scn, S - h, tau)
    f0 = price_scalar(model, cfg, scn, S, tau)
    return float((fp - fm) / (2 * h)), float((fp - 2 * f0 + fm) / (h ** 2))


def scenario_metrics(model: RealBarrierPINN, spec: VariantSpec, cfg: ExperimentConfig, scn: ch3.BarrierScenario):
    S_eval = ch3.chapter3_eval_spots(scn)
    V_true = ch3.down_and_out_call_rr(S_eval, scn.K, scn.B_d, scn.T, scn.r, scn.sigma, scn.delta)
    V_pred = np.array([price_scalar(model, cfg, scn, float(S), scn.T) for S in S_eval], dtype=float)
    abs_err = np.abs(V_pred - V_true)
    rel_err = 100.0 * abs_err / (np.abs(V_true) + 1e-12)
    delta_true, gamma_true = ch3.numerical_delta_gamma_rr(scn, scn.S0, scn.T, h=1e-3)
    delta_pred, gamma_pred = delta_gamma_fd(model, cfg, scn, scn.S0, scn.T)
    barrier_vals = np.array([abs(price_scalar(model, cfg, scn, scn.B_d, float(tau))) for tau in np.linspace(0.0, scn.T, 17)])
    grid_vals = []
    for S in np.linspace(scn.B_d + 1e-3, scn.S_max, 40):
        for tau in np.linspace(0.0, scn.T, 12):
            grid_vals.append(price_scalar(model, cfg, scn, float(S), float(tau)))
    grid_vals = np.array(grid_vals)
    sampler_rng = np.random.default_rng(cfg.seed + 7)
    batch, sampling_mode = sample_batch(
        cfg,
        spec,
        model if active_sampling_mode(cfg, spec, cfg.adam_epochs) in {"residual_refinement", "full_baac"} else None,
        [scn],
        None,
        next(model.parameters()).device,
        sampler_rng,
        cfg.adam_epochs,
    )
    S, tau, sigma, beta, r, q = batch["interior"]
    residual_q95 = float(np.quantile(np.abs(pde_residual(model, spec, cfg, S, tau, sigma, beta, r, q).detach().cpu().numpy().flatten()), 0.95))
    accept = ch4.acceptance_check({
        "price_q95_pct": float(np.quantile(rel_err, 0.95)),
        "gamma_q95_abs": abs(gamma_pred - gamma_true),
        "barrier_abs_max": float(np.max(barrier_vals)),
        "residual_q95": residual_q95,
        "positivity_violation_rate": float(np.mean(grid_vals < -1e-8)),
    }, ch4.FullConfig())
    return {
        "median_re": float(np.median(rel_err)),
        "q95_re": float(np.quantile(rel_err, 0.95)),
        "worst_re": float(np.max(rel_err)),
        "delta_err": abs(delta_pred - delta_true),
        "gamma_err": abs(gamma_pred - gamma_true),
        "barrier_residual": float(np.max(barrier_vals)),
        "positivity_violation": float(np.mean(grid_vals < -1e-8)),
        "success": float(accept),
    }


def validate_price_q95(model: RealBarrierPINN, cfg: ExperimentConfig, valid_scenarios: List[ch3.BarrierScenario]) -> float:
    re_vals = []
    for scn in valid_scenarios:
        for S in ch3.chapter3_eval_spots(scn):
            true = ch3.rr_price_scalar(scn, S=float(S), tau=scn.T)
            pred = price_scalar(model, cfg, scn, float(S), scn.T)
            re_vals.append(100.0 * abs(pred - true) / (abs(true) + 1e-12))
    return float(np.quantile(np.array(re_vals), 0.95))


def validate_metrics(model: RealBarrierPINN, cfg: ExperimentConfig, valid_scenarios: List[ch3.BarrierScenario]) -> Dict[str, float]:
    all_re = []
    low_rho_re = []
    low_rho_scenarios = [scn for scn in valid_scenarios if scn.rho_d <= cfg.low_rho_threshold]
    if not low_rho_scenarios:
        ranked = sorted(valid_scenarios, key=lambda scn: scn.rho_d)
        low_rho_scenarios = ranked[: max(1, len(ranked) // 3)]
    low_rho_ids = {id(scn) for scn in low_rho_scenarios}
    for scn in valid_scenarios:
        for S in ch3.chapter3_eval_spots(scn):
            true = ch3.rr_price_scalar(scn, S=float(S), tau=scn.T)
            pred = price_scalar(model, cfg, scn, float(S), scn.T)
            re = 100.0 * abs(pred - true) / (abs(true) + 1e-12)
            all_re.append(re)
            if id(scn) in low_rho_ids:
                low_rho_re.append(re)
    global_q95 = float(np.quantile(np.array(all_re), 0.95))
    low_rho_q95 = float(np.quantile(np.array(low_rho_re), 0.95))
    return {
        "global_q95": global_q95,
        "low_rho_q95": low_rho_q95,
        "composite_score": max(global_q95, low_rho_q95),
    }


def validate_residual_focus_q95(
    model: RealBarrierPINN,
    spec: VariantSpec,
    cfg: ExperimentConfig,
    valid_scenarios: List[ch3.BarrierScenario],
) -> float:
    device = next(model.parameters()).device
    per_scenario = []
    low_rho = []
    for idx, scn in enumerate(valid_scenarios):
        rng = np.random.default_rng(cfg.seed + 909 + idx)
        batch, _ = sample_batch(
            cfg,
            spec,
            model if active_sampling_mode(cfg, spec, cfg.adam_epochs) in {"residual_refinement", "full_baac"} else None,
            [scn],
            None,
            device,
            rng,
            cfg.adam_epochs,
        )
        S, tau, sigma, beta, r, q = batch["interior"]
        q95 = float(np.quantile(np.abs(pde_residual(model, spec, cfg, S, tau, sigma, beta, r, q).detach().cpu().numpy().flatten()), 0.95))
        per_scenario.append(q95)
        if scn.rho_d <= cfg.low_rho_threshold:
            low_rho.append(q95)
    overall = float(np.median(per_scenario))
    low_focus = float(max(low_rho)) if low_rho else overall
    return max(overall, low_focus)


def train_variant(cfg: ExperimentConfig, spec: VariantSpec, train_scenarios: List[ch3.BarrierScenario], valid_scenarios: List[ch3.BarrierScenario], output_dir: Path):
    device = torch.device(cfg.device)
    model = RealBarrierPINN(cfg, spec).to(device)
    rng = np.random.default_rng(cfg.seed + stable_text_seed(spec.name))
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.adam_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=cfg.adam_gamma)
    history = {
        "epoch": [],
        "stage": [],
        "sampling_mode": [],
        "total": [],
        "pde": [],
        "boundary": [],
        "positivity": [],
        "w_pde": [],
        "w_terminal": [],
        "w_anchor": [],
        "w_farfield": [],
        "w_boundary": [],
        "w_positivity": [],
        "grad_norm": [],
        "val_price_q95": [],
        "val_low_rho_q95": [],
        "val_residual_q95": [],
        "val_composite": [],
        "best_val_composite": [],
    }
    best_state = None
    best_val = math.inf
    patience = 0
    loss_ema: Dict[str, float] = {}
    anchor_cache = build_anchor_cache(cfg, train_scenarios)

    t0 = time.perf_counter()
    for epoch in range(1, cfg.adam_epochs + 1):
        sampling_mode = active_sampling_mode(cfg, spec, epoch)
        batch, sampling_mode = sample_batch(
            cfg,
            spec,
            model if sampling_mode in {"residual_refinement", "full_baac"} else None,
            train_scenarios,
            anchor_cache,
            device,
            rng,
            epoch,
        )
        optim.zero_grad()
        losses = compute_loss_terms(model, spec, cfg, batch, sampling_mode, epoch)
        if not loss_ema:
            update_loss_ema(cfg, loss_ema, losses)
        weights = adaptive_term_weights(cfg, spec, loss_ema, epoch)
        total_loss = combine_losses(losses, weights)
        total_loss.backward()
        gnorm = grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        scheduler.step()
        update_loss_ema(cfg, loss_ema, losses)
        if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.adam_epochs:
            val_metrics = validate_metrics(model, cfg, valid_scenarios)
            val_residual_q95 = validate_residual_focus_q95(model, spec, cfg, valid_scenarios)
            if val_metrics["composite_score"] < best_val:
                best_val = val_metrics["composite_score"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            history["epoch"].append(epoch)
            history["stage"].append("adam")
            history["sampling_mode"].append(sampling_mode)
            history["total"].append(float(total_loss.item()))
            history["pde"].append(float(losses["pde"].item()))
            history["boundary"].append(float(losses["boundary"].item()))
            history["positivity"].append(float(losses["positivity"].item()))
            history["w_pde"].append(weights.get("pde", 0.0))
            history["w_terminal"].append(weights.get("terminal", 0.0))
            history["w_anchor"].append(weights.get("anchor", 0.0))
            history["w_farfield"].append(weights.get("farfield", 0.0))
            history["w_boundary"].append(weights.get("boundary", 0.0))
            history["w_positivity"].append(weights.get("positivity", 0.0))
            history["grad_norm"].append(float(gnorm))
            history["val_price_q95"].append(val_metrics["global_q95"])
            history["val_low_rho_q95"].append(val_metrics["low_rho_q95"])
            history["val_residual_q95"].append(val_residual_q95)
            history["val_composite"].append(val_metrics["composite_score"])
            history["best_val_composite"].append(best_val)
            if epoch >= cfg.min_epochs_before_early_stop and patience >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if spec.optimizer_mode == "hybrid" and cfg.lbfgs_rounds > 0:
        lbfgs_patience = 0
        for lbfgs_round in range(1, cfg.lbfgs_rounds + 1):
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=cfg.lbfgs_lr,
                max_iter=cfg.lbfgs_chunk_steps,
                history_size=cfg.lbfgs_history_size,
                line_search_fn="strong_wolfe",
            )
            frozen_epoch = cfg.adam_epochs + lbfgs_round
            frozen_sampling_mode = active_sampling_mode(cfg, spec, frozen_epoch)
            frozen_batch, frozen_sampling_mode = sample_batch(
                cfg,
                spec,
                model if frozen_sampling_mode in {"residual_refinement", "full_baac"} else None,
                train_scenarios,
                anchor_cache,
                device,
                rng,
                frozen_epoch,
            )
            frozen_weights = adaptive_term_weights(cfg, spec, loss_ema, frozen_epoch)

            def closure():
                lbfgs.zero_grad()
                losses = compute_loss_terms(model, spec, cfg, frozen_batch, frozen_sampling_mode, frozen_epoch)
                total = combine_losses(losses, frozen_weights)
                total.backward()
                return total

            lbfgs.step(closure)
            losses_eval = compute_loss_terms(model, spec, cfg, frozen_batch, frozen_sampling_mode, frozen_epoch)
            update_loss_ema(cfg, loss_ema, losses_eval)
            val_metrics = validate_metrics(model, cfg, valid_scenarios)
            val_residual_q95 = validate_residual_focus_q95(model, spec, cfg, valid_scenarios)
            if val_metrics["composite_score"] < best_val:
                best_val = val_metrics["composite_score"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                lbfgs_patience = 0
            else:
                lbfgs_patience += 1
            history["epoch"].append(cfg.adam_epochs + lbfgs_round * cfg.lbfgs_chunk_steps)
            history["stage"].append("lbfgs")
            history["sampling_mode"].append(frozen_sampling_mode)
            history["total"].append(float(combine_losses(losses_eval, frozen_weights).item()))
            history["pde"].append(float(losses_eval["pde"].item()))
            history["boundary"].append(float(losses_eval["boundary"].item()))
            history["positivity"].append(float(losses_eval["positivity"].item()))
            history["w_pde"].append(frozen_weights.get("pde", 0.0))
            history["w_terminal"].append(frozen_weights.get("terminal", 0.0))
            history["w_anchor"].append(frozen_weights.get("anchor", 0.0))
            history["w_farfield"].append(frozen_weights.get("farfield", 0.0))
            history["w_boundary"].append(frozen_weights.get("boundary", 0.0))
            history["w_positivity"].append(frozen_weights.get("positivity", 0.0))
            history["grad_norm"].append(float("nan"))
            history["val_price_q95"].append(val_metrics["global_q95"])
            history["val_low_rho_q95"].append(val_metrics["low_rho_q95"])
            history["val_residual_q95"].append(val_residual_q95)
            history["val_composite"].append(val_metrics["composite_score"])
            history["best_val_composite"].append(best_val)
            if lbfgs_patience >= 2:
                break

    if cfg.residual_polish_enabled and is_residual_polish_target(spec):
        if best_state is not None:
            model.load_state_dict(best_state)
        baseline_polish_metrics = validate_metrics(model, cfg, valid_scenarios)
        baseline_polish_residual = validate_residual_focus_q95(model, spec, cfg, valid_scenarios)
        polish_optim = torch.optim.AdamW(model.parameters(), lr=cfg.residual_polish_lr, weight_decay=cfg.weight_decay)
        polish_best_score = baseline_polish_residual + 0.02 * baseline_polish_metrics["composite_score"]
        polish_best_state = None
        polish_patience = 0
        polish_batch = None
        polish_mode = "full_baac"
        polish_eval_every = max(6, cfg.eval_every // 2)
        polish_epoch_base = cfg.adam_epochs + cfg.lbfgs_rounds * cfg.lbfgs_chunk_steps
        for polish_epoch in range(1, cfg.residual_polish_epochs + 1):
            effective_epoch = polish_epoch_base + polish_epoch
            if polish_batch is None or (polish_epoch - 1) % cfg.residual_polish_resample_every == 0:
                polish_batch, polish_mode = sample_batch(
                    cfg,
                    spec,
                    model,
                    train_scenarios,
                    anchor_cache,
                    device,
                    rng,
                    effective_epoch,
                )
            polish_optim.zero_grad()
            losses = compute_loss_terms(model, spec, cfg, polish_batch, polish_mode, effective_epoch)
            tuned_weights = residual_polish_weights(cfg, adaptive_term_weights(cfg, spec, loss_ema, effective_epoch))
            hotspot = residual_hotspot_penalty(model, spec, cfg, polish_batch)
            total_loss = combine_losses(losses, tuned_weights) + cfg.residual_polish_hotspot_weight * hotspot
            total_loss.backward()
            gnorm = grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            polish_optim.step()
            update_loss_ema(cfg, loss_ema, losses)

            if polish_epoch == 1 or polish_epoch % polish_eval_every == 0 or polish_epoch == cfg.residual_polish_epochs:
                val_metrics = validate_metrics(model, cfg, valid_scenarios)
                val_residual_q95 = validate_residual_focus_q95(model, spec, cfg, valid_scenarios)
                polish_score = val_residual_q95 + 0.02 * val_metrics["composite_score"]
                price_guard = val_metrics["composite_score"] <= min(baseline_polish_metrics["composite_score"] * 1.01, baseline_polish_metrics["composite_score"] + 0.25)
                residual_gain = val_residual_q95 <= min(baseline_polish_residual * 0.92, baseline_polish_residual - 0.004)
                if price_guard and residual_gain and polish_score < polish_best_score:
                    polish_best_score = polish_score
                    polish_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_state = polish_best_state
                    best_val = min(best_val, val_metrics["composite_score"])
                    polish_patience = 0
                else:
                    polish_patience += 1

                history["epoch"].append(effective_epoch)
                history["stage"].append("polish")
                history["sampling_mode"].append(polish_mode)
                history["total"].append(float(total_loss.item()))
                history["pde"].append(float(losses["pde"].item()))
                history["boundary"].append(float(losses["boundary"].item()))
                history["positivity"].append(float(losses["positivity"].item()))
                history["w_pde"].append(tuned_weights.get("pde", 0.0))
                history["w_terminal"].append(tuned_weights.get("terminal", 0.0))
                history["w_anchor"].append(tuned_weights.get("anchor", 0.0))
                history["w_farfield"].append(tuned_weights.get("farfield", 0.0))
                history["w_boundary"].append(tuned_weights.get("boundary", 0.0))
                history["w_positivity"].append(tuned_weights.get("positivity", 0.0))
                history["grad_norm"].append(float(gnorm))
                history["val_price_q95"].append(val_metrics["global_q95"])
                history["val_low_rho_q95"].append(val_metrics["low_rho_q95"])
                history["val_residual_q95"].append(val_residual_q95)
                history["val_composite"].append(val_metrics["composite_score"])
                history["best_val_composite"].append(best_val)
                if polish_patience >= cfg.residual_polish_patience:
                    break

        if polish_best_state is not None:
            model.load_state_dict(polish_best_state)

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.perf_counter() - t0
    ensure_dir(output_dir)
    torch.save(model.state_dict(), output_dir / "model.pt")
    if best_state is not None:
        torch.save(best_state, output_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    with open(output_dir / "variant_spec.json", "w", encoding="utf-8") as f:
        json.dump(asdict(spec), f, indent=2)
    return model, pd.DataFrame(history), elapsed


def evaluate_variant(model: RealBarrierPINN, spec: VariantSpec, cfg: ExperimentConfig, scenarios: List[ch3.BarrierScenario]) -> Dict[str, float]:
    ms = [scenario_metrics(model, spec, cfg, scn) for scn in scenarios]
    return {
        "median_re": float(np.median([m["median_re"] for m in ms])),
        "q95_re": float(np.median([m["q95_re"] for m in ms])),
        "worst_re": float(np.max([m["worst_re"] for m in ms])),
        "delta_err": float(np.median([m["delta_err"] for m in ms])),
        "gamma_err": float(np.median([m["gamma_err"] for m in ms])),
        "barrier_residual": float(np.max([m["barrier_residual"] for m in ms])),
        "positivity_violation": float(np.max([m["positivity_violation"] for m in ms])),
        "success_rate": float(np.mean([m["success"] for m in ms])),
    }


def heatmap_for_variant(model: RealBarrierPINN, cfg: ExperimentConfig) -> np.ndarray:
    Z = np.zeros((len(cfg.sigma_grid), len(cfg.rho_grid)))
    for i, sig in enumerate(cfg.sigma_grid):
        for j, rho in enumerate(cfg.rho_grid):
            scn = scenario(cfg, sig, rho)
            true = ch3.rr_price_scalar(scn, S=scn.S0, tau=scn.T)
            pred = price_scalar(model, cfg, scn, scn.S0, scn.T)
            Z[i, j] = 100.0 * abs(pred - true) / (abs(true) + 1e-12)
    return Z


def plot_training_pathology(pathology_logs: Dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4), sharey=True)
    for ax, (title, df) in zip(axes, pathology_logs.items()):
        ax.plot(df["epoch"], np.maximum(df["pde"], 1e-10), color="#1f77b4", linewidth=2.0, label="PDE loss")
        ax.plot(df["epoch"], np.maximum(df["boundary"], 1e-10), color="#d62728", linewidth=2.0, linestyle="--", label="Boundary loss")
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


def plot_coordinate_choice(coord_histories: Dict[str, pd.DataFrame], summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))
    colors = {"Raw S-space": "#d62728", "x = S/K": "#ff7f0e", "y = ln(S/K)": "#1f77b4"}
    ax = axes[0]
    for label, df in coord_histories.items():
        ax.plot(df["epoch"], np.maximum(df["val_price_q95"], 1e-6), linewidth=2.0, label=label, color=colors[label])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation price q95 RE (%)")
    ax.set_title("Convergence")
    ax.grid(alpha=0.28, linestyle="--")
    ax.legend(frameon=True)

    sub = summary_df[summary_df["Variant"].isin(colors.keys())].set_index("Variant")
    axes[1].bar(sub.index.tolist(), sub["Median RE (%)"].tolist(), color=[colors[v] for v in sub.index], alpha=0.82)
    axes[1].set_ylabel("Median RE (%)")
    axes[1].set_title("Final error")
    axes[1].grid(axis="y", alpha=0.28, linestyle="--")
    gnorms = [float(np.median(coord_histories[k]["grad_norm"])) for k in colors.keys()]
    axes[2].bar(list(colors.keys()), gnorms, color=[colors[v] for v in colors.keys()], alpha=0.82)
    axes[2].set_ylabel("Median gradient norm")
    axes[2].set_title("Gradient scale")
    axes[2].grid(axis="y", alpha=0.28, linestyle="--")
    fig.suptitle("Figure 19. Effect of coordinate choice", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ansatz_effect(summary_df: pd.DataFrame, output_path: Path) -> None:
    labels = ["Soft BC", "Hard barrier only", "Hard barrier + positivity"]
    sub = summary_df[summary_df["Variant"].isin(labels)].set_index("Variant").loc[labels].reset_index()
    x = np.arange(len(labels))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    ax = axes[0]
    ax.bar(x - width / 2, sub["Median RE (%)"], width=width, color="#1f77b4", alpha=0.84, label="Median RE (%)")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, sub["Barrier residual"], width=width, color="#d62728", alpha=0.76, label="Barrier residual")
    ax2.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median RE (%)")
    ax2.set_ylabel("Barrier residual")
    ax.set_title("Boundary enforcement vs price quality")
    ax.grid(axis="y", alpha=0.28, linestyle="--")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)
    axes[1].bar(labels, sub["Positivity violation"], color=["#ff7f0e", "#2ca02c", "#9467bd"], alpha=0.82)
    axes[1].set_ylabel("Positivity violation rate")
    axes[1].set_title("Economic admissibility")
    axes[1].grid(axis="y", alpha=0.28, linestyle="--")
    fig.suptitle("Figure 20. Effect of hard-constrained ansatz", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_baac_effect(heatmaps: Dict[str, np.ndarray], cfg: ExperimentConfig, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.8))
    axes = axes.flatten()
    vmax = max(np.max(v) for v in heatmaps.values())
    for ax, (title, Z) in zip(axes, heatmaps.items()):
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r"Barrier proximity $\rho_d$")
        ax.set_ylabel(r"Volatility $\sigma$")
        ax.set_xticks(range(len(cfg.rho_grid)))
        ax.set_xticklabels([f"{v:.3f}" for v in cfg.rho_grid], rotation=35, ha="right")
        ax.set_yticks(range(len(cfg.sigma_grid)))
        ax.set_yticklabels([f"{v:.2f}" for v in cfg.sigma_grid])
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.86)
    cbar.set_label("Price RE (%) at $S_0$")
    fig.suptitle("Figure 21. Effect of BAAC", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_table12(df: pd.DataFrame, output_dir: Path) -> None:
    df.to_csv(output_dir / "table12_ablation_summary_matrix.csv", index=False)
    with open(output_dir / "table12_ablation_summary_matrix.tex", "w", encoding="utf-8") as f:
        f.write(
            df.to_latex(
                index=False,
                escape=False,
                caption="Ablation summary matrix based on real Chapter 7 experiments.",
                label="tab:ablation_summary_matrix_real",
            )
        )
    ch4.save_dataframe_as_png(df, output_dir / "table12_ablation_summary_matrix.png", "Table 12. Ablation summary matrix")


def main():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    ensure_dir(out)
    ensure_dir(out / "variants")

    train_scenarios, valid_scenarios, report_scenarios = build_panels(cfg)
    save_panel_summary(
        {
            "train": train_scenarios,
            "validation": valid_scenarios,
            "report": report_scenarios,
        },
        out / "panel_summary.csv",
    )

    # Figure 17 remains conceptual but is generated alongside the real experiment outputs.
    demo.plot_failure_taxonomy(out / "figure17_failure_taxonomy.png")

    # Figure 18: naive PINN pathology in two representative regimes.
    pathology_logs: Dict[str, pd.DataFrame] = {}
    pathology_cfg = replace(
        cfg,
        adam_epochs=max(180, cfg.adam_epochs // 3),
        eval_every=max(15, cfg.eval_every // 2),
        min_epochs_before_early_stop=max(90, cfg.min_epochs_before_early_stop // 2),
        early_stop_patience=max(3, cfg.early_stop_patience - 1),
        lbfgs_rounds=0,
        n_interior=max(700, cfg.n_interior // 2),
        n_terminal=max(240, cfg.n_terminal // 2),
        n_farfield=max(180, cfg.n_farfield // 2),
        n_boundary=max(180, cfg.n_boundary // 2),
        n_barrier=max(320, cfg.n_barrier // 2),
        n_strike=max(220, cfg.n_strike // 2),
        n_refine=max(220, cfg.n_refine // 2),
        candidate_refine_pool=max(900, cfg.candidate_refine_pool // 2),
    )
    for title, scn in {
        "Extreme curvature regime": scenario(cfg, 0.15, 0.002),
        "Smoother regime": scenario(cfg, 0.40, 0.150),
    }.items():
        spec = VariantSpec("Naive PINN pathology", "Failure baseline", "raw_s", "soft_bc", "uniform", "adam_only")
        pdir = out / "pathology" / title.lower().replace(" ", "_")
        model, hist, _ = train_variant(pathology_cfg, spec, [scn], [scn], pdir)
        pathology_logs[title] = hist
        del model
    plot_training_pathology(pathology_logs, out / "figure18_training_pathology_naive_pinn.png")

    summary_rows = []
    coord_histories: Dict[str, pd.DataFrame] = {}
    baac_heatmaps: Dict[str, np.ndarray] = {}

    for spec in default_variants():
        vdir = out / "variants" / spec.name.lower().replace(" ", "_").replace("=", "").replace("/", "_")
        model, hist, elapsed = train_variant(cfg, spec, train_scenarios, valid_scenarios, vdir)
        metrics = evaluate_variant(model, spec, cfg, report_scenarios)

        summary_rows.append({
            "Variant": spec.name,
            "Group": spec.group,
            "Median RE (%)": metrics["median_re"],
            "95th RE (%)": metrics["q95_re"],
            "Worst-case RE (%)": metrics["worst_re"],
            "Delta error": metrics["delta_err"],
            "Gamma error": metrics["gamma_err"],
            "Barrier residual": metrics["barrier_residual"],
            "Positivity violation": metrics["positivity_violation"],
            "Success rate": metrics["success_rate"],
            "Training time (s)": elapsed,
        })

        if spec.name in {"Raw S-space", "x = S/K", "y = ln(S/K)"}:
            coord_histories[spec.name] = hist
        if spec.name in {"No refinement", "Static oversampling", "Residual refinement", "Full BAAC"}:
            baac_heatmaps[spec.name] = heatmap_for_variant(model, cfg)

    df_summary = pd.DataFrame(summary_rows)
    save_table12(df_summary, out)

    plot_coordinate_choice(coord_histories, df_summary, out / "figure19_effect_coordinate_choice.png")
    plot_ansatz_effect(df_summary, out / "figure20_effect_hard_constrained_ansatz.png")
    plot_baac_effect(baac_heatmaps, cfg, out / "figure21_effect_baac.png")

    summary = {
        "status": "chapter7 formal real ablation script executed",
        "config": asdict(cfg),
        "train_panel_size": len(train_scenarios),
        "valid_panel_size": len(valid_scenarios),
        "report_panel_size": len(report_scenarios),
        "table12_csv": str((out / "table12_ablation_summary_matrix.csv").resolve()),
        "figure17": str((out / "figure17_failure_taxonomy.png").resolve()),
        "figure18": str((out / "figure18_training_pathology_naive_pinn.png").resolve()),
        "figure19": str((out / "figure19_effect_coordinate_choice.png").resolve()),
        "figure20": str((out / "figure20_effect_hard_constrained_ansatz.png").resolve()),
        "figure21": str((out / "figure21_effect_baac.png").resolve()),
    }
    with open(out / "chapter7_real_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Chapter 7 real ablation and failure diagnostics")
    print("=" * 72)
    print("Exported:")
    print("  - Figure 17: failure taxonomy")
    print("  - Figure 18: naive PINN pathology from real training logs")
    print("  - Figure 19: coordinate-choice ablation from real training/evaluation")
    print("  - Figure 20: ansatz ablation from real training/evaluation")
    print("  - Figure 21: BAAC ablation heatmaps from real evaluation")
    print("  - Table 12: ablation summary matrix")
    print(f"Output directory: {out.resolve()}")


if __name__ == "__main__":
    main()
