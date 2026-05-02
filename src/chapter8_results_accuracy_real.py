import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
import chapter7_ablation_failure_diagnostics_real as ch7


@dataclass
class Chapter8Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "results_chapter8_only"
    pinn_model_path: str = r"E:\results_chapter7_real_formal\full_baac_guard_probe\best_model.pt"
    pinn_variant_spec_path: str = r"E:\results_chapter7_real_formal\full_baac_guard_probe\variant_spec.json"
    train_label_samples_per_scenario: int = 56
    valid_label_samples_per_scenario: int = 24
    supervised_epochs: int = 520
    supervised_lr: float = 6e-4
    supervised_gamma: float = 0.9985
    supervised_weight_decay: float = 1e-6
    supervised_batch_size: int = 256
    supervised_patience: int = 5
    differential_price_weight: float = 1.0
    differential_delta_weight: float = 0.08
    differential_gamma_weight: float = 0.0
    differential_boundary_weight: float = 0.18
    supervised_hard_barrier: bool = True
    supervised_fourier_features: bool = True
    supervised_barrier_kappa: float = 14.0
    supervised_relative_loss_weight: float = 0.35
    supervised_price_rel_floor: float = 0.05
    supervised_barrier_weight: float = 1.0
    supervised_strike_weight: float = 0.8
    supervised_small_price_weight: float = 0.6
    supervised_small_price_threshold: float = 0.25
    supervised_rel_ramp_start: float = 0.20
    supervised_rel_ramp_width: float = 0.35
    supervised_focus_ramp_start: float = 0.15
    supervised_focus_ramp_width: float = 0.35
    differential_delta_start: float = 0.42
    differential_delta_width: float = 0.18
    differential_gamma_start: float = 0.90
    differential_gamma_width: float = 0.08
    differential_gamma_max_scale: float = 0.0
    differential_min_gamma_evals: int = 0
    differential_use_supervised_warmstart: bool = True
    fdm_Nx: int = 800
    fdm_Nt: int = 2000
    sigma_grid: Tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
    rho_grid: Tuple[float, ...] = (0.002, 0.010, 0.030, 0.060, 0.100, 0.150)
    rho_zoom_grid: Tuple[float, ...] = (0.002, 0.004, 0.006, 0.010, 0.020, 0.030)
    representative_slice_sigma: float = 0.25
    representative_slice_rho: float = 0.002
    representative_slice_T: float = 1.0
    boundary_tau_points: int = 25
    positivity_S_points: int = 30
    positivity_tau_points: int = 12
    residual_quantiles: Tuple[float, ...] = (0.50, 0.90, 0.95, 0.99)
    certified_bounds_enabled: bool = False
    force_retrain_surrogates: bool = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scenario_key(scn: ch3.BarrierScenario, tau: float) -> Tuple[float, float, float, float]:
    return (
        round(float(scn.sigma), 6),
        round(float(scn.rho_d), 6),
        round(float(scn.T), 6),
        round(float(tau), 6),
    )


def region_label(scn: ch3.BarrierScenario, S: float) -> str:
    if S <= scn.B_d + 0.06 * scn.K:
        return "near_barrier"
    if abs(S - scn.K) <= 0.08 * scn.K:
        return "near_strike"
    if S >= min(scn.S_max, 1.35 * scn.K):
        return "far_field"
    return "smooth"


def build_eval_panels(cfg: Chapter8Config) -> Tuple[List[ch3.BarrierScenario], List[ch3.BarrierScenario], List[ch3.BarrierScenario]]:
    ch7_cfg = ch7.ExperimentConfig()
    train, valid, report = ch7.build_panels(ch7_cfg)
    return train, valid, report


def core_scenarios(cfg: Chapter8Config) -> List[ch3.BarrierScenario]:
    return [replace(scn, T=cfg.representative_slice_T) for scn in ch3.CORE_SCENARIOS]


def make_grid_scenario(cfg: Chapter8Config, sigma: float, rho_d: float, T: float) -> ch3.BarrierScenario:
    return ch3.BarrierScenario(
        sigma=float(sigma),
        rho_d=float(rho_d),
        S0=100.0,
        K=100.0,
        T=float(T),
        r=0.10,
        delta=0.0,
        S_max=200.0,
    )


def model_feature_torch(S, tau, sigma, beta, r, q):
    K = 100.0
    x = S / K
    y = torch.log(x)
    barrier = torch.log(beta)
    # Zero-preserving smooth distance keeps d(B)=0 while remaining smooth enough
    # for stable higher-order autodiff near the barrier.
    d = F.softplus(y - barrier, beta=100.0) - (math.log(2.0) / 100.0)
    d_fast = 1.0 - torch.exp(-12.0 * d)
    d_slow = 1.0 - torch.exp(-1.2 * d)
    p = torch.cat([
        (sigma - 0.25) / 0.15,
        (beta - 0.90) / 0.10,
        (r - 0.05) / 0.10,
        (q - 0.00) / 0.05,
    ], dim=1)
    return torch.cat([y, x, tau, d, d_fast, d_slow, p], dim=1)


def label_feature_torch(S, tau, sigma, beta, r, q, use_fourier: bool = True):
    base = model_feature_torch(S, tau, sigma, beta, r, q)
    if not use_fourier:
        return base
    x = S / 100.0
    y = torch.log(x)
    barrier = torch.log(beta)
    d = F.softplus(y - barrier, beta=100.0) - (math.log(2.0) / 100.0)
    freq = 2.0 * math.pi
    extra = torch.cat(
        [
            torch.sin(freq * y),
            torch.cos(freq * y),
            torch.sin(0.5 * freq * d),
            torch.cos(0.5 * freq * d),
        ],
        dim=1,
    )
    return torch.cat([base, extra], dim=1)


class SurrogateMLP(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_layers: int = 4, width: int = 128):
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


class LabelSurrogate(nn.Module):
    """
    Label-driven surrogates for the supervised and differential baselines.
    """
    def __init__(self, cfg: Chapter8Config):
        super().__init__()
        self.cfg = cfg
        input_dim = 14 if cfg.supervised_fourier_features else 10
        self.core = SurrogateMLP(input_dim=input_dim)

    def forward(self, S, tau, sigma, beta, r, q):
        feats = label_feature_torch(S, tau, sigma, beta, r, q, use_fourier=self.cfg.supervised_fourier_features)
        amp = F.softplus(self.core(feats))
        if not self.cfg.supervised_hard_barrier:
            return amp
        barrier = torch.log(beta)
        y = torch.log(S / 100.0)
        d = F.softplus(y - barrier, beta=100.0) - (math.log(2.0) / 100.0)
        psi = 1.0 - torch.exp(-self.cfg.supervised_barrier_kappa * d)
        return psi * amp


class BaseAdapter:
    name: str

    def price(self, scn: ch3.BarrierScenario, S: float, tau: float) -> float:
        raise NotImplementedError

    def delta_gamma(self, scn: ch3.BarrierScenario, S: float, tau: float, h: float = 1e-2) -> Tuple[float, float]:
        fp = self.price(scn, S + h, tau)
        fm = self.price(scn, S - h, tau)
        f0 = self.price(scn, S, tau)
        delta = (fp - fm) / (2.0 * h)
        gamma = (fp - 2.0 * f0 + fm) / (h ** 2)
        return float(delta), float(gamma)


class TruthAdapter(BaseAdapter):
    name = "Truth"

    def price(self, scn: ch3.BarrierScenario, S: float, tau: float) -> float:
        return ch3.rr_price_scalar(scn, S=S, tau=tau)

    def delta_gamma(self, scn: ch3.BarrierScenario, S: float, tau: float, h: float = 1e-3) -> Tuple[float, float]:
        return ch3.numerical_delta_gamma_rr(scn, S, tau, h=h)


class FDMAdapter(BaseAdapter):
    name = "FDM"

    def __init__(self, cfg: Chapter8Config):
        self.cfg = cfg
        self._cache: Dict[Tuple[float, float, float, float], Tuple[np.ndarray, np.ndarray]] = {}

    def _grid(self, scn: ch3.BarrierScenario, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        key = scenario_key(scn, tau)
        if key not in self._cache:
            scn_tau = replace(scn, T=float(tau))
            self._cache[key] = ch3.fdm_solve_rannacher_cn_grid(scn_tau, Nt=self.cfg.fdm_Nt, Nx=self.cfg.fdm_Nx)
        return self._cache[key]

    def price(self, scn: ch3.BarrierScenario, S: float, tau: float) -> float:
        if tau <= 0.0:
            return float(max(S - scn.K, 0.0))
        S_grid, V_grid = self._grid(scn, tau)
        return float(np.interp(S, S_grid, V_grid))

    def delta_gamma(self, scn: ch3.BarrierScenario, S: float, tau: float, h: float = 1e-2) -> Tuple[float, float]:
        if tau <= 0.0:
            delta = 1.0 if S > scn.K else 0.0
            return float(delta), 0.0
        S_grid, V_grid = self._grid(scn, tau)
        return ch3.fdm_local_delta_gamma(S_grid, V_grid, S)


class PINNAdapter(BaseAdapter):
    name = "PINN"

    def __init__(self, cfg: Chapter8Config):
        self.device = torch.device(cfg.device)
        self.available = False
        try:
            with open(cfg.pinn_variant_spec_path, "r", encoding="utf-8") as f:
                variant_data = json.load(f)
            self.exp_cfg = ch7.ExperimentConfig()
            self.spec = ch7.VariantSpec(**variant_data)
            self.model = ch7.RealBarrierPINN(self.exp_cfg, self.spec).to(self.device)
            state = torch.load(cfg.pinn_model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.available = True
        except Exception as exc:
            print(f"Warning: PINN model not loaded. Exception: {exc}")

    def price(self, scn: ch3.BarrierScenario, S: float, tau: float) -> float:
        if not self.available:
            return 0.0
        return ch7.price_scalar(self.model, self.exp_cfg, scn, S, tau)

    def delta_gamma(self, scn: ch3.BarrierScenario, S: float, tau: float, h: float = 1e-2) -> Tuple[float, float]:
        if not self.available:
            return 0.0, 0.0
        return ch7.delta_gamma_fd(self.model, self.exp_cfg, scn, S, tau, h=h)


class NeuralAdapter(BaseAdapter):
    def __init__(self, name: str, model: LabelSurrogate, device: torch.device):
        self.name = name
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def price(self, scn: ch3.BarrierScenario, S: float, tau: float) -> float:
        with torch.no_grad():
            out = self.model(
                torch.tensor([[S]], dtype=torch.float32, device=self.device),
                torch.tensor([[tau]], dtype=torch.float32, device=self.device),
                torch.tensor([[scn.sigma]], dtype=torch.float32, device=self.device),
                torch.tensor([[scn.B_d / scn.K]], dtype=torch.float32, device=self.device),
                torch.tensor([[scn.r]], dtype=torch.float32, device=self.device),
                torch.tensor([[scn.delta]], dtype=torch.float32, device=self.device),
            )
        return float(out.item())

    def delta_gamma(self, scn: ch3.BarrierScenario, S: float, tau: float, h: float = 1e-2) -> Tuple[float, float]:
        S_t = torch.tensor([[S]], dtype=torch.float32, device=self.device, requires_grad=True)
        tau_t = torch.tensor([[tau]], dtype=torch.float32, device=self.device)
        sigma_t = torch.tensor([[scn.sigma]], dtype=torch.float32, device=self.device)
        beta_t = torch.tensor([[scn.B_d / scn.K]], dtype=torch.float32, device=self.device)
        r_t = torch.tensor([[scn.r]], dtype=torch.float32, device=self.device)
        q_t = torch.tensor([[scn.delta]], dtype=torch.float32, device=self.device)
        value = self.model(S_t, tau_t, sigma_t, beta_t, r_t, q_t)
        delta = torch.autograd.grad(value, S_t, grad_outputs=torch.ones_like(value), create_graph=True)[0]
        gamma = torch.autograd.grad(delta, S_t, grad_outputs=torch.ones_like(delta), create_graph=False)[0]
        return float(delta.item()), float(gamma.item())


def sample_labeled_points(
    scenarios: Sequence[ch3.BarrierScenario],
    n_per_scenario: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for scn in scenarios:
        for _ in range(n_per_scenario):
            u = rng.random()
            if u < 0.28:
                S = rng.uniform(scn.B_d + 1e-4, min(scn.S_max, scn.B_d + 0.08 * scn.K))
            elif u < 0.52:
                S = rng.uniform(max(scn.B_d + 1e-4, scn.K - 0.08 * scn.K), min(scn.S_max, scn.K + 0.08 * scn.K))
            elif u < 0.68:
                S = scn.B_d
            elif u < 0.84:
                S = scn.S_max
            else:
                S = rng.uniform(scn.B_d + 1e-4, scn.S_max)
            if S == scn.B_d:
                tau = rng.uniform(0.0, scn.T)
                price = 0.0
                delta = 0.0
                gamma = 0.0
            elif S == scn.S_max:
                tau = rng.uniform(0.0, scn.T)
                price = float(ch3.upper_bc(scn.S_max, scn.K, tau, scn.r, scn.delta))
                delta = 1.0
                gamma = 0.0
            else:
                if rng.random() < 0.22:
                    tau = 0.0
                else:
                    tau = scn.T * float(rng.beta(0.85, 1.15))
                if tau <= 0.0:
                    price = float(max(S - scn.K, 0.0))
                    delta = 1.0 if S > scn.K else 0.0
                    gamma = 0.0
                else:
                    price = ch3.rr_price_scalar(scn, S=float(S), tau=float(tau))
                    delta, gamma = ch3.numerical_delta_gamma_rr(scn, float(S), float(tau), h=1e-3)
            rows.append(
                {
                    "S": float(S),
                    "tau": float(tau),
                    "sigma": float(scn.sigma),
                    "beta": float(scn.B_d / scn.K),
                    "r": float(scn.r),
                    "q": float(scn.delta),
                    "price": float(price),
                    "delta": float(delta),
                    "gamma": float(gamma),
                }
            )
    return pd.DataFrame(rows)


def df_to_tensor(df: pd.DataFrame, cols: Sequence[str], device: torch.device) -> torch.Tensor:
    return torch.tensor(df[list(cols)].values, dtype=torch.float32, device=device)


def load_or_train_label_surrogate(
    cfg: Chapter8Config,
    name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    differential: bool,
    output_dir: Path,
) -> LabelSurrogate:
    ensure_dir(output_dir)
    model_path = output_dir / "best_model.pt"
    meta_path = output_dir / "training_meta.json"
    device = torch.device(cfg.device)
    model = LabelSurrogate(cfg).to(device)

    if model_path.exists() and not cfg.force_retrain_surrogates:
        try:
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            return model
        except RuntimeError:
            pass

    warmstart_used = False
    if differential and cfg.differential_use_supervised_warmstart:
        warmstart_path = output_dir.parent / "supervised_surrogate" / "best_model.pt"
        if warmstart_path.exists():
            try:
                warm_state = torch.load(warmstart_path, map_location=device)
                model.load_state_dict(warm_state, strict=True)
                warmstart_used = True
                print(f"Warm-started {name} from supervised checkpoint.")
            except Exception as exc:
                print(f"Warning: warm-start for {name} failed: {exc}")

    x_train = {k: df_to_tensor(train_df, [k], device) for k in ["S", "tau", "sigma", "beta", "r", "q"]}
    y_train = df_to_tensor(train_df, ["price", "delta", "gamma"], device)
    x_valid = {k: df_to_tensor(valid_df, [k], device) for k in ["S", "tau", "sigma", "beta", "r", "q"]}
    y_valid = df_to_tensor(valid_df, ["price", "delta", "gamma"], device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.supervised_lr, weight_decay=cfg.supervised_weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=cfg.supervised_gamma)

    batch_size = min(cfg.supervised_batch_size, len(train_df))
    best_val = math.inf
    best_state = None
    best_stage = 1
    best_state_stage2 = None
    best_val_stage2 = math.inf
    best_state_stage3 = None
    best_val_stage3 = math.inf
    patience = 0
    history = []

    def forward_batch(xd: Dict[str, torch.Tensor]) -> torch.Tensor:
        return model(xd["S"], xd["tau"], xd["sigma"], xd["beta"], xd["r"], xd["q"])

    def ramp(progress: float, start: float, width: float) -> float:
        if width <= 0.0:
            return 1.0 if progress >= start else 0.0
        value = (progress - start) / width
        return float(max(0.0, min(1.0, value)))

    def stage_from_scales(delta_scale: float, gamma_scale: float) -> int:
        if gamma_scale > 0.0:
            return 3
        if delta_scale > 0.0:
            return 2
        return 1

    def weighted_price_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        S: torch.Tensor,
        beta: torch.Tensor,
        progress: float,
    ) -> torch.Tensor:
        abs_term = (pred - target).pow(2)
        rel_term = ((pred - target) / (target.abs() + cfg.supervised_price_rel_floor)).pow(2)
        barrier_S = beta * 100.0
        rel_scale = ramp(progress, cfg.supervised_rel_ramp_start, cfg.supervised_rel_ramp_width)
        focus_scale = ramp(progress, cfg.supervised_focus_ramp_start, cfg.supervised_focus_ramp_width)
        weights = torch.ones_like(target)
        weights = weights + focus_scale * cfg.supervised_barrier_weight * (S <= barrier_S + 0.08 * 100.0).float()
        weights = weights + focus_scale * cfg.supervised_strike_weight * (torch.abs(S - 100.0) <= 0.08 * 100.0).float()
        weights = weights + focus_scale * cfg.supervised_small_price_weight * (target.abs() <= cfg.supervised_small_price_threshold).float()
        combined = (
            (1.0 - cfg.supervised_relative_loss_weight * rel_scale) * abs_term
            + (cfg.supervised_relative_loss_weight * rel_scale) * rel_term
        )
        return (weights * combined).mean()

    current_stage = 1
    gamma_eval_count = 0

    for epoch in range(1, cfg.supervised_epochs + 1):
        progress = epoch / max(cfg.supervised_epochs, 1)
        if differential:
            delta_scale = ramp(progress, cfg.differential_delta_start, cfg.differential_delta_width)
            gamma_scale = cfg.differential_gamma_max_scale * ramp(progress, cfg.differential_gamma_start, cfg.differential_gamma_width)
            stage = stage_from_scales(delta_scale, gamma_scale)
        else:
            delta_scale = 0.0
            gamma_scale = 0.0
            stage = 1
        if differential and stage != current_stage:
            current_stage = stage
            patience = 0
        perm = torch.randperm(len(train_df), device=device)
        model.train()
        for start in range(0, len(train_df), batch_size):
            idx = perm[start:start + batch_size]
            xb = {k: v[idx].clone() for k, v in x_train.items()}
            target = y_train[idx]
            optim.zero_grad()
            xb["S"].requires_grad_(differential)
            pred = forward_batch(xb)
            loss = weighted_price_loss(pred, target[:, :1], xb["S"], xb["beta"], progress)
            boundary_mask = xb["S"] <= (xb["beta"] * 100.0 + 1e-6)
            if boundary_mask.any():
                loss = loss + cfg.differential_boundary_weight * F.mse_loss(pred[boundary_mask], torch.zeros_like(pred[boundary_mask]))
            if differential:
                delta_pred = torch.autograd.grad(pred, xb["S"], grad_outputs=torch.ones_like(pred), create_graph=True)[0]
                gamma_pred = torch.autograd.grad(delta_pred, xb["S"], grad_outputs=torch.ones_like(delta_pred), create_graph=True)[0]
                valid_delta_mask = (xb["tau"] > 1e-3).float()
                valid_gamma_mask = (xb["tau"] > 1e-3).float()
                loss = (
                    cfg.differential_price_weight * loss
                    + delta_scale * cfg.differential_delta_weight * ((((delta_pred - target[:, 1:2]) / (target[:, 1:2].abs() + 0.05)).pow(2)) * valid_delta_mask).mean()
                    + gamma_scale * cfg.differential_gamma_weight * ((((gamma_pred - target[:, 2:3]) / (target[:, 2:3].abs() + 0.02)).pow(2)) * valid_gamma_mask).mean()
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
        scheduler.step()

        if epoch == 1 or epoch % 20 == 0 or epoch == cfg.supervised_epochs:
            model.eval()
            with torch.enable_grad():
                xv = {k: v.clone() for k, v in x_valid.items()}
                xv["S"].requires_grad_(differential)
                pred_v = forward_batch(xv)
                val_loss = weighted_price_loss(pred_v, y_valid[:, :1], xv["S"], xv["beta"], progress)
                if differential:
                    delta_v = torch.autograd.grad(pred_v, xv["S"], grad_outputs=torch.ones_like(pred_v), create_graph=True)[0]
                    gamma_v = torch.autograd.grad(delta_v, xv["S"], grad_outputs=torch.ones_like(delta_v), create_graph=False)[0]
                    valid_delta_mask_v = (xv["tau"] > 1e-3).float()
                    valid_gamma_mask_v = (xv["tau"] > 1e-3).float()
                    val_loss = (
                        cfg.differential_price_weight * val_loss
                        + delta_scale * cfg.differential_delta_weight * ((((delta_v - y_valid[:, 1:2]) / (y_valid[:, 1:2].abs() + 0.05)).pow(2)) * valid_delta_mask_v).mean()
                        + gamma_scale * cfg.differential_gamma_weight * ((((gamma_v - y_valid[:, 2:3]) / (y_valid[:, 2:3].abs() + 0.02)).pow(2)) * valid_gamma_mask_v).mean()
                    )
            val_item = float(val_loss.item())
            if differential and stage == 3:
                gamma_eval_count += 1
            history.append(
                {
                    "epoch": epoch,
                    "val_loss": val_item,
                    "progress": progress,
                    "stage": stage,
                    "delta_scale": delta_scale if differential else 0.0,
                    "gamma_scale": gamma_scale if differential else 0.0,
                }
            )
            if val_item < best_val:
                best_val = val_item
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_stage = stage
                patience = 0
            else:
                patience += 1
            if differential:
                if stage >= 2 and val_item < best_val_stage2:
                    best_val_stage2 = val_item
                    best_state_stage2 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if stage == 3 and val_item < best_val_stage3:
                    best_val_stage3 = val_item
                    best_state_stage3 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if cfg.differential_gamma_max_scale > 0.0:
                    allow_stop = stage == 3 and gamma_eval_count >= cfg.differential_min_gamma_evals
                else:
                    allow_stop = stage >= 2
            else:
                allow_stop = True
            if allow_stop and patience >= cfg.supervised_patience:
                break

    chosen_state = best_state
    chosen_stage = best_stage
    chosen_val = best_val
    if differential:
        if best_state_stage3 is not None:
            chosen_state = best_state_stage3
            chosen_stage = 3
            chosen_val = best_val_stage3
        elif best_state_stage2 is not None:
            chosen_state = best_state_stage2
            chosen_stage = 2
            chosen_val = best_val_stage2

    if chosen_state is not None:
        model.load_state_dict(chosen_state)
        torch.save(chosen_state, model_path)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "name": name,
                "differential": differential,
                "best_val_loss": chosen_val,
                "chosen_stage": chosen_stage,
                "warmstart_used": warmstart_used,
            },
            f,
            indent=2,
        )
    model.eval()
    return model


def measure_runtime_per_contract(adapter: BaseAdapter, scenarios: Sequence[ch3.BarrierScenario], repeats: int = 8) -> float:
    times = []
    for scn in scenarios:
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = adapter.price(scn, scn.S0, scn.T)
        dt = (time.perf_counter() - t0) / repeats
        times.append(dt)
    return float(np.median(times))


def finite_difference_residual(adapter: BaseAdapter, scn: ch3.BarrierScenario, S: float, tau: float) -> float:
    if tau <= 1e-5:
        tau = max(1e-4, 0.02 * scn.T)
    hS = max(1e-2, 1e-3 * max(S, 1.0))
    ht = max(1e-3, 1e-3 * max(tau, 0.2))
    S_lo = max(scn.B_d + 1e-4, S - hS)
    S_hi = min(scn.S_max, S + hS)
    if abs(S_hi - S_lo) < 1e-8:
        return 0.0
    tau_lo = max(0.0, tau - ht)
    tau_hi = min(max(scn.T, tau), tau + ht)
    if abs(tau_hi - tau_lo) < 1e-8:
        tau_hi = min(scn.T, tau + max(ht, 1e-3))
    v0 = adapter.price(scn, S, tau)
    vS_hi = adapter.price(scn, S_hi, tau)
    vS_lo = adapter.price(scn, S_lo, tau)
    vt_hi = adapter.price(scn, S, tau_hi)
    vt_lo = adapter.price(scn, S, tau_lo)
    dVdS = (vS_hi - vS_lo) / (S_hi - S_lo)
    h2 = 0.5 * (S_hi - S_lo)
    d2VdS2 = (vS_hi - 2.0 * v0 + vS_lo) / max(h2 ** 2, 1e-8)
    dVdtau = (vt_hi - vt_lo) / max(tau_hi - tau_lo, 1e-8)
    rhs = 0.5 * scn.sigma ** 2 * S ** 2 * d2VdS2 + (scn.r - scn.delta) * S * dVdS - scn.r * v0
    return float(abs(dVdtau - rhs))


def evaluate_model_point(adapter: BaseAdapter, truth: TruthAdapter, scn: ch3.BarrierScenario, S: float, tau: float) -> Dict[str, float]:
    true_price = truth.price(scn, S, tau)
    pred_price = adapter.price(scn, S, tau)
    true_delta, true_gamma = truth.delta_gamma(scn, S, tau)
    pred_delta, pred_gamma = adapter.delta_gamma(scn, S, tau)
    abs_err = abs(pred_price - true_price)
    rel_err = 100.0 * abs_err / (abs(true_price) + 1e-12)
    return {
        "price": pred_price,
        "truth": true_price,
        "ae": abs_err,
        "re_pct": rel_err,
        "delta_abs_err": abs(pred_delta - true_delta),
        "gamma_abs_err": abs(pred_gamma - true_gamma),
    }


def build_main_pricing_table(
    cfg: Chapter8Config,
    adapters: Dict[str, BaseAdapter],
    truth: TruthAdapter,
    scenarios: Sequence[ch3.BarrierScenario],
) -> pd.DataFrame:
    runtime_map = {name: measure_runtime_per_contract(adapter, scenarios) for name, adapter in adapters.items() if name != "Truth"}
    eval_names = [n for n in ["FDM", "PINN", "Supervised", "Differential"] if n in adapters]
    rows = []
    for scn in scenarios:
        row = {
            "Sigma": scn.sigma,
            "Rho_d": scn.rho_d,
            "B_d": scn.B_d,
            "T": scn.T,
            "Truth": truth.price(scn, scn.S0, scn.T),
        }
        for name in eval_names:
            metrics = evaluate_model_point(adapters[name], truth, scn, scn.S0, scn.T)
            row[f"V_{name}"] = metrics["price"]
            row[f"AE_{name}"] = metrics["ae"]
            row[f"RE_{name} (%)"] = metrics["re_pct"]
            row[f"Runtime_{name} (s)"] = runtime_map[name]
        rows.append(row)
    return pd.DataFrame(rows)


def heatmap_metric(
    adapter: BaseAdapter,
    truth: TruthAdapter,
    cfg: Chapter8Config,
    sigmas: Sequence[float],
    rhos: Sequence[float],
    metric: str,
    zoom: bool = False,
) -> np.ndarray:
    Z = np.zeros((len(sigmas), len(rhos)))
    for i, sigma in enumerate(sigmas):
        for j, rho in enumerate(rhos):
            scn = make_grid_scenario(cfg, sigma, rho, cfg.representative_slice_T)
            S_eval = scn.B_d + 0.04 * scn.K if zoom else scn.S0
            if metric == "price":
                Z[i, j] = evaluate_model_point(adapter, truth, scn, S_eval, scn.T)["re_pct"]
            elif metric == "delta":
                Z[i, j] = evaluate_model_point(adapter, truth, scn, scn.S0, scn.T)["delta_abs_err"]
            elif metric == "gamma":
                Z[i, j] = evaluate_model_point(adapter, truth, scn, scn.S0, scn.T)["gamma_abs_err"]
            else:
                raise ValueError(metric)
    return Z


def save_table(df: pd.DataFrame, tex_path: Path, csv_path: Path, png_path: Path, caption: str, label: str) -> None:
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False, caption=caption, label=label))
    ch4.save_dataframe_as_png(df, png_path, caption)


def plot_multi_model_heatmaps(
    heatmaps: Dict[str, np.ndarray],
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    output_path: Path,
    title: str,
    cbar_label: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.8), constrained_layout=True)
    axes = axes.flatten()
    vmax = max(np.max(arr) for arr in heatmaps.values())
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    x_step = np.min(np.diff(x_vals)) if len(x_vals) > 1 else 0.01
    y_step = np.min(np.diff(y_vals)) if len(y_vals) > 1 else 0.05
    extent = [
        float(x_vals[0] - 0.5 * x_step),
        float(x_vals[-1] + 0.5 * x_step),
        float(y_vals[0] - 0.5 * y_step),
        float(y_vals[-1] + 0.5 * y_step),
    ]
    for ax, (name, Z) in zip(axes, heatmaps.items()):
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax, extent=extent, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel(r"Barrier proximity $\rho_d$")
        ax.set_ylabel(r"Volatility $\sigma$")
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{v:.3f}" for v in x_vals], rotation=35, ha="right")
        ax.set_yticks(y_vals)
        ax.set_yticklabels([f"{v:.2f}" for v in y_vals])
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.90, pad=0.02)
    cbar.set_label(cbar_label)
    fig.suptitle(title, fontsize=15, y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gamma_heatmaps_and_slice(
    gamma_heatmaps: Dict[str, np.ndarray],
    adapters: Dict[str, BaseAdapter],
    truth: TruthAdapter,
    cfg: Chapter8Config,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(14.2, 10.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1])
    vmax = max(np.max(arr) for arr in gamma_heatmaps.values())
    x_vals = np.asarray(cfg.rho_grid, dtype=float)
    y_vals = np.asarray(cfg.sigma_grid, dtype=float)
    x_step = np.min(np.diff(x_vals)) if len(x_vals) > 1 else 0.01
    y_step = np.min(np.diff(y_vals)) if len(y_vals) > 1 else 0.05
    extent = [
        float(x_vals[0] - 0.5 * x_step),
        float(x_vals[-1] + 0.5 * x_step),
        float(y_vals[0] - 0.5 * y_step),
        float(y_vals[-1] + 0.5 * y_step),
    ]
    for idx, (name, Z) in enumerate(gamma_heatmaps.items()):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax, extent=extent, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel(r"Barrier proximity $\rho_d$")
        ax.set_ylabel(r"Volatility $\sigma$")
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{v:.3f}" for v in x_vals], rotation=35, ha="right")
        ax.set_yticks(y_vals)
        ax.set_yticklabels([f"{v:.2f}" for v in y_vals])
    fig.colorbar(im, ax=fig.axes[:4], shrink=0.86, pad=0.02).set_label("Gamma AE")

    ax_slice = fig.add_subplot(gs[2, :])
    scn = make_grid_scenario(cfg, cfg.representative_slice_sigma, cfg.representative_slice_rho, cfg.representative_slice_T)
    S_vals = np.linspace(scn.B_d + 0.002 * scn.K, min(scn.S_max, scn.B_d + 0.25 * scn.K), 80)
    _, gamma_true = zip(*[truth.delta_gamma(scn, float(S), scn.T) for S in S_vals])
    ax_slice.plot(S_vals, gamma_true, color="black", linewidth=2.2, label="Truth")
    colors = {"FDM": "#1f77b4", "PINN": "#ff7f0e", "Supervised": "#2ca02c", "Differential": "#d62728"}
    slice_names = [n for n in ["FDM", "PINN", "Supervised", "Differential"] if n in adapters]
    for name in slice_names:
        gammas = [adapters[name].delta_gamma(scn, float(S), scn.T)[1] for S in S_vals]
        ax_slice.plot(S_vals, gammas, linewidth=1.9, label=name, color=colors[name])
    ax_slice.set_title("Representative near-barrier Gamma slice")
    ax_slice.set_xlabel("Spot")
    ax_slice.set_ylabel("Gamma")
    ax_slice.grid(alpha=0.28, linestyle="--")
    ax_slice.legend(frameon=True, ncol=3)

    fig.suptitle("Figure 25. Gamma comparison heatmaps and representative slice", fontsize=15, y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def diagnostics_boundary_positivity(
    adapters: Dict[str, BaseAdapter],
    scenarios: Sequence[ch3.BarrierScenario],
    cfg: Chapter8Config,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    boundary_samples: Dict[str, List[float]] = {name: [] for name in adapters if name != "Truth"}
    positivity_rates: Dict[str, float] = {}
    for name, adapter in adapters.items():
        if name == "Truth":
            continue
        negatives = 0
        total = 0
        for scn in scenarios:
            for tau in np.linspace(0.0, scn.T, cfg.boundary_tau_points):
                boundary_samples[name].append(abs(adapter.price(scn, scn.B_d, float(tau))))
            for S in np.linspace(scn.B_d, scn.S_max, cfg.positivity_S_points):
                for tau in np.linspace(0.0, scn.T, cfg.positivity_tau_points):
                    total += 1
                    if adapter.price(scn, float(S), float(tau)) < -1e-8:
                        negatives += 1
        positivity_rates[name] = negatives / max(total, 1)
    return boundary_samples, positivity_rates


def plot_boundary_positivity(
    boundary_samples: Dict[str, List[float]],
    positivity_rates: Dict[str, float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=True)
    names = list(boundary_samples.keys())
    safe_samples = [np.maximum(np.asarray(boundary_samples[name], dtype=float), 1e-16) for name in names]
    axes[0].boxplot(safe_samples, tick_labels=names, showfliers=False)
    axes[0].set_yscale("log")
    axes[0].set_title("Boundary residual diagnostics")
    axes[0].set_ylabel(r"$|V(B_d,\tau)|$")
    axes[0].grid(axis="y", alpha=0.28, linestyle="--")
    axes[1].bar(names, [positivity_rates[name] for name in names], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], alpha=0.84)
    axes[1].set_title("Positivity violation rate")
    axes[1].set_ylabel("Violation fraction")
    axes[1].grid(axis="y", alpha=0.28, linestyle="--")
    fig.suptitle("Figure 26. Boundary residual and positivity diagnostics", fontsize=15, y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def residual_diagnostics(
    adapters: Dict[str, BaseAdapter],
    scenarios: Sequence[ch3.BarrierScenario],
    cfg: Chapter8Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    quant_rows = []
    regional_rows = []
    for name, adapter in adapters.items():
        if name == "Truth":
            continue
        residuals = []
        residual_by_region = {"near_barrier": [], "near_strike": [], "smooth": [], "far_field": []}
        for scn in scenarios:
            S_vals = np.concatenate([
                np.linspace(scn.B_d + 0.002 * scn.K, min(scn.S_max, scn.B_d + 0.08 * scn.K), 8),
                np.linspace(max(scn.B_d + 1e-4, scn.K - 0.08 * scn.K), min(scn.S_max, scn.K + 0.08 * scn.K), 6),
                np.linspace(max(scn.B_d + 0.10 * scn.K, scn.K + 0.12 * scn.K), min(scn.S_max, 1.30 * scn.K), 6),
            ])
            S_vals = np.unique(np.clip(S_vals, scn.B_d + 1e-4, scn.S_max))
            for S in S_vals:
                for tau in np.linspace(max(0.05, 0.10 * scn.T), scn.T, 8):
                    res = finite_difference_residual(adapter, scn, float(S), float(tau))
                    residuals.append(res)
                    residual_by_region[region_label(scn, float(S))].append(res)
        q_row = {"Model": name}
        for q in cfg.residual_quantiles:
            q_row[f"q{int(round(100*q))}"] = float(np.quantile(residuals, q))
        quant_rows.append(q_row)
        for region, vals in residual_by_region.items():
            regional_rows.append(
                {
                    "Model": name,
                    "Region": region,
                    "Residual q95": float(np.quantile(vals, 0.95)) if vals else 0.0,
                }
            )
    return pd.DataFrame(quant_rows), pd.DataFrame(regional_rows)


def plot_residual_diagnostics(
    quant_df: pd.DataFrame,
    regional_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), constrained_layout=True)
    q_cols = [c for c in quant_df.columns if c.startswith("q")]
    for _, row in quant_df.iterrows():
        axes[0].plot(q_cols, row[q_cols].values, marker="o", linewidth=2.0, label=row["Model"])
    axes[0].set_yscale("log")
    axes[0].set_title("Residual quantile profile")
    axes[0].set_ylabel("Residual magnitude")
    axes[0].grid(alpha=0.28, linestyle="--")
    axes[0].legend(frameon=True, ncol=2)

    pivot = regional_df.pivot(index="Region", columns="Model", values="Residual q95")
    pivot = pivot.loc[["near_barrier", "near_strike", "smooth", "far_field"]]
    pivot.plot(kind="bar", ax=axes[1], alpha=0.84)
    axes[1].set_yscale("log")
    axes[1].set_title("Regional residual q95")
    axes[1].set_ylabel("Residual q95")
    axes[1].grid(axis="y", alpha=0.28, linestyle="--")
    axes[1].legend(frameon=True)
    fig.suptitle("Figure 27. Residual quantiles and regional residual diagnostics", fontsize=15, y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def validation_scorecard(
    adapters: Dict[str, BaseAdapter],
    truth: TruthAdapter,
    scenarios: Sequence[ch3.BarrierScenario],
    cfg: Chapter8Config,
) -> pd.DataFrame:
    acceptance = ch4.FullConfig().acceptance
    rows = []
    boundary_samples, positivity_rates = diagnostics_boundary_positivity(adapters, scenarios, cfg)
    quant_df, _ = residual_diagnostics(adapters, scenarios, cfg)
    residual_q95_map = {row["Model"]: row["q95"] for _, row in quant_df.iterrows()}
    for name, adapter in adapters.items():
        if name == "Truth":
            continue
        price_res = []
        gamma_near_barrier = []
        for scn in scenarios:
            for S in ch3.chapter3_eval_spots(scn):
                m = evaluate_model_point(adapter, truth, scn, float(S), scn.T)
                price_res.append(m["re_pct"])
            if scn.rho_d <= 0.03:
                gamma_near_barrier.append(evaluate_model_point(adapter, truth, scn, scn.S0, scn.T)["gamma_abs_err"])
        price_q95 = float(np.quantile(price_res, 0.95))
        gamma_q95 = float(np.quantile(gamma_near_barrier, 0.95)) if gamma_near_barrier else float("nan")
        barrier_max = float(np.max(boundary_samples[name]))
        positivity_rate = float(positivity_rates[name])
        residual_q95 = float(residual_q95_map[name])
        pass_price = price_q95 <= acceptance.max_price_q95_pct
        pass_gamma = gamma_q95 <= acceptance.max_gamma_q95_abs
        pass_barrier = barrier_max <= acceptance.max_barrier_abs
        pass_residual = residual_q95 <= acceptance.max_residual_q95
        pass_positive = positivity_rate <= acceptance.max_positivity_violation_rate
        rows.append(
            {
                "Model": name,
                "Price q95 (%)": price_q95,
                "Pass price": "PASS" if pass_price else "FAIL",
                "Near-barrier Gamma q95": gamma_q95,
                "Pass gamma": "PASS" if pass_gamma else "FAIL",
                "Barrier residual max": barrier_max,
                "Pass barrier": "PASS" if pass_barrier else "FAIL",
                "Residual q95": residual_q95,
                "Pass residual": "PASS" if pass_residual else "FAIL",
                "Positivity violation rate": positivity_rate,
                "Pass positivity": "PASS" if pass_positive else "FAIL",
                "Overall": "PASS" if all([pass_price, pass_gamma, pass_barrier, pass_residual, pass_positive]) else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def optional_certified_bounds(
    adapters: Dict[str, BaseAdapter],
    truth: TruthAdapter,
    scenarios: Sequence[ch3.BarrierScenario],
    output_dir: Path,
) -> None:
    rows = []
    for name, adapter in adapters.items():
        if name == "Truth":
            continue
        observed = []
        for scn in scenarios:
            observed.append(evaluate_model_point(adapter, truth, scn, scn.S0, scn.T)["ae"])
        rows.append(
            {
                "Model": name,
                "Observed median AE": float(np.median(observed)),
                "Heuristic bound": float(1.25 * np.quantile(observed, 0.95)),
            }
        )
    df = pd.DataFrame(rows)
    save_table(
        df,
        output_dir / "optional_certified_bounds.tex",
        output_dir / "optional_certified_bounds.csv",
        output_dir / "optional_certified_bounds.png",
        "Optional heuristic certified-style bounds versus observed errors.",
        "tab:optional_certified_bounds",
    )


def main() -> None:
    cfg = Chapter8Config()
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    ensure_dir(out)
    ensure_dir(out / "models")

    train_scenarios, valid_scenarios, report_scenarios = build_eval_panels(cfg)
    truth = TruthAdapter()
    fdm = FDMAdapter(cfg)
    pinn = PINNAdapter(cfg)

    train_df = sample_labeled_points(train_scenarios, cfg.train_label_samples_per_scenario, seed=cfg.seed + 501)
    valid_df = sample_labeled_points(valid_scenarios, cfg.valid_label_samples_per_scenario, seed=cfg.seed + 502)

    supervised_model = load_or_train_label_surrogate(
        cfg,
        "Supervised surrogate",
        train_df,
        valid_df,
        differential=False,
        output_dir=out / "models" / "supervised_surrogate",
    )
    differential_model = load_or_train_label_surrogate(
        cfg,
        "Differential surrogate",
        train_df,
        valid_df,
        differential=True,
        output_dir=out / "models" / "differential_surrogate",
    )

    adapters: Dict[str, BaseAdapter] = {
        "Truth": truth,
        "FDM": fdm,
        "Supervised": NeuralAdapter("Supervised", supervised_model, torch.device(cfg.device)),
        "Differential": NeuralAdapter("Differential", differential_model, torch.device(cfg.device)),
    }
    if pinn.available:
        adapters["PINN"] = pinn

    pd.DataFrame(
        {
            "split": ["train", "validation", "report"],
            "n": [len(train_scenarios), len(valid_scenarios), len(report_scenarios)],
        }
    ).to_csv(out / "panel_sizes.csv", index=False)

    table13 = build_main_pricing_table(cfg, adapters, truth, core_scenarios(cfg))
    save_table(
        table13,
        out / "table13_main_pricing_comparison.tex",
        out / "table13_main_pricing_comparison.csv",
        out / "table13_main_pricing_comparison.png",
        "Main pricing comparison against analytical truth.",
        "tab:main_pricing_comparison",
    )

    price_heatmaps = {
        name: heatmap_metric(adapters[name], truth, cfg, cfg.sigma_grid, cfg.rho_grid, metric="price")
        for name in ["FDM", "PINN", "Supervised", "Differential"]
    }
    plot_multi_model_heatmaps(
        price_heatmaps,
        cfg.rho_grid,
        cfg.sigma_grid,
        out / "figure22_pricing_error_heatmaps.png",
        "Figure 22. Pricing error heatmaps",
        "Price RE (%)",
    )

    zoom_heatmaps = {
        name: heatmap_metric(adapters[name], truth, cfg, cfg.sigma_grid, cfg.rho_zoom_grid, metric="price", zoom=True)
        for name in ["FDM", "PINN", "Supervised", "Differential"]
    }
    plot_multi_model_heatmaps(
        zoom_heatmaps,
        cfg.rho_zoom_grid,
        cfg.sigma_grid,
        out / "figure23_near_barrier_zoom_price_error_map.png",
        "Figure 23. Near-barrier zoomed price error map",
        "Near-barrier price RE (%)",
    )

    delta_heatmaps = {
        name: heatmap_metric(adapters[name], truth, cfg, cfg.sigma_grid, cfg.rho_grid, metric="delta")
        for name in ["FDM", "PINN", "Supervised", "Differential"]
    }
    plot_multi_model_heatmaps(
        delta_heatmaps,
        cfg.rho_grid,
        cfg.sigma_grid,
        out / "figure24_delta_comparison_heatmaps.png",
        "Figure 24. Delta comparison heatmaps",
        "Delta AE",
    )

    gamma_heatmaps = {
        name: heatmap_metric(adapters[name], truth, cfg, cfg.sigma_grid, cfg.rho_grid, metric="gamma")
        for name in ["FDM", "PINN", "Supervised", "Differential"]
    }
    plot_gamma_heatmaps_and_slice(
        gamma_heatmaps,
        adapters,
        truth,
        cfg,
        out / "figure25_gamma_heatmaps_and_slices.png",
    )

    boundary_samples, positivity_rates = diagnostics_boundary_positivity(adapters, report_scenarios, cfg)
    plot_boundary_positivity(boundary_samples, positivity_rates, out / "figure26_boundary_positivity_diagnostics.png")

    quant_df, regional_df = residual_diagnostics(adapters, report_scenarios, cfg)
    plot_residual_diagnostics(quant_df, regional_df, out / "figure27_residual_diagnostics.png")
    quant_df.to_csv(out / "residual_quantiles.csv", index=False)
    regional_df.to_csv(out / "regional_residuals.csv", index=False)

    table14 = validation_scorecard(adapters, truth, report_scenarios, cfg)
    save_table(
        table14,
        out / "table14_validation_scorecard.tex",
        out / "table14_validation_scorecard.csv",
        out / "table14_validation_scorecard.png",
        "Validation scorecard under the Chapter 5 protocol.",
        "tab:validation_scorecard",
    )

    if cfg.certified_bounds_enabled:
        optional_certified_bounds(adapters, truth, report_scenarios, out)

    summary = {
        "status": "chapter8 accuracy and error-control workflow prepared",
        "config": asdict(cfg),
        "outputs": {
            "table13": str((out / "table13_main_pricing_comparison.csv").resolve()),
            "figure22": str((out / "figure22_pricing_error_heatmaps.png").resolve()),
            "figure23": str((out / "figure23_near_barrier_zoom_price_error_map.png").resolve()),
            "figure24": str((out / "figure24_delta_comparison_heatmaps.png").resolve()),
            "figure25": str((out / "figure25_gamma_heatmaps_and_slices.png").resolve()),
            "figure26": str((out / "figure26_boundary_positivity_diagnostics.png").resolve()),
            "figure27": str((out / "figure27_residual_diagnostics.png").resolve()),
            "table14": str((out / "table14_validation_scorecard.csv").resolve()),
        },
    }
    with open(out / "chapter8_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Chapter 8 results workflow: accuracy, Greeks, boundary consistency")
    print("=" * 72)
    print("Exported:")
    print("  - Table 13: main pricing comparison")
    print("  - Figure 22: pricing heatmaps")
    print("  - Figure 23: near-barrier zoom price map")
    print("  - Figure 24: Delta heatmaps")
    print("  - Figure 25: Gamma heatmaps and slice")
    print("  - Figure 26: boundary residual and positivity diagnostics")
    print("  - Figure 27: residual diagnostics")
    print("  - Table 14: validation scorecard")
    print(f"Output directory: {out.resolve()}")


if __name__ == "__main__":
    main()
