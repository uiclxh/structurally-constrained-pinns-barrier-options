import json
import math
import time
from dataclasses import dataclass, asdict, field
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


# ============================================================
# Utilities
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_dataframe_as_png(df: pd.DataFrame, output_path: Path, title: str) -> None:
    def sanitize_cell(val):
        if isinstance(val, str):
            text = val
            replacements = {
                "$": "",
                "\\mathcal": "L",
                "\\mathrm": "",
                "\\mathbb": "E",
                "\\frac{1}{2}": "0.5",
                "\\frac12": "0.5",
                "\\tau": "tau",
                "\\sigma": "sigma",
                "\\beta": "beta",
                "\\Gamma": "Gamma",
                "\\hat": "",
                "\\left": "",
                "\\right": "",
                "\\!": "",
                "\\exp": "exp",
                "\\e": "e",
                "\\,": " ",
                "\\": "",
                "{": "",
                "}": "",
            }
            for src, dst in replacements.items():
                text = text.replace(src, dst)
            return text
        return val

    fig_h = 1.6 + 0.55 * len(df)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.axis("off")

    table_df = df.copy()
    for col in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[col]):
            table_df[col] = table_df[col].map(lambda x: f"{x:.6g}")
        else:
            table_df[col] = table_df[col].map(sanitize_cell)

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.8)
    tbl.scale(1.08, 1.35)

    plt.title(title, fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Configuration
# ============================================================
@dataclass
class ProblemConfig:
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.10
    q: float = 0.00
    sigma: float = 0.25
    B_d: float = 85.0
    S_max: float = 200.0

    @property
    def beta(self) -> float:
        return self.B_d / self.K

    @property
    def y_barrier(self) -> float:
        return math.log(self.beta)

    @property
    def y_max(self) -> float:
        return math.log(self.S_max / self.K)


@dataclass
class NetworkConfig:
    input_dim: int = 6               # y, tau, sigma, beta, r, q
    hidden_layers: int = 4
    width: int = 128
    activation: str = "SiLU"
    initialization: str = "Xavier uniform"
    output_positive_map: str = "Softplus"
    barrier_factor_kappa: float = 14.0
    use_parametric_input: bool = True


@dataclass
class SamplingConfig:
    n_interior: int = 4000
    n_barrier: int = 2000
    n_strike: int = 1000
    n_terminal: int = 1500
    n_farfield: int = 1000
    n_refine: int = 1500
    candidate_refine_pool: int = 8000
    resample_every: int = 10
    barrier_band_width: float = 0.06
    strike_band_halfwidth: float = 0.08


@dataclass
class OptimizationConfig:
    adam_epochs: int = 3000
    adam_lr: float = 1e-3
    adam_lr_gamma: float = 0.995
    lbfgs_steps: int = 250
    lbfgs_lr: float = 1.0
    lbfgs_history_size: int = 100
    grad_clip: float = 5.0


@dataclass
class LossWeightConfig:
    w_pde: float = 1.0
    w_terminal: float = 25.0
    w_farfield: float = 5.0
    w_monotonicity: float = 0.1
    w_gamma_smooth: float = 0.02
    use_monotonicity: bool = True
    use_gamma_smooth: bool = True


@dataclass
class AcceptanceConfig:
    max_price_q95_pct: float = 2.0
    max_gamma_q95_abs: float = 0.15
    max_barrier_abs: float = 1e-5
    max_residual_q95: float = 1e-2
    max_positivity_violation_rate: float = 0.0


@dataclass
class FullConfig:
    seed: int = 42
    device: str = "cpu"
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    loss: LossWeightConfig = field(default_factory=LossWeightConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)


# ============================================================
# Coordinate transforms and structural ansatz
# ============================================================
def normalize_param(x: torch.Tensor, center: float, scale: float) -> torch.Tensor:
    return (x - center) / scale


def build_parametric_input(
    y: torch.Tensor,
    tau: torch.Tensor,
    sigma: torch.Tensor,
    beta: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    feats = [
        y,
        tau,
        normalize_param(sigma, center=0.25, scale=0.15),
        normalize_param(beta, center=0.90, scale=0.10),
        normalize_param(r, center=0.05, scale=0.10),
        normalize_param(q, center=0.00, scale=0.05),
    ]
    return torch.cat(feats, dim=1)


def barrier_factor(y: torch.Tensor, y_barrier: torch.Tensor, kappa: float) -> torch.Tensor:
    d = torch.clamp(y - y_barrier, min=0.0)
    return 1.0 - torch.exp(-kappa * d)


def multi_scale_features(y: torch.Tensor, y_barrier: torch.Tensor, gamma: float = 12.0) -> torch.Tensor:
    d = torch.clamp(y - y_barrier, min=0.0)
    d_fast = 1.0 - torch.exp(-gamma * d)
    d_slow = 1.0 - torch.exp(-0.1 * gamma * d)
    return torch.cat([d, d_fast, d_slow], dim=1)


# ============================================================
# Neural core and barrier-preserving model
# ============================================================
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
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BarrierSurrogate(nn.Module):
    """
    Barrier-preserving neural ansatz:
        V_hat(y, tau; mu) = psi_bar(y; mu) * Softplus(A_theta(y, tau, mu))
    """
    def __init__(self, cfg: FullConfig):
        super().__init__()
        self.cfg = cfg
        core_input_dim = cfg.network.input_dim + 3  # + multiscale barrier features
        self.core = SmoothMLP(
            input_dim=core_input_dim,
            hidden_layers=cfg.network.hidden_layers,
            width=cfg.network.width,
        )

    def forward(
        self,
        y: torch.Tensor,
        tau: torch.Tensor,
        sigma: torch.Tensor,
        beta: torch.Tensor,
        r: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        y_barrier = torch.log(beta)
        base_in = build_parametric_input(y, tau, sigma, beta, r, q)
        ms_feats = multi_scale_features(y, y_barrier)
        x = torch.cat([base_in, ms_feats], dim=1)

        amp = self.core(x)
        psi = barrier_factor(y, y_barrier, self.cfg.network.barrier_factor_kappa)
        return psi * F.softplus(amp)


# ============================================================
# PDE residual and losses
# ============================================================
def pde_residual(
    model: BarrierSurrogate,
    y: torch.Tensor,
    tau: torch.Tensor,
    sigma: torch.Tensor,
    beta: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """
    Transformed PDE in y = ln(S/K), tau = T - t, for scaled value:
        u_tau = 0.5 sigma^2 u_yy + (r-q-0.5 sigma^2) u_y - r u
    residual := u_tau - RHS
    """
    y.requires_grad_(True)
    tau.requires_grad_(True)

    u = model(y, tau, sigma, beta, r, q)

    ones = torch.ones_like(u)
    u_tau = torch.autograd.grad(u, tau, grad_outputs=ones, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    drift = (r - q - 0.5 * sigma**2)
    rhs = 0.5 * sigma**2 * u_yy + drift * u_y - r * u
    return u_tau - rhs


def terminal_payoff_dimless(y: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.exp(y) - 1.0, min=0.0)


def farfield_target(y: torch.Tensor, tau: torch.Tensor, r: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    x = torch.exp(y)
    return x * torch.exp(-q * tau) - torch.exp(-r * tau)


def monotonicity_penalty(
    model: BarrierSurrogate,
    y: torch.Tensor,
    tau: torch.Tensor,
    sigma: torch.Tensor,
    beta: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    y.requires_grad_(True)
    u = model(y, tau, sigma, beta, r, q)
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return F.relu(-u_y).pow(2).mean()


def gamma_smoothness_penalty(
    model: BarrierSurrogate,
    y: torch.Tensor,
    tau: torch.Tensor,
    sigma: torch.Tensor,
    beta: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    y.requires_grad_(True)
    u = model(y, tau, sigma, beta, r, q)
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return u_yy.pow(2).mean()


def compute_loss_dict(model: BarrierSurrogate, batch: Dict[str, Tuple[torch.Tensor, ...]], cfg: FullConfig) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}

    y, tau, sigma, beta, r, q = batch["interior"]
    res = pde_residual(model, y, tau, sigma, beta, r, q)
    losses["pde"] = res.pow(2).mean()

    y, tau, sigma, beta, r, q = batch["terminal"]
    pred = model(y, tau, sigma, beta, r, q)
    target = terminal_payoff_dimless(y)
    losses["terminal"] = (pred - target).pow(2).mean()

    y, tau, sigma, beta, r, q = batch["farfield"]
    pred = model(y, tau, sigma, beta, r, q)
    target = farfield_target(y, tau, r, q)
    losses["farfield"] = (pred - target).pow(2).mean()

    if cfg.loss.use_monotonicity:
        y, tau, sigma, beta, r, q = batch["interior_aux"]
        losses["monotonicity"] = monotonicity_penalty(model, y, tau, sigma, beta, r, q)
    else:
        losses["monotonicity"] = torch.tensor(0.0, device=pred.device)

    if cfg.loss.use_gamma_smooth:
        y, tau, sigma, beta, r, q = batch["barrier_band"]
        losses["gamma_smooth"] = gamma_smoothness_penalty(model, y, tau, sigma, beta, r, q)
    else:
        losses["gamma_smooth"] = torch.tensor(0.0, device=pred.device)

    losses["total"] = (
        cfg.loss.w_pde * losses["pde"]
        + cfg.loss.w_terminal * losses["terminal"]
        + cfg.loss.w_farfield * losses["farfield"]
        + cfg.loss.w_monotonicity * losses["monotonicity"]
        + cfg.loss.w_gamma_smooth * losses["gamma_smooth"]
    )
    return losses


# ============================================================
# BAAC sampler
# ============================================================
class BAACSampler:
    """
    Barrier-Aware Adaptive Collocation:
      1) stratified initialization
      2) near-barrier and near-strike quotas
      3) residual-triggered refinement
      4) periodic resampling
    """
    def __init__(self, cfg: FullConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def _const_param_tensors(self, n: int) -> Tuple[torch.Tensor, ...]:
        p = self.cfg.problem
        sigma = torch.full((n, 1), p.sigma, device=self.device)
        beta = torch.full((n, 1), p.beta, device=self.device)
        r = torch.full((n, 1), p.r, device=self.device)
        q = torch.full((n, 1), p.q, device=self.device)
        return sigma, beta, r, q

    def sample_interior(self, n: int) -> Tuple[torch.Tensor, ...]:
        p = self.cfg.problem
        y = torch.empty((n, 1), device=self.device).uniform_(p.y_barrier + 1e-4, p.y_max)
        tau = torch.empty((n, 1), device=self.device).uniform_(0.0, p.T)
        sigma, beta, r, q = self._const_param_tensors(n)
        return y, tau, sigma, beta, r, q

    def sample_near_barrier(self, n: int) -> Tuple[torch.Tensor, ...]:
        p = self.cfg.problem
        w = self.cfg.sampling.barrier_band_width
        y = torch.empty((n, 1), device=self.device).uniform_(p.y_barrier + 1e-4, p.y_barrier + w)
        tau = torch.empty((n, 1), device=self.device).uniform_(0.0, p.T)
        sigma, beta, r, q = self._const_param_tensors(n)
        return y, tau, sigma, beta, r, q

    def sample_near_strike(self, n: int) -> Tuple[torch.Tensor, ...]:
        p = self.cfg.problem
        w = self.cfg.sampling.strike_band_halfwidth
        y = torch.empty((n, 1), device=self.device).uniform_(-w, w)
        y = torch.clamp(y, min=p.y_barrier + 1e-4, max=p.y_max)
        tau = torch.empty((n, 1), device=self.device).uniform_(0.0, p.T)
        sigma, beta, r, q = self._const_param_tensors(n)
        return y, tau, sigma, beta, r, q

    def sample_terminal(self, n: int) -> Tuple[torch.Tensor, ...]:
        p = self.cfg.problem
        y = torch.empty((n, 1), device=self.device).uniform_(p.y_barrier + 1e-4, p.y_max)
        tau = torch.zeros((n, 1), device=self.device)
        sigma, beta, r, q = self._const_param_tensors(n)
        return y, tau, sigma, beta, r, q

    def sample_farfield(self, n: int) -> Tuple[torch.Tensor, ...]:
        p = self.cfg.problem
        y = torch.full((n, 1), p.y_max, device=self.device)
        tau = torch.empty((n, 1), device=self.device).uniform_(0.0, p.T)
        sigma, beta, r, q = self._const_param_tensors(n)
        return y, tau, sigma, beta, r, q

    @torch.enable_grad()
    def sample_residual_refinement(self, model: BarrierSurrogate, n: int) -> Tuple[torch.Tensor, ...]:
        pool = self.cfg.sampling.candidate_refine_pool
        cand = self.sample_interior(pool)
        y, tau, sigma, beta, r, q = [t.clone().detach().requires_grad_(True) for t in cand]
        res = pde_residual(model, y, tau, sigma, beta, r, q).detach().abs().flatten()
        topk = torch.topk(res, k=n, largest=True).indices
        picked = [t.detach()[topk] for t in cand]
        return tuple(picked)

    def sample_batch(self, model: Optional[BarrierSurrogate] = None) -> Dict[str, Tuple[torch.Tensor, ...]]:
        batch = {
            "interior": self.sample_interior(self.cfg.sampling.n_interior),
            "interior_aux": self.sample_interior(max(1000, self.cfg.sampling.n_interior // 2)),
            "barrier_band": self.sample_near_barrier(self.cfg.sampling.n_barrier),
            "strike_band": self.sample_near_strike(self.cfg.sampling.n_strike),
            "terminal": self.sample_terminal(self.cfg.sampling.n_terminal),
            "farfield": self.sample_farfield(self.cfg.sampling.n_farfield),
        }
        if model is not None:
            batch["refine"] = self.sample_residual_refinement(model, self.cfg.sampling.n_refine)
        else:
            batch["refine"] = self.sample_interior(self.cfg.sampling.n_refine)
        return batch


# ============================================================
# Optimizer schedule and training
# ============================================================
def train_barrier_surrogate(
    cfg: FullConfig,
    output_dir: Path,
    dry_run_epochs: int = 10,
) -> Tuple[BarrierSurrogate, Dict[str, List[float]]]:
    """
    Chapter 4 workflow trainer.
    By default, you can use dry_run_epochs for a fast smoke test.
    """
    device = torch.device(cfg.device)
    sampler = BAACSampler(cfg, device)
    model = BarrierSurrogate(cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimization.adam_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.optimization.adam_lr_gamma)

    history = {"total": [], "pde": [], "terminal": [], "farfield": []}

    adam_epochs = min(cfg.optimization.adam_epochs, dry_run_epochs)
    for epoch in range(adam_epochs):
        model.train()
        batch = sampler.sample_batch(model if epoch % cfg.sampling.resample_every == 0 else None)

        optimizer.zero_grad()
        losses = compute_loss_dict(model, batch, cfg)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimization.grad_clip)
        optimizer.step()
        scheduler.step()

        history["total"].append(float(losses["total"].item()))
        history["pde"].append(float(losses["pde"].item()))
        history["terminal"].append(float(losses["terminal"].item()))
        history["farfield"].append(float(losses["farfield"].item()))

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=cfg.optimization.lbfgs_lr,
        max_iter=min(cfg.optimization.lbfgs_steps, 15),
        history_size=cfg.optimization.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad()
        batch = sampler.sample_batch(model)
        losses = compute_loss_dict(model, batch, cfg)
        losses["total"].backward()
        return losses["total"]

    model.train()
    lbfgs.step(closure)

    artifact_dir = output_dir / "artifact"
    ensure_dir(artifact_dir)
    torch.save(model.state_dict(), artifact_dir / "barrier_surrogate.pt")
    with open(artifact_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    return model, history


# ============================================================
# Acceptance rule skeleton
# ============================================================
def acceptance_check(metrics: Dict[str, float], cfg: FullConfig) -> bool:
    acc = cfg.acceptance
    checks = [
        metrics.get("price_q95_pct", np.inf) <= acc.max_price_q95_pct,
        metrics.get("gamma_q95_abs", np.inf) <= acc.max_gamma_q95_abs,
        metrics.get("barrier_abs_max", np.inf) <= acc.max_barrier_abs,
        metrics.get("residual_q95", np.inf) <= acc.max_residual_q95,
        metrics.get("positivity_violation_rate", np.inf) <= acc.max_positivity_violation_rate,
    ]
    return bool(all(checks))


# ============================================================
# Table 6 and Table 7 builders
# ============================================================
def build_table6_architecture(cfg: FullConfig) -> pd.DataFrame:
    rows = [
        ("Input state", r"$[y,\tau,\sigma,\beta,r,q]$", "Parametric transformed coordinate input"),
        ("Hidden layers", cfg.network.hidden_layers, "Depth of the smooth neural core"),
        ("Width", cfg.network.width, "Neurons per hidden layer"),
        ("Activation", cfg.network.activation, "Smooth activation for stable derivative propagation"),
        ("Initialization", cfg.network.initialization, "Weight initialization of linear layers"),
        ("Output map", cfg.network.output_positive_map, "Positive amplitude map inside the structural ansatz"),
        ("Barrier ansatz", r"$\hat V=\psi_{\mathrm{bar}}(x;\mu)\,\mathrm{Softplus}(A_\theta)$",
         "Hard enforcement of the absorbing boundary"),
        ("Barrier factor sharpness", cfg.network.barrier_factor_kappa, "Controls how fast the barrier factor activates"),
        ("Adam schedule", f"{cfg.optimization.adam_epochs} epochs @ lr={cfg.optimization.adam_lr}",
         "Warm-up training stage"),
        ("Adam decay", cfg.optimization.adam_lr_gamma, "Exponential learning-rate decay"),
        ("L-BFGS refinement", f"{cfg.optimization.lbfgs_steps} steps @ lr={cfg.optimization.lbfgs_lr}",
         "Second-stage refinement"),
        ("Interior collocation", cfg.sampling.n_interior, "Broad PDE enforcement"),
        ("Near-barrier quota", cfg.sampling.n_barrier, "Localized barrier-layer resolution"),
        ("Near-strike quota", cfg.sampling.n_strike, "Payoff kink neighborhood coverage"),
        ("Terminal samples", cfg.sampling.n_terminal, "Terminal condition enforcement"),
        ("Far-field samples", cfg.sampling.n_farfield, "Asymptotic boundary consistency"),
        ("Residual refinement quota", cfg.sampling.n_refine, "BAAC residual-triggered refinement"),
        ("Resampling frequency", f"every {cfg.sampling.resample_every} epochs", "Dynamic refresh of collocation"),
    ]
    return pd.DataFrame(rows, columns=["Component", "Setting", "Role"])


def build_table7_losses(cfg: FullConfig) -> pd.DataFrame:
    rows = [
        (
            "Absorbing boundary loss",
            r"$\text{none}$",
            "Not used",
            "The barrier is enforced structurally by the ansatz rather than by a soft penalty.",
        ),
        (
            "PDE residual",
            r"$\mathcal{L}_{\mathrm{PDE}}=\mathbb{E}\!\left[|u_\tau-\frac12\sigma^2u_{yy}-(r-q-\frac12\sigma^2)u_y+ru|^2\right]$",
            "Always on",
            "Enforces the transformed Black--Scholes PDE in the interior domain.",
        ),
        (
            "Terminal residual",
            r"$\mathcal{L}_{\mathrm{term}}=\mathbb{E}[|\hat u(y,0)-(\mathrm{e}^y-1)_+|^2]$",
            "Always on",
            "Matches the payoff at maturity.",
        ),
        (
            "Far-field consistency",
            r"$\mathcal{L}_{\mathrm{far}}=\mathbb{E}[|\hat u(y_{\max},\tau)-(\mathrm{e}^y\mathrm{e}^{-q\tau}-\mathrm{e}^{-r\tau})|^2]$",
            "Always on",
            "Stabilizes the upper boundary truncation.",
        ),
        (
            "Monotonicity regularizer",
            r"$\mathcal{L}_{\mathrm{mono}}=\mathbb{E}[\mathrm{ReLU}(-u_y)^2]$",
            "Optional",
            "Discourages violations of price monotonicity with respect to the underlying.",
        ),
        (
            "Gamma smoothness regularizer",
            r"$\mathcal{L}_{\Gamma}=\mathbb{E}[|u_{yy}|^2]$",
            "Optional",
            "Controls excessive curvature noise in difficult regions, especially near the barrier.",
        ),
        (
            "Total objective",
            r"$\mathcal{L}=w_{\mathrm{PDE}}\mathcal{L}_{\mathrm{PDE}}+w_{\mathrm{term}}\mathcal{L}_{\mathrm{term}}+w_{\mathrm{far}}\mathcal{L}_{\mathrm{far}}+w_{\mathrm{mono}}\mathcal{L}_{\mathrm{mono}}+w_{\Gamma}\mathcal{L}_{\Gamma}$",
            "Always on",
            "Combines structural fitting and optional regularization under fixed weights.",
        ),
    ]
    return pd.DataFrame(rows, columns=["Term", "Mathematical form", "Status", "Purpose"])


def export_table6_and_table7(cfg: FullConfig, output_dir: Path) -> None:
    ensure_dir(output_dir)

    table6 = build_table6_architecture(cfg)
    table7 = build_table7_losses(cfg)

    table6.to_csv(output_dir / "table6_architecture_hyperparameters.csv", index=False)
    table7.to_csv(output_dir / "table7_loss_terms.csv", index=False)

    with open(output_dir / "table6_architecture_hyperparameters.tex", "w", encoding="utf-8") as f:
        f.write(
            table6.to_latex(
                index=False,
                escape=False,
                caption="Neural architecture and hyperparameters of the deployable barrier surrogate.",
                label="tab:chapter4_architecture",
            )
        )

    with open(output_dir / "table7_loss_terms.tex", "w", encoding="utf-8") as f:
        f.write(
            table7.to_latex(
                index=False,
                escape=False,
                caption="Loss terms and their roles in the barrier neural-surrogate framework.",
                label="tab:chapter4_losses",
            )
        )

    save_dataframe_as_png(
        table6,
        output_dir / "table6_architecture_hyperparameters.png",
        "Table 6. Neural architecture and hyperparameters",
    )
    save_dataframe_as_png(
        table7,
        output_dir / "table7_loss_terms.png",
        "Table 7. Loss terms and their roles",
    )


# ============================================================
# Main
# ============================================================
def main():
    cfg = FullConfig()
    set_seed(cfg.seed)

    base_dir = Path("results/results_chapter4_only")
    ensure_dir(base_dir)

    export_table6_and_table7(cfg, base_dir)

    t0 = time.perf_counter()
    _, history = train_barrier_surrogate(cfg, base_dir, dry_run_epochs=8)
    elapsed = time.perf_counter() - t0

    pd.DataFrame(history).to_csv(base_dir / "chapter4_dry_run_history.csv", index=False)

    summary = {
        "status": "workflow scaffold completed",
        "device": cfg.device,
        "dry_run_seconds": elapsed,
        "artifact_dir": (base_dir / "artifact").as_posix(),
        "table6_csv": (base_dir / "table6_architecture_hyperparameters.csv").as_posix(),
        "table7_csv": (base_dir / "table7_loss_terms.csv").as_posix(),
    }
    with open(base_dir / "chapter4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Chapter 4 deployable barrier neural-surrogate framework")
    print("=" * 72)
    print("Exported:")
    print("  - Table 6 (architecture + hyperparameters)")
    print("  - Table 7 (loss terms + roles)")
    print("  - Dry-run training artifact")
    print(f"Output directory: {base_dir.as_posix()}")


if __name__ == "__main__":
    main()
