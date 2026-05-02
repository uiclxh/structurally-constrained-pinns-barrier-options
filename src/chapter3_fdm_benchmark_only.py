import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as si
from scipy.sparse.linalg import factorized


# ============================================================
# Chapter 3 only: High-Precision Implicit FDM Benchmark
# Standalone script extracted and micro-adjusted from the user's
# original barrier_option_study_final.py
# ============================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def time_call(fn, *args, repeats: int = 1, **kwargs):
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return out, best


# ============================================================
# 1. Problem setup and analytical benchmark
# ============================================================
@dataclass
class BarrierScenario:
    sigma: float
    rho_d: float
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.10
    delta: float = 0.0
    S_max: float = 200.0

    @property
    def B_d(self) -> float:
        return self.S0 * (1.0 - self.rho_d)

    @property
    def x_min(self) -> float:
        return self.B_d / self.K


CORE_SCENARIOS: List[BarrierScenario] = [
    BarrierScenario(sigma=0.15, rho_d=0.002),
    BarrierScenario(sigma=0.15, rho_d=0.150),
    BarrierScenario(sigma=0.25, rho_d=0.002),
    BarrierScenario(sigma=0.25, rho_d=0.150),
    BarrierScenario(sigma=0.40, rho_d=0.002),
    BarrierScenario(sigma=0.40, rho_d=0.150),
]


def down_and_out_call_rr(S: np.ndarray, K: float, H: float, T: float, r: float,
                         sigma: float, delta: float = 0.0) -> np.ndarray:
    """
    Reiner-Rubinstein continuous-monitoring down-and-out call, zero rebate,
    under the H <= K regime used in the paper.
    """
    S = np.asarray(S, dtype=float)
    out = np.zeros_like(S)
    alive = S > H
    if H > K:
        raise ValueError("This implementation assumes H <= K.")
    if not np.any(alive):
        return out
    Sa = S[alive]
    mu = (r - delta - 0.5 * sigma**2) / sigma**2
    x1 = np.log(Sa / K) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y1 = np.log(H**2 / (Sa * K)) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    A = Sa * np.exp(-delta * T) * si.norm.cdf(x1) - K * np.exp(-r * T) * si.norm.cdf(x1 - sigma * np.sqrt(T))
    B = (
        Sa * np.exp(-delta * T) * (H / Sa) ** (2 * (mu + 1)) * si.norm.cdf(y1)
        - K * np.exp(-r * T) * (H / Sa) ** (2 * mu) * si.norm.cdf(y1 - sigma * np.sqrt(T))
    )
    out[alive] = A - B
    return out


def rr_price_scalar(scn: BarrierScenario, S: Optional[float] = None, tau: Optional[float] = None) -> float:
    S_use = scn.S0 if S is None else S
    tau_use = scn.T if tau is None else tau
    return float(
        down_and_out_call_rr(
            S=np.array([S_use]),
            K=scn.K,
            H=scn.B_d,
            T=tau_use,
            r=scn.r,
            sigma=scn.sigma,
            delta=scn.delta,
        )[0]
    )


def numerical_delta_gamma_rr(scn: BarrierScenario, S: float, tau: float, h: float = 1e-3) -> Tuple[float, float]:
    f_plus = rr_price_scalar(scn, S=S + h, tau=tau)
    f_minus = rr_price_scalar(scn, S=S - h, tau=tau)
    f0 = rr_price_scalar(scn, S=S, tau=tau)
    delta = (f_plus - f_minus) / (2 * h)
    gamma = (f_plus - 2 * f0 + f_minus) / (h**2)
    return float(delta), float(gamma)


# ============================================================
# 2. Chapter 3 benchmark FDM: log-space aligned Rannacher-CN
# ============================================================
def upper_bc(S_max: float, K: float, tau: float, r: float, delta: float = 0.0) -> float:
    return S_max * np.exp(-delta * tau) - K * np.exp(-r * tau)


def build_logspace_operator(Nx: int, dz: float, sigma: float, r: float, delta: float = 0.0):
    a = 0.5 * sigma**2
    b = r - delta - 0.5 * sigma**2
    lower = a / dz**2 - b / (2 * dz)
    diag = -2 * a / dz**2 - r
    upper = a / dz**2 + b / (2 * dz)
    N = Nx - 1
    L = sp.diags(
        diagonals=[np.full(N - 1, lower), np.full(N, diag), np.full(N - 1, upper)],
        offsets=[-1, 0, 1],
        format="csc",
    )
    return L, lower, upper


def fdm_solve_rannacher_cn_grid(scn: BarrierScenario, Nt: int = 2000, Nx: int = 800) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same solver as the original Chapter 3 benchmark, but returns the full grid
    so that price/Delta/Gamma diagnostics can be computed for the revised table.
    """
    z_min = np.log(scn.B_d)
    z_max = np.log(scn.S_max)
    dz = (z_max - z_min) / Nx
    dt = scn.T / Nt

    z_grid = np.linspace(z_min, z_max, Nx + 1)
    S_grid = np.exp(z_grid)

    V = np.maximum(S_grid - scn.K, 0.0)
    tau = 0.0

    L, _, upper = build_logspace_operator(Nx, dz, scn.sigma, scn.r, scn.delta)
    N = Nx - 1
    I = sp.eye(N, format="csc")

    # 4 half-steps backward Euler (Rannacher smoothing)
    dt_half = 0.5 * dt
    A_half = (I - dt_half * L).tocsc()
    solve_half = factorized(A_half)

    for _ in range(4):
        tau_new = tau + dt_half
        rhs = V[1:-1].copy()
        rhs[-1] += dt_half * upper * upper_bc(scn.S_max, scn.K, tau_new, scn.r, scn.delta)
        V[1:-1] = solve_half(rhs)
        V[0] = 0.0
        V[-1] = upper_bc(scn.S_max, scn.K, tau_new, scn.r, scn.delta)
        tau = tau_new

    # Crank-Nicolson
    A_cn = (I - 0.5 * dt * L).tocsc()
    B_cn = (I + 0.5 * dt * L).tocsc()
    solve_cn = factorized(A_cn)

    for _ in range(max(Nt - 2, 0)):
        tau_new = tau + dt
        rhs = B_cn.dot(V[1:-1])
        g_upper_n = upper_bc(scn.S_max, scn.K, tau, scn.r, scn.delta)
        g_upper_np1 = upper_bc(scn.S_max, scn.K, tau_new, scn.r, scn.delta)
        rhs[-1] += 0.5 * dt * upper * (g_upper_n + g_upper_np1)
        V[1:-1] = solve_cn(rhs)
        V[0] = 0.0
        V[-1] = g_upper_np1
        tau = tau_new

    return S_grid, V


def fdm_price_rannacher_cn(scn: BarrierScenario, Nt: int = 2000, Nx: int = 800) -> float:
    S_grid, V = fdm_solve_rannacher_cn_grid(scn, Nt=Nt, Nx=Nx)
    return float(np.interp(scn.S0, S_grid, V))


# ============================================================
# 2A. Chapter 3 upgraded benchmark helpers
# ============================================================
def chapter3_eval_spots(scn: BarrierScenario) -> np.ndarray:
    """
    Evaluation panel for Chapter 3 benchmark verification.
    Covers near-barrier, near-strike, current state, and smoother region.
    """
    candidates = [
        scn.B_d + 0.01 * scn.K,
        scn.B_d + 0.03 * scn.K,
        scn.B_d + 0.06 * scn.K,
        max(scn.B_d + 0.08 * scn.K, 0.95 * scn.K),
        scn.S0,
        scn.K,
        1.10 * scn.K,
        1.25 * scn.K,
    ]
    pts = sorted(set(float(s) for s in candidates if scn.B_d < s <= scn.S_max))
    return np.array(pts, dtype=float)


def fdm_local_delta_gamma(S_grid: np.ndarray, V_grid: np.ndarray, S_query: float, window: int = 5) -> Tuple[float, float]:
    """
    Estimate Delta and Gamma at S_query using a local quadratic fit on the FDM grid.
    """
    idx = int(np.searchsorted(S_grid, S_query))
    n = len(S_grid)

    left = max(0, idx - window // 2 - 1)
    right = min(n, left + window)
    left = max(0, right - window)

    xs = S_grid[left:right]
    ys = V_grid[left:right]

    if len(xs) < 3:
        i = max(1, min(n - 2, idx))
        delta = (V_grid[i + 1] - V_grid[i - 1]) / (S_grid[i + 1] - S_grid[i - 1])
        gamma = 2.0 * (
            (V_grid[i + 1] - V_grid[i]) / (S_grid[i + 1] - S_grid[i])
            - (V_grid[i] - V_grid[i - 1]) / (S_grid[i] - S_grid[i - 1])
        ) / (S_grid[i + 1] - S_grid[i - 1])
        return float(delta), float(gamma)

    a, b, _ = np.polyfit(xs, ys, deg=2)
    delta = 2.0 * a * S_query + b
    gamma = 2.0 * a
    return float(delta), float(gamma)


def save_dataframe_as_png(df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig_h = 1.8 + 0.55 * len(df)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    ax.axis("off")

    table_df = df.copy()
    for col in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[col]):
            table_df[col] = table_df[col].map(lambda x: f"{x:.6f}")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.2)
    tbl.scale(1.05, 1.35)

    plt.title(title, fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 3. Chapter 3 benchmark table
# ============================================================
def run_fdm_convergence(output_dir: Path) -> pd.DataFrame:
    """
    Upgraded Chapter 3 benchmark verification table, using the exact
    scenarios and solver settings from the original script.

    Output columns:
      Nx, Nt,
      Max Abs Error,
      Median Relative Error (%),
      95th Relative Error (%),
      Delta AE,
      Gamma AE,
      Runtime per Contract (s)
    """
    grid_list = [(200, 500), (400, 1000), (800, 2000), (1200, 3000), (1600, 4000)]

    summary_rows = []
    detail_rows = []

    for Nx, Nt in grid_list:
        abs_errors = []
        rel_errors = []
        delta_abs_errors = []
        gamma_abs_errors = []
        runtimes = []

        for scn in CORE_SCENARIOS:
            (S_grid, V_grid), runtime = time_call(
                fdm_solve_rannacher_cn_grid, scn, Nt=Nt, Nx=Nx, repeats=1
            )
            runtimes.append(runtime)

            # price diagnostics over evaluation panel
            S_eval = chapter3_eval_spots(scn)
            V_true = down_and_out_call_rr(
                S=S_eval,
                K=scn.K,
                H=scn.B_d,
                T=scn.T,
                r=scn.r,
                sigma=scn.sigma,
                delta=scn.delta,
            )
            V_fdm = np.interp(S_eval, S_grid, V_grid)

            abs_err = np.abs(V_fdm - V_true)
            rel_err = 100.0 * abs_err / (np.abs(V_true) + 1e-12)

            abs_errors.extend(abs_err.tolist())
            rel_errors.extend(rel_err.tolist())

            # Greek diagnostics at S0
            delta_true, gamma_true = numerical_delta_gamma_rr(scn, scn.S0, scn.T, h=1e-3)
            delta_fdm, gamma_fdm = fdm_local_delta_gamma(S_grid, V_grid, scn.S0)

            delta_abs_errors.append(abs(delta_fdm - delta_true))
            gamma_abs_errors.append(abs(gamma_fdm - gamma_true))

            detail_rows.append({
                "Nx": Nx,
                "Nt": Nt,
                "Sigma": scn.sigma,
                "Rho_d": scn.rho_d,
                "B_d": scn.B_d,
                "Price True @S0": rr_price_scalar(scn),
                "Price FDM @S0": float(np.interp(scn.S0, S_grid, V_grid)),
                "Delta True @S0": delta_true,
                "Delta FDM @S0": delta_fdm,
                "Gamma True @S0": gamma_true,
                "Gamma FDM @S0": gamma_fdm,
                "Runtime (s)": runtime,
            })

        summary_rows.append({
            "Nx": Nx,
            "Nt": Nt,
            "Max Abs Error": float(np.max(abs_errors)),
            "Median Relative Error (%)": float(np.median(rel_errors)),
            "95th Relative Error (%)": float(np.quantile(rel_errors, 0.95)),
            "Delta AE": float(np.median(delta_abs_errors)),
            "Gamma AE": float(np.median(gamma_abs_errors)),
            "Runtime per Contract (s)": float(np.mean(runtimes)),
        })

    df = pd.DataFrame(summary_rows)
    df_detail = pd.DataFrame(detail_rows)

    # CSV outputs
    df.to_csv(output_dir / "fdm_convergence.csv", index=False)
    df_detail.to_csv(output_dir / "fdm_convergence_detail.csv", index=False)

    # LaTeX table
    with open(output_dir / "fdm_convergence.tex", "w", encoding="utf-8") as f:
        f.write(
            df.to_latex(
                index=False,
                escape=False,
                caption=(
                    "FDM convergence and benchmark verification across the six core scenarios. "
                    "Price errors are computed on the scenario evaluation panel, while "
                    "Delta and Gamma errors are computed at $S_0$ and aggregated across scenarios."
                ),
                label="tab:fdm_convergence",
                float_format=lambda x: f"{x:.6f}",
            )
        )

    # PNG table for quick insertion in drafts
    save_dataframe_as_png(
        df,
        output_dir / "fdm_convergence.png",
        "FDM Convergence and Benchmark Verification",
    )

    return df


# ============================================================
# 4. Script entry
# ============================================================
def main():
    output_dir = Path("results_chapter3_only")
    ensure_dir(output_dir)

    print("=" * 72)
    print("Chapter 3 only: FDM convergence and benchmark verification")
    print("=" * 72)

    df = run_fdm_convergence(output_dir)

    print("\n[Done] Chapter 3 benchmark table:\n")
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
