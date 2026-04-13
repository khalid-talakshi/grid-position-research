"""
grid_position_model.py
======================
Bayesian Ordinal Regression  —  Grid Position -> Finishing Position
Framework: PyMC

Model
-----
    finish_i  ~ OrderedLogistic(phi_i, cutpoints)

    phi_i      = beta_grid * grid_std_i
                 + u_driver[driver_i]
                 + u_team[team_i]
                 + u_circuit[circuit_i]

    beta_grid   ~ Normal(0, 1)
    cutpoints   ~ Normal(linspace(-2, 2, K-1), 1.5)   [ordered transform]
    u_*[j]      ~ Normal(0, sigma_*)
    sigma_*     ~ HalfNormal(0.5)

Expected CSV columns
--------------------
    grid      integer starting position  (1 = pole)
    finish    integer finishing position (1 = winner)
    driver    string label
    team      string label
    circuit   string label

Usage
-----
    python grid_position_model.py --data race_data.csv
    python grid_position_model.py --data race_data.csv --draws 2000 --tune 1500
    python grid_position_model.py --data race_data.csv --out-dir reports/
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bayesian ordinal regression: grid position -> finishing position"
    )
    p.add_argument(
        "--data",
        type=str,
        default="race_data.csv",
        help="Path to CSV file (default: race_data.csv)",
    )
    p.add_argument(
        "--draws",
        type=int,
        default=2000,
        help="Posterior draws per chain (default: 2000)",
    )
    p.add_argument(
        "--tune", type=int, default=1000, help="Tuning steps per chain (default: 1000)"
    )
    p.add_argument(
        "--chains", type=int, default=4, help="Number of MCMC chains (default: 4)"
    )
    p.add_argument(
        "--target-accept",
        type=float,
        default=0.9,
        help="NUTS target acceptance rate (default: 0.9)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory for output files (default: .)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading & encoding
# ---------------------------------------------------------------------------


def load_and_encode(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load CSV, validate columns, encode categoricals as 0-indexed integers,
    and z-score standardise grid position.

    Returns
    -------
    df   : enriched DataFrame with *_idx and grid_std columns added
    meta : sizes, category lists, and grid scaling constants
    """
    required = {"grid", "finish", "driver", "team", "circuit"}
    df = pd.read_csv(path)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for col in ("grid", "finish"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    if df["grid"].min() < 1 or df["finish"].min() < 1:
        raise ValueError("grid and finish positions must be >= 1")

    for col in ("driver", "team", "circuit"):
        cat = pd.Categorical(df[col])
        df[f"{col}_idx"] = cat.codes
        df[col] = cat  # keep Categorical so .cat.categories works

    grid_mean = df["grid"].mean()
    grid_sd = df["grid"].std()
    df["grid_std"] = (df["grid"] - grid_mean) / grid_sd

    K = int(df["finish"].max())

    meta = {
        "K": K,
        "n_drivers": df["driver_idx"].nunique(),
        "n_teams": df["team_idx"].nunique(),
        "n_circuits": df["circuit_idx"].nunique(),
        "grid_mean": grid_mean,
        "grid_sd": grid_sd,
        "driver_cats": df["driver"].cat.categories.tolist(),
        "team_cats": df["team"].cat.categories.tolist(),
        "circuit_cats": df["circuit"].cat.categories.tolist(),
    }

    print(
        f"[data]  {len(df):,} rows | "
        f"{meta['n_drivers']} drivers | {meta['n_teams']} teams | "
        f"{meta['n_circuits']} circuits | K={K} finishing positions"
    )
    return df, meta


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


def build_model(df: pd.DataFrame, meta: dict) -> pm.Model:
    """
    Construct the PyMC cumulative-logit ordinal regression model.

    The likelihood is:
        P(finish <= k | phi_i) = logistic(cutpoints[k] - phi_i)
        P(finish  = k | phi_i) = P(finish <= k) - P(finish <= k-1)

    A positive beta_grid means worse grid position (higher number) shifts
    the predicted finishing position distribution toward worse (higher) values.
    """
    K = meta["K"]

    # PyMC's OrderedLogistic expects 0-indexed categories
    finish_obs = df["finish"].values - 1

    coords = {
        "driver": meta["driver_cats"],
        "team": meta["team_cats"],
        "circuit": meta["circuit_cats"],
    }

    with pm.Model(coords=coords) as model:
        # -- Data containers -------------------------------------------------
        # Wrapping arrays in pm.Data allows swapping in new data for prediction
        # without rebuilding the model.
        grid_data = pm.Data("grid", df["grid_std"].values, dims="obs")
        driver_idx = pm.Data("driver_idx", df["driver_idx"].values, dims="obs")
        team_idx = pm.Data("team_idx", df["team_idx"].values, dims="obs")
        circuit_idx = pm.Data("circuit_idx", df["circuit_idx"].values, dims="obs")

        # -- Fixed effect: grid position -------------------------------------
        beta_grid = pm.Normal("beta_grid", mu=0, sigma=1)

        # -- Hierarchical random effects -------------------------------------
        # Each group (driver / team / circuit) gets a shared scale parameter,
        # achieving partial pooling: small groups borrow strength from larger ones.
        sigma_driver = pm.HalfNormal("sigma_driver", sigma=0.5)
        u_driver = pm.Normal("u_driver", mu=0, sigma=sigma_driver, dims="driver")

        sigma_team = pm.HalfNormal("sigma_team", sigma=0.5)
        u_team = pm.Normal("u_team", mu=0, sigma=sigma_team, dims="team")

        sigma_circuit = pm.HalfNormal("sigma_circuit", sigma=0.5)
        u_circuit = pm.Normal("u_circuit", mu=0, sigma=sigma_circuit, dims="circuit")

        # -- Linear predictor phi_i -----------------------------------------
        phi = (
            beta_grid * grid_data
            + u_driver[driver_idx]
            + u_team[team_idx]
            + u_circuit[circuit_idx]
        )

        # -- Ordered cutpoints alpha_1 < alpha_2 < ... < alpha_{K-1} --------
        # Spread initial mu values across [-2, 2] so each cutpoint starts near
        # a sensible region. PyMC's ordered transform enforces strict ordering.
        cutpoints = pm.Normal(
            "cutpoints",
            mu=np.linspace(-2, 2, K - 1),
            sigma=1.5,
            shape=K - 1,
            transform=pm.distributions.transforms.ordered,
        )

        # -- Likelihood ------------------------------------------------------
        pm.OrderedLogistic(
            "finish",
            eta=phi,
            cutpoints=cutpoints,
            observed=finish_obs,
        )

    return model


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_model(model: pm.Model, args: argparse.Namespace) -> az.InferenceData:
    with model:
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=args.seed,
            progressbar=True,
        )
        idata.extend(pm.sample_posterior_predictive(idata))
    return idata


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def run_diagnostics(idata: az.InferenceData, out_dir: Path) -> None:
    """Print posterior summary, flag R-hat issues, save trace plots."""
    key_vars = ["beta_grid", "sigma_driver", "sigma_team", "sigma_circuit"]

    summary = az.summary(idata, var_names=key_vars, round_to=3)
    print("\n-- Posterior summary " + "-" * 50)
    print(summary.to_string())

    bad_rhat = summary["r_hat"][summary["r_hat"] > 1.05]
    if not bad_rhat.empty:
        warnings.warn(
            f"R-hat > 1.05 detected for: {bad_rhat.index.tolist()} "
            "-- consider increasing --tune or --draws"
        )
    else:
        print("\n[ok]  All R-hat values <= 1.05")

    axes = az.plot_trace(idata, var_names=key_vars, compact=True)
    fig = axes.ravel()[0].get_figure()
    fig.tight_layout()
    path = out_dir / "trace_plot.png"
    fig.savefig(path, dpi=150)
    print(f"[plot]  Trace plot -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Posterior prediction
# ---------------------------------------------------------------------------


def posterior_finish_probs(
    idata: az.InferenceData,
    meta: dict,
    grid_pos: int,
    driver_idx: int = 0,
    team_idx: int = 0,
    circuit_idx: int = 0,
) -> np.ndarray:
    """
    Compute posterior predictive P(finish = k) for a single scenario.

    Parameters
    ----------
    grid_pos     : raw (unstandardised) grid position, e.g. 1 or 10
    driver_idx   : 0-indexed integer into the driver dimension
    team_idx     : 0-indexed integer into the team dimension
    circuit_idx  : 0-indexed integer into the circuit dimension

    Returns
    -------
    cell_probs : ndarray of shape (n_posterior_samples, K)
        Row i is a valid probability vector over K finishing positions.
    """
    grid_s = (grid_pos - meta["grid_mean"]) / meta["grid_sd"]
    post = idata.posterior

    def flat(name: str) -> np.ndarray:
        """Flatten (chain, draw, ...) -> (n_samples, ...)."""
        arr = post[name].values
        return arr.reshape(-1, *arr.shape[2:])

    beta = flat("beta_grid")  # (S,)
    cuts = flat("cutpoints")  # (S, K-1)
    u_d = flat("u_driver")[:, driver_idx]  # (S,)
    u_t = flat("u_team")[:, team_idx]  # (S,)
    u_c = flat("u_circuit")[:, circuit_idx]  # (S,)

    phi_s = beta * grid_s + u_d + u_t + u_c  # (S,)

    # Cumulative probs: P(finish <= k) = logistic(alpha_k - phi)
    cum = 1.0 / (1.0 + np.exp(-(cuts - phi_s[:, None])))  # (S, K-1)

    # Prepend 0 and append 1, then diff to get cell probabilities
    S = len(phi_s)
    cell_probs = np.diff(
        np.concatenate([np.zeros((S, 1)), cum, np.ones((S, 1))], axis=1),
        axis=1,
    )  # (S, K)

    return cell_probs


def print_prediction_table(idata: az.InferenceData, meta: dict) -> None:
    """Print P(finish=1) and P(top-5) for every grid position."""
    K = meta["K"]

    print(
        "\n-- Grid -> Finish prediction table (baseline driver/team/circuit) " + "-" * 5
    )
    print(f"{'Grid':>6}  {'P(P1)':>7}  {'89% HDI P(P1)':>17}  {'P(top 5)':>10}")
    print("-" * 48)

    for gp in range(1, K + 1):
        probs = posterior_finish_probs(idata, meta, grid_pos=gp)
        mean_p = probs.mean(axis=0)
        hdi = az.hdi(probs[:, 0], hdi_prob=0.89)
        top5 = mean_p[: min(5, K)].sum()
        print(
            f"{gp:>6}  {mean_p[0]:>7.3f}  "
            f"[{hdi[0]:.3f}, {hdi[1]:.3f}]       "
            f"{top5:>10.3f}"
        )


def plot_finish_distributions(
    idata: az.InferenceData,
    meta: dict,
    out_dir: Path,
    grid_positions: list[int] | None = None,
) -> None:
    """
    Bar charts of posterior mean P(finish = k) with 89% HDI error bars,
    one subplot per grid position.
    """
    K = meta["K"]
    if grid_positions is None:
        step = max(1, K // 5)
        grid_positions = list(range(1, K + 1, step))[:6]

    n_cols = min(3, len(grid_positions))
    n_rows = int(np.ceil(len(grid_positions) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharey=True,
        sharex=True,
    )
    axes = np.array(axes).flatten()
    positions = np.arange(1, K + 1)

    for ax, gp in zip(axes, grid_positions):
        probs = posterior_finish_probs(idata, meta, grid_pos=gp)
        mean_p = probs.mean(axis=0)
        hdis = np.array([az.hdi(probs[:, k], hdi_prob=0.89) for k in range(K)])
        yerr = np.stack([mean_p - hdis[:, 0], hdis[:, 1] - mean_p], axis=0)

        ax.bar(positions, mean_p, color="steelblue", alpha=0.75)
        ax.errorbar(
            positions,
            mean_p,
            yerr=yerr,
            fmt="none",
            color="black",
            capsize=2,
            linewidth=0.8,
        )
        ax.set_title(f"Grid {gp}", fontsize=11)
        ax.set_xlabel("Finishing position")
        ax.set_ylabel("Probability")

    for ax in axes[len(grid_positions) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Posterior  P(finish = k | grid)   [89 % HDI error bars]",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / "finish_distributions.png"
    fig.savefig(path, dpi=150)
    print(f"[plot]  Finish distributions -> {path}")
    plt.close(fig)


def plot_random_effects(idata: az.InferenceData, meta: dict, out_dir: Path) -> None:
    """
    Forest plot of driver random effects.
    Negative u_driver => driver tends to finish better than grid predicts.
    """
    fig, ax = plt.subplots(figsize=(8, max(4, meta["n_drivers"] * 0.4)))
    az.plot_forest(idata, var_names=["u_driver"], combined=True, hdi_prob=0.89, ax=ax)
    ax.axvline(0, color="tomato", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(
        "Driver random effects  (u_driver)\n"
        "Negative = outperforms grid position; Positive = underperforms",
        fontsize=10,
    )
    fig.tight_layout()
    path = out_dir / "driver_effects.png"
    fig.savefig(path, dpi=150)
    print(f"[plot]  Driver effects -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & encode data
    df, meta = load_and_encode(args.data)

    # 2. Build model
    model = build_model(df, meta)

    # 3. Sample
    print("\n[mcmc]  Sampling posterior ...")
    idata = sample_model(model, args)

    # 4. Save InferenceData (can be reloaded with az.from_netcdf)
    nc_path = out_dir / "idata.nc"
    idata.to_netcdf(str(nc_path))
    print(f"[save]  InferenceData -> {nc_path}")

    # 5. Diagnostics
    run_diagnostics(idata, out_dir)

    # 6. Predictions
    print_prediction_table(idata, meta)
    plot_finish_distributions(idata, meta, out_dir)

    # 7. Driver ability ranking
    if meta["n_drivers"] > 1:
        plot_random_effects(idata, meta, out_dir)

    print("\n[done]")


if __name__ == "__main__":
    main()
