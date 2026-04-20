from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt


def load_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["GridPosition", "ClassifiedPosition"])
    df = df.astype({"GridPosition": int, "ClassifiedPosition": int})

    circuit_type = pd.read_csv(Path("./data/circuit-type.csv"))
    df["circuitType"] = df["circuitId"].map(
        circuit_type.set_index("circuitId")["trackType"]
    )
    df["IsStreetCircuit"] = (df["circuitType"] == "street").astype("int")

    grid = df["GridPosition"].to_numpy(dtype=float)
    finish = df["ClassifiedPosition"].to_numpy(dtype=int)
    is_street = df["IsStreetCircuit"].to_numpy(dtype=int)

    grid_mean = grid.mean()
    grid_std = grid.std()
    grid_z = (grid - grid_mean) / grid_std

    finish_0idx = finish - 1  # OrderedLogistic expects 0-indexed categories

    print(f"Observations : {len(grid)}")
    print(f"Grid range   : {grid.min():.0f} – {grid.max():.0f}")
    print(f"Finish range : {finish.min()} – {finish.max()}")
    print(f"K categories : {finish_0idx.max() + 1}")

    return (
        grid_z,
        finish_0idx,
        grid_mean,
        grid_std,
        is_street
    )


def sample_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 42,
) -> az.InferenceData:
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
        )
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    return idata


def ilogit(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_probabilities(
    idata: az.InferenceData,
    grid_pos: np.ndarray,
    grid_mean: float,
    grid_std: float,
    is_street: np.ndarray | None = None,
) -> np.ndarray:
    posterior = idata.posterior
    beta_samples = posterior["beta"].values.flatten()
    cutpoints_samples = posterior["cutpoints"].values.reshape(
        -1, posterior["cutpoints"].shape[-1]
    )
    grid_z = (grid_pos - grid_mean) / grid_std
    eta_samples = beta_samples * grid_z
    if is_street is not None:
        beta_street_samples = posterior["beta_street"].values.flatten()
        eta_samples += beta_street_samples * is_street
    cum_probs = ilogit(cutpoints_samples - eta_samples[:, np.newaxis])
    zeros = np.zeros((cum_probs.shape[0], 1))
    ones = np.ones((cum_probs.shape[0], 1))
    cum_full = np.hstack([zeros, cum_probs, ones])
    pos_probs = np.diff(cum_full, axis=1)
    return pos_probs.mean(axis=0)


def print_diagnostics(
    idata: az.InferenceData, var_names: list[str] = ["beta", "cutpoints"]
) -> None:
    summary = az.summary(idata, var_names=var_names, round_to=3)
    print("\n--- Posterior summary ---")
    print(summary.to_string())

    rhats = summary["r_hat"]
    if (rhats > 1.05).any():
        print("\n⚠  Some Rhat values > 1.05 — consider more tuning steps.")
    else:
        print("\n✓ All Rhat ≤ 1.05 — chains have converged.")


def save_idata(idata: az.InferenceData, path: Path) -> None:
    idata.to_netcdf(path)


def load_idata(path: Path) -> az.InferenceData:
    return az.from_netcdf(path)


def plot_heatmap(prob_df: pd.DataFrame, out_path: Path | None = None) -> None:
    pivot = prob_df.pivot(
        index="ClassifiedPosition", columns="GridPosition", values="Probability"
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", origin="upper")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Grid position")
    ax.set_ylabel("Finish position")
    ax.set_title("P(finish = y | grid = x)")

    plt.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)


def generate_prob_df(
    idata: az.InferenceData,
    grid_mean: float,
    grid_std: float,
    include_street: bool = False,
) -> pd.DataFrame:
    grid_pos = np.arange(1, 21)  # Grid positions from 1 to 20
    prob_df = pd.DataFrame()
    grid_col = []
    classified_col = []
    is_street_col = []
    prob_col = []

    def generate_record(is_street: int | None = None):
        probs = generate_probabilities(idata, pos, grid_mean, grid_std, is_street)
        for k in range(len(probs)):
            grid_col.append(pos)
            classified_col.append(k + 1)  # Convert back to 1-indexed
            is_street_col.append(is_street) if is_street is not None else None
            prob_col.append(probs[k])


    for pos in grid_pos:
        if include_street:
            generate_record(is_street=1)
            generate_record(is_street=0)
        else:
            generate_record()
        
    prob_df["GridPosition"] = grid_col
    prob_df["ClassifiedPosition"] = classified_col
    if include_street:
        prob_df["IsStreet"] = is_street_col
    prob_df["Probability"] = prob_col

    return prob_df
