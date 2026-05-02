from model.shared import save_idata
from pathlib import Path
from model.shared import (
    load_data,
    sample_model,
    print_diagnostics,
    generate_prob_df,
)
import pymc as pm
import numpy as np


def build_model(
    grid_z: np.ndarray, finish_idx: np.ndarray, is_street: np.ndarray
) -> pm.Model:
    K = int(finish_idx.max()) + 1  # number of finish categories
    with pm.Model() as model:
        grid_z_data = pm.Data("grid_z", grid_z, dims="obs")
        is_street_data = pm.Data("is_street", is_street, dims="obs")

        # Priors
        beta = pm.Normal("beta", mu=0, sigma=1)
        beta_street = pm.Normal("beta_street", mu=0, sigma=1)
        cutpoints = pm.Normal(
            "cutpoints",
            mu=np.linspace(-2, 2, K - 1),
            sigma=1.5,
            transform=pm.distributions.transforms.ordered,
            initval=np.linspace(-2, 2, K - 1),
            shape=K - 1,
        )

        # Linear predictor
        eta = pm.Deterministic(
            "eta", beta * grid_z_data + beta_street * is_street_data, dims="obs"
        )

        # Likelihood
        pm.OrderedLogistic(
            "finish",
            eta=eta,
            cutpoints=cutpoints,
            observed=finish_idx,
            dims="obs",
        )

    return model


def run_model():
    print("Loading data...")
    grid_z, finish_0idx, grid_mean, grid_std, is_street_circuit, _ = load_data(
        Path("./data/grid-results.csv")
    )

    print("Building model...")
    model = build_model(grid_z, finish_0idx, is_street_circuit)
    print("Model built successfully!")

    print("Sampling model...")
    idata, ppc = sample_model(model)
    print("Model sampling completed!")

    prob_df = generate_prob_df(idata, grid_mean, grid_std, model_type="street")

    print(prob_df[prob_df["GridPosition"] == 1])

    print_diagnostics(idata, var_names=["beta", "beta_street", "cutpoints"])

    save_idata(idata, Path("./model/results/street-grid-model.nc"))
    save_idata(ppc, Path("./model/results/street-grid-model-ppc.nc"))


if __name__ == "__main__":
    run_model()
