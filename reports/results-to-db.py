import os
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.shared import generate_prob_df

data = pd.read_csv(PROJECT_ROOT / "data/grid-results.csv")
circuit_types = pd.read_csv(PROJECT_ROOT / "data/circuit-type.csv")
data["GridPosition"] = data["GridPosition"].astype("int")
data["ClassifiedPosition"] = data["ClassifiedPosition"].astype("int")
data["circuitType"] = data["circuitId"].map(
    circuit_types.set_index("circuitId")["trackType"]
)
data["IsStreetCircuit"] = (data["circuitType"] == "street").astype("int")
data["IsGroundEffectEra"] = (data["year"] >= 2022).astype("int")
data[["GridPosition", "ClassifiedPosition", "IsStreetCircuit", "IsGroundEffectEra"]]

grid_mean = data["GridPosition"].mean()
grid_std = data["GridPosition"].std()

split = "basic"
model_idata = az.from_netcdf(PROJECT_ROOT / f"model/results/{split}-grid-model.nc")
prob_df = generate_prob_df(model_idata, grid_mean, grid_std, model_type=split)

if split == "basic":
    prob_df["grid_position"] = prob_df["GridPosition"]
    prob_df["classified_position"] = prob_df["ClassifiedPosition"]
    prob_df["probability"] = prob_df["Probability"]
    prob_df["split_type"] = "ALL"
elif split == "era":
    prob_df["grid_position"] = prob_df["GridPosition"]
    prob_df["classified_position"] = prob_df["ClassifiedPosition"]
    prob_df["probability"] = prob_df["Probability"]
    prob_df["split_type"] = np.where(
        prob_df["IsGroundEffect"] == 1, "GroundEffectEra", "TurboHybridEra"
    )
elif split == "street":
    prob_df["grid_position"] = prob_df["GridPosition"]
    prob_df["classified_position"] = prob_df["ClassifiedPosition"]
    prob_df["probability"] = prob_df["Probability"]
    prob_df["split_type"] = np.where(
        prob_df["IsStreet"] == 1, "StreetCircuit", "ConventionalCircuit"
    )

prob_df = prob_df[["grid_position", "classified_position", "split_type", "probability"]]

database_url = os.environ["DB_URL"]
engine = create_engine(database_url)

prob_df.to_sql(
    "grid_position_probability", engine, if_exists="append", index=False, schema="f1"
)
