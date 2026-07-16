import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

data = pd.read_csv("data/circuit-type.csv")
data["circuit_id"] = data["circuitId"]
data["is_street"] = data["trackType"] == "street"

data = data[["circuit_id", "is_street"]]


print(data.head())

database_url = os.environ["DB_URL"]
engine = create_engine(database_url)

data.to_sql("circuit_type", engine, if_exists="replace", index=False, schema="f1")
