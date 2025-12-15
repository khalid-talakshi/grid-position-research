import fastf1
import pandas as pd
from fastf1.ergast import Ergast
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

ergast = Ergast()

years = [i for i in range(2018, 2025)]

df = None

fastf1.Cache.enable_cache("./data/cache", use_requests_cache=True)

def downalod_grid_position_data(year):
    logger.info(f"fetching data for {year}")
    schedule = fastf1.get_event_schedule(year)
    events = schedule.loc[schedule["EventFormat"] != "testing", "RoundNumber"]

    for event in events:
        logger.info(f"RoundNumber: {event}")
        event_race = fastf1.get_event(year, event).get_race()
        event_race.load(laps = False, telemetry=False, weather=False, messages=False)

        circuit_info = ergast.get_circuits(season = year, round = event)

        logger.debug(circuit_info)

        event_results = event_race.results

        grid_results = event_results.loc[
            (event_results["Status"] == "Finished") & (event_results["GridPosition"] > 0),
            ["GridPosition", "ClassifiedPosition", "DriverId"],
        ]
        grid_results["year"] = year
        grid_results["circuitId"] = circuit_info['circuitId'].item()

        if df is None:
            df = grid_results
        else:
            df = pd.concat([df, grid_results])
        
        time.sleep(5)

for year in years:
    downalod_grid_position_data(year)

if df is not None:
    df.to_csv("./data/grid-results.csv")

