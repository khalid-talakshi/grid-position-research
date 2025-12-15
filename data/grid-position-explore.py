import fastf1
from fastf1.ergast import Ergast

year = 2023
event = "Dutch Grand Prix"

schedule = fastf1.get_event_schedule(2023)

event_race = fastf1.get_event(year, event).get_race()
event_race.load(laps = False, telemetry=False, weather=False, messages=False)

event_results = event_race.results
event_results

event_info = event_race.event
event_info.RoundNumber.item()

ergast = Ergast()

ergast.get_circuits(season = year, round = event_info.RoundNumber.item())
