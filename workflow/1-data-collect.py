import requests
import json

url = "https://archive-api.open-meteo.com/v1/archive"

# Parameters
params = {
    "latitude": 48.0019,
    "longitude": 11.3442,
    "start_date": "1940-01-01",
    "end_date": "2026-03-15",
    "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall", "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]
}

response = requests.get(url, params={**params, "format": "csv"})
response.raise_for_status()

with open("./data/data_raw.csv", "w") as f:
    f.write(response.text.split("\n\n", 1)[1])

with open("./data/area_data.json", "w") as f:
    json.dump({
        "lat": params["latitude"],
        "lon": params["longitude"]
    }, f)