import requests

# Koordinaten Starnberg
lat = 48.0
lon = 11.3

# Hier kriegen wir die Daten her
url = "https://api.open-meteo.com/v1/forecast"

# Diese daten wollen wir
params = {
    "latitude": lat,
    "longitude": lon,
    "hourly": ["surface_pressure"], # Luftdruck
    "current": ["surface_pressure"],
    "past_hours": 3, # Luftdruck vor drei Stunden
    "forecast_hours": 0
}

data = requests.get(url, params=params).json()

luftdruck_jetzt = data["current"]["surface_pressure"]
luftdruck_vorher = data["hourly"]["surface_pressure"][0]

delta = luftdruck_jetzt - luftdruck_vorher

if delta <= -6:
    print("Regen/Sturm wahrscheinlich")
elif delta <= -2:
    print("Wetter verschlechtert sich")
elif delta < 2:
    print("Wetter bleibt ähnlich")
elif delta < 6:
    print("Wetter verbessert sich")
else:
    print("Das Wetter wird schön")
