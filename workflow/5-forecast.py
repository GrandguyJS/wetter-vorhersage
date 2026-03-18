import argparse
import os
import json
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--forecast", type=str, required=True, default=None)

args = parser.parse_args()

model_path = f"./models/{args.model}"
model_data_path = "." + "".join(model_path.split(".")[:-1]) + ".json"

if not os.path.exists(model_path) or not os.path.exists(model_data_path):
    raise Exception("Please supply a model and a model data file!")

with open(model_data_path, "r", encoding="utf-8") as f:
    model_data = json.load(f)
INPUT_WIDTH, LABEL_WIDTH, SHIFT, LABEL_COLUMNS = model_data["input_width"], model_data["label_width"], model_data["shift"], model_data["label_columns"]

with open("./data/area_data.json", "r") as f:
    data = json.load(f)

# Ortsdaten
lat = data["lat"]
lon = data["lon"]

# Initialisiere WindowGenerator
# Dieses Objekt stellt Trainings-, Evaluations- und Testdaten im richtigen Format zur Verfügung

from window import WindowGenerator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

window = WindowGenerator(
    input_width=INPUT_WIDTH, 
    label_width=LABEL_WIDTH, 
    shift=SHIFT, 

    label_columns=LABEL_COLUMNS, # Spalten, die das Model hervorsagen soll

    train_df=pd.read_csv("./data/train.csv"), # Trainingsdaten
    val_df=pd.read_csv("./data/validate.csv"), # Evaluationsdaten
    test_df=pd.read_csv("./data/test.csv"), # Testdaten

    train_mean=pd.read_csv("./data/train_mean.csv"),
    train_std=pd.read_csv("./data/train_std.csv")
)

# Trainingsfunktion
import tensorflow as tf
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    raise Exception("Please supply a model!")

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": lat,
    "longitude": lon,
    "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
}

if args.forecast == "now":
    params["forecast_hours"] = 0
    params["past_hours"] = INPUT_WIDTH
elif len(args.forecast) == 10:
    time = datetime.strptime(args.forecast, '%Y-%m-%d').date()
    params["end_date"] = time - timedelta(hours=24)
    params["start_date"] = time - timedelta(hours=INPUT_WIDTH)
else:
    raise("Invalid forecast parameter. Either 'now' or 'YYYY-mm-dd'!")

# Daten von OpenMeteo nehmen
response = requests.get(url, params=params)
data = response.json()["hourly"]

last_hour = datetime.strptime(data["time"][-1], "%Y-%m-%dT%H:%M")

print(f"Vorhersage von {last_hour + timedelta(hours=1)} bis {last_hour + timedelta(hours=SHIFT)}.")

df = pd.DataFrame(data)

# Datenverarbeitung wie in 1-3
# Spaltenübersetzung
translation = {
    "time": "Zeit",
    "temperature_2m": "Temperatur_2m (°C)",
    "relative_humidity_2m": "Relative_Luftfeuchtigkeit_2m (%)",
    "rain": "Regen (mm)",
    "snowfall": "Schneefall (cm)",
    "surface_pressure": "Luftdruck (hPa)",
    "cloud_cover": "Bewölkung (%)",
    "wind_speed_10m": "Windgeschwindigkeit_10m (km/h)",
    "wind_direction_10m": "Windrichtung_10m (°)",
    "wind_gusts_10m": "Windböen_10m (km/h)"
}

df = df.rename(columns=translation)

# Konvertiere die Zeitspalte zu einem Objekt mit Datentyp DateTime
df["Zeit"] = pd.to_datetime(df["Zeit"])

# Neuronale Netzwerke können periodische Strukturen wie Grad (0-360) oder Zeit (01.01-31.12; 0:00-24:00) schlecht verstehen
# Diese werden in Vektoren umgewandelt

wind_geschwindigkeit = df.pop("Windgeschwindigkeit_10m (km/h)")
wind_richtung = df.pop("Windrichtung_10m (°)")

df['Windgeschwindigkeit_10m_x (km/h)'] = wind_geschwindigkeit*np.cos(wind_richtung)
df['Windgeschwindigkeit_10m_y (km/h)'] = wind_geschwindigkeit*np.sin(wind_richtung)

windböen_geschwindigkeit = df.pop("Windböen_10m (km/h)")
df["Windböen_10m_x (km/h)"] = windböen_geschwindigkeit*np.cos(wind_richtung)
df["Windböen_10m_y (km/h)"] = windböen_geschwindigkeit*np.sin(wind_richtung)

# Konvertieren der Zeit in Sekunden
date_time = df.pop("Zeit")
timestamp_s = date_time.map(pd.Timestamp.timestamp)

# Sekundenanzahl
day = 24*60*60
year = (365.2425)*day

df['Tag_x'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Tag_y'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Jahr_x'] = np.cos(timestamp_s * (2 * np.pi / year))
df['Jahr_y'] = np.sin(timestamp_s * (2 * np.pi / year))

# Normalisierung
train_mean = pd.read_csv("./data/train_mean.csv")
train_std = pd.read_csv("./data/train_std.csv")

df = (df - train_mean["mean"].values) / train_std["std"].values

x = df.to_numpy()
x = np.expand_dims(x, axis=0)

prediction = model(x)
print(prediction[0, :, 0] * train_std["std"][0] + train_mean["mean"][0])

window.plot(
    model=model,
    plot_cols=window.label_columns,
    max_subplots=1,
    normed=False,
    inputs=x,
    show_history=False,
    show_y_labels=True
)