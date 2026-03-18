import pandas as pd

df = pd.read_csv("./data/data_raw.csv")

# Übersetze Spaltennamen
translation = {
    "time": "Zeit",
    "temperature_2m (°C)": "Temperatur_2m (°C)",
    "relative_humidity_2m (%)": "Relative_Luftfeuchtigkeit_2m (%)",
    "rain (mm)": "Regen (mm)",
    "snowfall (cm)": "Schneefall (cm)",
    "surface_pressure (hPa)": "Luftdruck (hPa)",
    "cloud_cover (%)": "Bewölkung (%)",
    "wind_speed_10m (km/h)": "Windgeschwindigkeit_10m (km/h)",
    "wind_direction_10m (°)": "Windrichtung_10m (°)",
    "wind_gusts_10m": "Windböen_10m (km/h)"
}

df = df.rename(columns=translation)

# Konvertiere die Zeitspalte zu einem Objekt mit Datentyp DateTime
df["Zeit"] = pd.to_datetime(df["Zeit"])

# Da am 01.01.1940 mehrere Werte unbekannt sind, wird dieser Tag entfernt
df = df.loc[(df['Zeit'] >= '1940-01-02')]

# Neuronale Netzwerke können periodische Strukturen wie Grad (0-360) oder Zeit (01.01-31.12; 0:00-24:00) schlecht verstehen
# Diese werden in Vektoren umgewandelt
import numpy as np

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

# Abspeicherung des konvertierten DatenFrames
df.to_csv("./data/data_converted.csv", index=False)