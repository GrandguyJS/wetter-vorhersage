# Der Luftdruck hat einen viel größeren numerischen Wert als die Temperatur
# Dieser Wert wird somit stärker im neuronalen Netzwerk vertreten, was zu verfälschten Prognosen führen kann
# Durch Normalisierung der Daten um 0 mit einer ähnlichen Amplitude kann dieser Effekt verringert werden

# Normalisierung = (Wert - Durchschnitt) / Standardabweichung

# Dabei ist es wichtig, dass nur Werte vom Trainingsdatenset in die Normalisierung kommen
# Deswegen wird die Aufteilung zuvor gemacht

import pandas as pd

df = pd.read_csv("./data/data_ready_de.csv")

# 70-20-10 Aufteilung der Trainings-, Validierungs- und Testdatensets
n = len(df)
train_df_unnormalized = df[0:int(n*0.7)]
val_df_unnormalized = df[int(n*0.7):int(n*0.9)]
test_df_unnormalized = df[int(n*0.9):]

# Durchschnitt und Standardabweichung berechnen
train_mean = train_df_unnormalized.mean()
train_std = train_df_unnormalized.std()

train_mean.to_frame(name="mean").to_csv("./data/train_mean.csv", index=False)
train_std.to_frame(name="std").to_csv("./data/train_std.csv", index=False)

train_df_normalized = (train_df_unnormalized - train_mean) / train_std
val_df_normalized = (val_df_unnormalized - train_mean) / train_std
test_df_normalized = (test_df_unnormalized - train_mean) / train_std

# Vergleiche durch Grafik
import matplotlib.pyplot as plt
plt.style.use("dark_background")

unnormalisiert = pd.DataFrame({
    "min": train_df_unnormalized.iloc[:, :6].min(),
    "max": train_df_unnormalized.iloc[:, :6].max(),
})

normalisiert = pd.DataFrame({
    "min": train_df_normalized.iloc[:, :6].min(),
    "max": train_df_normalized.iloc[:, :6].max(),
})

plt.figure(figsize=(28, 7))

plt.subplot(1, 2, 1)
unnormalisiert.plot(kind="bar", ax=plt.gca(), title="Min und Max unnormalisierte Daten", rot=45)
plt.ylabel("Wert")
plt.xlabel("Spalten")

plt.subplot(1, 2, 2)
normalisiert.plot(kind="bar", ax=plt.gca(), title="Min und Max normalisierte Daten", rot=45)
plt.ylabel("Wert")
plt.xlabel("Spalten")

plt.tight_layout()
plt.show()

# Speicherung der normalisierten Datensets
train_df_normalized.to_csv("./data/train.csv", index=False)
val_df_normalized.to_csv("./data/validate.csv", index=False)
test_df_normalized.to_csv("./data/test.csv", index=False)