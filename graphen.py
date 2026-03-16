# Imports
import pandas as pd # lese .csv Dateien
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

plt.style.use("dark_background")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

"""
# Lade die Daten

df = pd.read_csv("./data/data_raw.csv")
print(df.agg(["min", "max"]))

# Verarbeite Daten
df["time"] = pd.to_datetime(df["time"])

df_neu = df.drop(columns=["precipitation (mm)", "snow_depth (m)"])
df_neu = df_neu.loc[(df_neu['time'] >= '1940-01-02')]

# Speichere überarbeitete Daten
df_neu.to_csv("./data/data_processed.csv", index=False)
"""
"""
df = pd.read_csv("./data/data_processed.csv")

# Windgrad --> Windvektor
wind_geschwindigkeit = df.pop('wind_speed_10m (km/h)')
max_wind_geschwindigkeit = df.pop('wind_gusts_10m (km/h)') # Böen

# Convert to radians.
wind_geschwindigkeit_rad = df.pop('wind_direction_10m (°)')*np.pi / 180

# Berechne x- und y-Anteil des Windes
df['wind_speed_10m_x (km/h)'] = wind_geschwindigkeit*np.cos(wind_geschwindigkeit_rad)
df['wind_speed_10m_y (km/h)'] = wind_geschwindigkeit*np.sin(wind_geschwindigkeit_rad)

# Berechne x- und y-Anteil der Wind-Böen
df['wind_gusts_10m_x'] = max_wind_geschwindigkeit*np.cos(wind_geschwindigkeit_rad)
df['wind_gusts_10m_y'] = max_wind_geschwindigkeit*np.sin(wind_geschwindigkeit_rad)


# Datum und Uhrzeit --> trigonometrischr Verlauf
date_time = pd.to_datetime(df.pop("time"))
timestamp_s = date_time.map(pd.Timestamp.timestamp)

# Sekundenanzahl
day = 24*60*60
year = (365.2425)*day

df['day_x'] = np.cos(timestamp_s * (2 * np.pi / day))
df['day_y'] = np.sin(timestamp_s * (2 * np.pi / day))
df['year_x'] = np.cos(timestamp_s * (2 * np.pi / year))
df['year_y'] = np.sin(timestamp_s * (2 * np.pi / year))

df.to_csv("./data/data_ready.csv", index=False)
"""
"""
df = pd.read_csv("./data/data_ready.csv")

translation = {
    "temperature_2m (°C)": "Temperatur_2m (°C)",
    "relative_humidity_2m (%)": "Relative_Luftfeuchtigkeit_2m (%)",
    "rain (mm)": "Regen (mm)",
    "snowfall (cm)": "Schneefall (cm)",
    "surface_pressure (hPa)": "Luftdruck (hPa)",
    "cloud_cover (%)": "Bewölkung (%)",
    "wind_speed_10m_x (km/h)": "Windgeschwindigkeit_10m_x (km/h)",
    "wind_speed_10m_y (km/h)": "Windgeschwindigkeit_10m_y (km/h)",
    "wind_gusts_10m_x": "Windböen_10m_x (km/h)",
    "wind_gusts_10m_y": "Windböen_10m_y (km/h)",
    "day_x": "Tag_x",
    "day_y": "Tag_y",
    "year_x": "Jahr_x",
    "year_y": "Jahr_y"
}

df = df.rename(columns=translation)
df.to_csv("./data/data_ready_de.csv", index=False)
"""
"""
df = pd.read_csv("./data/data_ready_de.csv")

# Aufteilung in Trainieren, Validieren und Testen
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

# Anzahl Spalten
num_features = df.shape[1]

# Normalisiere die Daten --> Circa gleiche Amplitude mit Durchschnitt bei 0
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

train_df.to_csv("./data/train_normalized.csv", index=False)
val_df.to_csv("./data/val_normalized.csv", index=False)
test_df.to_csv("./data/test_normalized.csv", index=False)
"""
"""
train_df = pd.read_csv("./data/train_normalized.csv")
val_df = pd.read_csv("./data/val_normalized.csv")
test_df = pd.read_csv("./data/test_normalized.csv")

df_ready = pd.read_csv("./data/data_ready_de.csv")

df_ready_num = df_ready.select_dtypes(include="number")
df_norm_num = train_df.select_dtypes(include="number")

ready_stats = pd.DataFrame({
    "min": df_ready_num.iloc[:, :6].min(),
    "max": df_ready_num.iloc[:, :6].max(),
})

norm_stats = pd.DataFrame({
    "min": df_norm_num.iloc[:, :6].min(),
    "max": df_norm_num.iloc[:, :6].max(),
})

plt.figure(figsize=(14, 7))
ready_stats.plot(kind="bar", figsize=(14, 7), title="Min und Max unnormalisierte Daten", rot=45)
plt.ylabel("Wert")
plt.xlabel("Spalten")
plt.tight_layout()
plt.savefig("./graphs/data_ready_de_min_max.png", dpi=200)
plt.close()

# Graph 2
plt.figure(figsize=(14, 7))
norm_stats.plot(kind="bar", figsize=(14, 7), title="Min and Max normalisierte Daten", rot=45)
plt.ylabel("Wert")
plt.xlabel("Spalten")
plt.tight_layout()
plt.savefig("./graphs/train_normalized_min_max.png", dpi=200)
plt.close()
"""

# Get norm factor
df = pd.read_csv("./data/data_ready_de.csv")
n = len(df)

train_df = df[0:int(n*0.7)]

# Normalisiere die Daten --> Circa gleiche Amplitude mit Durchschnitt bei 0
train_mean = train_df.mean()
train_std = train_df.std()

train_df = pd.read_csv("./data/train_normalized.csv")
val_df = pd.read_csv("./data/val_normalized.csv")
test_df = pd.read_csv("./data/test_normalized.csv")

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    return inputs, labels
  
  def plot(self, model=None, plot_col='Temperatur_2m (°C)', max_subplots=3, normed=True):
    inputs, labels = self.example

    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col} {"[normalisiert]" if normed else ""}')
      if normed:
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                label='Vergangenheit', marker='.', zorder=-10)
      else:
        plt.plot(self.input_indices, inputs[n, :, plot_col_index] * train_std.iloc[plot_col_index] + train_mean.iloc[plot_col_index],
                label='Vergangenheit', marker='.', zorder=-10)

      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue
      
      if normed:
        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Tatsächlich', c='#2ca02c', s=64)
      else:
        plt.scatter(self.label_indices, labels[n, :, label_col_index] * train_std.iloc[label_col_index] + train_mean.iloc[label_col_index],
                    edgecolors='k', label='Tatsächlich', c='#2ca02c', s=64)
      if model is not None:
        predictions = model(inputs)
        if normed:
          plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                      marker='X', edgecolors='k', label='Hervorsagen',
                      c='#ff7f0e', s=64)
        else:
          plt.scatter(self.label_indices, predictions[n, :, label_col_index] * train_std.iloc[label_col_index] + train_mean.iloc[label_col_index],
                      marker='X', edgecolors='k', label='Hervorsagen',
                      c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()

    plt.xlabel('Zeit [h]')
    plt.xticks(np.arange(0, self.total_window_size+1, 24))
    plt.show()

  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds
  
  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    # Get and cache an example batch of `inputs, labels` for plotting.
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

MAX_EPOCHS = 20
model_path = "model_complex.keras"

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
                                                    
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_path,
    monitor="val_loss",
    save_best_only=True
  )

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping, checkpoint])
  return history

# LSTM multi output - one shot 24h output
        
OUT_STEPS = 24

multi_window = WindowGenerator(
    input_width=120,
    label_width=OUT_STEPS,
    shift=OUT_STEPS,
    label_columns=["Temperatur_2m (°C)", "Relative_Luftfeuchtigkeit_2m (%)", "Regen (mm)", "Schneefall (cm)", "Luftdruck (hPa)", "Bewölkung (%)", "Windgeschwindigkeit_10m_x (km/h)", "Windgeschwindigkeit_10m_y (km/h)", "Windböen_10m_x (km/h)", "Windböen_10m_y (km/h)"]
)

if not multi_window.label_columns:
  num_features = train_df.shape[1]
else: 
  num_features = len(multi_window.label_columns)

if os.path.exists(model_path):
    print("Loading existing model...")
    multi_lstm_model = tf.keras.models.load_model(model_path)

    if input("Train (y | n): ").lower() == "y":
      history = compile_and_fit(multi_lstm_model, multi_window)

else:
    print("Training new model...")

    """multi_lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(OUT_STEPS * num_features,
                              kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])"""

    multi_lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(OUT_STEPS * num_features),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    history = compile_and_fit(multi_lstm_model, multi_window)

if input("Evaluate and test? (y | n)? ").lower() == "y":
  val_performance = multi_lstm_model.evaluate(multi_window.val, return_dict=True)
  test_performance = multi_lstm_model.evaluate(multi_window.test, return_dict=True)

  print("Validation:", val_performance)
  print("Test:", test_performance)

multi_window.plot(multi_lstm_model, plot_col="Temperatur_2m (°C)", normed=False, max_subplots=1)