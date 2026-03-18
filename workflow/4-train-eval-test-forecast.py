import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--train", action="store_true")
parser.add_argument("--validate", action="store_true")

args = parser.parse_args()

# Datenformat
INPUT_WIDTH = 120 # 120 Stunden Eingabe
LABEL_WIDTH = 24 # 24 Stunden Ausgabe
SHIFT = 24 # Kein Abstand zwischen Vergangenheit und Prognose (letzter Wert ist 24 Stunden nach letztem Vergangenheitswert)

LABEL_COLUMNS = ["Temperatur_2m (°C)"]

# Training
MAX_EPOCHS = 20 # Anzahl Epochen
model_path = f"./models/{args.model}"
model_data_path = "." + "".join(model_path.split(".")[:-1]) + ".json"

if os.path.exists(model_path) and os.path.exists(model_data_path):
    with open(model_data_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)
    INPUT_WIDTH, LABEL_WIDTH, SHIFT, LABEL_COLUMNS = model_data["input_width"], model_data["label_width"], model_data["shift"], model_data["label_columns"]
else:
    model_data = {
        "input_width": INPUT_WIDTH,
        "label_width": LABEL_WIDTH,
        "shift": SHIFT,
        "label_columns": LABEL_COLUMNS
    }
    with open(model_data_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, indent=2)

with open("./data/area_data.json", "r") as f:
    data = json.load(f)

# Ortsdaten
lat = data["lat"]
lon = data["lon"]

# Initialisiere WindowGenerator
# Dieses Objekt stellt Trainings-, Evaluations- und Testdaten im richtigen Format zur Verfügung

from window import WindowGenerator
import pandas as pd

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
  
  # Trainingsdaten mit window.train kriegen
  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping, checkpoint])
  return history

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(LABEL_WIDTH * len(window.label_columns),
                                kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([LABEL_WIDTH, len(window.label_columns)])
        ])

if args.train:
    history = compile_and_fit(model, window)
if args.validate:
    val_performance = model.evaluate(window.val, return_dict=True)
    test_performance = model.evaluate(window.test, return_dict=True)

window.plot(
    model=model,
    plot_cols=window.label_columns,
    max_subplots=3,
    normed=False
)