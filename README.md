# Wettervorhersage mit einem neuronalen Netzwerk

Dieses Projekt wendet den Workflow von (TensorFlow)[https://www.tensorflow.org/tutorials/structured_data/time_series] an, um Wetterdaten zu sammeln, für ein neuronales Netzwerk aufzubereiten, ein Modell zu trainieren, dessen Qualität zu bewerten und anschließend eine Vorhersage für die nächsten Stunden zu erzeugen. Die Pipeline ist in einzelne Python-Skripte im Ordner `workflow/` aufgeteilt, sodass jeder Schritt nachvollziehbar bleibt.

Im Fokus steht eine Zeitreihen-Vorhersage auf Basis historischer Stundenwerte.

## Überblick über den Workflow

Die Reihenfolge der Skripte ist:

1. `workflow/1-data-collect.py`
2. `workflow/2-data-convert.py`
3. `workflow/3-data-normalise-split.py`
4. `workflow/4-train-eval-test-forecast.py`
5. `workflow/5-forecast.py`

Die Idee dahinter ist:

1. Historische Wetterdaten für einen festen Ort laden
2. Rohdaten in ein lernbares numerisches Format umwandeln
3. Daten normalisieren und in Trainings-, Validierungs- und Testdaten aufteilen
4. Ein LSTM-Modell trainieren und evaluieren
5. Mit dem trainierten Modell eine echte Wettervorhersage erzeugen

## Verzeichnisstruktur

Wichtige Ordner und Dateien:

- `workflow/`: enthält die einzelnen Verarbeitungsschritte
- `data/`: enthält Rohdaten, aufbereitete Daten und Statistikdateien für die Normalisierung
- `models/`: enthält gespeicherte Keras-Modelle und zugehörige Metadaten
- `workflow/window.py`: Hilfsklasse für Zeitfenster, Datasets und Visualisierung

## Benötigte Bibliotheken

Das Projekt verwendet folgende packages:

- `requests`
- `pandas`
- `numpy`
- `matplotlib`
- `tensorflow`

Diese können mit `pip install -r requirements.txt` installiert werden.

## Schritt 1: Wetterdaten sammeln

Datei: `workflow/1-data-collect.py`

In diesem Schritt werden historische Wetterdaten über die Open-Meteo-Archiv-API geladen.


1. Definieren eines festen Ort über Breiten- und Längengrad. Beispiel:
   `latitude = 48.0019`, `longitude = 11.3442`
2. Forderung stündlicher Wetterdaten von `1940-01-01` bis `2026-03-15` (aktuelles Datum).
3. Geladene Wettermerkmale:
   `temperature_2m`, `relative_humidity_2m`, `precipitation`, `rain`, `snowfall`, `surface_pressure`, `cloud_cover`, `wind_speed_10m`, `wind_direction_10m`, `wind_gusts_10m`
4. Die API-Antwort wird als CSV gespeichert unter `data/data_raw.csv`.
5. Zusätzlich werden die Ortsdaten für spätere Prognosen unter `data/area_data.json` gespeichert.

Diese Daten sind notwendig, damit das neuronale Netzwerk Muster in den Wetterdaten erkennen kann, um eigene Prognosen zu treffen.

Aufruf:

```bash
python workflow/1-data-collect.py
```

## Schritt 2: Rohdaten in ML-taugliche Features umwandeln

Datei: `workflow/2-data-convert.py`

Rohdaten aus einer Wetter-API sind noch nicht ideal für ein neuronales Netzwerk. Dieses Skript bereitet sie deshalb in mehreren Schritten auf.

### 2.1 Spalten umbenennen

Zuerst werden englische API-Spaltennamen in sprechendere deutsche Namen übersetzt:
```
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
```

### 2.2 Zeitspalte in Objekte des Datentypen DateTime umwandeln

Die Spalte `Zeit` wird mit `pandas.to_datetime(...)` in echte Zeitstempel umgewandelt. Dadurch kann das Datum einfacher modifiziert werden.

### 2.3 Unvollständige Anfangsdaten entfernen

Hier wurde der erste Tag `1940-01-01` entfernt, weil dort mehrere Werte fehlen. Das vermeidet Störungen beim Training.

### 2.4 Zyklische Größen als Vektoren darstellen

Ein zentrales Problem bei Zeitreihen ist, dass manche Größen zyklisch sind:

- Windrichtung: `1°` und `360°` bedeuten fast dasselbe, numerisch liegen sie aber weit auseinander
- Uhrzeit und Jahresverlauf sind ebenfalls periodisch

Neuronale Netze verstehen diese Periodizität nicht automatisch. Deshalb werden solche Größen in Vektoren umgewandelt.

#### Windrichtung und Windgeschwindigkeit

Die ursprünglichen Spalten `Windgeschwindigkeit_10m (km/h)` und `Windrichtung_10m (°)` werden kombiniert zu:

- `Windgeschwindigkeit_10m_x (km/h)`
- `Windgeschwindigkeit_10m_y (km/h)`

Das Gleiche passiert mit den Windböen:

- `Windböen_10m_x (km/h)`
- `Windböen_10m_y (km/h)`

Die Windrichtung fällt weg, da Windgeschwindigkeit und Böen als Vektoren Richtung und Stärke des Windes darstellen.

#### Tages- und Jahreszyklus

Aus den Zeitstempeln werden zusätzliche Features erzeugt:

- `Tag_x`
- `Tag_y`
- `Jahr_x`
- `Jahr_y`

Diese kodieren:

- die Position innerhalb eines Tages
- die Position innerhalb eines Jahres

So kann das Modell lernen, dass 23:00 Uhr und 00:00 Uhr zeitlich nahe beieinander liegen, obwohl die eigentlichen Zahlenwerte stark unterschiedlich aussehen würden.

### 2.5 Ergebnis speichern

Die aufbereiteten Daten werden nach `data/data_converted.csv` geschrieben.

Aufruf:

```bash
python workflow/2-data-convert.py
```

## Schritt 3: Daten normalisieren und in Datensätze aufteilen

Datei: `workflow/3-data-normalise-split.py`

Nach der Feature-Erzeugung müssen die Daten für das Training skaliert und in mehrere Teilmengen zerlegt werden.

### 3.1 Warum normalisieren?

Wettermerkmale liegen auf sehr unterschiedlichen Skalen:

- Temperatur zwischen `-20` und `35`
- Luftdruck um `1000`
- Luftfeuchtigkeit zwischen `0` und `100`

Ohne Normalisierung würden Merkmale mit großen Zahlenbereichen das Training überproportional dominieren. Deshalb wird jede Spalte standardisiert:

```text
(Wert - Mittelwert) / Standardabweichung
```

### 3.2 Warum erst teilen und dann normalisieren?

Das Skript berechnet Mittelwert und Standardabweichung ausschließlich auf dem Trainingsdatensatz. Das ist wichtig, damit keine Information aus Validierungs- oder Testdaten unabsichtlich ins Training einfließt.

### 3.3 Aufteilung in 70-20-10

Die Daten werden zeitlich in drei Abschnitte geteilt:

- `70 %` Training
- `20 %` Validierung
- `10 %` Test

Zeitreihen werden hier der Reihenfolge nach getrennt, nicht zufällig gemischt. Das ist sinnvoll, weil Wetterdaten zeitliche Abhängigkeiten besitzen.

### 3.4 Gespeicherte Dateien

Das Skript erzeugt:

- `data/train.csv`
- `data/validate.csv`
- `data/test.csv`
- `data/train_mean.csv`
- `data/train_std.csv`

Die letzten beiden Dateien werden später benötigt, um die vorhergesagten Werte wieder zu "entnormalisieren".

### 3.5 Visualisierung der Normalisierung

Zusätzlich zeigt das Skript einen Vergleich der Min-/Max-Werte vor und nach der Normalisierung als zwei Graphen. So lässt sich direkt prüfen, ob die Merkmale nach der Standardisierung in ähnlichen Größenordnungen liegen.

Aufruf:

```bash
python workflow/3-data-normalise-split.py
```

## Schritt 4: Zeitfenster für das neuronale Netzwerk erzeugen

Datei: `workflow/window.py`

Bevor trainiert werden kann, müssen die fortlaufenden Zeitreihen in Eingabe- und Zielblöcke aufgeteilt werden. Genau das übernimmt die Klasse `WindowGenerator`.

### 4.1 Grundidee

Das Modell soll aus einer festen Anzahl vergangener Stunden die nächsten Stunden vorhersagen.

In unserem Fall:

- `INPUT_WIDTH = 120 # 5 Tage`
- `LABEL_WIDTH = 24 # 1 Tag`
- `SHIFT = 24`
- `LABEL_COLUMNS = ["Temperatur_2m (°C)"]`

In diesem Beispiel:

- Das Modell sieht die letzten `120` Stunden als Eingabe
- Es soll `24` Stunden an Temperaturwerten vorhersagen, die direkt nach den 120 Stunden liegen.
- Die Zielwerte liegen direkt im Anschluss an die Eingabe

### 4.2 Was `WindowGenerator` konkret macht

Die Klasse:

1. speichert Trainings-, Validierungs- und Testdaten
2. merkt sich die Positionen aller Spalten
3. trennt jede Zeitsequenz in Eingabe (`inputs`) und Ziel (`labels`)
4. baut daraus `tf.data.Dataset`-Objekte für TensorFlow
5. stellt Beispielbatches für Visualisierungen bereit
6. kann Vorhersagen und echte Werte direkt plotten

### 4.3 Warum Fenster nötig sind

Ein LSTM arbeitet nicht auf einer kompletten CSV-Datei auf einmal, sondern auf vielen kurzen Sequenzen gleicher Länge. Diese Sequenzen sind die Trainingsbeispiele, aus denen das Modell die Dynamik des Wetters lernt.

## Schritt 5: Modellarchitektur und Training

Datei: `workflow/4-train-eval-test-forecast.py`

Dieses Skript ist der Kern des Projekts. Es lädt die Daten, erzeugt die Fenster, definiert das Modell, trainiert es und kann es anschließend evaluieren.

### 5.1 Modell- und Metadateien

Das Modell wird unter `models/<name>` gespeichert, zum Beispiel:

- `models/model_complex_temp.keras`

Zusätzlich wird eine JSON-Datei mit den wichtigsten Modellparametern angelegt, zum Beispiel:

- `models/model_complex_temp.json`

Darin stehen:

- `input_width`
- `label_width`
- `shift`
- `label_columns`

Das ist wichtig, damit Forecasts später mit genau denselben Fenstergrößen erzeugt werden wie beim Training.

### 5.2 Architektur des neuronalen Netzes

Falls noch kein gespeichertes Modell existiert, wird ein neues Keras-Modell aufgebaut. Hier ein (Standard-Modell)[https://github.com/iamtekson/deep-learning-for-earth-observation/blob/main/Notebooks/05.%20time-series-analysis-LSTM/Weather_prediction_(LSTM).ipynb]:

1. `LSTM(128, return_sequences=True)`
2. `Dropout(0.2)`
3. `LSTM(64, return_sequences=True)`
4. `Dropout(0.2)`
5. `LSTM(32, return_sequences=False)`
6. `Dropout(0.2)`
7. `Dense(LABEL_WIDTH * Anzahl_Zielspalten)`
8. `Reshape([LABEL_WIDTH, Anzahl_Zielspalten])`

### 5.3 Warum genau diese Architektur?

Die Architektur ist für Sequenzdaten geeignet:

- Die LSTM-Schichten erkennen zeitliche Muster und Abhängigkeiten
- Mehrere LSTM-Schichten erlauben abstraktere Repräsentationen
- Dropout reduziert Overfitting
- Die Dense-Schicht projiziert die letzte interne Repräsentation auf die komplette Vorhersagesequenz
- `Reshape` formt die Ausgabe in `24` Zeitschritte mit `1` Zielgröße pro Stunde um (in diesem Beispiel)

### 5.4 Verlustfunktion und Optimierung

Das Modell wird kompiliert mit:

- Verlustfunktion: `MeanSquaredError`
- Optimierer: `Adam`
- Metrik: `MeanAbsoluteError`

`MeanSquaredError` eignet sich gut für Regressionsprobleme. `MeanAbsoluteError` ist zusätzlich leicht interpretierbar, weil er die durchschnittliche absolute Abweichung in der Zielgröße ausdrückt.

### 5.5 Trainingsstrategie

Das Training verwendet:

- bis zu `20` Epochen
- `EarlyStopping` mit `patience=2`
- `ModelCheckpoint`, um das beste Modell anhand von `val_loss` zu speichern

Damit wird nicht einfach nur das letzte Modell, sondern das beste validierte Modell gesichert.

### 5.6 Training starten

Beispiel:

```bash
python workflow/4-train-eval-test-forecast.py --model model_complex_temp.keras --train
```

### 5.7 Validierung und Test

Mit `--validate` wird das gespeicherte Modell auf Validierungs- und Testdaten ausgewertet:

```bash
python workflow/4-train-eval-test-forecast.py --model model_complex_temp.keras --validate
```

Dabei werden Verlust und mittlere absolute Abweichung berechnet.

### 5.8 Visualisierung

Am Ende plottet das Skript Beispielsequenzen:

- vergangene Werte
- echte Zielwerte
- Modellvorhersagen

Wenn `normed=False` gesetzt ist, werden die Daten mit `train_mean.csv` und `train_std.csv` zurück in die echten Werte umgerechnet.

## Schritt 6: Eine echte Vorhersage erzeugen

Datei: `workflow/5-forecast.py`

Dieses Skript lädt ein trainiertes Modell und erzeugt daraus eine Vorhersage für die nächsten 24 Stunden.

### 6.1 Was das Skript benötigt

Es setzt voraus, dass bereits vorhanden sind:

- ein trainiertes Modell in `models/`
- die zugehörige JSON-Datei mit den Fensterparametern
- `data/area_data.json`
- `data/train_mean.csv`
- `data/train_std.csv`

### 6.2 Welche Daten verwendet werden?

Das Skript lädt aktuelle oder historische Eingabedaten aus der normalen Open-Meteo-Forecast-API:

- bei `--forecast now`: die letzten `INPUT_WIDTH` Stunden bis jetzt
- bei `--forecast YYYY-MM-DD`: die letzten `INPUT_WIDTH` Stunden vor einem angegebenen Datum

### 6.3 Gleiche Vorverarbeitung wie im Training

Die neu geladenen Forecast-Eingaben durchlaufen dieselben Schritte wie die Trainingsdaten:

1. Spalten umbenennen
2. Zeit in Datumsobjekte umwandeln
3. Windrichtung, Windgeschwindigkeit, Tages- und Jahreszyklen in Vektoren umwandeln
4. mit `train_mean` und `train_std` normalisieren (Laden der .csv Dateien)

### 6.4 Vorhersage berechnen

Danach:

1. Umwandlung der Eingabewerte in ein `np.array`
2. Umwandeln des Array zur Form `(1, INPUT_WIDTH, ANZAHL_LABELS)` (Bsp: `(1, 24, 1)`)
3. Berechnen der Vorhersage mit dem Modell
4. Entnormalisierung der Ausgave

Zusätzlich wird der Vorhersagezeitraum ausgegeben, zum Beispiel von Stunde `0:00` bis Stunde `23:00` nach dem letzten bekannten Messwert.

### 6.5 Forecast starten

Aktuelle Vorhersage:

```bash
python workflow/5-forecast.py --model model_complex_temp.keras --forecast now
```

Vorhersage ab einem bestimmten Datum:

```bash
python workflow/5-forecast.py --model model_complex_temp.keras --forecast 2026-03-20
```

## Beispielablauf

Wenn alle Dateien und Abhängigkeiten vorhanden sind, sieht der typische Workflow so aus:

```bash
python workflow/1-data-collect.py
python workflow/2-data-convert.py
# Danach Dateiname angleichen: data_converted.csv <-> data_ready_de.csv
python workflow/3-data-normalise-split.py
python workflow/4-train-eval-test-forecast.py --model model_complex_temp.keras --train
python workflow/4-train-eval-test-forecast.py --model model_complex_temp.keras --validate
python workflow/5-forecast.py --model model_complex_temp.keras --forecast now
```

## Modelle

Das Projekt enthält schon 3 vortrainierte Modelle:

1. `models/model_simple.keras`: Sagt mit 120 Stunden Vergangenheit, 24 Stunden mit 10 Parametern hervor. 32-LSTM-Layer --> 240-Dense-Layer --> 24*10 Ausgabe (24 Stunden; 10 Parameter)
2. `models/model_temp.keras`: Sagt mit 120 Stunden Vergangenheit, 24 Stunden mit 1 Parameter (Temperatur) hervor. 32-LSTM-Layer --> 24-Dense-Layer --> 24*1 Ausgabe (24 Stunden; Temperatur)
3. `models/model_complex_temp.keras`: Sagt mit 120 Stunden Vergangenheit, 24 Stunden mit 1 Parameter (Temperatur) hervor. 128-LSTM-Layer --> 0.2 Dropout --> 64-LSTM-Layer --> 0.2 Dropout --> 32-LSTM-Layer --> 0.2 Dropout --> 24-Dense --> 24*1 Ausgabe (24 Stunden; Temperatur)

## Fazit

Dieses Projekt bildet einen vollständigen End-to-End-Workflow für Wettervorhersage mit einem neuronalen Netzwerk ab:

1. historische Wetterdaten laden
2. periodische und meteorologische Merkmale in lernbare Features umwandeln
3. Daten normalisieren und aufteilen
4. mit LSTM-Zeitfenstern trainieren
5. Modell evaluieren
6. reale 24-Stunden-Prognosen erzeugen
