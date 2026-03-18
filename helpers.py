data_simple = [1.6893854, 0.8360405, 0.05175877, -0.48978138, -0.7218008, -0.6839962, -0.26737309, 0.592648, 1.7472734, 3.017901, 4.317954, 5.5038385, 6.383465, 6.8271027, 6.769967, 6.2646775, 5.462652, 4.518154, 3.530665, 2.6266174, 1.8807006, 1.3420682, 0.8833151, 0.5153284]
data_temp = [0.6, 0.4, 0, -0.5, -0.8, -0.7, 0, 1.1, 2.6, 4.4, 6.0, 7.3, 8.1, 8.4, 8.2, 7.6, 6.8, 5.8, 4.7, 3.7, 2.8, 2.1, 1.5, 1.1]
data_app = [0, -0.5, -1, -1, -1, -0.5, 0, 0.5, 1, 2, 4, 5, 7, 8, 9, 10, 10.5, 10, 9, 8, 7, 6, 5.5, 5]

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

data_simple = np.array(data_simple)
data_temp = np.array(data_temp)

hours = [(0 + i) % 24 for i in range(24)]
labels = [f"{h:02d}:00" for h in hours]

x = np.arange(24)

plt.figure(figsize=(12, 5))

plt.plot(x, data_simple, color='orange', marker='x', label='Modell allgemein')
plt.plot(x, data_temp, color='green', marker='x', label='Modell Temperatur')
plt.plot(x, data_app, color='blue', marker='x', label='Wetter-App')

plt.xticks(x, labels, rotation=45)
plt.xlabel("Zeit [h]")
plt.ylabel("Temperatur [°C]")
plt.legend()

plt.tight_layout()
plt.show()