import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

import tensorflow as tf

class WindowGenerator():
  def __init__(self, input_width, label_width, shift, 
               train_df, val_df, test_df, 
               train_mean, train_std, 
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    self.train_mean = train_mean
    self.train_std = train_std

    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is None:
      self.label_columns_indices = self.column_indices
    else:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    

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
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    return inputs, labels

  def plot(self, model=None, plot_cols=['Temperatur_2m (°C)'], max_subplots=3, normed=True, inputs=None, show_history=True, show_y_labels=False):
    if inputs is None:
        inputs, labels = self.example
    else:
      labels = None

    fig = plt.figure(figsize=(12, 8))
    for plot_col in plot_cols:
      plot_col_index = self.column_indices[plot_col]
      max_n = min(max_subplots, len(inputs))
      for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        if normed:
          plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                  label='Vergangenheit', marker='.', zorder=-10)
        else:
          plt.plot(self.input_indices, inputs[n, :, plot_col_index] * self.train_std["std"].iloc[plot_col_index] + self.train_mean["mean"].iloc[plot_col_index],
                  label='Vergangenheit', marker='.', zorder=-10)

        # Wenn wir bestimmte Spalten ausgeben im Modell, dann sind diese in label_columns_indices gespeichert
        # Sonst werden alle Spalten  
        label_col_index = self.label_columns_indices.get(plot_col, None)
        if label_col_index is None:
          # Falls die gefordertte Spalte nicht in der Modelausgabe ist, wird diese überspringen
          continue
        
        if labels is not None:
          if normed:
              real_y = labels[n, :, label_col_index]
          else:
              real_y = labels[n, :, label_col_index] * self.train_std["std"].iloc[label_col_index] + self.train_mean["mean"].iloc[label_col_index]
          plt.scatter(self.label_indices, real_y,
                        edgecolors='k', label='Tatsächlich', c='#2ca02c', s=64)
          
          if show_y_labels:
              # y-Werte daneben schreiben
              for x, y in zip(self.label_indices, real_y):
                  plt.annotate(
                      f'{y:.1f}',
                      (x, y),
                      textcoords="offset points",
                      xytext=(5, -10),
                      fontsize=8
                  )
        
        if model is not None:
            predictions = model(inputs)
            if normed:
                pred_y = predictions[n, :, label_col_index]
            else:
                pred_y = (
                    predictions[n, :, label_col_index] * self.train_std["std"].iloc[label_col_index]
                    + self.train_mean["mean"].iloc[label_col_index]
                )

            plt.scatter(
                self.label_indices,
                pred_y,
                marker='X',
                edgecolors='k',
                label='Vorhersagen',
                c='#ff7f0e',
                s=64
            )

            if show_y_labels:
              # y-Werte daneben schreiben
              for x, y in zip(self.label_indices, pred_y):
                  plt.annotate(
                      f'{y:.1f}',
                      (x, y),
                      textcoords="offset points",
                      xytext=(5, -10),
                      fontsize=8
                  )

        if n == 0:
          plt.legend()

        x_positions = np.arange(self.input_width, self.input_width + self.shift)
        hour_labels = [f"{(0 + (x - self.input_width)) % 24:02d}:00" for x in x_positions]

        if not show_history:
          plt.xticks(x_positions, hour_labels, rotation=45)
          plt.xlim(self.input_width, self.input_width + self.shift - 1)

      plt.xlabel('Zeit')
      fig.supylabel(f'{plot_col} {"[normalisiert]" if normed else ""}')
      plt.tight_layout()
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