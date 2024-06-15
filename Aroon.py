import numpy as np


def aroon(df, window_size):
    aroon_up = np.zeros(len(df))
    aroon_down = np.zeros(len(df))

    for i in range(window_size, len(df)):
        high_window = df["High"][i - window_size : i]
        low_window = df["Low"][i - window_size : i]

        high_index = high_window.argmax()
        low_index = low_window.argmin()

        aroon_up[i] = (window_size - high_index) / window_size * 100
        aroon_down[i] = (window_size - low_index) / window_size * 100

    return aroon_up, aroon_down


# df['Aroon Up'], df['Aroon Down'] = aroon(df, window_size = 14)


def aroon_oscillator(df, window_size):
    aroon_up, aroon_down = aroon(df, window_size)
    aroon_oscillator = aroon_up - aroon_down
    return aroon_oscillator


# df['Aroon Oscillator'] = aroon_oscillator(df, window_size = 14)
