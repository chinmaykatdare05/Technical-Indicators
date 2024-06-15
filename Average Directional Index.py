import numpy as np


def adx(df, window_size):
    # Calculate True Range (TR)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))

    # Calculate Directional Movement (+DM, -DM)
    up_move = df["High"].diff()
    down_move = df["Low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Calculate smoothed TR, +DM, -DM
    atr = tr.rolling(window=window_size, min_periods=1).mean()
    smoothed_plus_dm = plus_dm.rolling(window=window_size, min_periods=1).mean()
    smoothed_minus_dm = minus_dm.rolling(window=window_size, min_periods=1).mean()

    # Calculate +DI, -DI
    plus_di = 100 * (smoothed_plus_dm / atr)
    minus_di = 100 * (smoothed_minus_dm / atr)

    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=window_size, min_periods=1).mean()

    return adx


# df['ADX'] = adx(df, window_size=14)
