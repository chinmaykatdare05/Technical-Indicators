import pandas as pd
import numpy as np

def directional_movement(df, period=14):
    # Calculate differences between high and low prices
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()

    # Calculate True Range (TR)
    TR = np.maximum(high_diff, low_diff)

    # Calculate Directional Movement (DM)
    DMplus = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    DMminus = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    # Calculate smoothed ATR and ADX using rolling windows
    ATR = TR.rolling(window=period, min_periods=1).mean()
    ADMplus = DMplus.rolling(window=period, min_periods=1).mean()
    ADMminus = DMminus.rolling(window=period, min_periods=1).mean()

    # Calculate +DI and -DI
    plus_DI = (ADMplus / ATR) * 100
    minus_DI = (ADMminus / ATR) * 100

    # Calculate DX and ADX
    DX = np.abs(plus_DI - minus_DI) / (plus_DI + minus_DI) * 100
    ADX = DX.rolling(window=period, min_periods=1).mean()
    return plus_DI, minus_DI, ADX

# df['+DI'], df['-DI'], df['ADX'] = directional_movement(df)
