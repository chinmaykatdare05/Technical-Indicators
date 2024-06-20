import pandas as pd

def ratiocator(df, column='Close', window=14):
    # Calculate SMA
    sma = df[column].rolling(window=window, min_periods=1).mean()
    
    # Calculate Ratiocator (RAT)
    rat = df[column] / sma
    
    return rat

# df['RAT'] = ratiocator(df)
