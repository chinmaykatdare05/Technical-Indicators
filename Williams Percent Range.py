import pandas as pd

def williams_r(df, window=14):
    highest_high = df['High'].rolling(window=window, min_periods=1).max()
    lowest_low = df['Low'].rolling(window=window, min_periods=1).min()
    
    williams_r = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
    
    return williams_r

# df['Williams %R'] = williams_r(df)
