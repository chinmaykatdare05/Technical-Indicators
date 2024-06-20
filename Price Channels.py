import pandas as pd

def donchian_channel(df, window=20):
    # Calculate upper and lower channel lines
    df['Upper Channel'] = df['High'].rolling(window=window, min_periods=1).max()
    df['Lower Channel'] = df['Low'].rolling(window=window, min_periods=1).min()
    
    return df

# df = donchian_channel(df)
