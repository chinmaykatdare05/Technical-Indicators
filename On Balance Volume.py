import pandas as pd

def obv(df):
    # Calculate price changes
    price_diff = df['Close'].diff()
    
    # Initialize OBV with 0
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = 0

    # Determine OBV values based on price changes
    for i in range(1, len(df)):
        if price_diff.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
        elif price_diff.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]

    return obv

# df['OBV'] = obv(df)
