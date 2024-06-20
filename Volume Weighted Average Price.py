import pandas as pd

def vwap(df):
    """
    Calculate the Volume Weighted Average Price (VWAP).

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close', 'Volume', 'High', and 'Low' columns.

    Returns:
    pd.DataFrame: DataFrame with an additional 'VWAP' column containing the VWAP values.
    """
    # Calculate VWAP
    df['Typical Price'] = (df['Close'] + df['High'] + df['Low']) / 3
    df['Cumulative TPV'] = (df['Typical Price'] * df['Volume']).cumsum()
    df['Cumulative Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative TPV'] / df['Cumulative Volume']
    
    return df

print(vwap(df))
