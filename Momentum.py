import pandas as pd


def momentum(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the momentum indicator, which measures the rate of change in price.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Close' column.
        period (int): The number of periods over which to calculate momentum. Default is 14.

    Returns:
        pd.Series: The momentum indicator values.
    """
    # Calculate the difference in closing prices
    return df["Close"].diff(period)
