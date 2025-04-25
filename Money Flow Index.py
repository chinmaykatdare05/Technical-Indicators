import pandas as pd
import numpy as np


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Money Flow Index (MFI), a volume-weighted version of the Relative Strength Index (RSI).

    Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', 'Close', and 'Volume' columns.
        period (int): The number of periods over which to calculate the MFI. Default is 14.

    Returns:
        pd.Series: The Money Flow Index (MFI) values.
    """
    # Calculate typical price
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3

    # Calculate raw money flow
    raw_money_flow = typical_price * df["Volume"]

    # Determine positive and negative money flows
    positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)

    # Calculate 14-period sums of positive and negative money flows
    positive_mf = pd.Series(positive_flow).rolling(window=period, min_periods=1).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period, min_periods=1).sum()

    # Calculate money flow ratio (MFR)
    mfr = positive_mf / negative_mf.replace(to_replace=0, method="ffill")

    # Calculate Money Flow Index (MFI)
    mfi = 100 - (100 / (1 + mfr))

    return mfi
