import pandas as pd


def psar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    initial_af: float = 0.02,
    max_af: float = 0.2,
    increment: float = 0.02,
) -> pd.Series:
    """
    Calculate the Parabolic SAR (PSAR) indicator.

    Parameters:
        high (pd.Series): The high prices of the security.
        low (pd.Series): The low prices of the security.
        close (pd.Series): The closing prices of the security.
        initial_af (float): The initial acceleration factor (default is 0.02).
        max_af (float): The maximum acceleration factor (default is 0.2).
        increment (float): The increment to increase the acceleration factor (default is 0.02).

    Returns:
        pd.Series: The Parabolic SAR values for each period.
    """
    length = len(close)
    psar = close.copy()
    bull = True  # Initial trend direction (True for Bullish, False for Bearish)
    af = initial_af
    ep = low.iloc[
        0
    ]  # Extreme Point (EP) for a bullish trend (initially the first low price)
    sar = high.iloc[0]  # Start with the first high price as the SAR value

    for i in range(1, length):
        previous_sar = sar

        if bull:
            sar = sar + af * (ep - sar)
            if low.iloc[i] < sar:
                bull = False
                sar = ep
                ep = low.iloc[i]
                af = initial_af
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + increment, max_af)
        else:
            sar = sar + af * (ep - sar)
            if high.iloc[i] > sar:
                bull = True
                sar = ep
                ep = high.iloc[i]
                af = initial_af
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + increment, max_af)

        psar.iloc[i] = sar if bull else previous_sar

    return psar
