import pandas as pd


def price_channels(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculate the Price Channel indicator, which consists of an Upper Channel Line (UCL)
    and a Lower Channel Line (LCL), based on the highest high and lowest low over the last n periods.

    Parameters:
    prices (pd.DataFrame): DataFrame containing 'High' and 'Low' columns for stock prices.
    period (int): The period for calculating the price channel (lookback period).

    Returns:
    pd.DataFrame: A DataFrame containing the Upper Channel Line (UCL) and Lower Channel Line (LCL).
    """
    # Calculate the Highest High and Lowest Low over the given period
    highest_high = prices["High"].rolling(window=period).max()
    lowest_low = prices["Low"].rolling(window=period).min()

    # Create the Price Channel with the Upper and Lower Channel Lines
    price_channel = pd.DataFrame({"UCL": highest_high, "LCL": lowest_low})

    return price_channel
