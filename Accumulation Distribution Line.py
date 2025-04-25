from pandas import DataFrame, Series


def calculate_adl(data: DataFrame) -> Series:
    """
    Calculate the Accumulation Distribution Line (ADL) for a given DataFrame.
    The ADL is a volume-based indicator that measures the cumulative flow of money into and out of a security.
    It is calculated using the following formula:
    ADL = Previous ADL + Money Flow Volume
    where:
    Money Flow Volume = Money Flow Multiplier * Volume
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    """
    required_cols = {"High", "Low", "Close", "Volume"}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    high_low_range = data["High"] - data["Low"]
    high_low_range = high_low_range.replace(0, 1e-10)  # prevent division by zero

    money_flow_multiplier = (
        (data["Close"] - data["Low"]) - (data["High"] - data["Close"])
    ) / high_low_range
    money_flow_volume = money_flow_multiplier * data["Volume"]
    adl = money_flow_volume.cumsum()

    return adl
