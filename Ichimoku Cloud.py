import pandas as pd


def ichimoku_cloud(
    df: pd.DataFrame, n1: int = 9, n2: int = 26, n3: int = 52
) -> pd.DataFrame:
    """
    Calculate the Ichimoku Cloud indicator for the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        n1 (int): Period for the Tenkan-sen (Conversion Line). Default is 9.
        n2 (int): Period for the Kijun-sen (Base Line) and Senkou Span A & B. Default is 26.
        n3 (int): Period for the Senkou Span B. Default is 52.

    Returns:
        pd.DataFrame: DataFrame with Ichimoku Cloud components:
            'Conversion Line', 'Base Line', 'Leading Span A', 'Leading Span B', 'Cloud Top', 'Cloud Bottom'.
    """
    # Tenkan-sen (Conversion Line)
    df["Conversion Line"] = (
        df["High"].rolling(window=n1).max() + df["Low"].rolling(window=n1).min()
    ) / 2

    # Kijun-sen (Base Line)
    df["Base Line"] = (
        df["High"].rolling(window=n2).max() + df["Low"].rolling(window=n2).min()
    ) / 2

    # Senkou Span A (Leading Span A)
    df["Leading Span A"] = ((df["Conversion Line"] + df["Base Line"]) / 2).shift(n2)

    # Senkou Span B (Leading Span B)
    df["Leading Span B"] = (
        (df["High"].rolling(window=n3).max() + df["Low"].rolling(window=n3).min()) / 2
    ).shift(n2)

    # Kumo (Cloud)
    df["Cloud Top"] = df[["Leading Span A", "Leading Span B"]].max(axis=1)
    df["Cloud Bottom"] = df[["Leading Span A", "Leading Span B"]].min(axis=1)

    return df[
        [
            "Conversion Line",
            "Base Line",
            "Leading Span A",
            "Leading Span B",
            "Cloud Top",
            "Cloud Bottom",
        ]
    ]
