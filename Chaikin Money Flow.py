def chaikin_money_flow(df, window_size):
    # Calculate the Money Flow Multiplier
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (
        df["High"] - df["Low"]
    )

    # Calculate the Money Flow Volume
    mfv = mfm * df["Volume"]

    # Calculate the Chaikin Money Flow
    cmf = (
        mfv.rolling(window=window_size, min_periods=1).sum()
        / df["Volume"].rolling(window=window_size, min_periods=1).sum()
    )

    return cmf


# df["Chaikin Money Flow"] = chaikin_money_flow(df, window_size=20)
