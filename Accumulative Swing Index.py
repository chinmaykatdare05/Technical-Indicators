import numpy as np


def swing_index(df, limit_move_value):
    close_yesterday = df["Close"].shift(1)

    k = np.maximum(
        df["High"] - close_yesterday,
        np.maximum(df["Low"] - close_yesterday, df["High"] - df["Low"]),
    )

    a = np.abs(df["High"] - close_yesterday)
    b = np.abs(df["Low"] - close_yesterday)
    c = np.abs(df["High"] - df["Low"])

    condition1 = (close_yesterday < df["High"]) & (close_yesterday > df["Low"])
    condition2 = (close_yesterday < df["High"]) & (close_yesterday <= df["Low"])

    swing_index = np.where(
        condition1,
        50
        * (
            df["Close"]
            - close_yesterday
            + 0.5 * (df["Close"] - df["Open"])
            + 0.25 * (close_yesterday - df["Open"])
        )
        / k,
        np.where(
            condition2,
            50
            * (
                df["Close"]
                - close_yesterday
                + 0.5 * (df["Close"] - df["Open"])
                - 0.25 * (close_yesterday - df["Open"])
            )
            / k,
            50
            * (
                df["Close"]
                - close_yesterday
                - 0.5 * (df["Close"] - df["Open"])
                - 0.25 * (close_yesterday - df["Open"])
            )
            / k,
        ),
    )

    swing_index /= limit_move_value
    swing_index *= 100

    return swing_index


# Limit Move Value: A constant representing the maximum price change expected for the stock
# df['ASI'] = swing_index(df, limit_move_value = 0.5)
