import numpy as np


def center_of_gravity(df, window_size):
    price = df["Close"].values
    weights = np.arange(1, window_size + 1)

    # Calculate the numerator (weighted sum of prices)
    numerator = np.convolve(price, weights[::-1], mode="valid")

    # Calculate the denominator (sum of weights)
    denominator = weights.sum()

    # Center of Gravity
    cog = np.concatenate([np.full(window_size - 1, np.nan), numerator / denominator])

    return cog


# df["Center of Gravity"] = center_of_gravity(df, window_size=10)
