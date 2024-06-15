import numpy as np


def sma(data, window_swing_indexze):
    weights = np.repeat(1.0, window_swing_indexze) / window_swing_indexze
    moving_averages = np.convolve(data, weights, mode="valid")
    return moving_averages


# sma(df["Close"], 20)


def ema(data, window_swing_indexze):
    weights = np.exp(np.linspace(-1.0, 0.0, window_swing_indexze))
    weights /= weights.sum()
    moving_averages = np.convolve(data, weights)[: len(data)]
    moving_averages[:window_swing_indexze] = moving_averages[window_swing_indexze]
    return moving_averages


# ema(df["Close"], 20)


def rsi(df, window=14):
    # Calculate price differences
    delta = df["Close"].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gains and losses over the specified window
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss.replace(to_replace=0, method="ffill").replace(0, np.nan)

    # Calculate RSI directly from RS using vectorized operations
    rsi = 100 - (100 / (1 + rs))

    return rsi


# df['RSI'] = rsi(df, window=14)


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


def atr(df, window_size):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=window_size, min_periods=1).mean()

    return atr


# df['ATR'] = atr(df, window_size = 14)


def adx(df, window_size):
    # Calculate True Range (TR)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))

    # Calculate Directional Movement (+DM, -DM)
    up_move = df["High"].diff()
    down_move = df["Low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Calculate smoothed TR, +DM, -DM
    atr = tr.rolling(window=window_size, min_periods=1).mean()
    smoothed_plus_dm = plus_dm.rolling(window=window_size, min_periods=1).mean()
    smoothed_minus_dm = minus_dm.rolling(window=window_size, min_periods=1).mean()

    # Calculate +DI, -DI
    plus_di = 100 * (smoothed_plus_dm / atr)
    minus_di = 100 * (smoothed_minus_dm / atr)

    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=window_size, min_periods=1).mean()

    return adx


# df['ADX'] = adx(df, window_size=14)


def aroon(df, window_size):
    aroon_up = np.zeros(len(df))
    aroon_down = np.zeros(len(df))

    for i in range(window_size, len(df)):
        high_window = df["High"][i - window_size : i]
        low_window = df["Low"][i - window_size : i]

        high_index = high_window.argmax()
        low_index = low_window.argmin()

        aroon_up[i] = (window_size - high_index) / window_size * 100
        aroon_down[i] = (window_size - low_index) / window_size * 100

    return aroon_up, aroon_down


# df['Aroon Up'], df['Aroon Down'] = aroon(df, window_size = 14)


def aroon_oscillator(df, window_size):
    aroon_up, aroon_down = aroon(df, window_size)
    aroon_oscillator = aroon_up - aroon_down
    return aroon_oscillator


# df['Aroon Oscillator'] = aroon_oscillator(df, window_size = 14)


def bollinger_bands(df, window_size, num_std_dev):
    rolling_mean = df["Close"].rolling(window=window_size, min_periods=1).mean()
    rolling_std = df["Close"].rolling(window=window_size, min_periods=1).std()

    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    return rolling_mean, upper_band, lower_band


# df['Middle Band'], df['Upper Band'], df['Lower Band'] = bollinger_bands(df, window_size = 20, num_std_dev = 2)


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


def chande_momentum_oscillator(df, window_size):
    close = df["Close"]

    # Calculate the differences in the closing prices
    diff = close.diff()

    # Calculate the sum of gains and losses
    gain = diff.where(diff > 0, 0).rolling(window=window_size, min_periods=1).sum()
    loss = -diff.where(diff < 0, 0).rolling(window=window_size, min_periods=1).sum()

    # Calculate the CMO
    cmo = 100 * (gain - loss) / (gain + loss)

    return cmo


# df['CMO'] = chande_momentum_oscillator(df, window_size=10)


def CCI(df, window_size):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    mean_deviation = typical_price.rolling(window=window_size).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    CCI = (typical_price - typical_price.rolling(window=window_size).mean()) / (
        0.015 * mean_deviation
    )
    return CCI


# df['CCI'] = CCI(df, window_size=14)


def keltner_channels(df, window_size=20, multiplier=2, ema_window=20):
    # Calculate Typical Price
    df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3

    # Calculate Middle Line (EMA of Typical Price)
    df["Middle_Line"] = (
        df["Typical_Price"].ewm(span=ema_window, min_periods=ema_window).mean()
    )

    # Calculate True Range (TR)
    df["TR"] = np.maximum.reduce(
        [
            df["High"] - df["Low"],
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift()),
        ]
    )

    # Calculate Average True Range (ATR)
    df["ATR"] = df["TR"].ewm(span=window_size, min_periods=window_size).mean()

    # Calculate Upper and Lower Bands
    df["Upper_Band"] = df["Middle_Line"] + multiplier * df["ATR"]
    df["Lower_Band"] = df["Middle_Line"] - multiplier * df["ATR"]

    return df


# df = keltner_channels(df)
