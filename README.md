# Technical Indicators: Formulas, Insights, and Applications

---

This comprehensive guide provides descriptions and formulas for various technical indicators used in financial analysis and trading. Understanding these indicators can help you analyze price movements, identify trends, and make informed trading decisions.

Explore the descriptions and formulas below to gain insights into each indicator's significance and how it can be applied to your trading strategies.

Now, let's dive into the world of technical indicators and enhance our trading skills!

---

## Accumulation Distribution Line

The Accumulation Distribution Line (ADL) is a momentum indicator that relates price and volume. It combines price and volume to show how much money is flowing in or out of a security. It is calculated by adding the money flow volume to the previous ADL.

```math
ADL = \text{Previous ADL} + \text{Current Money Flow Volume}
```

## Average Directional Index (ADX)

The Average Directional Index (ADX) measures the strength of a trend without regard to its direction. It is part of the Directional Movement System developed by J. Welles Wilder.

```math
\text{ADX} = \text{SMA}(\frac{\text{TR}_n}{\text{DMI}}, n)
```

## Aroon

Aroon is an indicator system that determines whether a stock is trending or not and how strong the trend is. It consists of two lines: Aroon Up and Aroon Down. Aroon Up measures the number of periods since the highest high, while Aroon Down measures the number of periods since the lowest low.

```math
\text{Aroon Up} = \left( \frac{\text{Period} - \text{Number of Periods Since Highest High}}{\text{Period}} \right) \times 100
```

```math
\text{Aroon Down} = \left( \frac{\text{Period} - \text{Number of Periods Since Lowest Low}}{\text{Period}} \right) \times 100
```

## Average True Range (ATR)

The Average True Range (ATR) measures market volatility by calculating the average range between successive periods. It considers any gaps in trading that may occur between periods.

```math
ATR = \text{SMA}(\left| \text{High} - \text{Low} \right|, n)
```

## Bollinger Bands

Bollinger Bands consist of a middle band (SMA) and two outer bands that are standard deviations away from the middle band. They provide a relative definition of high and low prices. The upper band indicates overbought conditions, while the lower band indicates oversold conditions.

```math
\text{Upper Band} = \text{SMA}(Close, n) + (k \times \text{StdDev}(Close, n))
```

```math
\text{Lower Band} = \text{SMA}(Close, n) - (k \times \text{StdDev}(Close, n))
```

## Bollinger Bands %B

%B measures a security's position relative to the bands. It quantifies a security's price relative to the upper and lower bands.
```math
\%B = \frac{(Close - \text{Lower Band})}{(\text{Upper Band} - \text{Lower Band})}
```

## Bollinger Bands Width

The Bollinger Bands Width quantifies the width of the Bollinger Bands. It is calculated as the percentage difference between the upper and lower bands relative to the middle band.

```math
BBW = \frac{(\text{Upper Band} - \text{Lower Band})}{\text{SMA}(Close, n)}
```

## Chaikin Money Flow

Chaikin Money Flow (CMF) is an oscillator that measures the accumulation/distribution of a security over a specified period. It combines price and volume to determine buying and selling pressure.

```math
CMF = \frac{((Close - Low) - (High - Close))}{(High - Low)} \times \text{Volume}
```

## Commodity Channel Index

The Commodity Channel Index (CCI) measures a security's deviation from its statistical average. It indicates overbought and oversold conditions and potential trend reversals.

```math
CCI = \frac{(Typical Price - \text{SMA}(Typical Price, n))}{(0.015 \times \text{Mean Deviation})}
```

## Directional Movement Indicator

The Directional Movement Indicator (DMI) consists of two lines: +DI and -DI. It helps identify trend direction and strength. The +DI measures a bullish movement, while the -DI measures a bearish movement.

```math
+DI = 100 \times \frac{(Smoothed +DM)}{ATR}
```

```math
-DI = 100 \times \frac{(Smoothed -DM)}{ATR}
```

## Exponential Moving Average (EMA)

The Exponential Moving Average (EMA) gives more weight to recent prices, making it more responsive to price changes compared to the SMA. It is calculated using a smoothing factor (α).

```math
EMA = \alpha \times Close + (1 - \alpha) \times \text{Previous EMA}
```

## Fibonacci Retracement

Fibonacci Retracement is a tool used in technical analysis to identify potential support and resistance levels based on the Fibonacci sequence.

```math
\text{Fibonacci Retracement Levels} = \text{High} - (\text{High} - \text{Low}) \times \{0, 0.236, 0.382, 0.5, 0.618, 1\}
```

## Heikin-Ashi

Heikin-Ashi charts smooth out price data, making trends and market direction easier to identify. They are useful for determining trends, potential reversal points, and entry/exit signals.

```math
\text{HA Close} = \frac{(Open + High + Low + Close)}{4}
```

```math
\text{HA Open} = \frac{(\text{Previous HA Open} + \text{Previous HA Close})}{2}
```

```math
\text{HA High} = \text{Max}(Open, High, Close)
```

```math
\text{HA Low} = \text{Min}(Open, Low, Close)
```

## Ichimoku Cloud (IKH)

The Ichimoku Cloud is a comprehensive indicator that provides information about support and resistance levels, trend direction, and momentum. It consists of several components, including the Tenkan-sen, Kijun-sen, Senkou Span A, and Senkou Span B.

```math
\text{Tenkan-sen} = \frac{(Highest High + Lowest Low)}{2}
```

```math
\text{Kijun-sen} = \frac{(Highest High + Lowest Low)}{2}
```

```math
\text{Senkou Span A} = \frac{(Tenkan-sen + Kijun-sen)}{2}
```

```math
\text{Senkou Span B} = \frac{(Highest High + Lowest Low)}{2}
```

## KDJ

The KDJ indicator is a derivative of the Stochastic Oscillator. It helps identify overbought and oversold conditions and potential trend reversals.

```math
RSV = \frac{(Close - Lowest Low)}{(Highest High - Lowest Low)} \times 100
```

```math
K = \text{SMA}(RSV, n)
```

```math
D = \text{SMA}(K, m)
```

```math
J = 3 \times K - 2 \times D
```

## Keltner Channels

Keltner Channels consist of an SMA line and two outer lines that are plotted above and below the SMA line. They help identify trend direction and potential reversal points.

```math
\text{Middle Line} = \text{EMA}(Close, n)
```

```math
\text{Upper Channel} = \text{Middle Line} + (ATR \times \text{ATRMultiplier})
```

```math
\text{Lower Channel} = \text{Middle Line} - (ATR \times \text{ATRMultiplier})
```

## Momentum

Momentum measures the rate of price change and helps identify trend strength. It is calculated as the difference between the current price and the price n periods ago.

```math
Momentum = Close - Close(n \text{ periods ago})
```

## Money Flow Index (MFI)

The Money Flow Index (MFI) measures buying and selling pressure by comparing positive and negative money flows. It helps identify overbought and oversold conditions.

```math
MFI = 100 - \left( \frac{100}{(1 + \text{Money Flow Ratio})} \right)
```

```math
\text{Money Flow Ratio} = \frac{(Positive Money Flow)}{(Negative Money Flow)}
```

## Moving Average Convergence Divergence (MACD)

The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price. It consists of the MACD Line and the Signal Line.

```math
\text{MACD Line} = \text{EMA}(Close, Shorter Period) - \text{EMA}(Close, Longer Period)
```

```math
\text{Signal Line} = \text{EMA}(\text{MACD Line}, \text{Signal Period})
```

## On Balance Volume (OBV)

On Balance Volume (OBV) measures buying and selling pressure by adding volume on up days and subtracting volume on down days. It helps confirm price trends and identify potential reversals.

```math
OBV = \text{Previous OBV} + \text{Volume if Close} > \text{Previous Close}
```

```math
OBV = \text{Previous OBV} - \text{Volume if Close} < \text{Previous Close}
```

```math
OBV = \text{Previous OBV} \text{ if Close} = \text{Previous Close}
```

## Parabolic SAR (PSAR)

The Parabolic SAR (PSAR) is a trend-following indicator that helps traders identify potential reversal points in price direction. It is plotted on the price chart and moves incrementally closer to the price as the trend extends.

```math
PSAR = \text{Previous PSAR} + AF \times (EP - \text{Previous PSAR})
```

## Pivot Points

Pivot Points are used to determine potential support and resistance levels based on the previous day's trading range.

```math
\text{Pivot Point} = \frac{\text{High} + \text{Low} + \text{Close}}{3}
```

```math
\text{Support and Resistance Levels} = \{ \text{R1, R2, R3, S1, S2, S3} \}
```

## Price Oscillator (PPO)

The Price Oscillator (PPO) is a momentum oscillator that measures the difference between two moving averages as a percentage. It helps identify trend direction and potential buy/sell signals.

```math
PPO = \left( \frac{(Shorter EMA - Longer EMA)}{Longer EMA} \right) \times 100
```

## Psychological Line (PSY)

The Psychological Line (PSY) measures market sentiment by calculating the percentage of periods closing higher than the previous period. It helps identify potential trend reversals.

```math
PSY = \left( \frac{\text{Number of periods closing higher}}{\text{Total number of periods}} \right) \times 100
```

## Rate of Change (ROC)

The Rate of Change (ROC) measures the percentage change in price over a specified period. It helps identify momentum and potential trend reversals.

```math
ROC = \left( \frac{(Close - \text{Close n periods ago})}{\text{Close n periods ago}} \right) \times 100
```

## Relative Strength Index (RSI)

The Relative Strength Index (RSI) measures the magnitude of recent price changes to evaluate overbought or oversold conditions in a security. It ranges from 0 to 100, with values above 70 indicating overbought conditions and values below 30 indicating oversold conditions.

```math
RSI = 100 - \left( \frac{100}{(1 + RS)} \right)
```

```math
RS = \text{Average of } x \text{ days' up closes} / \text{Average of } x \text{ days' down closes}
```

## Relative Vigor Index (RVI)

The Relative Vigor Index (RVI) measures the conviction of a recent price action and the level of conviction behind it.

```math
RVI = \frac{\text{Close} - \text{Open}}{\text{High} - \text{Low}}
```

## Running Moving Average (RMA)

The Running Moving Average (RMA) is a type of moving average that continually updates as new data becomes available. It helps identify trends and potential reversal points more accurately.

```math
RMA = \frac{(\text{Previous RMA} \times (n - 1)) + Close}{n}
```

## Simple Moving Average (SMA)

The Simple Moving Average (SMA) is the average of a security's closing prices over a specified period. It is used to identify trends and potential support/resistance levels.

```math
SMA = \frac{\text{Sum of Close Prices for } n \text{ periods}}{n}
```

## Smoothed Moving Average (SMMA)

The Smoothed Moving Average (SMMA) is similar to the Simple Moving Average (SMA) but with a smoother curve. It helps filter out noise and identify trends more accurately.

```math
SMMA = \frac{\text{Sum of Close Prices for } n \text{ periods}}{n}
```

## Stochastic Oscillator

The Stochastic Oscillator measures the relative position of a security's closing price within its price range over a specified period. It helps identify overbought and oversold conditions and potential trend reversals.
```math
\%K = \left( \frac{(Current Close - Lowest Low)}{(Highest High - Lowest Low)} \right) \times 100
```

```math
\%D = \text{SMA}(\%K, m)
```

## Stochastic RSI (StochRSI)

Stochastic RSI (StochRSI) measures the level of RSI relative to its range over a set period.

```math
\%K = \left( \frac{(Current RSI - Lowest RSI)}{(Highest RSI - Lowest RSI)} \right) \times 100
```

```math
\%D = \text{SMA}(\%K, m)
```

## Volume Weighted Average Price (VWAP)

The Volume Weighted Average Price (VWAP) determines the average price a security has traded at throughout the day, based on both volume and price.

```math
VWAP = \frac{\sum(\text{Price} \times \text{Volume})}{\sum(\text{Volume})}
```

## Williams %R

The Williams %R indicator measures overbought and oversold levels in a security. It ranges from 0 to -100, with values above -20 indicating overbought conditions and values below -80 indicating oversold conditions.
```math
\%R = \left( \frac{{\text{{Highest High}} - \text{{Close}}}}{{\text{{Highest High}} - \text{{Lowest Low}}}} \right) \times -100
```
