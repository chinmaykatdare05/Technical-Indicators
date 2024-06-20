# Technical Indicators: Formulas, Insights, and Applications

This guide offers descriptions and formulas for a variety of technical indicators crucial in financial analysis and trading. Understanding these indicators can empower you to analyze price movements, identify trends, and make informed trading decisions.

Explore the descriptions and formulas below to gain insights into each indicator's significance and its application in refining your trading strategies.

Let's delve into the world of technical indicators to enhance your trading skills!

---

## Accumulation Distribution Line (ADL)

The Accumulation Distribution Line (ADL) is a momentum indicator that combines price and volume to reflect the cumulative flow of money into or out of a security.

```math
ADL = \text{Previous ADL} + \text{Current Money Flow Volume}
```

## Aroon

Aroon identifies trend strength and direction using two lines: Aroon Up and Aroon Down.

```math
\text{Aroon Up} = \left( \frac{\text{Period} - \text{Number of Periods Since Highest High}}{\text{Period}} \right) \times 100
```

```math
\text{Aroon Down} = \left( \frac{\text{Period} - \text{Number of Periods Since Lowest Low}}{\text{Period}} \right) \times 100
```

## Average Directional Index (ADX)

The Average Directional Index (ADX) measures the strength of a trend, irrespective of its direction.

```math
\text{ADX} = \text{SMA}\left(\frac{\text{TR}_n}{\text{DMI}}, n\right)
```

## Average True Range (ATR)

ATR measures market volatility by calculating the average range between successive periods.

```math
ATR = \text{SMA}\left(\left| \text{High} - \text{Low} \right|, n\right)
```

## Bollinger Bands

Bollinger Bands use a middle band (SMA) and two outer bands to identify price volatility and potential overbought or oversold conditions.

```math
\text{Upper Band} = \text{SMA}(Close, n) + (k \times \text{StdDev}(Close, n))
```

```math
\text{Lower Band} = \text{SMA}(Close, n) - (k \times \text{StdDev}(Close, n))
```

## Bollinger Bands %B

%B indicates a security's price relative to the Bollinger Bands.

```math
\%B = \frac{(Close - \text{Lower Band})}{(\text{Upper Band} - \text{Lower Band})}
```

## Bollinger Bands Width (BBW)

BBW measures the width of the Bollinger Bands relative to the middle band.

```math
BBW = \frac{(\text{Upper Band} - \text{Lower Band})}{\text{SMA}(Close, n)}
```

## Chaikin Money Flow (CMF)

CMF uses price and volume to measure buying and selling pressure.

```math
CMF = \frac{((Close - Low) - (High - Close))}{(High - Low)} \times \text{Volume}
```

## Commodity Channel Index (CCI)

CCI identifies overbought and oversold conditions and trend reversals.

```math
CCI = \frac{(Typical Price - \text{SMA}(Typical Price, n))}{(0.015 \times \text{Mean Deviation})}
```

## Directional Movement Indicator (DMI)

DMI consists of +DI and -DI lines to determine trend direction and strength.

```math
+DI = 100 \times \frac{(Smoothed +DM)}{ATR}
```

```math
-DI = 100 \times \frac{(Smoothed -DM)}{ATR}
```

## Exponential Moving Average (EMA)

EMA provides more weight to recent prices for faster responsiveness.

```math
EMA = \alpha \times Close + (1 - \alpha) \times \text{Previous EMA}
```

## Fibonacci Retracement

Fibonacci Retracement identifies potential support and resistance levels based on the Fibonacci sequence.

```math
\text{Fibonacci Retracement Levels} = \text{High} - (\text{High} - \text{Low}) \times \{0, 0.236, 0.382, 0.5, 0.618, 1\}
```

## Heikin-Ashi

Heikin-Ashi charts smooth price data for trend identification.

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

IKH provides support, resistance, trend direction, and momentum signals.

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

## Keltner Channels

Keltner Channels use an SMA line and outer bands to identify trend and reversals.

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

Momentum measures the rate of price change.

```math
Momentum = Close - Close(n \text{ periods ago})
```

## Money Flow Index (MFI)

MFI measures buying and selling pressure.

```math
MFI = 100 - \left( \frac{100}{(1 + \text{Money Flow Ratio})} \right)
```

```math
\text{Money Flow Ratio} = \frac{(Positive Money Flow)}{(Negative Money Flow)}
```

## Moving Average Convergence Divergence (MACD)

MACD shows the relationship between two moving averages.

```math
\text{MACD Line} = \text{EMA}(Close, Shorter Period) - \text{EMA}(Close, Longer Period)
```

```math
\text{Signal Line} = \text{EMA}(\text{MACD Line}, \text{Signal Period})
```

## On Balance Volume (OBV)

OBV confirms trends using volume flow.

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

PSAR identifies potential reversal points.

```math
PSAR = \text{Previous PSAR} + AF \times (EP - \text{Previous PSAR})
```

## Pivot Points

Pivot Points determine support and resistance levels based on the previous day's trading range.

```math
\text{Pivot Point} = \frac{\text{High} + \text{Low} + \text{Close}}{3}
```

```math
\text{Support and Resistance Levels} = \{ \text{R1, R2, R3, S1, S2, S3} \}
```

## Rate of Change (ROC)

ROC measures percentage price change.

```math
ROC = \left( \frac{(Close - \text{Close n periods ago})}{\text{Close n periods ago}} \right) \times 100
```

## Relative Strength Index (RSI)

RSI identifies overbought or oversold conditions.

```math
RSI = 100 - \left( \frac{100}{(1 + RS)} \right)
```

```math
RS = \text{Average of } x \text{ days' up closes} / \text{Average of } x \text{ days' down closes}
```

## Relative Vigor Index (RVI)

RVI measures price momentum.

```math
RVI = \frac{\text{Close} - \text{Open}}{\text{High} - \text{Low}}
```

## Simple Moving Average (SMA)

SMA averages closing prices.

```math
SMA = \frac{\text{Sum of Close Prices for } n \text{ periods}}{n}
```

## Stochastic Oscillator

Stochastic Oscillator identifies overbought/oversold conditions.

```math
\%K = \left( \frac{(Current Close - Lowest Low)}{(Highest High - Lowest Low)} \right) \times 100
```

```math
\%D = \text{SMA}(\%K, m)
```

## Stochastic RSI (StochRSI)

StochRSI measures RSI relative to its range.

```math
\%K = \left( \frac{(Current RSI - Lowest RSI)}{(Highest RSI - Lowest RSI)} \right) \times 100
```

```math
\%D = \text{SMA}(\%K, m)
```

## Volume Weighted Average Price (VWAP)

VWAP averages price based on volume.

```math
VWAP = \frac{\sum(\text{Price} \times \text{Volume})}{\sum(\text{Volume})}
```

## Williams %R

Williams %R identifies overbought/oversold levels.

```math
\%R = \left( \frac{{\text{{Highest High}} - \text{{Close}}}}{{\text{{Highest High}} - \text{{Lowest Low}}}} \right) \times -100
```
