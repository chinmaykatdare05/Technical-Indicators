def pivot_points(high, low, close):
    pivot_point = (high + low + close) / 3
    support1 = (2 * pivot_point) - high
    support2 = pivot_point - (high - low)
    support3 = low - 2 * (high - pivot_point)
    resistance1 = (2 * pivot_point) - low
    resistance2 = pivot_point + (high - low)
    resistance3 = high + 2 * (pivot_point - low)

    return {
        "Pivot Point": pivot_point,
        "Support 1": support1,
        "Support 2": support2,
        "Support 3": support3,
        "Resistance 1": resistance1,
        "Resistance 2": resistance2,
        "Resistance 3": resistance3
    }

print(pivot_points(high, low, close))
