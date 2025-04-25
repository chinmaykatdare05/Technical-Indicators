from typing import Dict


def pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    pivot_point = (high + low + close) / 3
    range_high_low = high - low

    support1 = (2 * pivot_point) - high
    support2 = pivot_point - range_high_low
    support3 = low - 2 * (high - pivot_point)
    resistance1 = (2 * pivot_point) - low
    resistance2 = pivot_point + range_high_low
    resistance3 = high + 2 * (pivot_point - low)

    return {
        "Pivot Point": pivot_point,
        "Support 1": support1,
        "Support 2": support2,
        "Support 3": support3,
        "Resistance 1": resistance1,
        "Resistance 2": resistance2,
        "Resistance 3": resistance3,
    }
