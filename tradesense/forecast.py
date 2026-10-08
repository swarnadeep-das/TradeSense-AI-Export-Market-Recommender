"""Simple 3-year export value projection using the ML growth signal, damped
toward long-run mean growth (so forecasts don't compound unrealistically)."""
import numpy as np


def project(last_value: float, pred_growth: float, years: int = 3, damping: float = 0.7,
            long_run: float = 0.04) -> list[float]:
    out, g, v = [], pred_growth, last_value
    for _ in range(years):
        v *= 1 + g
        out.append(round(v, 2))
        g = long_run + damping * (g - long_run)
    return out
