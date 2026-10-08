"""Market Attractiveness Index (MAI): blends ML growth forecast with
market size, ease-of-access (tariff, FTA, distance) and risk (FX volatility)."""
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"growth": 0.35, "size": 0.25, "access": 0.25, "risk": 0.15}


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else s * 0 + 0.5


def market_attractiveness(latest: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}
    d = latest.copy()
    d["s_growth"] = _minmax(d["pred_growth"])
    d["s_size"] = _minmax(np.log(d["export_value_musd"]))
    access = (1 - _minmax(d["tariff_pct"])) * 0.5 + d["fta"] * 0.3 + (1 - _minmax(d["log_distance"])) * 0.2
    d["s_access"] = _minmax(access)
    d["s_risk"] = 1 - _minmax(d["fx_volatility"])   # higher = safer
    d["MAI"] = 100 * (w["growth"] * d["s_growth"] + w["size"] * d["s_size"]
                      + w["access"] * d["s_access"] + w["risk"] * d["s_risk"])
    return d.sort_values("MAI", ascending=False).reset_index(drop=True)
