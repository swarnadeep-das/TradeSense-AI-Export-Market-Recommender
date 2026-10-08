"""Feature engineering: next-year export growth as the supervised target."""
import numpy as np
import pandas as pd

FEATURES = [
    "growth_1y", "growth_3y_cagr", "gdp_growth", "log_gdp_pc", "tariff_pct",
    "tariff_change", "log_distance", "fta", "fx_volatility", "income_elasticity_proxy",
    "growth_trend", "post_shock_year",
]


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["country", "hs_code", "year"]).copy()
    g = d.groupby(["country", "hs_code"])
    d["growth_1y"] = g["export_value_musd"].pct_change()
    d["growth_3y_cagr"] = (d["export_value_musd"] / g["export_value_musd"].shift(3)) ** (1 / 3) - 1
    d["tariff_change"] = g["tariff_pct"].diff()
    d["log_gdp_pc"] = np.log(d["gdp_per_capita"])
    d["log_distance"] = np.log(d["distance_km"])
    d["income_elasticity_proxy"] = d["gdp_growth"] * d["log_gdp_pc"]
    # long-run trend: expanding mean of past growth (shifted so it uses only history)
    d["growth_trend"] = g["growth_1y"].transform(lambda s: s.shift(1).expanding().mean())
    d["post_shock_year"] = (d["year"] == 2021).astype(int)  # rebound after 2020 shock
    # target: growth in the FOLLOWING year (no look-ahead in features)
    d["target_growth_next"] = g["export_value_musd"].pct_change().groupby([d["country"], d["hs_code"]]).shift(-1)
    return d
