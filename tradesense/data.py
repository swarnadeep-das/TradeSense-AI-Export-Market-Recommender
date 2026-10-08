"""Data layer: synthetic sample generator + loader for real trade data.

The bundled dataset is SYNTHETIC (seeded, reproducible). It mimics the shape of
UN Comtrade / WITS / DGCI&S bilateral data so the pipeline can be demonstrated
offline. To use real data, supply a CSV with the same columns (see COLUMNS).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COLUMNS = [
    "year", "country", "region", "hs_code", "product", "export_value_musd",
    "gdp_growth", "gdp_per_capita", "tariff_pct", "distance_km", "fta", "fx_volatility",
]

# country, region, gdp_pc (USD k), distance_km from Delhi, FTA/CEPA with India, base demand scale
COUNTRIES = [
    ("United States", "North America", 80, 11500, 0, 1.00), ("UAE", "Middle East", 50, 2200, 1, 0.90),
    ("Germany", "Europe", 52, 6200, 0, 0.60), ("United Kingdom", "Europe", 48, 6700, 1, 0.55),
    ("Netherlands", "Europe", 58, 6400, 0, 0.50), ("Singapore", "ASEAN", 82, 4300, 1, 0.40),
    ("Japan", "East Asia", 34, 5800, 1, 0.38), ("Australia", "Oceania", 65, 8300, 1, 0.30),
    ("Saudi Arabia", "Middle East", 32, 3100, 0, 0.45), ("Bangladesh", "South Asia", 2.7, 1500, 0, 0.45),
    ("Vietnam", "ASEAN", 4.3, 3300, 1, 0.30), ("Brazil", "Latin America", 9, 14500, 0, 0.28),
    ("South Africa", "Africa", 6.7, 8500, 0, 0.22), ("Nigeria", "Africa", 2.2, 7900, 0, 0.18),
    ("Kenya", "Africa", 2.1, 5100, 0, 0.10), ("Mexico", "Latin America", 11, 14700, 0, 0.20),
    ("Indonesia", "ASEAN", 4.9, 5000, 1, 0.26), ("China", "East Asia", 12.7, 3800, 0, 0.55),
]
PRODUCTS = [  # hs_code, name, base value, tariff sensitivity, income elasticity
    ("30", "Pharmaceuticals", 160, 0.5, 0.8), ("71", "Gems & Jewellery", 140, 0.9, 1.2),
    ("85", "Electronics", 110, 0.7, 1.0), ("10", "Cereals (Rice)", 90, 1.2, 0.2),
    ("52", "Cotton & Textiles", 80, 1.1, 0.6), ("87", "Auto Components", 70, 0.8, 0.9),
]
YEARS = list(range(2015, 2025))


def generate_sample(seed: int = 45) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cname, region, gpc, dist, fta, scale in COUNTRIES:
        gdp_g = rng.uniform(1.0, 6.0)
        tariff0 = rng.uniform(2, 20) * (0.55 if fta else 1.0)
        for hs, pname, base, tsens, inc in PRODUCTS:
            level = base * scale * rng.uniform(0.7, 1.3)
            trend = rng.normal(0.05, 0.04) + 0.01 * fta
            tariff = tariff0 * rng.uniform(0.6, 1.4)
            for y in YEARS:
                gdp_g_y = float(np.clip(gdp_g + rng.normal(0, 1.2), -3, 9))
                fxv = float(np.clip(rng.gamma(2, 2.2), 0.5, 25))
                tariff = max(0.0, tariff + rng.normal(-0.3 * fta, 0.5))
                growth = (trend + 0.012 * gdp_g_y * inc - 0.0012 * tsens * tariff
                          - 0.0006 * fxv - 0.0000015 * dist + rng.normal(0, 0.03))
                level *= (1 + growth)
                if y == 2020:  # pandemic shock
                    level *= rng.uniform(0.78, 0.92)
                elif y == 2021:  # partial rebound
                    level *= rng.uniform(1.06, 1.15)
                rows.append((y, cname, region, hs, pname, round(max(level, 0.1), 2),
                             round(gdp_g_y, 2), round(gpc * (1.02 ** (y - 2015)), 2),
                             round(tariff, 2), dist, fta, round(fxv, 2)))
    return pd.DataFrame(rows, columns=COLUMNS)


def load(path: str | None = None) -> pd.DataFrame:
    """Load a CSV in the COLUMNS schema; fall back to the synthetic sample."""
    if path:
        df = pd.read_csv(path, dtype={"hs_code": str})
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        return df[COLUMNS]
    return generate_sample()
