"""Gradient-boosting model predicting next-year export growth per market-product."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from .features import FEATURES, build_panel


@dataclass
class TrainedModel:
    model: GradientBoostingRegressor
    metrics: pd.DataFrame  # backtest results
    importances: pd.Series
    panel: pd.DataFrame


def _gbr(seed: int) -> GradientBoostingRegressor:
    # Deliberately shallow: ~600 noisy panel rows overfit badly with deeper trees.
    return GradientBoostingRegressor(n_estimators=100, max_depth=1, learning_rate=0.05,
                                     min_samples_leaf=20, random_state=seed)


def backtest(panel: pd.DataFrame, test_years=(2021, 2022, 2023), seed: int = 45) -> pd.DataFrame:
    """Rolling-origin backtest: for each year, train on strictly earlier years only.

    Reports MAE of growth forecasts and Spearman rank correlation (what matters
    for a *ranking* product). A random split would leak future data, so none is used.
    """
    lab = panel.dropna(subset=FEATURES + ["target_growth_next"])
    rows = []
    for ty in test_years:
        tr, te = lab[lab.year < ty], lab[lab.year == ty]
        if te.empty or len(tr) < 50:
            continue
        gbr = _gbr(seed).fit(tr[FEATURES], tr["target_growth_next"])
        rdg = Ridge(alpha=1.0).fit(tr[FEATURES], tr["target_growth_next"])
        y = te["target_growth_next"]
        for name, pred in (("Gradient Boosting", gbr.predict(te[FEATURES])),
                           ("Ridge", rdg.predict(te[FEATURES])), ("Naive (last year)", te["growth_1y"].values)):
            rows.append({"test_year": ty, "model": name, "n": len(te),
                         "mae": mean_absolute_error(y, pred),
                         "spearman": pd.Series(pred, index=y.index).corr(y, method="spearman")})
    return pd.DataFrame(rows)


def train(df: pd.DataFrame, seed: int = 45) -> TrainedModel:
    panel = build_panel(df)
    labelled = panel.dropna(subset=FEATURES + ["target_growth_next"])
    bt = backtest(panel, seed=seed)
    gbr = _gbr(seed).fit(labelled[FEATURES], labelled["target_growth_next"])  # final fit: all data
    imp = pd.Series(gbr.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return TrainedModel(gbr, bt, imp, panel)


def predict_latest(tm: TrainedModel, hs_code: str) -> pd.DataFrame:
    """Forecast growth for the year after the latest observed year."""
    p = tm.panel
    latest = p[(p.year == p.year.max()) & (p.hs_code == hs_code)].dropna(subset=FEATURES).copy()
    latest["pred_growth"] = tm.model.predict(latest[FEATURES])
    return latest
