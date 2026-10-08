import pandas as pd
from tradesense import data, features, forecast, model, scoring


def test_sample_is_reproducible_and_valid():
    a, b = data.generate_sample(), data.generate_sample()
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == data.COLUMNS and (a.export_value_musd > 0).all()


def test_no_target_leakage():
    p = features.build_panel(data.generate_sample())
    last = p[p.year == p.year.max()]
    assert last["target_growth_next"].isna().all()


def test_backtest_is_time_ordered_and_useful():
    tm = model.train(data.generate_sample())
    bt = tm.metrics
    assert set(bt.test_year) == {2021, 2022, 2023}
    g = bt[bt.model == "Gradient Boosting"]
    assert g.spearman.mean() > 0.5                       # ranks markets meaningfully
    y21 = bt[bt.test_year == 2021].set_index("model").mae
    assert y21["Gradient Boosting"] < y21["Naive (last year)"]  # wins when conditions shift


def test_ranking_sorted_and_bounded():
    tm = model.train(data.generate_sample())
    r = scoring.market_attractiveness(model.predict_latest(tm, "30"))
    assert r["MAI"].is_monotonic_decreasing and r["MAI"].between(0, 100).all()


def test_forecast_damps_toward_long_run():
    out = forecast.project(100, 0.20)
    assert len(out) == 3 and out[2] / out[1] - 1 < out[1] / out[0] - 1
