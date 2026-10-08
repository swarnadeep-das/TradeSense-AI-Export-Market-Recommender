# 🌐 TradeSense — AI Export Market Recommender

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Final-year AI project** · MBA (International Business), Indian Institute of Foreign Trade (IIFT), Delhi
> **Author:** Swarnadeep Das · Roll No. 45C

> **Data disclaimer:** the bundled dataset is **synthetic** (seeded, reproducible) and only mimics the shape of
> UN Comtrade / WITS / DGCI&S bilateral trade data so the pipeline runs offline. It is not real trade
> data and results must not be used for real business decisions. Swap in real data via the CSV uploader.

## Table of Contents
- [The Problem](#the-problem) · [How It Works](#how-it-works) · [Features](#features) · [Backtest Results](#backtest-results)
- [Getting Started](#getting-started) · [Project Structure](#project-structure) · [Using Real Data](#using-real-data) · [Limitations](#limitations) · [Author & Project Details](#author--project-details)

## The Problem
India's exporters — especially MSMEs — usually pick destination markets by habit or by following competitors.
Choosing among 190+ countries needs growth outlook, tariff barriers, FTA access, distance and currency risk
weighed together. TradeSense turns that into a ranked, explainable shortlist per product (HS code).

## How It Works
```
Bilateral trade CSV ──► Feature engineering ──► Gradient-boosting model ──► Next-year growth forecast
 (country × HS × year)   (lags, CAGR, tariff     (time-ordered backtest)            │
                          change, GDP, FX, FTA)                                     ▼
                                                  Market Attractiveness Index (MAI, 0–100)
                                                  = growth + market size + ease of access + low FX risk
                                                    (weights adjustable by the user)  ──► Streamlit dashboard
```
1. **Features** – 1-yr growth, 3-yr CAGR, long-run trend, GDP growth, income level, tariff & tariff change, log-distance, FTA flag, FX volatility.
2. **Model** – shallow Gradient Boosting predicts next-year export growth for every market–product pair.
3. **Scoring** – forecast is blended with size, access (tariff/FTA/distance) and risk into the **MAI**; the exporter's strategy weights drive the ranking.
4. **Projection** – a 3-year value path with damping towards long-run growth, so forecasts don't compound unrealistically.

## Features
- Product selector and live **strategy-weight sliders** (growth-led vs. safety-led exporter)
- Ranked market table + chart, downloadable as CSV
- Per-market 3-year export projection
- Transparent **Model** tab: backtest vs. baselines and feature importance
- Upload your own CSV

## Backtest Results
Rolling-origin backtest (each year predicted using only earlier years; no random split, to avoid look-ahead leakage), on the synthetic sample:

| Test year | Model | MAE ↓ | Spearman ↑ |
|---|---|---|---|
| 2021 | **Gradient Boosting** | **0.094** | **0.77** |
| 2021 | Naive (last year) | 0.116 | 0.63 |
| 2022 | Gradient Boosting | 0.055 | 0.52 |
| 2022 | Naive (last year) | 0.034 | 0.72 |
| 2023 | Gradient Boosting | 0.040 | 0.74 |
| 2023 | Naive (last year) | 0.034 | 0.75 |

The model clearly helps when conditions shift (2021 rebound) but only roughly matches a "same as last year" baseline in calm years. Reported as-is rather than tuned to look better.

## Getting Started
```bash
git clone https://github.com/swarnadeep-das/TradeSense-AI-Export-Market-Recommender.git
cd TradeSense-AI-Export-Market-Recommender
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
pytest -q          # run the tests
```

## Project Structure
```
app.py                  Streamlit dashboard
tradesense/data.py      Synthetic generator + CSV loader
tradesense/features.py  Feature engineering (no look-ahead)
tradesense/model.py     Gradient boosting, backtest
tradesense/scoring.py   Market Attractiveness Index
tradesense/forecast.py  Damped 3-year projection
tests/                  pytest suite
```

## Using Real Data
Provide a CSV with columns: `year, country, region, hs_code, product, export_value_musd, gdp_growth, gdp_per_capita, tariff_pct, distance_km, fta, fx_volatility`
(`hs_code` as text). Sources: UN Comtrade, WITS (tariffs), World Bank (GDP), CEPII (distance), Ministry of Commerce DGCI&S.

## Limitations
- Demo data is synthetic; real-world accuracy is untested.
- No sanctions, non-tariff measures, logistics cost or competitor-share data.
- Annual data only; no sub-annual seasonality.

## Author & Project Details
- **Author:** Swarnadeep Das
- **Roll No / ID:** 45C
- **Program:** MBA (International Business)
- **Institution:** Indian Institute of Foreign Trade (IIFT), Delhi
- **Email:** swarnadeepdas@icloud.com
- **GitHub:** [@swarnadeep-das](https://github.com/swarnadeep-das)
- **LinkedIn:** [linkedin.com/in/swarnadeepdas](https://linkedin.com/in/swarnadeepdas)

© 2026 Swarnadeep Das · MIT License
