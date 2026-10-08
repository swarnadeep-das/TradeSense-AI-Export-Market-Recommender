"""TradeSense dashboard - run with: streamlit run app.py
Author: Swarnadeep Das | MBA (International Business), IIFT Delhi | Roll No. 45C
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from tradesense import __author__, data, forecast, model, scoring

st.set_page_config(page_title="TradeSense | Swarnadeep Das", page_icon="🌐", layout="wide")


@st.cache_data
def get_data(upload):
    return data.load(upload) if upload else data.load()


@st.cache_resource
def get_model(df_key, df):
    return model.train(df)


st.title("🌐 TradeSense — AI Export Market Recommender")
st.caption("Which overseas markets should an Indian exporter target next? · Built by "
           f"**{__author__}**, MBA (International Business), IIFT Delhi · Roll No. 45C")

with st.sidebar:
    st.header("Inputs")
    up = st.file_uploader("Trade data CSV (optional)", type="csv")
    df = get_data(up)
    if up is None:
        st.info("Using the bundled **synthetic** sample dataset.")
    products = df[["hs_code", "product"]].drop_duplicates()
    label = st.selectbox("Product", products["product"])
    hs = products.loc[products["product"] == label, "hs_code"].iloc[0]
    st.subheader("Strategy weights")
    w = {"growth": st.slider("Growth", 0, 100, 35), "size": st.slider("Market size", 0, 100, 25),
         "access": st.slider("Ease of access", 0, 100, 25), "risk": st.slider("Low FX risk", 0, 100, 15)}
    top_n = st.slider("Top N markets", 3, 12, 8)

tm = get_model(hash(df.to_json()), df)
ranked = scoring.market_attractiveness(model.predict_latest(tm, hs), w if sum(w.values()) else None)

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Recommendations", "📈 Projection", "🧠 Model", "ℹ️ About"])

with tab1:
    top = ranked.head(top_n)
    c1, c2, c3 = st.columns(3)
    c1.metric("Top market", top.iloc[0]["country"], f"MAI {top.iloc[0]['MAI']:.0f}/100")
    c2.metric("Fastest forecast growth", top.sort_values("pred_growth").iloc[-1]["country"],
              f"{top['pred_growth'].max():.1%}")
    c3.metric("Markets scored", len(ranked))
    fig = px.bar(top.iloc[::-1], x="MAI", y="country", orientation="h", color="region",
                 title=f"Market Attractiveness Index — {label}")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(top[["country", "region", "MAI", "pred_growth", "export_value_musd", "tariff_pct", "fta",
                      "fx_volatility"]].rename(columns={"pred_growth": "forecast_growth",
                      "export_value_musd": "last_export_musd"}), width="stretch", hide_index=True)
    st.download_button("Download ranking (CSV)", ranked.to_csv(index=False), f"tradesense_{hs}.csv")

with tab2:
    pick = st.selectbox("Market", ranked["country"])
    row = ranked[ranked.country == pick].iloc[0]
    yrs = int(df.year.max())
    hist = df[(df.country == pick) & (df.hs_code == hs)][["year", "export_value_musd"]].assign(kind="Actual")
    proj = forecast.project(row["export_value_musd"], row["pred_growth"])
    fut = pd.DataFrame({"year": [yrs + i for i in (1, 2, 3)], "export_value_musd": proj, "kind": "Projected"})
    st.plotly_chart(px.line(pd.concat([hist, fut]), x="year", y="export_value_musd", color="kind",
                    markers=True, title=f"{label} exports to {pick} (USD m)"), width="stretch")

with tab3:
    bt = tm.metrics
    st.write("Rolling-origin backtest: each year is predicted using **only earlier years**. "
             "Lower MAE is better; higher Spearman means better *ranking* of markets.")
    st.dataframe(bt.round(3), width="stretch", hide_index=True)
    st.plotly_chart(px.bar(bt, x="test_year", y="spearman", color="model", barmode="group",
                    title="Rank correlation with actual next-year growth"), width="stretch")
    st.plotly_chart(px.bar(tm.importances.iloc[::-1], orientation="h", title="Feature importance"),
                    width="stretch")
    st.caption("Honest note: on the synthetic sample the model clearly wins in the shock year (2021) but "
               "only roughly matches a naive last-year baseline afterwards. Real data will differ.")

with tab4:
    st.markdown(open("README.md", encoding="utf-8").read().split("## Getting Started")[0].split("## The Problem")[1])
