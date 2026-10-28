import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Delhi PM2.5 Forecast", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl"), joblib.load("features.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/app_data.csv", parse_dates=["datetime"])


def category(pm):
    # CPCB PM2.5 (24-hr) categories
    if pm <= 30: return "Good"
    if pm <= 60: return "Satisfactory"
    if pm <= 90: return "Moderate"
    if pm <= 120: return "Poor"
    if pm <= 250: return "Very Poor"
    return "Severe"


model, features = load_model()
df = load_data()

st.title("Delhi PM2.5 Forecast (24 hours ahead)")
st.caption("XGBoost model trained on hourly 2025 data from 6 Delhi stations.")

st.sidebar.header("Select input")
station = st.sidebar.selectbox("Station", sorted(df["location"].unique()))
sub = df[df["location"] == station]

min_d, max_d = sub["datetime"].min().date(), sub["datetime"].max().date()
date = st.sidebar.date_input("Date", value=min_d, min_value=min_d, max_value=max_d)
hour = st.sidebar.slider("Hour of day", 0, 23, 12)

when = pd.Timestamp(date) + pd.Timedelta(hours=hour)
row = sub[sub["datetime"] == when]

if row.empty:
    st.warning("No data for this selection. Try another date or hour.")
else:
    pred = float(model.predict(row[features])[0])
    now_pm = float(row["pm2_5"].iloc[0])
    actual = row["target"].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("PM2.5 at selected time", f"{now_pm:.1f} µg/m³")
    c2.metric("Predicted PM2.5 (+24h)", f"{pred:.1f} µg/m³", category(pred))
    c3.metric("Actual PM2.5 (+24h)", f"{actual:.1f} µg/m³",
              f"{pred - actual:+.1f} error", delta_color="off")

    window = sub[(sub["datetime"] >= when - pd.Timedelta(days=3)) &
                 (sub["datetime"] <= when + pd.Timedelta(days=2))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["datetime"], y=window["pm2_5"],
                             mode="lines", name="Observed PM2.5"))
    fig.add_trace(go.Scatter(x=[when + pd.Timedelta(hours=24)], y=[pred],
                             mode="markers", name="Prediction",
                             marker=dict(size=12, color="red")))
    fig.update_layout(title=f"{station}: PM2.5 around selected time",
                      xaxis_title="Time", yaxis_title="PM2.5 (µg/m³)")
    st.plotly_chart(fig, use_container_width=True)