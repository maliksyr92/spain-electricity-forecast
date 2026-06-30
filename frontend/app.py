"""
frontend/app.py
===============
A live Streamlit dashboard. It reads live/state.json (written every hour by
the simulator) and draws three things:

  * the actual electricity demand so far,
  * the model's 1-hour-ahead prediction overlaid on that history,
  * the current 24-hour-ahead forecast extending into the future,

plus a few live metrics (current demand, next-hour forecast, running error).

The page refreshes itself every couple of seconds, so the chart grows live.

Run it from the PROJECT ROOT with:
    streamlit run frontend/app.py
A browser tab opens automatically at http://localhost:8501
"""

import os
import json
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

LIVE_DIR = os.environ.get("LIVE_DIR", "live")
STATE_PATH = os.path.join(LIVE_DIR, "state.json")
REFRESH_SECONDS = 2

st.set_page_config(page_title="Spain Electricity Demand — Live",
                   page_icon="⚡", layout="wide")
st.title("⚡ التنبؤ اللحظي بالطلب على الكهرباء — إسبانيا")
st.caption("الطلب الفعلي مقابل المتنبَّأ + توقّع الـ24 ساعة القادمة (محاكاة بيانات 2018)")


def load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


state = load_state()

# Friendly waiting screen if the simulator hasn't produced data yet.
if not state or not state.get("history"):
    st.info("⏳ في انتظار بدء المحاكي...\n\n"
            "شغّل الخادم ثم المحاكي، وستظهر البيانات هنا تلقائياً.")
    time.sleep(REFRESH_SECONDS)
    st.rerun()

# ---- build tables -----------------------------------------------------------
hist = pd.DataFrame(state["history"])
fc = pd.DataFrame(state["forecast"])
hist["timestamp"] = pd.to_datetime(hist["timestamp"])
fc["timestamp"] = pd.to_datetime(fc["timestamp"])

# ---- live metrics -----------------------------------------------------------
latest_actual = float(hist["actual"].iloc[-1])
next_hour = float(fc["predicted"].iloc[0])
paired = hist.dropna(subset=["predicted"])      # hours we can score
mae = (paired["actual"] - paired["predicted"]).abs().mean() if len(paired) else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("الوقت الحالي (محاكاة)", str(state["now"])[:16])
c2.metric("الطلب الفعلي الآن", f"{latest_actual:,.0f} MW")
c3.metric("تنبؤ الساعة القادمة", f"{next_hour:,.0f} MW")
c4.metric("متوسط الخطأ المباشر (MAE)",
          f"{mae:,.0f} MW" if pd.notna(mae) else "—")

# ---- chart ------------------------------------------------------------------
fig = go.Figure()

# 1) actual demand (history)
fig.add_trace(go.Scatter(
    x=hist["timestamp"], y=hist["actual"],
    name="الطلب الفعلي", mode="lines",
    line=dict(color="#2563eb", width=2.5)))

# 2) predicted overlay on history (1-hour-ahead, dotted)
if len(paired):
    fig.add_trace(go.Scatter(
        x=paired["timestamp"], y=paired["predicted"],
        name="المتنبَّأ (تاريخي)", mode="lines",
        line=dict(color="#f59e0b", width=2, dash="dot")))

# 3) the 24h forecast, connected to the last actual point so it flows on
fc_x = [hist["timestamp"].iloc[-1]] + list(fc["timestamp"])
fc_y = [latest_actual] + list(fc["predicted"])
fig.add_trace(go.Scatter(
    x=fc_x, y=fc_y,
    name="توقّع 24 ساعة", mode="lines",
    line=dict(color="#16a34a", width=2.5, dash="dash")))

fig.update_layout(
    height=540, xaxis_title="الوقت", yaxis_title="الطلب (MW)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(t=40, r=20, l=20, b=20), hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)
st.caption(f"يتحدّث تلقائياً كل {REFRESH_SECONDS} ثانية · المصدر: {STATE_PATH}")

# ---- auto-refresh -----------------------------------------------------------
time.sleep(REFRESH_SECONDS)
st.rerun()
