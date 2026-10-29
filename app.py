"""Streamlit demo of the ML1 focus-state model: describe the last 15 minutes at the PC, get focus/distracted/break."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from focus_features import CATEGORIES, FEATURE_COLUMNS, build_features

TZ = "Asia/Seoul"
MONDAY = pd.Timestamp("2026-10-26", tz=TZ)
LABEL_KO = {"focus": "집중", "distracted": "딴짓", "break": "휴식"}
CATEGORY_KO = {"ide": "IDE (코딩)", "browser": "브라우저", "terminal": "터미널", "game": "게임",
               "chat": "메신저", "other": "기타"}
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "models" / "focus_lgbm.pkl")


def history(early_app, recent_app, switches, idle_min, weekday, hour):
    """16 one-minute rows (t-15 .. t), shaped like the training data; the model reads only t-15 .. t-1."""
    now = MONDAY + pd.Timedelta(days=weekday, hours=hour)
    rows = []
    for k in range(15, -1, -1):  # k minutes before now
        idle = max(0, idle_min - k) * 60  # idle_sec is a running count: no input for the last idle_min minutes
        rows.append({"device_id": "demo", "ts": now - pd.Timedelta(minutes=k),
                     "app_category": recent_app if k <= 5 else early_app,
                     "idle_sec": idle, "switch_count": 0 if idle else switches})
    return pd.DataFrame(rows)


def predict(model, samples):
    x = build_features(samples, TZ)[FEATURE_COLUMNS].iloc[[-1]].astype(float)
    return pd.Series(model.predict_proba(x)[0], index=model.classes_), x


st.set_page_config(page_title="집중 상태 예측", page_icon="🎯")
st.title("🎯 집중 상태 예측 (ML1)")
st.caption("최근 15분 동안의 PC 사용 패턴으로 지금이 집중 · 딴짓 · 휴식 중 어느 상태인지 예측합니다. "
           "모델: LightGBM, BEHACOM 공개 데이터의 규칙 라벨로 학습.")

with st.sidebar:
    st.header("최근 15분 입력")
    fmt = CATEGORY_KO.get
    early_app = st.selectbox("10~15분 전에 주로 쓴 앱", CATEGORIES, format_func=fmt)
    recent_app = st.selectbox("최근 5분 동안 주로 쓴 앱", CATEGORIES, format_func=fmt)
    switches = st.slider("분당 창 전환 횟수", 0, 20, 2)
    idle_min = st.slider("마지막 입력 후 지난 시간 (분)", 0, 15, 0)
    weekday = st.selectbox("요일", range(7), format_func=WEEKDAYS.__getitem__)
    hour = st.slider("시각 (시)", 0, 23, 14)

proba, x = predict(load_model(), history(early_app, recent_app, switches, idle_min, weekday, hour))
best = proba.idxmax()
st.metric("예측 상태", LABEL_KO[best], f"확률 {proba[best]:.0%}")
st.bar_chart(proba.rename(index=LABEL_KO), horizontal=True)

with st.expander("모델에 들어간 피처 28개 보기"):
    st.dataframe(x.T.rename(columns=lambda _: "값"))
