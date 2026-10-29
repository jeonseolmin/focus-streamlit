"""Streamlit demo of the ML1 focus-state model: describe the last 15 minutes at the PC, get focus/distracted/break."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from focus_features import build_features

HERE = Path(__file__).parent
TZ = "Asia/Seoul"
MONDAY = pd.Timestamp("2026-10-26", tz=TZ)
LABEL_KO = {"focus": "집중", "distracted": "딴짓", "break": "휴식"}
CATEGORY_KO = {"ide": "IDE (코딩)", "browser": "브라우저", "terminal": "터미널", "game": "게임",
               "chat": "메신저", "other": "기타"}
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]

st.set_page_config(page_title="집중 상태 예측 시스템", page_icon="🎯", layout="wide")


# 1. 학습 모델과 메타 정보 로드 (dump_model.py가 joblib으로 저장한 파일)
@st.cache_resource
def load_ml_objects():
    model = joblib.load(HERE / "model/focus_model.joblib")
    meta = joblib.load(HERE / "model/features_meta.joblib")
    return model, meta


# 대시보드용 학습 데이터 (피처 28개 + 규칙 라벨)
@st.cache_data
def load_raw_data():
    df = pd.read_csv(HERE / "dataset/focus_features.csv")
    df["hour"] = (np.degrees(np.arctan2(df["hour_sin"], df["hour_cos"])) / 15 % 24).round().astype(int) % 24
    df["state"] = df["label"].map(LABEL_KO)
    return df


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


def predict(model, meta, samples):
    x = build_features(samples, TZ)[meta["features"]].iloc[[-1]].astype(float)
    return pd.Series(model.predict_proba(x)[0], index=model.classes_), x


def main():
    try:
        model, meta = load_ml_objects()
    except FileNotFoundError:
        st.error("학습된 모델 파일(joblib)을 찾을 수 없습니다. 먼저 dump_model.py로 모델을 저장해주세요.")
        st.stop()
    df = load_raw_data()

    st.title("🎯 집중 상태 예측 시스템 (ML1)")
    st.write("최근 15분 동안의 PC 사용 패턴으로 지금이 집중 · 딴짓 · 휴식 중 어느 상태인지 예측합니다. "
             "모델: LightGBM, BEHACOM 공개 데이터(11명, 약 2만 분)의 규칙 라벨로 학습.")

    main_tab1, main_tab2 = st.tabs(["집중 상태 시뮬레이터", "데이터 인사이트 대시보드"])

    # ==================== TAB 1: 예측 시뮬레이터 ====================
    with main_tab1:
        st.markdown("### 지금 내 상태 예측")
        st.write("아래에 최근 15분의 PC 사용 패턴을 입력한 후 '예측 실행' 버튼을 눌러주세요.")
        col1, col2 = st.columns([2, 1.1])

        with col1:
            st.subheader("사용 패턴 입력")
            with st.container(border=True):
                fmt = CATEGORY_KO.get
                sub1, sub2 = st.columns(2)
                with sub1:
                    early_app = st.selectbox("10~15분 전에 주로 쓴 앱", meta["categories"], format_func=fmt)
                with sub2:
                    recent_app = st.selectbox("최근 5분 동안 주로 쓴 앱", meta["categories"], format_func=fmt)
                switches = st.slider("분당 창 전환 횟수", 0, 20, 2)
                idle_min = st.slider("마지막 입력 후 지난 시간 (분)", 0, 15, 0)
                sub3, sub4 = st.columns(2)
                with sub3:
                    weekday = st.selectbox("요일", range(7), format_func=WEEKDAYS.__getitem__)
                with sub4:
                    hour = st.number_input("시각 (시)", min_value=0, max_value=23, value=14)

        with col2:
            st.subheader("예측 결과")
            proba, x = predict(model, meta, history(early_app, recent_app, switches, idle_min, weekday, hour))
            with st.container(border=True):
                if st.button("집중 상태 예측 실행", use_container_width=True, type="primary"):
                    best = proba.idxmax()
                    st.metric(label="예측 상태 결과", value=LABEL_KO[best])
                    st.markdown("---")
                    message = f"(확률: **{proba[best] * 100:.2f}%**)"
                    if best == "focus":
                        st.success(f"**지금 집중하고 있습니다.**\n\n{message}")
                    elif best == "distracted":
                        st.error(f"**딴짓 중일 가능성이 높습니다.**\n\n{message}")
                    else:
                        st.info(f"**휴식 중입니다.**\n\n{message}")
                    st.bar_chart(proba.rename(index=LABEL_KO), horizontal=True)
                else:
                    st.info("왼쪽 정보를 채우신 후 버튼을 누르면 AI 분석 결과가 표시됩니다.")
            with st.expander("모델에 들어간 피처 28개 보기"):
                st.dataframe(x.T.rename(columns=lambda _: "값"))

    # ==================== TAB 2: 데이터 인사이트 대시보드 ====================
    with main_tab2:
        st.markdown("### 학습 데이터 시각화 및 인사이트")
        st.write("모델이 학습한 데이터(분 단위 PC 사용 기록)를 바탕으로 판단 기준과 상태 분포를 살펴봅니다.")
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["AI 특성 중요도 분석", "앱 사용 패턴 분석", "시간대별 분포"])

        with sub_tab1:
            st.subheader("상태 판단에 영향을 주는 주요 피처 (Feature Importance)")
            st.caption("모델이 상태를 판단할 때 가장 많이 기여한 상위 10개 피처입니다 (gain 기준, 전체 피처 합계를 1로 본 비율).")
            gain = pd.Series(model.booster_.feature_importance("gain"), index=meta["features"])
            st.bar_chart((gain / gain.sum()).sort_values(ascending=False).head(10), horizontal=True, sort=False)

        with sub_tab2:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.subheader("상태별 데이터 수")
                st.bar_chart(df["state"].value_counts())
            with col_t2:
                st.subheader("상태별 최근 15분 앱 사용 비율")
                ratios = df.groupby("state")[[f"ratio_{c}_15m" for c in meta["categories"]]].mean()
                st.bar_chart(ratios.rename(columns=lambda c: CATEGORY_KO[c.split("_")[1]]))
            st.subheader("상태별 평균 입력 없음 시간 (최근 5분, 초)")
            st.bar_chart(df.groupby("state")["idle_mean_5m"].mean())

        with sub_tab3:
            st.subheader("시간대별 상태 적층 분포")
            st.bar_chart(pd.crosstab(df["hour"], df["state"]))


if __name__ == "__main__":
    main()
