# 집중 상태 예측 (ML1) — Streamlit

최근 15분 동안의 PC 사용 패턴(앱 종류, 창 전환, 입력 없는 시간, 시각)으로
지금이 **집중 / 딴짓 / 휴식** 중 어느 상태인지 예측하는 데모입니다.

- 모델: LightGBM (`models/focus_lgbm.pkl`, joblib 덤프), BEHACOM 공개 데이터의 규칙 라벨로 학습
- 피처: `focus_features.py` (학습 때와 같은 코드, 피처 28개)
- 배포: Streamlit Community Cloud — <배포 URL>

## 실행

```sh
pip install -r requirements.txt
streamlit run app.py
python test_app.py   # 모델 로드 + 대표 케이스 확인
```

## 모델 덤프

학습 코드와 데이터는 DevDesk 저장소(`ml/`)에 있습니다. 그 폴더에서:

```sh
uv run --group ml python ../focus-streamlit/dump_model.py
```

`requirements.txt`의 버전은 덤프할 때의 버전으로 고정했습니다(피클은 버전이 달라지면 열리지 않을 수 있음).
