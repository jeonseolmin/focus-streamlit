# 집중 상태 예측 시스템 (ML1) — Streamlit

최근 15분 동안의 PC 사용 패턴(앱 종류, 창 전환, 입력 없는 시간, 시각)으로
지금이 **집중 / 딴짓 / 휴식** 중 어느 상태인지 예측하는 데모입니다.

- 배포: https://focusml1.streamlit.app (Streamlit Community Cloud)
- 모델: LightGBM, BEHACOM 공개 데이터(11명, 약 2만 분)의 규칙 라벨로 학습
- 탭 1 **집중 상태 시뮬레이터**: 사용 패턴 입력 → 예측 버튼 → 예측 상태와 확률
- 탭 2 **데이터 인사이트 대시보드**: 특성 중요도, 상태별 앱 사용 패턴, 시간대별 분포

## 구조

```
app.py                       Streamlit 앱
focus_features.py            피처 생성 (학습 때와 같은 코드, 피처 28개)
model/focus_model.joblib     학습된 모델 (joblib 덤프)
model/features_meta.joblib   피처 순서, 클래스, 앱 카테고리
dataset/focus_features.csv   대시보드용 학습 데이터 (피처 + 라벨)
dump_model.py                위 3개 파일을 만드는 스크립트
```

LightGBM은 트리 모델이라 스케일러가 필요 없고, 앱 종류는 `focus_features.py`가 원핫으로 바꾸므로
별도 인코더 파일이 없습니다.

## 실행

```sh
pip install -r requirements.txt
streamlit run app.py
python test_app.py   # 모델 로드 + 대표 케이스 + 화면 렌더링 확인
```

## 모델 덤프

학습 코드와 원본 데이터는 DevDesk 저장소(`ml/`)에 있습니다. 그 폴더에서:

```sh
uv run --group ml python ../focus-streamlit/dump_model.py
```

`requirements.txt`의 버전은 덤프할 때의 버전으로 고정했습니다(joblib 파일은 버전이 달라지면 열리지 않을 수 있음).

## 보고서 숫자와 다른 점

- 학습 데이터: 21,786행(break 18,619 / focus 2,007 / distracted 1,160), 사용자 11명
- 보고서(AI개발 수행내역서)의 분석은 22,112행입니다. 그때는 앱 분류표가 `java`·`javaw`를 IDE로 봤고,
  지금 분류표는 Java로 도는 게임 등과 섞이지 않도록 둘을 IDE에서 뺐습니다. 차이 326행은 모두 focus입니다.
- 저장 형식: 이 배포본은 joblib으로 저장하고, DevDesk 앱은 LightGBM 텍스트 모델(`focus_lgbm.txt`)을
  불러옵니다. 피처 28개와 하이퍼파라미터는 같습니다.
