# 🍷 레드와인 품질 데이터 분석

UCI Wine Quality(레드와인) 데이터셋을 `kagglehub`로 내려받아 탐색적 분석(EDA)과
품질 예측 모델링을 수행하고, **Jupyter 노트북 · Streamlit 웹 대시보드 · 분석 보고서**를 제공합니다.

## 📦 산출물

| 파일 | 설명 |
|---|---|
| `notebook/wine_quality_analysis.ipynb` | 분석 전체 과정 (실행 결과·차트 포함) |
| `app.py` | **웹 대시보드** (Streamlit) |
| `report.md` | 분석 보고서 (수치·그래프 포함) |
| `analysis.py` | 지표 계산 및 보고서용 그림 생성 스크립트 |
| `figures/` | 보고서 삽입 그림 8종 |
| `results.json` | 계산된 통계·모델 성능 결과 |

## 🚀 실행 방법

```bash
pip install -r requirements.txt

# (선택) 그림·results.json 재생성
python analysis.py

# 노트북 실행
jupyter notebook notebook/wine_quality_analysis.ipynb

# 웹 대시보드 실행 (기본 http://localhost:8501)
streamlit run app.py
```

## 📊 분석 요약

- 데이터: 레드와인 **1,599개**, 화학적 변수 11개 + 품질 점수(3~8).
- 품질 분포: 5·6점에 집중(82%), 7점 이상 우수 와인은 13.6%.
- 주요 변수: **알코올(+0.48)**, 황산염(+0.25), 구연산(+0.23) / **휘발성 산도(−0.39)**.
- 모델(우수 와인 ≥7 분류): **RandomForest Accuracy 0.925 / ROC-AUC 0.943**
  (5-fold 교차검증 ROC-AUC 0.866 ± 0.033).

자세한 내용은 [`report.md`](report.md)를 참고하세요.

## 대시보드 기능

- 품질 범위·변수 필터, KPI 지표
- 품질 분포·변수별 분포·상관 히트맵·산점도 (Plotly 인터랙티브)
- 모델 성능 비교·혼동행렬·변수 중요도
- 측정값을 입력하면 **우수 와인 확률을 예측**하는 예측기

## 데이터 출처

- UCI Machine Learning Repository — Wine Quality Data Set (Cortez et al., 2009)
- Kaggle: `uciml/red-wine-quality-cortez-et-al-2009`
