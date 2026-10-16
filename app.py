"""레드와인 품질 분석 웹 대시보드 (Streamlit).

실행:
    streamlit run app.py
"""
import os
import glob

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="레드와인 품질 분석",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

KOR = {
    "fixed_acidity": "고정 산도",
    "volatile_acidity": "휘발성 산도",
    "citric_acid": "구연산",
    "residual_sugar": "잔당",
    "chlorides": "염화물",
    "free_sulfur_dioxide": "유리 이산화황",
    "total_sulfur_dioxide": "총 이산화황",
    "density": "밀도",
    "pH": "pH",
    "sulphates": "황산염",
    "alcohol": "알코올",
    "quality": "품질",
}


@st.cache_data(show_spinner=False)
def load_data():
    local = os.path.join(os.path.dirname(__file__), "data", "winequality-red.csv")
    if os.path.exists(local):
        csv_path = local
    else:
        import kagglehub

        path = kagglehub.dataset_download(
            "uciml/red-wine-quality-cortez-et-al-2009"
        )
        csv_path = glob.glob(os.path.join(path, "*.csv"))[0]
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


@st.cache_resource(show_spinner=False)
def train_models(df):
    X = df.drop(columns=["quality"])
    y = (df["quality"] >= 7).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(scaler.transform(X_tr), y_tr)
    pred_lr = lr.predict(scaler.transform(X_te))
    prob_lr = lr.predict_proba(scaler.transform(X_te))[:, 1]

    rf = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
    rf.fit(X_tr, y_tr)
    pred_rf = rf.predict(X_te)
    prob_rf = rf.predict_proba(X_te)[:, 1]

    metrics = pd.DataFrame(
        {
            "LogisticRegression": [
                accuracy_score(y_te, pred_lr),
                f1_score(y_te, pred_lr),
                roc_auc_score(y_te, prob_lr),
            ],
            "RandomForest": [
                accuracy_score(y_te, pred_rf),
                f1_score(y_te, pred_rf),
                roc_auc_score(y_te, prob_rf),
            ],
        },
        index=["Accuracy", "F1", "ROC-AUC"],
    ).T

    cv = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
    importances = pd.Series(
        rf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    cm = confusion_matrix(y_te, pred_rf)
    return {
        "metrics": metrics,
        "cv_mean": cv.mean(),
        "cv_std": cv.std(),
        "importances": importances,
        "cm": cm,
        "model": rf,
        "features": list(X.columns),
    }


def fmt(name):
    return KOR.get(name, name)


# ------------------------- 헤더 -------------------------
st.title("🍷 레드와인 품질 분석 대시보드")
st.caption(
    "데이터: UCI Wine Quality (Cortez et al., 2009) · "
    "레드와인 1,599개 · 화학적 변수 11개 + 품질 점수"
)

df = load_data()
models = train_models(df)

# ------------------------- 사이드바 -------------------------
st.sidebar.header("⚙️ 필터")
q_min, q_max = int(df["quality"].min()), int(df["quality"].max())
quality_range = st.sidebar.slider(
    "품질 점수 범위", q_min, q_max, (q_min, q_max)
)
feature_cols = [c for c in df.columns if c != "quality"]
selected = st.sidebar.multiselect(
    "분석할 변수", feature_cols, default=feature_cols
)
st.sidebar.markdown("---")
st.sidebar.info("노트북: `notebook/wine_quality_analysis.ipynb`\n\n보고서: `report.md`")

fdf = df[df["quality"].between(*quality_range)]

# ------------------------- KPI -------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("표본 수", f"{len(fdf):,}")
c2.metric("평균 품질", f"{fdf['quality'].mean():.2f}")
c3.metric("평균 알코올", f"{fdf['alcohol'].mean():.2f}%")
good = (fdf["quality"] >= 7).mean()
c4.metric("우수 와인(≥7) 비율", f"{good:.1%}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 데이터 개요", "🔍 탐색 분석", "🤖 모델 성능", "🎯 품질 예측"]
)

# ------------------------- 탭1: 개요 -------------------------
with tab1:
    st.subheader("품질 점수 분포")
    counts = fdf["quality"].value_counts().sort_index().reset_index()
    counts.columns = ["quality", "count"]
    fig = px.bar(
        counts, x="quality", y="count", text="count",
        color="count", color_continuous_scale="Reds",
    )
    fig.update_layout(showlegend=False, height=380,
                      xaxis_title="품질 점수", yaxis_title="빈도")
    st.plotly_chart(fig, width="stretch")

    st.subheader("데이터 미리보기 및 기술 통계")
    st.dataframe(fdf.head(20), width="stretch")

    desc = fdf[selected + ["quality"]].describe().T
    desc["median"] = fdf[selected + ["quality"]].median()
    desc["skew"] = fdf[selected + ["quality"]].skew()
    st.dataframe(desc.round(3), width="stretch")

# ------------------------- 탭2: 탐색 -------------------------
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("변수별 분포")
        feat = st.selectbox("변수 선택", selected, index=0)
        fig = px.histogram(
            fdf, x=feat, nbins=40, marginal="box",
            color_discrete_sequence=["#3b6ea5"],
        )
        fig.update_layout(height=380, xaxis_title=fmt(feat), yaxis_title="빈도")
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.subheader("품질별 분포")
        fig = px.box(
            fdf, x="quality", y=feat, color="quality",
            color_discrete_sequence=px.colors.sequential.Viridis,
        )
        fig.update_layout(height=380, showlegend=False,
                          xaxis_title="품질 점수", yaxis_title=fmt(feat))
        st.plotly_chart(fig, width="stretch")

    st.subheader("상관관계 히트맵")
    corr = fdf[selected + ["quality"]].corr().round(2)
    labels = [fmt(c) for c in corr.columns]
    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        x=labels, y=labels,
    )
    fig.update_layout(height=620)
    st.plotly_chart(fig, width="stretch")

    st.subheader("품질과의 상관계수")
    cq = (
        df[selected + ["quality"]].corr()["quality"]
        .drop("quality").sort_values()
    )
    fig = px.bar(
        x=cq.values, y=[fmt(i) for i in cq.index], orientation="h",
        color=cq.values, color_continuous_scale="RdYlGn",
        labels={"x": "상관계수", "y": ""},
    )
    fig.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("산점도 탐색")
    sc_a, sc_b = st.columns(2)
    x_ax = sc_a.selectbox("X축", selected, index=feature_cols.index("alcohol"))
    y_ax = sc_b.selectbox(
        "Y축", selected, index=feature_cols.index("volatile_acidity")
    )
    fig = px.scatter(
        fdf, x=x_ax, y=y_ax, color="quality",
        color_continuous_scale="Viridis", opacity=0.7,
        labels={x_ax: fmt(x_ax), y_ax: fmt(y_ax)},
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, width="stretch")

# ------------------------- 탭3: 모델 -------------------------
with tab3:
    st.subheader("분류 모델 성능 (우수 와인 = 품질 ≥ 7)")
    st.dataframe(
        models["metrics"].round(4).style.background_gradient(
            cmap="Greens", axis=1
        ),
        width="stretch",
    )
    st.success(
        f"5-fold 교차검증 ROC-AUC: **{models['cv_mean']:.4f}** "
        f"(± {models['cv_std']:.4f})"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("RandomForest 혼동행렬")
        cm = models["cm"]
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=["보통", "우수"], y=["보통", "우수"],
        )
        fig.update_layout(
            xaxis_title="예측", yaxis_title="실제", height=400,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, width="stretch")
    with col_b:
        st.subheader("변수 중요도")
        imp = models["importances"].sort_values()
        fig = px.bar(
            x=imp.values, y=[fmt(i) for i in imp.index], orientation="h",
            color=imp.values, color_continuous_scale="Blues",
            labels={"x": "중요도", "y": ""},
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")

# ------------------------- 탭4: 예측 -------------------------
with tab4:
    st.subheader("🎯 와인 품질 예측기")
    st.caption(
        "측정값을 입력하면 우수 와인(품질 ≥ 7)일 확률을 예측합니다. "
        "(RandomForest 모델)"
    )
    ranges = df[feature_cols].agg(["min", "mean", "max"])
    inputs = {}
    cols = st.columns(3)
    for i, f in enumerate(feature_cols):
        lo = float(ranges.loc["min", f])
        hi = float(ranges.loc["max", f])
        mean = float(ranges.loc["mean", f])
        step = (hi - lo) / 100 if hi > lo else 0.01
        inputs[f] = cols[i % 3].slider(
            fmt(f), lo, hi, mean, step=step, format="%.4f"
        )

    if st.button("예측하기", type="primary"):
        row = pd.DataFrame([inputs])[models["features"]]
        prob = models["model"].predict_proba(row)[0, 1]
        pred = "우수 와인 🍷" if prob >= 0.5 else "보통 와인"
        g1, g2 = st.columns([1, 2])
        g1.metric("우수 와인 확률", f"{prob:.1%}")
        if prob >= 0.5:
            g2.success(f"예측 결과: **{pred}**")
        else:
            g2.warning(f"예측 결과: **{pred}**")
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                title={"text": "우수 와인 확률"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#8e2f36"},
                    "steps": [
                        {"range": [0, 50], "color": "#f0e6d2"},
                        {"range": [50, 100], "color": "#d9ead3"},
                    ],
                },
            )
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.caption(
    "© 2026 레드와인 품질 분석 · 데이터 출처: UCI ML Repository / Kaggle"
)
