import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "winequality-red.csv"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

results = {}


def load_data():
    df = pd.read_csv(DATA, sep=None, engine="python")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def descriptive(df):
    results["shape"] = list(df.shape)
    results["columns"] = list(df.columns)
    results["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    results["missing"] = int(df.isna().sum().sum())
    results["duplicates"] = int(df.duplicated().sum())
    desc = df.describe().T
    desc["median"] = df.median()
    desc["skew"] = df.skew()
    results["describe"] = desc.round(3).reset_index().rename(
        columns={"index": "feature"}
    ).to_dict(orient="records")
    results["corr_with_quality"] = (
        df.corr(numeric_only=True)["quality"].drop("quality")
        .sort_values(ascending=False).round(3).to_dict()
    )
    quality_counts = df["quality"].value_counts().sort_index()
    results["quality_counts"] = {int(k): int(v) for k, v in quality_counts.items()}
    return df, desc


def plot_quality_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    counts = df["quality"].value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values, color="#8e2f36")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v),
                ha="center", fontsize=9)
    ax.set_title("품질(Quality) 점수 분포")
    ax.set_xlabel("품질 점수")
    ax.set_ylabel("빈도")
    fig.savefig(FIG / "01_quality_distribution.png")
    plt.close(fig)


def plot_feature_histograms(df):
    feats = ["fixed_acidity", "volatile_acidity", "citric_acid",
             "residual_sugar", "chlorides", "free_sulfur_dioxide",
             "total_sulfur_dioxide", "density", "pH", "sulphates",
             "alcohol"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, f in zip(axes.ravel(), feats):
        sns.histplot(df[f], kde=True, ax=ax, color="#3b6ea5")
        ax.set_title(f, fontsize=10)
        ax.set_xlabel("")
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle("변수별 분포", fontsize=14, y=1.01)
    fig.savefig(FIG / "02_feature_histograms.png")
    plt.close(fig)


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=.5,
                cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("변수 간 상관관계 히트맵")
    fig.savefig(FIG / "03_correlation_heatmap.png")
    plt.close(fig)


def plot_box_by_quality(df):
    feats = ["volatile_acidity", "alcohol", "sulphates", "citric_acid",
             "total_sulfur_dioxide", "density"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, f in zip(axes.ravel(), feats):
        sns.boxplot(data=df, x="quality", y=f, ax=ax, color="#6a9a8b")
        ax.set_title(f"{f} vs quality", fontsize=11)
    fig.suptitle("품질 점수별 주요 변수 분포", fontsize=14, y=1.01)
    fig.savefig(FIG / "04_boxplots_by_quality.png")
    plt.close(fig)


def plot_pairplot(df):
    cols = ["volatile_acidity", "alcohol", "sulphates", "citric_acid", "quality"]
    g = sns.pairplot(df[cols], hue="quality", palette="viridis",
                     corner=True, height=1.8)
    g.figure.suptitle("주요 변수 산점도 행렬", y=1.02)
    g.figure.savefig(FIG / "05_pairplot.png")
    plt.close(g.figure)


def train_models(df):
    X = df.drop(columns=["quality"])
    y = df["quality"]

    # binary target: good wine = quality >= 7
    y_bin = (y >= 7).astype(int)
    results["good_wine_ratio"] = float(round(y_bin.mean(), 4))
    results["good_wine_count"] = int(y_bin.sum())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"),
    }
    model_metrics = {}
    for name, model in models.items():
        if name == "LogisticRegression":
            model.fit(X_tr_s, y_tr)
            pred = model.predict(X_te_s)
            proba = model.predict_proba(X_te_s)[:, 1]
        else:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            proba = model.predict_proba(X_te)[:, 1]
        model_metrics[name] = {
            "accuracy": float(round(accuracy_score(y_te, pred), 4)),
            "f1": float(round(f1_score(y_te, pred), 4)),
            "roc_auc": float(round(roc_auc_score(y_te, proba), 4)),
        }
        if name == "RandomForest":
            cm = confusion_matrix(y_te, pred)
            results["cm"] = cm.tolist()
            results["report_rf"] = classification_report(
                y_te, pred, target_names=["보통", "우수"], output_dict=True)
            importances = pd.Series(
                model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)
            results["feature_importance"] = importances.round(4).to_dict()

    # cross-validation on RandomForest
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=300, random_state=42,
                               class_weight="balanced"),
        X, y_bin, cv=5, scoring="roc_auc")
    results["cv_roc_auc_mean"] = float(round(cv.mean(), 4))
    results["cv_roc_auc_std"] = float(round(cv.std(), 4))
    results["model_metrics"] = model_metrics

    # plot feature importance
    imp = pd.Series(results["feature_importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, color="#4f7cac")
    ax.set_title("RandomForest 변수 중요도 (우수 와인 분류)")
    ax.set_xlabel("중요도")
    fig.savefig(FIG / "06_feature_importance.png")
    plt.close(fig)

    # confusion matrix plot
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(results["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["보통", "우수"], yticklabels=["보통", "우수"], ax=ax)
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    ax.set_title("RandomForest 혼동행렬")
    fig.savefig(FIG / "07_confusion_matrix.png")
    plt.close(fig)

    # correlation bar
    fig, ax = plt.subplots(figsize=(8, 5))
    cq = pd.Series(results["corr_with_quality"]).sort_values()
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in cq.values]
    ax.barh(cq.index, cq.values, color=colors)
    ax.set_title("각 변수와 품질의 상관계수")
    ax.axvline(0, color="black", linewidth=.8)
    fig.savefig(FIG / "08_corr_with_quality.png")
    plt.close(fig)

    return model_metrics


def main():
    df = load_data()
    df, desc = descriptive(df)
    plot_quality_distribution(df)
    plot_feature_histograms(df)
    plot_correlation(df)
    plot_box_by_quality(df)
    plot_pairplot(df)
    train_models(df)
    with open(BASE / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Analysis complete. Shape:", results["shape"])
    print("Corr with quality:", results["corr_with_quality"])
    print("Model metrics:", results["model_metrics"])
    print("CV ROC-AUC:", results["cv_roc_auc_mean"], "+-", results["cv_roc_auc_std"])


if __name__ == "__main__":
    main()
