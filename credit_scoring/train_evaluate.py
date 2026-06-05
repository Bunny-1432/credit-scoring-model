import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)

MODEL_COLORS = {
    "Logistic Regression": "#5c9de0",
    "Decision Tree":       "#f0a045",
    "Random Forest":       "#5cb85c",
    "XGBoost":             "#d35400",
    "LightGBM":            "#8e44ad",
}


def load_or_generate_data():
    csv_path = DATA_DIR / "credit_data.csv"
    if csv_path.exists():
        print(f"Loading existing data from {csv_path}")
        return pd.read_csv(csv_path)
    print("Generating synthetic dataset...")
    from data_generator import generate_credit_dataset
    df = generate_credit_dataset()
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return df


def plot_eda(df):
    print("  Generating EDA plots...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Credit Dataset - Exploratory Data Analysis", fontsize=16, fontweight="bold")

    colors = ["#5c9de0", "#e05c5c"]

    ax = axes[0, 0]
    counts = df["default"].value_counts()
    bars = ax.bar(["No Default (0)", "Default (1)"], counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                f"{val:,}\n({val/len(df):.1%})", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")

    ax = axes[0, 1]
    for cls, col, lbl in [(0, colors[0], "No Default"), (1, colors[1], "Default")]:
        vals = df.loc[df["default"] == cls, "annual_income"]
        ax.hist(np.log1p(vals), bins=40, alpha=0.65, color=col, label=lbl, edgecolor="none")
    ax.set_title("Log Annual Income by Class")
    ax.set_xlabel("log1p(Income)")
    ax.legend()

    ax = axes[0, 2]
    for cls, col, lbl in [(0, colors[0], "No Default"), (1, colors[1], "Default")]:
        vals = df.loc[df["default"] == cls, "credit_utilization"]
        ax.hist(vals, bins=40, alpha=0.65, color=col, label=lbl, edgecolor="none")
    ax.set_title("Credit Utilisation by Class")
    ax.set_xlabel("Utilisation Ratio")
    ax.legend()

    ax = axes[1, 0]
    dti = df["total_debt"] / (df["annual_income"] + 1)
    for cls, col, lbl in [(0, colors[0], "No Default"), (1, colors[1], "Default")]:
        vals = dti[df["default"] == cls].clip(0, 5)
        ax.hist(vals, bins=40, alpha=0.65, color=col, label=lbl, edgecolor="none")
    ax.set_title("Debt-to-Income Ratio by Class")
    ax.set_xlabel("DTI")
    ax.legend()

    ax = axes[1, 1]
    late = df.groupby(["late_payments_6m", "default"]).size().unstack(fill_value=0)
    late.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Late Payments (6m) vs Default")
    ax.set_xlabel("Late Payments")
    ax.legend(["No Default", "Default"])

    ax = axes[1, 2]
    rate = df.groupby("num_derogatory_marks")["default"].mean()
    ax.bar(rate.index, rate.values, color="#a278d4", edgecolor="white")
    ax.set_title("Default Rate by Derogatory Marks")
    ax.set_xlabel("# Derogatory Marks")
    ax.set_ylabel("Default Rate")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: eda_overview.png")


def build_models(class_weights):
    scale_pos_weight = class_weights[1] / class_weights[0] if 0 in class_weights and 1 in class_weights else 1.0
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight=class_weights,
                C=0.5,
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=20,
                class_weight=class_weights,
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=10,
                class_weight=class_weights,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "LightGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
    }


def evaluate_model(name, pipeline, X_test, y_test):
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "model":     name,
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),                   4),
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    }


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, lw=2.5, color=MODEL_COLORS[r["model"]],
                label=f"{r['model']} (AUC = {r['roc_auc']:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Models", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved: roc_curves.png")


def plot_pr_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"Random (P={baseline:.2f})")
    for r in results:
        prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
        ax.plot(rec, prec, lw=2.5, color=MODEL_COLORS[r["model"]], label=r["model"])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves - All Models", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved: pr_curves.png")


def plot_confusion_matrices(results, y_test):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_test, r["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(r["model"], fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved: confusion_matrices.png")


def plot_metrics_comparison(results):
    metrics = ["precision", "recall", "f1", "roc_auc"]
    labels  = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, r in enumerate(results):
        vals = [r[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=r["model"],
                      color=MODEL_COLORS[r["model"]], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved: metrics_comparison.png")


def plot_feature_importance(feature_names, fitted_models):
    n_models = len(fitted_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 7))
    if n_models == 1:
        axes = [axes]
    fig.suptitle("Feature Importance / Coefficients", fontsize=14, fontweight="bold")

    importance_dfs = {}
    for ax, (name, pipeline) in zip(axes, fitted_models.items()):
        clf = pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            title = "Feature Importance"
        else:
            imp = np.abs(clf.coef_[0])
            title = "|Coefficient| (Logistic Reg)"
        imp_series = pd.Series(imp, index=feature_names).sort_values(ascending=True)
        importance_dfs[name] = imp_series.sort_values(ascending=False)
        imp_series.tail(15).plot(kind="barh", ax=ax, color=MODEL_COLORS[name], edgecolor="white")
        ax.set_title(f"{name}\n{title}", fontweight="bold")
        ax.set_xlabel("Importance")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved: feature_importance.png")
    return importance_dfs


def plot_threshold_analysis(best_result, y_test):
    y_proba = best_result["y_proba"]
    thresholds = np.linspace(0.05, 0.95, 100)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

    best_t = thresholds[np.argmax(f1s)]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, precisions, lw=2.5, label="Precision", color="#5c9de0")
    ax.plot(thresholds, recalls,    lw=2.5, label="Recall",    color="#e05c5c")
    ax.plot(thresholds, f1s,        lw=2.5, label="F1-Score",  color="#5cb85c")
    ax.axvline(best_t, color="gray", linestyle="--", lw=1.5, label=f"Best F1 threshold = {best_t:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Analysis - {best_result['model']}", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: threshold_analysis.png  (optimal threshold ~ {best_t:.2f})")
    return best_t


def main():
    print("\n" + "="*60)
    print("  CREDIT SCORING MODEL PIPELINE")
    print("="*60)

    print("\n[1/6] Loading data...")
    df = load_or_generate_data()
    print(f"      Rows: {len(df):,}  |  Default rate: {df['default'].mean():.1%}")

    print("\n[2/6] EDA plots...")
    plot_eda(df)

    print("\n[3/6] Feature engineering...")
    from feature_engineering import build_features, feature_summary
    X, y = build_features(df)
    print(f"      Feature matrix: {X.shape}")
    feat_sum = feature_summary()
    feat_sum.to_csv(OUTPUT_DIR / "feature_summary.csv", index=False)
    print("      Saved: feature_summary.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes, weights))
    print(f"      Class weights: {cw_dict}")

    print("\n[4/6] Training models...")
    models        = build_models(cw_dict)
    results       = []
    fitted_models = {}

    for name, pipeline in models.items():
        print(f"  -> {name}...", end=" ")
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        r = evaluate_model(name, pipeline, X_test, y_test)
        results.append(r)
        print(f"  F1={r['f1']:.3f}  AUC={r['roc_auc']:.3f}")

    print("\n[5/6] Generating evaluation plots...")
    plot_roc_curves(results, y_test)
    plot_pr_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_metrics_comparison(results)
    imp_dfs = plot_feature_importance(list(X.columns), fitted_models)

    best_result = max(results, key=lambda r: r["roc_auc"])
    optimal_threshold = plot_threshold_analysis(best_result, y_test)

    print("\n[6/6] Saving tabular results...")
    metrics_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("y_pred", "y_proba")}
        for r in results
    ])
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    print("      Saved: model_metrics.csv")

    rf_imp = imp_dfs["XGBoost"].reset_index()
    rf_imp.columns = ["feature", "importance"]
    rf_imp.to_csv(OUTPUT_DIR / "xgb_feature_importance.csv", index=False)
    print("      Saved: xgb_feature_importance.csv")

    with open(OUTPUT_DIR / "classification_reports.txt", "w") as f:
        for r in results:
            f.write(f"\n{'='*55}\n{r['model']}\n{'='*55}\n")
            f.write(classification_report(y_test, r["y_pred"], target_names=["No Default", "Default"]))

    summary = {
        "best_model": best_result["model"],
        "optimal_threshold": round(float(optimal_threshold), 2),
        "metrics": {
            r["model"]: {k: v for k, v in r.items() if k not in ("model", "y_pred", "y_proba")}
            for r in results
        },
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("      Saved: summary.json")

    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(metrics_df.to_string(index=False))
    print(f"\n  Best model : {best_result['model']}")
    print(f"  ROC-AUC    : {best_result['roc_auc']:.4f}")
    print(f"  Optimal threshold : {optimal_threshold:.2f}")
    print(f"\n  All outputs saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
