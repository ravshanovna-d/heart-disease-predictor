"""Train the heart disease model, evaluate it, and export it for the web demo.

Usage:  python train.py
Outputs: web/model.onnx, web/meta.json, web/assets/*.png
"""
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from export_utils import check_onnx, examples, export_onnx, feature_schema, write_meta

ROOT = Path(__file__).parent
DATA_URL = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
WEB, ASSETS = ROOT / "web", ROOT / "web" / "assets"
SEED = 42

# The CSV stores categories as integer codes. These mappings were checked row by row
# against the original UCI Cleveland file (processed.cleveland.data).
CODES = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {0: "Asymptomatic", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Typical angina"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Left ventricular hypertrophy", 1: "Normal", 2: "ST-T wave abnormality"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Downsloping", 1: "Flat", 2: "Upsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},          # 4 = missing in this file
    "thal": {1: "Fixed defect", 2: "Normal", 3: "Reversible defect"},  # 0 = missing
}
RENAME = {
    "cp": "chest_pain", "trestbps": "resting_bp", "chol": "cholesterol",
    "fbs": "high_fasting_sugar", "restecg": "rest_ecg", "thalach": "max_heart_rate",
    "exang": "exercise_angina", "oldpeak": "st_depression", "slope": "st_slope",
    "ca": "major_vessels", "thal": "thallium_test",
}
NUM_COLS = ["age", "resting_bp", "cholesterol", "max_heart_rate", "st_depression"]
LABELS = {
    "age": "Age", "sex": "Sex", "chest_pain": "Chest pain type",
    "resting_bp": "Resting blood pressure", "cholesterol": "Serum cholesterol",
    "high_fasting_sugar": "Fasting blood sugar > 120 mg/dl", "rest_ecg": "Resting ECG",
    "max_heart_rate": "Max heart rate (exercise test)", "exercise_angina": "Chest pain during exercise",
    "st_depression": "ST depression", "st_slope": "ST segment slope (peak exercise)",
    "major_vessels": "Major vessels seen on fluoroscopy", "thallium_test": "Thallium stress test",
}
HELPS = {"age": "years", "resting_bp": "mm Hg", "cholesterol": "mg/dl", "max_heart_rate": "bpm",
         "st_depression": "mm, exercise vs rest", "major_vessels": "0–3"}


def fetch(url: str, dest: Path, tries: int = 4) -> Path:
    """Download once into data/ (retries for flaky connections)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(exist_ok=True)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                dest.write_bytes(r.read())
            return dest
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def load_data() -> pd.DataFrame:
    raw = pd.read_csv(fetch(DATA_URL, ROOT / "data" / "heart.csv"), encoding="utf-8-sig")
    # In this file target=1 means *no* disease (it matches num == 0 in the UCI original),
    # so flip it: 1 = heart disease (>50% narrowing of a major vessel).
    raw["heart_disease"] = 1 - raw.pop("target")
    # 7 rows have missing ca (coded 4) or thal (coded 0): fill with the most common value.
    raw.loc[raw["ca"] == 4, "ca"] = 0
    raw.loc[raw["thal"] == 0, "thal"] = 2
    for col, mapping in CODES.items():
        raw[col] = raw[col].map(mapping)
    df = raw.rename(columns=RENAME)
    for c in NUM_COLS:
        df[c] = df[c].astype(float)
    return df


def build_models(num_cols, cat_cols, cv_inner):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    tree_grid = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__min_samples_leaf": [1, 5, 10, 20],
    }
    return {
        "Logistic Regression": Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, C=0.3))]),
        "Decision Tree (tuned)": GridSearchCV(
            Pipeline([("prep", prep), ("model", DecisionTreeClassifier(random_state=SEED))]),
            tree_grid, cv=cv_inner, scoring="roc_auc", n_jobs=2),
        "Random Forest": Pipeline([("prep", prep), ("model", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, max_features="sqrt", random_state=SEED, n_jobs=-1))]),
    }


def pretty_name(n: str) -> str:
    n = n.split("__", 1)[1]
    for col, lab in LABELS.items():
        if n.startswith(col + "_"):
            return f"{lab} = {n[len(col) + 1:]}"
        if n == col:
            return lab
    return n


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X, y = df.drop(columns="heart_disease"), df["heart_disease"]
    cat_cols = [c for c in X.columns if c not in NUM_COLS]
    print(f"{len(df)} patients, {y.mean():.1%} with heart disease")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # 1) Compare models with repeated 5-fold CV (the tree is tuned inside each fold = nested CV)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    cv_inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = build_models(NUM_COLS, cat_cols, cv_inner)
    rows = []
    for name, model in models.items():
        r = cross_validate(model, X_train, y_train, cv=cv, n_jobs=2,
                           scoring={"ROC-AUC": "roc_auc", "F1": "f1", "Accuracy": "accuracy"})
        rows.append({"model": name, **{k: float(np.mean(r[f"test_{k}"])) for k in ("ROC-AUC", "F1", "Accuracy")}})
        print(f"{name:24s} ROC-AUC={rows[-1]['ROC-AUC']:.3f}  F1={rows[-1]['F1']:.3f}  Acc={rows[-1]['Accuracy']:.3f}")
    comparison = sorted(rows, key=lambda r: -r["ROC-AUC"])
    # Prefer the simplest model (dict order = simplest first) within 0.005 ROC-AUC of the top score.
    top = comparison[0]["ROC-AUC"]
    best_name = next(r["model"] for r in rows if r["ROC-AUC"] >= top - 0.005)

    tree_search = models["Decision Tree (tuned)"].fit(X_train, y_train)
    tree = tree_search.best_estimator_
    tree_params = {k.replace("model__", ""): v for k, v in tree_search.best_params_.items()}
    print("Best tree:", tree_params)
    best = tree if best_name.startswith("Decision Tree") else models[best_name].fit(X_train, y_train)

    # 2) Final evaluation on the untouched test set
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }
    tree_auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
    print(f"Test ({best_name}):", {k: round(v, 3) for k, v in test_metrics.items()})
    print(f"Test ROC-AUC of the tuned tree: {tree_auc:.3f}")

    # 3) Figures
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Healthy", "Disease"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(ASSETS / "confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, name=best_name, ax=ax)
    RocCurveDisplay.from_predictions(y_test, tree.predict_proba(X_test)[:, 1], name="Decision Tree", ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_title("ROC curve (test set)")
    fig.tight_layout(); fig.savefig(ASSETS / "roc_curve.png", dpi=130); plt.close(fig)

    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=2)
    imp = pd.Series(imp.importances_mean, index=[LABELS[c] for c in X_test.columns]).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    imp.plot.barh(ax=ax, color="#4f46e5"); ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Most important features")
    fig.tight_layout(); fig.savefig(ASSETS / "feature_importance.png", dpi=130); plt.close(fig)

    rate = df.groupby("chest_pain")["heart_disease"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    rate.plot.barh(ax=ax, color="#4f46e5"); ax.set_xlabel("Share with heart disease"); ax.set_ylabel("")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Heart disease rate by chest pain type")
    fig.tight_layout(); fig.savefig(ASSETS / "disease_by_chest_pain.png", dpi=130); plt.close(fig)

    dt = tree.named_steps["model"]
    names = [pretty_name(n) for n in tree.named_steps["prep"].get_feature_names_out()]
    # Scaled thresholds are hard to read, so refit the same tree on unscaled numbers just for the plot.
    shown_depth = min(dt.get_depth(), 4)
    fig, ax = plt.subplots(figsize=(min(22, 4 + 3.2 * 2 ** (shown_depth - 1)), 2 * shown_depth + 2))
    raw_prep = ColumnTransformer([("num", "passthrough", NUM_COLS),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    tree_plot = Pipeline([("prep", raw_prep), ("model", DecisionTreeClassifier(
        random_state=SEED, **tree_params))]).fit(X_train, y_train)
    plot_tree(tree_plot.named_steps["model"], max_depth=shown_depth,
              feature_names=[pretty_name(n) for n in tree_plot.named_steps["prep"].get_feature_names_out()],
              class_names=["Healthy", "Disease"], filled=True, rounded=True, impurity=False,
              proportion=True, fontsize=8, ax=ax)
    ax.set_title(f"Tuned decision tree (depth {dt.get_depth()}, {dt.get_n_leaves()} leaves)")
    fig.tight_layout(); fig.savefig(ASSETS / "decision_tree.png", dpi=110); plt.close(fig)

    # 4) Export for the browser and verify it matches scikit-learn
    export_onnx(best, X_train, WEB / "model.onnx")
    err = check_onnx(WEB / "model.onnx", best, X_test)
    print(f"ONNX check OK (max diff {err:.2e})")

    write_meta(WEB / "meta.json", {
        "title": "Heart Disease Risk Predictor",
        "subtitle": ("Estimates the chance of coronary heart disease from 13 routine clinical "
                     "measurements, using the classic UCI Cleveland patient data."),
        "tags": ["Classification", "scikit-learn", best_name, "Decision Tree", "Nested CV", "ONNX"],
        "task": "classification",
        "classes": [0, 1], "positive_class": 1,
        "output_label": "Estimated probability of heart disease",
        "verdicts": ["Low estimated risk", "Moderate estimated risk", "High estimated risk"],
        "thresholds": [0.3, 0.6],
        "result_note": ("<b>Not medical advice.</b> This is a learning project trained on 303 patients "
                        "from a 1988 study; it cannot diagnose anyone. If you have heart symptoms, see a doctor."),
        "features": feature_schema(X_train, LABELS, HELPS),
        "examples": examples(X_test),
        "model_name": best_name,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "metric_formats": {k: "percent" for k in test_metrics if k != "ROC-AUC"},
        "metrics_note": f"Scores on a held-out test set of {len(y_test)} patients the model never saw during training.",
        "figures": [
            {"src": "assets/decision_tree.png", "caption": "The tuned decision tree: readable if-then rules (top levels)."},
            {"src": "assets/feature_importance.png", "caption": "What drives the prediction the most."},
            {"src": "assets/disease_by_chest_pain.png", "caption": "Patients with no chest pain (asymptomatic) were most often diagnosed."},
            {"src": "assets/roc_curve.png", "caption": "ROC curve on the test set: final model vs the single tree."},
            {"src": "assets/confusion_matrix.png", "caption": "Correct and wrong predictions on the test set."},
        ],
        "steps": [
            f"Loaded {len(df)} patients (UCI Cleveland) and decoded the numeric codes into readable categories, "
            "checking each mapping against the original UCI file (the target turned out to be inverted).",
            "Built one scikit-learn Pipeline: scaling for numbers, one-hot encoding for categories.",
            "Tuned a Decision Tree with GridSearchCV inside each fold (nested CV) and compared it with "
            "Logistic Regression and Random Forest using 5×5 repeated stratified cross-validation.",
            f"Picked {best_name} by ROC-AUC and evaluated it once on a held-out 20% test set.",
            "Visualised the tuned tree and explained the model with permutation importance.",
            "Exported the whole pipeline to ONNX so it runs in the browser — no server needed.",
        ],
        "dataset": {"name": "UCI Heart Disease (Cleveland)", "url": "https://archive.ics.uci.edu/dataset/45/heart+disease", "rows": len(df)},
        "repo_url": "https://github.com/ravshanovna-d/heart-disease-predictor",
        "notebook_url": "https://github.com/ravshanovna-d/heart-disease-predictor/blob/main/notebook.ipynb",
        "footer": "Built by <a href=\"https://github.com/ravshanovna-d\">ravshanovna-d</a>",
    })
    print("Wrote web/model.onnx and web/meta.json")


if __name__ == "__main__":
    main()
