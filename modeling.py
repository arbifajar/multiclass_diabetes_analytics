"""Reproducible, non-diagnostic multiclass model evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import FEATURES, NOTEBOOK_FEATURES, CODES, CLASS_NAMES, sequential_notebook_clean

SEED = 42
MODE_NOTEBOOK = "Replikasi notebook • 228 data, 5 fitur tetap"
MODE_STRICT = "Evaluasi lebih ketat • 264 data, RFE di pipeline"
MODES = (MODE_NOTEBOOK, MODE_STRICT)
MODEL_NAMES = ("Logistic Regression", "Random Forest", "Gradient Boosting")


@dataclass
class EvaluationBundle:
    mode: str
    samples: int
    input_features: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    models: dict[str, Pipeline]
    summary: pd.DataFrame
    cv_folds: pd.DataFrame
    reports: dict[str, pd.DataFrame]
    confusion: dict[str, np.ndarray]
    test_predictions: dict[str, np.ndarray]
    test_probabilities: dict[str, np.ndarray]
    selected_features: dict[str, list[str]]


def pipeline_for(model_name: str, strict: bool) -> Pipeline:
    algorithms = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, solver="lbfgs", class_weight="balanced", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        # GradientBoostingClassifier has no class_weight parameter in sklearn.
        # We intentionally preserve notebook behavior rather than falsely imply it is weighted.
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }
    if model_name not in algorithms:
        raise ValueError(f"Algoritma tidak dikenal: {model_name}")
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if strict:
        steps.append(("rfe", RFE(
            estimator=LogisticRegression(max_iter=3000, random_state=SEED),
            n_features_to_select=5,
        )))
    steps.append(("model", algorithms[model_name]))
    return Pipeline(steps)


def build_evaluation(df: pd.DataFrame, mode: str = MODE_NOTEBOOK) -> EvaluationBundle:
    if mode not in MODES:
        raise ValueError(f"Mode tidak dikenal: {mode}")
    strict = mode == MODE_STRICT
    working = df.copy() if strict else sequential_notebook_clean(df)[0]
    inputs = FEATURES if strict else NOTEBOOK_FEATURES
    X = working[inputs]
    y = working["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models, scores, cvs, reports, confusions, preds, probas, chosen = {}, [], [], {}, {}, {}, {}, {}
    for name in MODEL_NAMES:
        template = pipeline_for(name, strict)
        # CV uses ONLY the training split. For strict mode, both scaler & RFE
        # are fitted anew within each CV fold, preventing feature-selection leakage.
        fold_scores = []
        for i, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
            fold_model = clone(template)
            fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            y_val = fold_model.predict(X_train.iloc[val_idx])
            fold_score = float(f1_score(y_train.iloc[val_idx], y_val, average="macro", zero_division=0))
            fold_scores.append(fold_score)
            cvs.append({"Algoritma": name, "Fold": i, "Macro F1": fold_score})
        fitted = clone(template).fit(X_train, y_train)
        prediction = fitted.predict(X_test)
        probability = fitted.predict_proba(X_test)
        cls = np.asarray(fitted.classes_)
        if not np.array_equal(cls, np.asarray(CODES)):
            raise ValueError("Urutan kelas tidak sesuai; klasifikasi harus berisi 0, 1, 2.")
        metrics = {
            "Algoritma": name,
            "Akurasi": float(accuracy_score(y_test, prediction)),
            "Macro Recall": float(recall_score(y_test, prediction, average="macro", zero_division=0)),
            "Macro F1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
            "Macro ROC-AUC": float(roc_auc_score(y_test, probability, labels=list(CODES), multi_class="ovr", average="macro")),
            "CV Macro F1 (mean)": float(np.mean(fold_scores)),
            "CV Macro F1 (std)": float(np.std(fold_scores, ddof=0)),
        }
        scores.append(metrics)
        models[name] = fitted
        reports[name] = pd.DataFrame(classification_report(
            y_test, prediction, labels=list(CODES),
            target_names=[CLASS_NAMES[x] for x in CODES],
            output_dict=True, zero_division=0,
        )).T
        confusions[name] = confusion_matrix(y_test, prediction, labels=list(CODES))
        preds[name] = prediction
        probas[name] = probability
        if strict:
            chosen[name] = [f for f, keep in zip(inputs, fitted.named_steps["rfe"].support_) if keep]
        else:
            chosen[name] = list(inputs)
    summary = pd.DataFrame(scores).sort_values(
        ["Macro F1", "Macro ROC-AUC", "Macro Recall"], ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    return EvaluationBundle(
        mode=mode, samples=len(working), input_features=list(inputs),
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        models=models, summary=summary, cv_folds=pd.DataFrame(cvs),
        reports=reports, confusion=confusions, test_predictions=preds,
        test_probabilities=probas, selected_features=chosen,
    )


def validate_prediction_rows(df: pd.DataFrame, input_features: list[str]) -> pd.DataFrame:
    missing = set(input_features) - set(df.columns)
    if missing:
        raise ValueError("CSV prediksi kekurangan kolom: " + ", ".join(sorted(missing)))
    out = df[input_features].copy()
    for c in input_features:
        out[c] = pd.to_numeric(out[c], errors="raise")
    if out.isna().any().any() or not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("Data prediksi harus numerik, tidak boleh kosong/inf.")
    if "Gender" in input_features and not out["Gender"].isin([0, 1]).all():
        raise ValueError("Nilai Gender harus kode 0 atau 1.")
    return out
