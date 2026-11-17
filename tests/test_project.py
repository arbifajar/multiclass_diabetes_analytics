from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import (FEATURES, NOTEBOOK_FEATURES, CLASS_NAMES, read_dataset,
                        sequential_notebook_clean, class_counts, outlier_summary)
from modeling import (MODE_NOTEBOOK, MODE_STRICT, MODEL_NAMES, build_evaluation,
                      pipeline_for, validate_prediction_rows)


@pytest.fixture(scope="module")
def source_df():
    return read_dataset()


@pytest.fixture(scope="module")
def replicate(source_df):
    return build_evaluation(source_df, MODE_NOTEBOOK)


@pytest.fixture(scope="module")
def strict(source_df):
    return build_evaluation(source_df, MODE_STRICT)


def test_dataset_schema_and_counts(source_df):
    assert source_df.shape == (264, 12)
    assert source_df.columns.tolist() == FEATURES + ["Class"]
    assert source_df.isna().sum().sum() == 0
    assert source_df.duplicated().sum() == 0
    assert class_counts(source_df)["Jumlah"].tolist() == [96, 40, 128]


def test_notebook_cleaning_is_auditable(source_df):
    clean, audit = sequential_notebook_clean(source_df)
    assert clean.shape == (228, 12)
    assert audit["Dihapus"].tolist() == [25, 11]
    assert class_counts(clean)["Jumlah"].tolist() == [90, 37, 101]
    assert not set(clean.index) - set(source_df.index)
    assert len(source_df) - len(clean) == 36


def test_outliers_are_described_not_silently_removed(source_df):
    report = outlier_summary(source_df)
    assert report["Fitur"].nunique() == 10
    assert "Gender" not in report["Fitur"].tolist()
    assert source_df["Cr"].max() == 800
    assert source_df["VLDL"].max() == 31.8


def test_modes_reproducible_and_all_models(replicate, strict):
    assert (replicate.samples, len(replicate.y_train), len(replicate.y_test)) == (228, 182, 46)
    assert (strict.samples, len(strict.y_train), len(strict.y_test)) == (264, 211, 53)
    assert replicate.input_features == NOTEBOOK_FEATURES
    assert strict.input_features == FEATURES
    for bundle in (replicate, strict):
        assert set(bundle.summary["Algoritma"]) == set(MODEL_NAMES)
        assert len(bundle.cv_folds) == 15
        assert all(bundle.summary[m].between(0, 1).all() for m in ["Akurasi", "Macro F1", "Macro Recall", "Macro ROC-AUC"])
        assert all(len(v) == 5 for v in bundle.selected_features.values())
        assert all(c.shape == (3, 3) for c in bundle.confusion.values())
        for name in MODEL_NAMES:
            assert bundle.confusion[name].sum() == len(bundle.y_test)
            assert bundle.test_probabilities[name].shape == (len(bundle.y_test), 3)
            assert np.allclose(bundle.test_probabilities[name].sum(axis=1), 1)


def test_notebook_metrics_match_reference(replicate):
    result = replicate.summary.set_index("Algoritma")
    assert result.loc["Random Forest", "Macro F1"] == pytest.approx(0.935821, abs=0.00002)
    assert result.loc["Random Forest", "Macro Recall"] == pytest.approx(0.95, abs=0.00001)
    assert result.loc["Random Forest", "Macro ROC-AUC"] == pytest.approx(0.985111, abs=0.00002)


def test_rfe_is_in_pipeline_only_for_strict_mode(strict, replicate):
    for model in MODEL_NAMES:
        assert "rfe" in strict.models[model].named_steps
        assert "scale" in strict.models[model].named_steps
        assert "rfe" not in replicate.models[model].named_steps
        assert len(strict.selected_features[model]) == 5


def test_prediction_validation(strict, source_df):
    data = source_df[FEATURES].head(2)
    checked = validate_prediction_rows(data, FEATURES)
    assert checked.shape == (2, 11)
    bad = data.drop(columns=["Cr"])
    with pytest.raises(ValueError, match="Cr"):
        validate_prediction_rows(bad, FEATURES)
    with pytest.raises(ValueError, match="Gender"):
        validate_prediction_rows(data.assign(Gender=3), FEATURES)
    for model in MODEL_NAMES:
        preds = strict.models[model].predict(checked)
        assert set(preds).issubset({0, 1, 2})
