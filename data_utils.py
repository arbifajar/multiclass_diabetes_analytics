"""Dataset, labels, provenance, data validation and auditable IQR filtering."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Multiclass Diabetes Dataset.csv"
FEATURES = ["Gender", "AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"]
NOTEBOOK_FEATURES = ["Gender", "HbA1c", "TG", "VLDL", "BMI"]
CODES = (0, 1, 2)
CLASS_NAMES = {0: "Non-Diabetic", 1: "Diabetic", 2: "Prediabetic"}
CLASS_LABELS = {k: f"{v} ({k})" for k, v in CLASS_NAMES.items()}
CLASS_COLORS = {0: "#2a9d8f", 1: "#e76f51", 2: "#e9a23b"}
CODE_COLOR = {CLASS_LABELS[k]: CLASS_COLORS[k] for k in CODES}
DESCRIPTION = {
    "Gender": "Kode gender 0/1; pemetaan jenis kelamin tidak dikonfirmasi oleh sumber",
    "AGE": "Usia individu",
    "Urea": "Urea darah",
    "Cr": "Kreatinin",
    "HbA1c": "Hemoglobin A1c",
    "Chol": "Kolesterol total",
    "TG": "Trigliserida",
    "HDL": "High-Density Lipoprotein",
    "LDL": "Low-Density Lipoprotein",
    "VLDL": "Very Low-Density Lipoprotein",
    "BMI": "Indeks massa tubuh",
    "Class": "Label klasifikasi menurut notebook asli (0, 1, 2)",
}


def read_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read local CSV and fail descriptively if its schema is incompatible."""
    df = pd.read_csv(path)
    expected = FEATURES + ["Class"]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(sorted(missing))}")
    df = df[expected].copy()
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df.isna().any().any() or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Dataset memuat nilai kosong atau angka tidak finite; lakukan audit terlebih dahulu.")
    if not set(df["Class"].unique()).issubset(set(CODES)):
        raise ValueError("Label Class hanya boleh terdiri dari kode 0, 1, dan 2.")
    if not {0, 1, 2}.issubset(set(df["Class"].unique())):
        raise ValueError("Ketiga kelas (0, 1, 2) harus ada untuk evaluasi multikelas.")
    if not df["Gender"].isin([0, 1]).all():
        raise ValueError("Gender harus berupa kode 0 atau 1.")
    return df


def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    return (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))


def sequential_notebook_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replicate notebook cleaning exactly: IQR filtering on Cr then VLDL."""
    result = df.copy()
    records = []
    for column in ("Cr", "VLDL"):
        before = len(result)
        lower, upper = iqr_bounds(result[column])
        keep = result[column].between(lower, upper, inclusive="both")
        result = result.loc[keep].copy()
        records.append({
            "Tahap": f"Filter IQR: {column}",
            "Sebelum": before, "Dihapus": int((~keep).sum()),
            "Sesudah": len(result), "Batas bawah": lower, "Batas atas": upper,
        })
    return result, pd.DataFrame.from_records(records)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in FEATURES:
        if c == "Gender":
            continue  # A binary category is not an appropriate IQR outlier target.
        lo, hi = iqr_bounds(df[c])
        count = int((~df[c].between(lo, hi)).sum())
        rows.append({
            "Fitur": c, "Outlier IQR": count,
            "Persen": 100.0 * count / len(df),
            "Min": float(df[c].min()), "Maks": float(df[c].max()),
            "Batas bawah": lo, "Batas atas": hi,
        })
    return pd.DataFrame(rows).sort_values("Outlier IQR", ascending=False)


def class_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = df["Class"].value_counts().reindex(CODES, fill_value=0).rename_axis("Kode").reset_index(name="Jumlah")
    out["Kelas"] = out["Kode"].map(CLASS_LABELS)
    out["Persen"] = (100 * out["Jumlah"] / len(df)).round(1)
    return out


def median_by_class(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    med = df.groupby("Class")[features].median().reindex(CODES).round(2)
    med.index = [CLASS_LABELS[int(i)] for i in med.index]
    return med
