"""Multiclass Diabetes Analytics — Streamlit educational portfolio.

Run: streamlit run app.py
Built from the supplied CSV, notebook, and BNSP portfolio. Not a medical device.
"""
from __future__ import annotations

from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, roc_curve

from data_utils import (BASE_DIR, DATA_PATH, FEATURES, NOTEBOOK_FEATURES, CODES,
                        CLASS_LABELS, CLASS_NAMES, CLASS_COLORS, CODE_COLOR,
                        DESCRIPTION, read_dataset, sequential_notebook_clean,
                        outlier_summary, class_counts, median_by_class)
from modeling import (MODES, MODEL_NAMES, MODE_NOTEBOOK, MODE_STRICT,
                      build_evaluation, validate_prediction_rows)

st.set_page_config(
    page_title="Diabetes Analytics | Multiclass ML",
    page_icon="🩺",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2.3rem; max-width: 1480px;}
.app-hero {background:linear-gradient(110deg,#12364f,#147c83);border-radius:18px;
  padding:24px 30px 23px;color:white;margin-bottom:16px;}
.app-hero h1 {margin:0;color:white;font-size:2.1rem;font-weight:750;line-height:1.25;}
.app-hero p {color:#d4eff0;margin:8px 0 0;line-height:1.6;font-size:.99rem;}
.section-label {font-size:.81rem;letter-spacing:.06em;text-transform:uppercase;
  color:#177d85;font-weight:700;margin-top:14px;}
.warning-small {border-left:4px solid #c58a31;padding:10px 14px;background:#fff8e9;
 border-radius:0 9px 9px 0;font-size:.94rem;margin-bottom:12px;}
[data-testid="stMetric"] {background:white;border:1px solid #e0e8ed;padding:12px 15px;
 border-radius:13px;min-height:103px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return read_dataset(DATA_PATH)


@st.cache_resource(show_spinner="Menghitung evaluasi 3 algoritma dan 5-fold cross-validation...")
def get_bundle(mode: str):
    return build_evaluation(read_dataset(DATA_PATH), mode)


def headline(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="app-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=405):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=18, r=14, t=35, b=24),
        font=dict(family="Arial", size=12, color="#234052"),
        legend_title_text="", hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def with_class_label(df: pd.DataFrame) -> pd.DataFrame:
    copied = df.copy()
    copied["Kategori"] = copied["Class"].map(CLASS_LABELS)
    copied["Kode gender"] = copied["Gender"].astype(str)
    return copied


def source_disclaimer() -> None:
    st.info(
        "**Batas penggunaan:** Aplikasi ini merupakan demonstrasi analisis dataset dan "
        "Machine Learning, **bukan alat diagnosis, skrining klinis, atau pemberi saran medis**. "
        "Makna kode kelas mengikuti notebook sumber; belum ada verifikasi label, satuan, "
        "representativitas sampel, maupun validasi klinis eksternal."
    )


try:
    original = get_data()
    cleaned, audit = sequential_notebook_clean(original)
except Exception as exc:
    st.error(f"Data tidak dapat dibaca: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("### 🩺 Diabetes Analytics")
    st.caption("Portofolio data mining / BNSP")
    page = st.radio(
        "Navigasi aplikasi",
        ["🏠 Dashboard", "📊 Eksplorasi Data", "🧹 Audit Data", "🧠 Evaluasi Model",
         "🔎 Prediksi Eksperimental", "📚 Metodologi & Sumber"],
    )
    st.divider()
    mode = st.selectbox(
        "Skenario pemodelan", MODES,
        help="Replikasi mengikuti pilihan fitur/outlier notebook. Mode ketat memasukkan RFE dalam pipeline, tanpa filter outlier.",
    )
    st.caption("Pilihan skenario digunakan pada Evaluasi Model dan Prediksi Eksperimental.")
    st.divider()
    st.caption("**Data**: 264 observasi · 12 kolom")
    st.caption("**Sumber**: CSV + notebook + portofolio BNSP")
    st.caption("**Catatan**: Tidak untuk keputusan klinis")

if page == "🏠 Dashboard":
    headline("Multiclass Diabetes | Analytics Dashboard",
             "Eksplorasi 264 catatan, analisis kualitas data, dan evaluasi klasifikasi tiga kategori berdasarkan portofolio BNSP.")
    source_disclaimer()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observasi asli", f"{len(original):,}")
    c2.metric("Fitur input", len(FEATURES))
    c3.metric("Data hilang", f"{int(original.isna().sum().sum()):,}")
    c4.metric("Setelah filter notebook", f"{len(cleaned)}", delta=f"-{len(original)-len(cleaned)} baris", delta_color="off")

    a, b = st.columns([1, 1.1], gap="large")
    with a:
        st.subheader("Komposisi kelas pada data asli")
        dist = class_counts(original)
        fig = px.pie(
            dist, names="Kelas", values="Jumlah", color="Kelas", color_discrete_map=CODE_COLOR,
            hole=.55,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label", hovertemplate="%{label}<br>%{value} observasi<br>%{percent}<extra></extra>")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.caption("Kode label mengikuti notebook: 0 = Non-Diabetic, 1 = Diabetic, 2 = Prediabetic.")
    with b:
        st.subheader("Jumlah data menurut kelas")
        fig = px.bar(
            dist, x="Kelas", y="Jumlah", color="Kelas", text="Jumlah",
            color_discrete_map=CODE_COLOR, custom_data=["Persen"],
        )
        fig.update_traces(textposition="outside", hovertemplate="%{x}<br>%{y} baris (%{customdata[0]:.1f}%)<extra></extra>")
        fig.update_yaxes(range=[0, max(dist["Jumlah"]) * 1.24], title_text="Observasi")
        fig.update_xaxes(title_text="")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.caption("Distribusi tidak seimbang; metrik Macro F1 dan Macro Recall perlu ditampilkan bersama akurasi.")

    st.subheader("Ringkasan indikator berdasarkan kelas")
    key_features = ["AGE", "HbA1c", "BMI", "TG", "VLDL"]
    st.dataframe(median_by_class(original, key_features), use_container_width=True)
    st.caption("Angka di tabel merupakan **median**, bukan ambang diagnosis atau hubungan sebab-akibat. Satuan pengukuran belum diverifikasi.")

    st.subheader("Apa yang tersedia di aplikasi?")
    ca, cb, cc = st.columns(3, gap="medium")
    with ca:
        st.markdown("**📊 Eksplorasi interaktif**")
        st.write("Filter kelompok, amati distribusi dan hubungan antar fitur, serta ekspor tampilan data.")
    with cb:
        st.markdown("**🧹 Audit transparan**")
        st.write("Periksa missing value, duplikasi, outlier IQR, dan 36 observasi yang dikeluarkan notebook.")
    with cc:
        st.markdown("**🧠 Perbandingan terukur**")
        st.write("Bandingkan 3 algoritma pada test set yang sama, confusion matrix, ROC, dan 5-fold CV.")

elif page == "📊 Eksplorasi Data":
    headline("Eksplorasi Data", "Jelajahi pola variabel tanpa menjadikannya kesimpulan medis atau kausal.")
    p1, p2, p3 = st.columns([1.3, 1, 1])
    with p1:
        variant = st.radio("Basis data", ["Asli (264)", "Setelah filter notebook (228)"], horizontal=True)
    active = original if variant.startswith("Asli") else cleaned
    with p2:
        classes = st.multiselect("Kelas", options=list(CODES), default=list(CODES),
                                 format_func=lambda code: CLASS_LABELS[code])
    with p3:
        gender = st.multiselect("Gender (kode belum terverifikasi)", [0, 1], default=[0, 1],
                                format_func=lambda code: f"Kode {code}")
    min_age, max_age = int(active["AGE"].min()), int(active["AGE"].max())
    ages = st.slider("Rentang usia (AGE)", min_age, max_age, (min_age, max_age))
    filtered = active[active["Class"].isin(classes) & active["Gender"].isin(gender) & active["AGE"].between(*ages)]
    if filtered.empty:
        st.warning("Tidak ada data untuk filter tersebut. Ubah kelas, kode gender, atau rentang usia.")
        st.stop()
    st.caption(f"Menampilkan {len(filtered)} dari {len(active)} observasi pada basis data yang dipilih.")
    dff = with_class_label(filtered)
    tab1, tab2, tab3, tab4 = st.tabs(["Distribusi", "Perbandingan kelas", "Hubungan fitur", "Tabel data"])
    with tab1:
        target = st.selectbox("Variabel yang diperiksa", [x for x in FEATURES if x != "Gender"], index=3)
        fig = px.histogram(dff, x=target, color="Kategori", color_discrete_map=CODE_COLOR,
                           nbins=22, barmode="overlay", opacity=.76,
                           labels={"Kategori": "Kelas", "count": "Jumlah"})
        fig.update_yaxes(title_text="Jumlah observasi")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.caption(f"{DESCRIPTION[target]}. Histogram memperlihatkan frekuensi sampel, bukan probabilitas diagnosis.")
    with tab2:
        select = st.selectbox("Bandingkan fitur menurut kelas", [c for c in FEATURES if c != "Gender"], index=4)
        fig = px.box(dff, x="Kategori", y=select, color="Kategori", color_discrete_map=CODE_COLOR,
                     category_orders={"Kategori": list(CODE_COLOR)}, points="outliers")
        fig.update_xaxes(title_text="Kategori dataset")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.dataframe(median_by_class(filtered, [select]).dropna(), use_container_width=True)
        st.caption("Kotak menunjukkan kuartil dan median; titik ekstrem belum tentu kesalahan data.")
    with tab3:
        plot_left, plot_right = st.columns(2)
        with plot_left:
            xcol = st.selectbox("Sumbu X", [c for c in FEATURES if c != "Gender"], index=3)
        with plot_right:
            ycol = st.selectbox("Sumbu Y", [c for c in FEATURES if c != "Gender"], index=9)
        fig = px.scatter(dff, x=xcol, y=ycol, color="Kategori", color_discrete_map=CODE_COLOR,
                         opacity=.8, hover_data=["AGE", "Gender"])
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.subheader("Korelasi Pearson antarf fitur")
        corr = filtered[FEATURES].drop(columns=["Gender"]).corr(numeric_only=True)
        fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", origin="lower",
                        aspect="auto", text_auto=".2f")
        st.plotly_chart(chart_layout(fig, height=585), use_container_width=True)
        st.caption("Target Class sengaja tidak dimasukkan karena kode 0, 1, 2 adalah kategori, bukan skala risiko kontinu.")
    with tab4:
        st.dataframe(dff[FEATURES + ["Class", "Kategori"]], hide_index=True, use_container_width=True)
        st.download_button("Unduh CSV data terfilter", filtered.to_csv(index=False).encode("utf-8-sig"),
                           "diabetes_data_terfilter.csv", "text/csv")
        with st.expander("Kamus data"):
            st.dataframe(pd.DataFrame({"Kolom": list(DESCRIPTION), "Deskripsi": list(DESCRIPTION.values())}),
                         hide_index=True, use_container_width=True)

elif page == "🧹 Audit Data":
    headline("Audit Data & Pembersihan", "Menunjukkan perubahan baris data dan nilai ekstrem secara eksplisit, bukan menghapusnya secara tersembunyi.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data awal", len(original))
    c2.metric("Nilai kosong", int(original.isna().sum().sum()))
    c3.metric("Duplikasi persis", int(original.duplicated().sum()))
    c4.metric("Data dihapus pada notebook", len(original) - len(cleaned),
              f"{(len(original)-len(cleaned))/len(original)*100:.1f}%", delta_color="off")
    st.subheader("Jejak filter IQR yang diterapkan notebook")
    st.dataframe(audit.style.format({"Batas bawah": "{:.3f}", "Batas atas": "{:.3f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("Filter berurutan: pertama Cr (25 baris), kemudian VLDL (11 baris pada data tersisa). Total 36 baris dihapus.")
    v1, v2 = st.columns([1.05, 1], gap="large")
    with v1:
        st.subheader("Distribusi kelas sebelum vs sesudah")
        frame = []
        for name, data in [("Asli", original), ("Setelah IQR", cleaned)]:
            sub = class_counts(data)
            sub["Dataset"] = name
            frame.append(sub)
        both = pd.concat(frame, ignore_index=True)
        fig = px.bar(both, x="Kelas", y="Jumlah", color="Dataset", barmode="group", text="Jumlah",
                     color_discrete_map={"Asli": "#197d85", "Setelah IQR": "#e9a23b"})
        fig.update_xaxes(title_text="")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with v2:
        st.subheader("Bandingkan sebaran nilai")
        variable = st.selectbox("Variabel numerik", [c for c in FEATURES if c != "Gender"], index=2)
        compared = pd.concat([original.assign(Dataset="Asli"), cleaned.assign(Dataset="Setelah IQR")], ignore_index=True)
        fig = px.box(compared, x="Dataset", y=variable, color="Dataset", points="outliers",
                     color_discrete_map={"Asli": "#197d85", "Setelah IQR": "#e9a23b"})
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    st.subheader("Deteksi outlier IQR pada dataset asli")
    report = outlier_summary(original)
    st.dataframe(report.style.format({"Persen": "{:.1f}%", "Batas bawah": "{:.2f}",
                                      "Batas atas": "{:.2f}", "Min": "{:.2f}", "Maks": "{:.2f}"}),
                 hide_index=True, use_container_width=True)
    st.warning(
        "**Pertimbangan kualitas:** Dataset asli memiliki nilai ekstrem, termasuk Cr hingga 800 "
        "dan VLDL hingga 31,8, serta Chol bernilai 0. Satuan dan validitas biologis "
        "belum dapat dipastikan dari arsip. Deteksi IQR bersifat statistik, **bukan bukti data salah**."
    )
    st.markdown(
        "**Catatan metodologi:** Replikasi menggunakan filter IQR pada seluruh dataset sebelum "
        "train/test split, sebagaimana notebook. Langkah ini dapat menyebabkan *data leakage* "
        "pada proses evaluasi. Mode 'Evaluasi lebih ketat' mempertahankan semua 264 observasi "
        "dan menempatkan StandardScaler serta RFE di dalam pipeline pelatihan/validasi."
    )
    with st.expander("Lihat observasi yang dikeluarkan filter notebook"):
        dropped = original.loc[~original.index.isin(cleaned.index)]
        st.dataframe(dropped, use_container_width=True)
        st.download_button("Unduh CSV baris yang dikeluarkan", dropped.to_csv(index=False).encode("utf-8-sig"),
                           "observasi_terfilter_iqr.csv", "text/csv")

elif page == "🧠 Evaluasi Model":
    headline("Perbandingan 3 Algoritma", "Hasil dihitung ulang dari data yang disertakan. Seluruh model diuji pada split yang sama.")
    source_disclaimer()
    bundle = get_bundle(mode)
    best = str(bundle.summary.iloc[0]["Algoritma"])
    st.markdown(f"**Skenario aktif:** {mode}")
    if mode == MODE_NOTEBOOK:
        st.warning(
            "Mode replikasi memakai 228 observasi dan **5 fitur tetap hasil RFE notebook** "
            "(Gender, HbA1c, TG, VLDL, BMI). Pemilihan 5 fitur itu semula dilakukan "
            "pada seluruh data, sehingga estimasi test set tidak sepenuhnya bebas bias seleksi fitur."
        )
    else:
        st.success(
            "Mode lebih ketat menggunakan 264 observasi, tanpa membuang outlier otomatis. "
            "RFE 5 fitur dan StandardScaler dipelajari ulang dalam tiap fold CV."
        )
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Model terbaik pada uji ini", best)
    q2.metric("Macro F1 (test)", f"{bundle.summary.iloc[0]['Macro F1']:.3f}")
    q3.metric("Macro Recall (test)", f"{bundle.summary.iloc[0]['Macro Recall']:.3f}")
    q4.metric("Data latih / uji", f"{len(bundle.y_train)} / {len(bundle.y_test)}")
    st.caption("Model terbaik dipilih menurut Macro F1 test, lalu Macro ROC-AUC sebagai pemecah seri. "
               "Pemilihan berbasis test set ini hanya untuk demonstrasi, bukan estimasi performa model final yang independen.")
    st.subheader("Tabel evaluasi test set")
    metric_cols = ["Akurasi", "Macro Recall", "Macro F1", "Macro ROC-AUC", "CV Macro F1 (mean)", "CV Macro F1 (std)"]
    st.dataframe(bundle.summary.set_index("Algoritma").style.format({m: "{:.3f}" for m in metric_cols}),
                 use_container_width=True)
    st.download_button("Unduh metrik semua model (CSV)",
                       bundle.summary.to_csv(index=False).encode("utf-8-sig"),
                       "evaluasi_model_diabetes.csv", "text/csv")
    st.subheader("Perbandingan visual")
    comparative = bundle.summary.melt(id_vars="Algoritma", value_vars=["Akurasi", "Macro Recall", "Macro F1", "Macro ROC-AUC"],
                                      var_name="Metrik", value_name="Skor")
    fig = px.bar(comparative, x="Metrik", y="Skor", color="Algoritma", barmode="group",
                 text_auto=".3f", color_discrete_sequence=["#197d85", "#e9a23b", "#5079a4"])
    fig.update_yaxes(range=[0, 1.11], title_text="Skor (0–1)")
    fig.update_xaxes(title_text="")
    st.plotly_chart(chart_layout(fig, height=470), use_container_width=True)
    st.caption("Macro Recall mengukur rata-rata sensitivitas setiap kelas. Macro F1 menggabungkan precision dan recall per kelas.")

    st.subheader("Detail satu algoritma")
    chosen = st.selectbox("Pilih model untuk pemeriksaan mendalam", MODEL_NAMES,
                          index=MODEL_NAMES.index(best))
    dt1, dt2 = st.columns([1, 1.05], gap="large")
    with dt1:
        st.markdown("**Confusion matrix (test set)**")
        normalized = st.toggle("Normalisasi menurut kelas sebenarnya", value=False)
        cm = bundle.confusion[chosen]
        z = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalized else cm
        fig = px.imshow(z, color_continuous_scale="Blues", text_auto=".2f" if normalized else ".0f",
                        x=[CLASS_NAMES[c] for c in CODES],
                        y=[CLASS_NAMES[c] for c in CODES],
                        labels={"x": "Prediksi", "y": "Aktual", "color": "Proporsi" if normalized else "Jumlah"},
                        aspect="auto")
        st.plotly_chart(chart_layout(fig, height=390), use_container_width=True)
        st.caption("Diagonal = benar; sel di luar diagonal = salah klasifikasi.")
    with dt2:
        st.markdown("**Classification report (test set)**")
        report = bundle.reports[chosen].copy()
        report.index = [CLASS_NAMES.get(int(x), x) if str(x).isdigit() else x for x in report.index]
        st.dataframe(report.style.format("{:.3f}"), use_container_width=True)
        st.caption("'support' menunjukkan jumlah observasi aktual pada setiap kelas.")

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("**ROC One-vs-Rest untuk model terpilih**")
        fig = go.Figure()
        for i, code in enumerate(CODES):
            binary = (bundle.y_test.to_numpy() == code).astype(int)
            fpr, tpr, _ = roc_curve(binary, bundle.test_probabilities[chosen][:, i])
            score = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{CLASS_NAMES[code]} · AUC {score:.3f}",
                                     line=dict(color=CLASS_COLORS[code], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Acak (AUC 0,5)",
                                 line=dict(color="#8b97a6", dash="dash")))
        fig.update_xaxes(title_text="False positive rate", range=[0, 1])
        fig.update_yaxes(title_text="True positive rate", range=[0, 1])
        st.plotly_chart(chart_layout(fig, 440), use_container_width=True)
    with right:
        st.markdown("**Cross-validation (5 fold, hanya data latih)**")
        cv_selected = bundle.cv_folds[bundle.cv_folds["Algoritma"] == chosen]
        fig = px.line(cv_selected, x="Fold", y="Macro F1", markers=True,
                      color_discrete_sequence=["#197d85"])
        fig.add_hline(y=float(cv_selected["Macro F1"].mean()),
                      line_dash="dot", line_color="#e9a23b",
                      annotation_text=f"Mean = {cv_selected['Macro F1'].mean():.3f}")
        fig.update_yaxes(range=[0, 1.06])
        st.plotly_chart(chart_layout(fig, 440), use_container_width=True)
        st.caption("Cross-validation dipakai untuk melihat variasi antar fold, bukan membuktikan generalisasi klinis.")
    st.markdown(f"**Fitur yang digunakan model {chosen}:** {', '.join(bundle.selected_features[chosen])}")
    if mode == MODE_STRICT:
        st.caption("Pada mode lebih ketat, RFE memilih fitur dari data latih untuk model akhir; "
                   "pilihan fitur dalam setiap fold CV bisa berbeda.")

    with st.expander("Bandingkan dengan angka yang tercantum dalam notebook sumber"):
        reference = pd.DataFrame({
            "Algoritma": list(MODEL_NAMES),
            "Macro Recall (sumber)": [0.896, 0.950, 0.950],
            "Macro F1 (sumber)": [0.859, 0.936, 0.936],
            "Macro ROC-AUC (sumber)": [0.953, 0.985, 0.965],
        })
        st.dataframe(reference, hide_index=True, use_container_width=True)
        st.caption("Ini adalah **angka historis yang dilaporkan** notebook, bukan hasil aplikasi saat ini. "
                   "Perbedaan dapat muncul karena versi library, langkah pemodelan, atau konfigurasi CV.")

elif page == "🔎 Prediksi Eksperimental":
    headline("Prediksi Eksperimental", "Mengilustrasikan keluaran model dari nilai masukan numerik — bukan diagnosis atau penilaian risiko kesehatan.")
    source_disclaimer()
    bundle = get_bundle(mode)
    algorithm = st.selectbox("Model yang digunakan untuk simulasi", MODEL_NAMES,
                             index=MODEL_NAMES.index(str(bundle.summary.iloc[0]["Algoritma"])))
    st.caption(f"Skenario: {mode} · Input diperlukan: {len(bundle.input_features)} fitur. "
               f"Fitur efektif hasil seleksi: {', '.join(bundle.selected_features[algorithm])}.")
    with st.form("manual_prediction"):
        st.subheader("Masukkan angka berdasarkan skema dataset")
        st.caption("Nilai bawaan adalah median dataset. Satuan dan batas normal klinis tidak terverifikasi.")
        values = {}
        inputs = st.columns(3)
        for idx, col in enumerate(bundle.input_features):
            with inputs[idx % 3]:
                if col == "Gender":
                    values[col] = st.selectbox("Gender (kode 0/1)", [0, 1],
                                               index=int(original["Gender"].median()),
                                               help="Tidak diasumsikan kode ini merepresentasikan gender tertentu.")
                elif col in ("AGE", "Cr"):
                    values[col] = st.number_input(col, value=int(round(float(original[col].median()))),
                                                  step=1, help=DESCRIPTION[col])
                else:
                    values[col] = st.number_input(col, value=float(round(float(original[col].median()), 2)),
                                                  step=0.1, format="%.2f", help=DESCRIPTION[col])
        submitted = st.form_submit_button("Jalankan simulasi klasifikasi", type="primary", use_container_width=True)
    if submitted:
        sample = pd.DataFrame([values], columns=bundle.input_features)
        try:
            checked = validate_prediction_rows(sample, bundle.input_features)
            proba = bundle.models[algorithm].predict_proba(checked)[0]
            code = int(bundle.models[algorithm].predict(checked)[0])
            st.success(f"Hasil model: **{CLASS_LABELS[code]}**")
            st.write("**Skor keluaran model per kelas (bukan probabilitas klinis terkalibrasi)**")
            result = pd.DataFrame({"Kelas": [CLASS_LABELS[c] for c in CODES], "Skor model": proba})
            fig = px.bar(result, x="Skor model", y="Kelas", orientation="h", color="Kelas",
                         color_discrete_map=CODE_COLOR, text_auto=".1%")
            fig.update_xaxes(range=[0, 1.09], tickformat=".0%")
            st.plotly_chart(chart_layout(fig, 285), use_container_width=True)
            outside = [c for c in bundle.input_features
                       if c != "Gender" and (checked.at[0, c] < original[c].min() or checked.at[0, c] > original[c].max())]
            if outside:
                st.warning("Nilai di luar rentang data latih asal: " + ", ".join(outside) + ". Hasil ekstrapolasi sangat tidak pasti.")
            st.caption("Prediksi dapat salah; angka tidak mengukur keparahan penyakit, tidak untuk keputusan pasien.")
        except ValueError as e:
            st.error(f"Data masukan tidak valid: {e}")

    st.divider()
    st.subheader("Prediksi banyak baris dari CSV (opsional)")
    st.caption("Hanya gunakan **data sintetis atau data yang diizinkan**. Jangan unggah identitas atau rekam medis pribadi.")
    example = original[bundle.input_features].head(3).copy()
    st.download_button("Unduh contoh format CSV", example.to_csv(index=False).encode("utf-8-sig"),
                       "template_input_diabetes.csv", "text/csv")
    uploaded = st.file_uploader("Unggah CSV berisi fitur yang diperlukan", type="csv", key="batch_csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            if len(batch) > 5000:
                raise ValueError("Maksimal 5.000 baris per unggahan agar aplikasi tetap responsif.")
            if len(batch) == 0:
                raise ValueError("CSV tidak memiliki baris data.")
            validated = validate_prediction_rows(batch, bundle.input_features)
            pred = bundle.models[algorithm].predict(validated)
            prob = bundle.models[algorithm].predict_proba(validated)
            output = validated.copy()  # Exclude arbitrary extra fields / potential identifiers from export
            output["Predicted_Class"] = pred
            output["Predicted_Label"] = [CLASS_NAMES[int(v)] for v in pred]
            for j, c in enumerate(CODES):
                output[f"Model_Score_Class_{c}"] = prob[:, j].round(6)
            st.dataframe(output.head(100), use_container_width=True)
            st.download_button("Unduh hasil simulasi CSV", output.to_csv(index=False).encode("utf-8-sig"),
                               "hasil_simulasi_diabetes.csv", "text/csv")
            st.warning("Hasil bersifat eksperimen statistika. Skor model belum dikalibrasi sebagai probabilitas klinis.")
        except Exception as exc:
            st.error(f"CSV tidak dapat diproses: {exc}")

elif page == "📚 Metodologi & Sumber":
    headline("Metodologi & Transparansi", "Dokumentasi alur, pilihan pemodelan, dan batas interpretasi terhadap tiga sumber asli.")
    st.subheader("Ringkasan sumber yang disertakan")
    sources = pd.DataFrame([
        ("Multiclass Diabetes Dataset.csv", "264 baris × 12 kolom; tidak ditemukan missing value"),
        ("Namirus_Project_BNSP(Multiclass_Diabetes_Dataset).ipynb", "Notebook 89 sel; EDA, IQR, RFE, dan 3 algoritma"),
        ("Namirus_Portofolio_BNSP.pdf", "59 halaman; portofolio dan presentasi hasil penelitian"),
    ], columns=["Sumber", "Peran"])
    st.dataframe(sources, hide_index=True, use_container_width=True)
    with st.expander("Unduh sumber asli dari ZIP yang diberikan"):
        st.download_button("Unduh CSV asli", DATA_PATH.read_bytes(), DATA_PATH.name, mime="text/csv")
        notebook_path = BASE_DIR / "referensi" / "Namirus_Project_BNSP(Multiclass_Diabetes_Dataset).ipynb"
        portfolio_path = BASE_DIR / "referensi" / "Namirus_Portofolio_BNSP.pdf"
        if notebook_path.exists():
            st.download_button("Unduh notebook asli", notebook_path.read_bytes(), notebook_path.name,
                               mime="application/x-ipynb+json")
        if portfolio_path.exists():
            st.download_button("Unduh portofolio PDF", portfolio_path.read_bytes(), portfolio_path.name,
                               mime="application/pdf")
    st.subheader("Tahapan pemodelan")
    st.markdown("""
1. **Audit awal:** verifikasi 12 kolom, nilai kosong, duplikasi, label 0/1/2, dan rentang statistik.
2. **Skenario replikasi:** filter IQR Cr kemudian VLDL → 228 baris; gunakan 5 fitur RFE yang **telah dipilih di notebook**.
3. **Skenario lebih ketat:** pertahankan semua 264 baris; masukkan StandardScaler dan **RFE lima fitur di dalam pipeline** setiap pembagian pelatihan.
4. **Pembagian 80:20:** `train_test_split(..., stratify=y, random_state=42)`. Model berbagi data latih dan test yang sama.
5. **Tiga model:** Logistic Regression, Random Forest, dan Gradient Boosting.
6. **Penilaian:** Akurasi, Macro Recall, Macro F1, ROC-AUC multikelas One-vs-Rest, classification report, dan confusion matrix.
7. **5-fold stratified cross-validation:** hanya pada data latih; scaler/RFE di-fit ulang setiap fold dalam mode ketat. Test set tidak dipakai untuk CV.
""")
    st.subheader("Perbedaan penting dari notebook")
    st.markdown("""
- Notebook menghitung RFE **sebelum** train/test split; hal itu dapat membocorkan informasi label test ke pemilihan fitur. Mode lebih ketat memperbaiki urutan tersebut.
- Notebook melakukan filter IQR pada keseluruhan data; mode replikasi sengaja mempertahankannya untuk keterlacakan. Mode lebih ketat tidak membuang outlier otomatis.
- Notebook membagi data terstratifikasi 80:20 dan menstandardisasi fitur berdasarkan data latih. Aplikasi juga melakukannya lewat pipeline.
- Notebook menulis semua model menggunakan `class_weight='balanced'`, tetapi implementasi Gradient Boosting di notebook **tidak** menetapkan parameter tersebut. Aplikasi mempertahankan implementasi aktual: Logistic Regression dan Random Forest berbobot kelas; Gradient Boosting tanpa class weighting.
- Hasil CV di notebook memproses skala data latih sebelum CV dan menggunakan fold default. Aplikasi melatih scaler per fold melalui pipeline dan memakai shuffled stratified folds, sehingga angka CV dapat berbeda.
- Peringkat algoritma pada tabel test set **bukan bukti pemilihan model yang tidak bias**. Untuk klaim generalisasi yang kuat, diperlukan prosedur seleksi model terpisah dan evaluasi eksternal.
""")
    st.subheader("Kamus data dan interpretasi aman")
    st.dataframe(pd.DataFrame({"Kolom": list(DESCRIPTION), "Keterangan": list(DESCRIPTION.values())}),
                 hide_index=True, use_container_width=True)
    st.markdown("""
**Keterbatasan penelitian:** Skema tidak mencantumkan metadata satuan baku dan pemetaan gender. Kode kelas disimpulkan dari implementasi notebook, bukan label diagnosis klinis yang sudah diverifikasi. Dataset kecil, tidak memiliki riwayat pengumpulan/representativitas yang dapat dinilai, dan mengandung nilai ekstrem. Fitur yang terpilih bersifat prediktif, **bukan** bukti kausalitas. Hasil test set dari satu split dan CV pada data yang sama belum cukup untuk menyatakan kesiapan implementasi klinis.
""")
    st.info("**Kesimpulan penggunaan:** Cocok sebagai portofolio pembelajaran data mining, EDA, "
            "audit praproses, dan perbandingan model. Tidak cocok dijadikan penentu status kesehatan seseorang.")
