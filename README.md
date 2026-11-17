# Multiclass Diabetes Analytics — Streamlit

Aplikasi Streamlit berbahasa Indonesia untuk mengeksplorasi, mengaudit, dan mengevaluasi model klasifikasi diabetes multikelas berdasarkan **tiga dokumen dalam arsip asli**: dataset CSV, notebook Jupyter/Colab, dan portofolio BNSP.

> **Penting:** Proyek ini merupakan demonstrasi analisis dan pembelajaran machine learning, **bukan** alat diagnosis, skrining klinis, atau rekomendasi medis. Tidak ada bukti pada arsip untuk memvalidasi label, satuan pengukuran, atau generalisasi klinis model.

## Fitur aplikasi

- **Dashboard:** ringkasan observasi dan fitur, jumlah data kosong, komposisi tiga kelas, median indikator menurut kelas.
- **Eksplorasi Data:** filter kelas, gender (kode 0/1), usia; histogram, boxplot, scatter plot, heatmap korelasi, tabel dan ekspor CSV.
- **Audit Data:** pemeriksaan missing values, duplikasi, outlier IQR, rincian 36 baris yang dihapus notebook, dan perbandingan data sebelum/sesudah pembersihan.
- **Evaluasi Model:** Logistic Regression, Random Forest, Gradient Boosting; akurasi, Macro Recall, Macro F1, Macro ROC-AUC, confusion matrix, classification report, ROC OvR, dan 5-fold cross-validation pada data latih.
- **Prediksi Eksperimental:** input nilai manual atau CSV untuk mengilustrasikan keluaran model, lengkap dengan pengingat bahwa skor bukan probabilitas klinis terkalibrasi.
- **Metodologi & Sumber:** perbandingan pendekatan notebook vs versi evaluasi yang lebih ketat, kamus data, batas interpretasi, dan unduhan ulang sumber asli.

## Persyaratan

Python 3.10–3.13 direkomendasikan (gunakan versi Python yang kompatibel dengan paket pada `requirements.txt`). Internet dibutuhkan saat menginstal paket untuk pertama kali.

## Instalasi dan menjalankan

**Cara paling mudah:** pada Windows, klik dua kali `JALANKAN_WINDOWS.bat`. Pada Ubuntu/macOS, jalankan `bash JALANKAN_UBUNTU_MAC.sh` di folder proyek. Kedua skrip membuat virtual environment, memasang paket, lalu memulai Streamlit.

**Cara manual:**

1. Ekstrak ZIP proyek.
2. Buka terminal di folder yang memiliki `app.py` dan `requirements.txt`.
3. Buat virtual environment:

   **Windows (PowerShell):**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **Ubuntu/macOS:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Instal dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Jalankan Streamlit:
   ```bash
   streamlit run app.py
   ```
6. Buka alamat lokal yang ditampilkan Streamlit (umumnya `http://localhost:8501`).

**Jika PowerShell tidak mengizinkan aktivasi virtual environment:** gunakan `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` dan `.\.venv\Scripts\python.exe -m streamlit run app.py` tanpa aktivasi.

## Struktur berkas

```text
multiclass-diabetes-streamlit/
├── app.py                         # Streamlit (6 halaman navigasi)
├── data_utils.py                  # Validasi dan cleaning IQR serta kamus data
├── modeling.py                    # Pipeline, 3 model, metrik, CV, prediksi
├── JALANKAN_WINDOWS.bat            # Menjalankan mudah di Windows
├── JALANKAN_UBUNTU_MAC.sh         # Menjalankan mudah di Linux/macOS
├── requirements.txt               # Dependencies
├── .streamlit/config.toml         # Tema aplikasi
├── data/
│   └── Multiclass Diabetes Dataset.csv
├── referensi/
│   ├── Namirus_Project_BNSP(Multiclass_Diabetes_Dataset).ipynb
│   └── Namirus_Portofolio_BNSP.pdf
└── tests/
    └── test_project.py
```

## Dataset dan makna label

Dataset memiliki **264 baris × 12 kolom**; 11 fitur (`Gender`, `AGE`, `Urea`, `Cr`, `HbA1c`, `Chol`, `TG`, `HDL`, `LDL`, `VLDL`, `BMI`) dan target `Class`.

| Kode kelas | Nama menurut notebook | Data asli | Setelah pembersihan |
|---|---|---:|---:|
| 0 | Non-Diabetic | 96 | 90 |
| 1 | Diabetic | 40 | 37 |
| 2 | Prediabetic / Predict Diabetic | 128 | 101 |
| **Total** | | **264** | **228** |

Makna `Gender=0/1` tidak bisa dipastikan dari sumber; aplikasi **tidak** mengubahnya menjadi laki-laki/perempuan. Satuan semua indikator laboratorium belum terverifikasi.

## Dua skenario pemodelan

| | Replikasi notebook | Evaluasi lebih ketat |
|---|---|---|
| Data | 228 observasi setelah filter IQR `Cr` kemudian `VLDL` | Semua 264 observasi |
| Masukan model | 5 fitur tetap dari RFE notebook: `Gender`, `HbA1c`, `TG`, `VLDL`, `BMI` | Semua 11 fitur sebagai input pipeline |
| Seleksi fitur | Fitur tetap; RFE **tidak** dihitung ulang setelah train/test split | RFE lima fitur dilakukan *di dalam pipeline*, setelah scaling |
| Scaling | StandardScaler di-fit pada latih | StandardScaler di-fit pada latih dan masing-masing CV fold |
| Split | 80:20, stratified, random_state 42 | Sama |
| CV | Stratified 5-fold, shuffle, random_state 42 pada data latih | Sama; RFE ikut dilatih ulang dalam tiap fold |
| Keterbatasan | Mewarisi pemilihan fitur dari keseluruhan dataset di notebook, berpotensi seleksi-bias/leakage | Lebih tepat sebagai eksperimen internal, belum ada validasi eksternal |

**Penting:** Fitur RFE notebook awalnya dipilih menggunakan seluruh data (termasuk data yang kemudian dijadikan test). Dengan demikian, mode replikasi **tidak** boleh dianggap estimasi performa yang sepenuhnya independen. Penyaringan outlier sebelum split juga mendatangkan risiko kebocoran informasi distribusi data.

### Metrik terverifikasi pada lingkungan pengembangan

Mode replikasi (228 observasi, 182 latih, 46 uji) memberikan:

| Algoritma | Akurasi | Macro Recall | Macro F1 | Macro ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0,870 | 0,896 | 0,859 | 0,953 |
| Random Forest | 0,935 | 0,950 | 0,936 | 0,985 |
| Gradient Boosting | 0,935 | 0,950 | 0,936 | 0,965 |

Angka ini konsisten dengan keluaran test set notebook yang dilampirkan. **Skor CV sengaja tidak diklaim identik**, karena implementasi aplikasi memakai prosedur yang lebih tepat: tiap fold melatih ulang pipeline pada subset latih dan mengacak stratified folds dengan seed yang tetap.

Mode lebih ketat (264 observasi, 211 latih, 53 uji), pada lingkungan pengembangan:

| Algoritma | Akurasi | Macro Recall | Macro F1 | Macro ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0,887 | 0,827 | 0,835 | 0,966 |
| Random Forest | 0,943 | 0,933 | 0,930 | 0,996 |
| Gradient Boosting | 0,925 | 0,920 | 0,906 | 0,990 |

**Perbedaan skor antarmode tidak berarti bahwa menghapus atau mempertahankan outlier lebih baik secara klinis.** Keduanya mengevaluasi kumpulan data dan prosedur yang berbeda. Hasil bisa bergeser sedikit pada versi paket yang berbeda.

### Catatan penting atas implementasi sumber

- Keluaran notebook menunjukkan `GradientBoostingClassifier(random_state=42)` **tanpa class_weight**, walaupun penjelasan menyatakan semua model berbobot kelas. Aplikasi mempertahankan perilaku kode asli. `LogisticRegression` dan `RandomForestClassifier` memakai `class_weight='balanced'`.
- Output notebook mencantumkan rata-rata CV F1 Random Forest sekitar 0,953; aplikasi dapat menghasilkan skor CV berbeda karena prosedur fold yang lebih ketat.
- Perhitungan Macro ROC-AUC menggunakan metode One-vs-Rest.
- Semua kolom dan label kelas dipertahankan apa adanya, tanpa mengubah kode ke tingkat keparahan penyakit yang tidak didukung sumber.
- 36 observasi yang dibuang adalah hasil filter statistik IQR, **bukan 36 rekam medis yang terbukti salah**.
- Pemilihan "model terbaik" berdasarkan metrik test set pada aplikasi hanya ilustrasi perbandingan. Model yang benar-benar akan diterapkan membutuhkan model selection pada validation/CV yang terpisah dan evaluasi pada test eksternal yang tidak dipakai untuk memilih model.

## Menjalankan pengujian

```bash
python -m pip install pytest
python -m pytest -q
```

Pengujian memeriksa skema data, jumlah observasi kelas, pembersihan notebook, pilihan fitur, keluaran prediksi, metrik, dan pipeline tanpa kebocoran RFE pada skenario ketat.

## Sitasi sumber utama dalam paket

1. `data/Multiclass Diabetes Dataset.csv` — observasi dan fitur.
2. `referensi/Namirus_Project_BNSP(Multiclass_Diabetes_Dataset).ipynb` — preprocessing, EDA, RFE, kode model dan metrik.
3. `referensi/Namirus_Portofolio_BNSP.pdf` — ringkasan presentasi portofolio.

Hak dan lisensi asli dataset maupun materi sumber tidak dinyatakan dalam berkas yang diterima; proyek ini tidak mengklaim lisensi baru atas sumber-sumber tersebut.
