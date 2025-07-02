## Submission 2: Diabetes Detection dengan TFX Pipeline

**Nama:** Wahyu Rizky Cahyana  
**Username Dicoding:** elchilz

|                          | Deskripsi                                                                 |
|--------------------------|---------------------------------------------------------------------------|
| **Dataset**              | [Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset) dengan 8 kolom fitur dan 1 label `Outcome`. |
| **Masalah**              | Mendeteksi apakah pasien terindikasi **diabetes** atau **tidak** berdasarkan jumlah kehamilan, kadar glukosa dalam darah, tekanan darah, ketebalan lipatan kulit, kadar insulin dalam darah, massa tubuh, risiko diabetes berdasarkan silsilah keluarga dan usia pasien. |
| **Solusi machine learning** | Membuat pipeline **TFX** end-to-end menggunakan TensorFlow untuk preprocessing, pelatihan model klasifikasi, evaluasi, dan deployment model deteksi diabetes. |
| **Metode pengolahan**    | Scaling `0-1` pada fitur numerik menggunakan `tft.scale_to_0_1`, cast label ke `int64`, split 80-20 train-eval. |
| **Arsitektur model**     | TensorFlow Functional API dengan dense layers (`relu`) dan `sigmoid` pada output, menggunakan BatchNormalization, Dropout, L2 regularization, serta hyperparameter tuning dengan KerasTuner. |
| **Metrik evaluasi**      | `BinaryAccuracy` (threshold > 0.5), `AUC`, `Precision`, `Recall` dan `ExampleCount`.  |
| **Performa model**       | Model mencapai **AUC: 0.814** dengan jumlah data evaluasi 182 dan akurasi stabil pada evaluasi menggunakan TFMA. |
| **Opsi deployment**      | Model dideploy menggunakan **TensorFlow Serving** pada Railway untuk serving REST API prediksi diabetes. |
| **Web app**              | [Diabetes Detection API](https://mlops2-production-fce5.up.railway.app/v1/models/cc-model/metadata) untuk prediksi dengan JSON payload. |
| **Monitoring**           | Monitoring model serving dilakukan menggunakan **Prometheus** dengan exporter di container TensorFlow Serving untuk memantau latency, throughput dan health endpoint. |
