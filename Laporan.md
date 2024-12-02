# **Laporan Proyek Machine Learning - Muhamad Rifqi Afriansyah**

---

## **Domain Proyek**

Penyakit jantung terus menjadi salah satu penyebab utama kematian global, menyebabkan jutaan kematian setiap tahunnya. Deteksi dini risiko penyakit jantung memainkan peran penting dalam mengurangi angka kematian dan meningkatkan kualitas hidup pasien. Namun, metode diagnosa tradisional sering kali memakan waktu, membutuhkan biaya yang besar, dan sangat bergantung pada keahlian medis.

Dalam konteks ini, machine learning memberikan peluang untuk memanfaatkan data medis seperti usia, tekanan darah, kadar kolesterol, dan jenis nyeri dada untuk membangun model prediksi yang cepat dan akurat. Pendekatan ini tidak hanya menghemat waktu tetapi juga dapat meningkatkan efisiensi diagnosa, terutama di daerah dengan sumber daya kesehatan yang terbatas.

**Referensi**:
- [Centers for Disease Control and Prevention (CDC)](https://www.cdc.gov/heartdisease/index.htm) - Overview of Heart Disease.
- [World Health Organization (WHO)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)) - Cardiovascular Diseases.

---

## **Business Understanding**

### **Problem Statements**
1. Penyakit jantung adalah penyebab utama kematian di dunia, tetapi metode diagnosa tradisional lambat dan mahal.
2. Data medis pasien sering kali tidak dimanfaatkan secara optimal untuk mendukung prediksi berbasis machine learning.
3. Model prediksi yang kurang akurat dapat menyebabkan diagnosa yang salah dan membahayakan pasien.

### **Goals**
1. Membangun model machine learning yang dapat memprediksi risiko penyakit jantung dengan cepat dan efisien.
2. Memanfaatkan dataset medis pasien untuk membangun model prediksi berbasis data.
3. Mengidentifikasi model terbaik berdasarkan evaluasi metrik seperti akurasi, precision, recall, dan F1-score.

### **Solution Statements**
1. Menggunakan algoritma seperti Logistic Regression, Random Forest, Support Vector Machine (SVM), dan Deep Learning untuk membangun baseline model.
2. Membandingkan kinerja model menggunakan metrik evaluasi yang sesuai.
3. Melakukan hyperparameter tuning pada model terbaik untuk meningkatkan performa.
4. Menggunakan validasi silang untuk memastikan performa yang stabil.

---

## **Data Understanding**

### **Dataset**
Dataset yang digunakan berasal dari [Kaggle - Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). Dataset ini berisi informasi medis pasien dengan berbagai parameter untuk memprediksi risiko penyakit jantung.

### **Informasi Dataset**
Dataset memiliki **918 baris** dan **12 kolom**, dengan deskripsi fitur sebagai berikut:

| **Nama Fitur**       | **Tipe Data** | **Deskripsi** |
|----------------------|---------------|---------------|
| `Age`                | int64         | Usia pasien dalam tahun. |
| `Sex`                | object        | Jenis kelamin pasien (M: Laki-laki, F: Perempuan). |
| `ChestPainType`      | object        | Jenis nyeri dada (TA, ATA, NAP, ASY). |
| `RestingBP`          | int64         | Tekanan darah saat istirahat (mm Hg). |
| `Cholesterol`        | int64         | Kadar kolesterol serum (mg/dl). |
| `FastingBS`          | int64         | Gula darah puasa > 120 mg/dl (1 = ya, 0 = tidak). |
| `RestingECG`         | object        | Hasil elektrokardiografi (Normal, ST, LVH). |
| `MaxHR`              | int64         | Detak jantung maksimum yang dicapai. |
| `ExerciseAngina`     | object        | Angina yang diinduksi olahraga (Y = ya, N = tidak). |
| `Oldpeak`            | float64       | Depresi segmen ST selama tes olahraga. |
| `ST_Slope`           | object        | Kemiringan segmen ST (Up, Flat, Down). |
| `HeartDisease`       | int64         | Target (1: Risiko tinggi, 0: Risiko rendah). |

### **Kondisi Data (Missing Values)**
Setelah memeriksa data, ditemukan bahwa tidak ada nilai yang hilang pada dataset ini. Berikut adalah rincian mengenai missing values:

| **Nama Fitur**       | **Jumlah Missing Values** | **Persentase Missing** |
|----------------------|---------------------------|------------------------|
| `Age`                | 0                         | 0%                     |
| `Sex`                | 0                         | 0%                     |
| `ChestPainType`      | 0                         | 0%                     |
| `RestingBP`          | 0                         | 0%                     |
| `Cholesterol`        | 0                         | 0%                     |
| `FastingBS`          | 0                         | 0%                     |
| `RestingECG`         | 0                         | 0%                     |
| `MaxHR`              | 0                         | 0%                     |
| `ExerciseAngina`     | 0                         | 0%                     |
| `Oldpeak`            | 0                         | 0%                     |
| `ST_Slope`           | 0                         | 0%                     |
| `HeartDisease`       | 0                         | 0%                     |

---

## **Data Preparation**

### **Langkah-langkah Data Preparation**
1. **Mengganti Nilai `?` dengan NaN**:
   - Proses: Nilai `?` dalam dataset diubah menjadi `NaN` menggunakan fungsi `replace`.
   - Alasan: Nilai `?` dianggap sebagai data yang hilang (*missing value*), sehingga perlu diubah menjadi format yang dapat dikenali oleh library Python.

2. **Menghapus Baris dengan Missing Values**:
   - Proses: Baris yang mengandung nilai `NaN` dihapus menggunakan fungsi `dropna`.
   - Alasan: Baris dengan *missing values* dapat memengaruhi hasil analisis dan pelatihan model, sehingga dihapus untuk menjaga kualitas data.

3. **Encoding Variabel Kategorikal**:
   - Proses: Variabel kategorikal seperti `Sex`, `ChestPainType`, `RestingECG`, dan `ST_Slope` diubah menjadi numerik menggunakan *One-Hot Encoding*.
   - Alasan: Algoritma machine learning hanya dapat bekerja dengan data numerik, sehingga variabel kategorikal perlu diubah.

4. **Normalisasi Fitur Numerik**:
   - Proses: Data numerik dinormalisasi menggunakan teknik *Min-Max Scaling* agar setiap fitur memiliki skala yang sama.
   - Alasan: Normalisasi diperlukan untuk mencegah bias pada fitur tertentu yang memiliki skala lebih besar dibandingkan fitur lainnya.

5. **Membagi Dataset menjadi Data Latih dan Uji**:
   - Proses: Data dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split`.
   - Alasan: Pembagian ini diperlukan untuk mengevaluasi performa model secara objektif pada data yang belum pernah dilihat model sebelumnya.

---

## **Modeling**

### **Algoritma yang Digunakan**

#### 1. **Logistic Regression**
- **Parameter utama**: `penalty='l2'`, `C=1.0`.
- **Kelebihan**: Cepat, sederhana, dan mudah diinterpretasikan.
- **Kekurangan**: Kurang efektif untuk pola data yang kompleks atau hubungan non-linear.
- **Cara Kerja**:
   Logistic Regression digunakan untuk klasifikasi biner, di mana model ini memprediksi probabilitas suatu kejadian (misalnya, serangan jantung) dengan menggunakan fungsi logistik.

   - **Fungsi Logistik (Sigmoid)**: Logistic Regression menggunakan fungsi sigmoid untuk mengubah nilai yang dihitung oleh model (kombinasi linear dari fitur dan bobot) menjadi probabilitas antara 0 dan 1. Fungsi sigmoid memiliki bentuk:
     $$
     \sigma(z) = \frac{1}{1 + e^{-z}}
     $$

     Dimana \(z\) adalah hasil perkalian linear dari fitur input \(X = (X_1, X_2, \dots, X_n)\) dan bobot model \(w = (w_0, w_1, \dots, w_n)\), yang dirumuskan sebagai:

     $$
     z = w_0 + w_1 X_1 + w_2 X_2 + \dots + w_n X_n
     $$


   - **Proses Pelatihan dan Perubahan Bobot**: Selama proses pelatihan, model Logistic Regression meminimalkan kesalahan (loss) menggunakan metode optimasi seperti **gradient descent**. Perubahan bobot terjadi berdasarkan gradien dari fungsi loss (log-loss), yang mengukur perbedaan antara probabilitas yang diprediksi dan label yang sebenarnya. Bobot diupdate untuk mengurangi error secara iteratif.
   
   - **Prediksi**: Setelah model dilatih, hasil prediksi berupa probabilitas (output dari fungsi sigmoid). Jika probabilitas lebih besar dari 0.5, data diklasifikasikan sebagai kelas positif (misalnya, memiliki risiko serangan jantung), dan jika kurang dari 0.5, diklasifikasikan sebagai kelas negatif.

#### 2. **Random Forest**
- **Parameter utama**: `n_estimators=100`, `max_depth=None`.
- **Kelebihan**: Mampu menangani data yang kompleks dan resisten terhadap overfitting.
- **Kekurangan**: Memerlukan lebih banyak sumber daya komputasi, terutama dengan jumlah pohon yang besar.
- **Cara Kerja**:
   Random Forest adalah algoritma ensemble yang menggunakan banyak pohon keputusan (decision trees) yang dilatih pada subset acak dari data pelatihan.
   
   - **Bagging**: Dalam Random Forest, data dibagi menjadi subset acak, dan setiap pohon keputusan dilatih pada subset tersebut. Proses ini dikenal dengan teknik *Bootstrap Aggregating* atau *Bagging*.
   
   - **Voting Mayoritas**: Setiap pohon memberikan prediksi (hasil klasifikasi), dan Random Forest menentukan hasil akhir dengan cara "voting" mayoritas di antara pohon-pohon tersebut.
   
   - **Prediksi**: Untuk setiap data yang diberikan, Random Forest mengumpulkan prediksi dari semua pohon dan memilih kelas yang paling sering dipilih oleh mayoritas pohon.

#### 3. **Support Vector Machine (SVM)**
- **Parameter utama**: `kernel='rbf'`, `C=1.0`.
- **Kelebihan**: Efektif untuk dataset berdimensi tinggi dan dapat menangani data non-linear dengan baik.
- **Kekurangan**: Memerlukan lebih banyak waktu dan sumber daya komputasi pada dataset besar.
- **Cara Kerja**:
   SVM adalah algoritma yang mencoba mencari hyperplane terbaik yang memisahkan dua kelas dalam data.
   
   - **Hyperplane dan Margin**: SVM berusaha menemukan hyperplane (garis atau bidang) yang memaksimalkan margin antara dua kelas. Margin ini diukur sebagai jarak antara hyperplane dan titik data terdekat dari masing-masing kelas, yang disebut *support vectors*.
   
   - **Kernel**: Ketika data tidak dapat dipisahkan dengan hyperplane linear, SVM menggunakan kernel untuk memetakan data ke ruang dimensi lebih tinggi, di mana pemisahan kelas menjadi lebih mudah. Kernel RBF (Radial Basis Function) adalah salah satu yang sering digunakan untuk data non-linear.

#### 4. **Neural Network (Deep Learning)**
- **Arsitektur**: Dua hidden layer, optimizer Adam, dan fungsi loss Binary Crossentropy.
- **Kelebihan**: Dapat menangkap pola kompleks dan non-linear dalam data.
- **Kekurangan**: Membutuhkan dataset besar dan waktu pelatihan yang lebih lama.
- **Cara Kerja**:
   Neural Network adalah model yang terinspirasi dari struktur otak manusia, yang terdiri dari lapisan-lapisan neuron yang saling terhubung untuk memproses informasi.

   - **Neuron dan Lapisan**: Jaringan syaraf terdiri dari beberapa lapisan:
     - **Input Layer**: Menerima fitur data.
     - **Hidden Layers**: Lapisan-lapisan di antara input dan output yang terdiri dari neuron-neuron. Setiap neuron menerima input, mengalikannya dengan bobot, dan menerapkan fungsi aktivasi.
     - **Output Layer**: Lapisan terakhir yang menghasilkan prediksi (probabilitas untuk klasifikasi biner).
     
     Setiap neuron dalam hidden layer melakukan operasi matematis berupa perkalian input dengan bobot, ditambah dengan bias, lalu menerapkan fungsi aktivasi (seperti **ReLU** pada hidden layers dan **sigmoid** pada output layer).

   - **Fungsi Aktivasi**: Fungsi **ReLU** (Rectified Linear Unit) digunakan pada hidden layers untuk menangkap pola non-linear dalam data. Fungsi **sigmoid** pada output layer digunakan untuk menghasilkan nilai probabilitas antara 0 dan 1, yang menentukan kelas prediksi.
   
   - **Proses Pelatihan (Backpropagation)**: Neural Network dilatih menggunakan algoritma **backpropagation**, di mana kesalahan (loss) dihitung di output layer dan kemudian disebarkan kembali ke lapisan-lapisan sebelumnya untuk memperbarui bobot dan bias menggunakan teknik optimasi, seperti **Adam**.
   
     Proses ini dilakukan secara iteratif, dengan tujuan meminimalkan fungsi loss (**Binary Crossentropy**), yang mengukur perbedaan antara prediksi model dan label yang sebenarnya.
   
   - **Prediksi**: Setelah pelatihan selesai, model dapat memprediksi probabilitas kelas berdasarkan data input baru.

---

## **Evaluation**

### **Hasil Evaluasi Model**

| **Model**             | **Akurasi** | **Precision** | **Recall** | **F1 Score** |
|------------------------|-------------|---------------|------------|--------------|
| Logistic Regression    | 0.8641      | 0.902         | 0.860      | 0.880        |
| Random Forest          | 0.8859      | 0.913         | 0.888      | 0.900        |
| Support Vector Machine | 0.8424      | 0.890         | 0.832      | 0.860        |
| Deep Learning          | 0.8533      | 0.908         | 0.832      | 0.868        |

### **Confusion Matrix**
Berikut adalah visualisasi confusion matrix untuk setiap model yang digunakan:

![Confusion Matrix](https://github.com/user-attachments/assets/3b47c2be-4c9d-4454-83bb-0091ebbab775)

**Deskripsi Gambar**:
- **Logistic Regression**:
  - **True Positives (TP)**: 92
  - **True Negatives (TN)**: 67
  - **False Positives (FP)**: 10
  - **False Negatives (FN)**: 15

- **Random Forest**:
  - **True Positives (TP)**: 95
  - **True Negatives (TN)**: 68
  - **False Positives (FP)**: 9
  - **False Negatives (FN)**: 12

- **SVM**:
  - **True Positives (TP)**: 89
  - **True Negatives (TN)**: 66
  - **False Positives (FP)**: 11
  - **False Negatives (FN)**: 18

- **Deep Learning**:
  - **True Positives (TP)**: 89
  - **True Negatives (TN)**: 68
  - **False Positives (FP)**: 9
  - **False Negatives (FN)**: 18

### **Analisis Hasil**
- **Random Forest** menunjukkan performa terbaik dengan **True Positives (TP)** dan **True Negatives (TN)** yang paling tinggi serta jumlah **False Positives (FP)** dan **False Negatives (FN)** yang paling rendah.
- **Logistic Regression** juga memberikan performa yang baik, meskipun jumlah kesalahan sedikit lebih tinggi dibandingkan Random Forest.
- **SVM** dan **Deep Learning** menunjukkan performa yang serupa, tetapi memiliki jumlah False Negatives yang lebih tinggi dibandingkan model lainnya.

### **Dampak Model**
- **Random Forest** adalah model yang paling andal untuk prediksi risiko penyakit jantung karena dapat mengurangi jumlah kesalahan prediksi, sehingga memberikan dampak yang signifikan dalam mendukung diagnosa medis yang lebih akurat dan efisien.

---

## **Kesimpulan dan Rekomendasi**

### **Kesimpulan**
1. **Random Forest** adalah model terbaik untuk prediksi risiko penyakit jantung berdasarkan hasil evaluasi.
2. Model ini dapat membantu mempercepat proses diagnosa dan mendukung pengambilan keputusan klinis yang lebih efisien.

### **Rekomendasi**
1. Implementasikan **Random Forest** sebagai model utama untuk prediksi risiko penyakit jantung.
2. Lakukan hyperparameter tuning untuk lebih meningkatkan akurasi model.
3. Eksplorasi dataset yang lebih besar untuk mengoptimalkan performa Deep Learning.
