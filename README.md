# (SKRIPSI-S1) Deteksi Pes Planus pada Citra Telapak Kaki Menggunakan Deep Learning dengan Pelabelan Berbasis Staheli Arch Index
# Classification of Pes Planus Based on Fluorescent Footprint Images Using Deep Learning

## Overview

Penelitian ini bertujuan mengembangkan sistem klasifikasi kondisi kaki **Non-Flatfoot** dan **Flatfoot (*Pes Planus*)** berdasarkan citra *footprint* fluoresensi menggunakan pendekatan *deep learning*.

Berbeda dengan pelabelan berdasarkan anotasi dataset awal, penelitian ini menggunakan **Staheli Arch Index (SAI)** sebagai parameter biomekanik dalam menentukan kelas kondisi kaki. Pendekatan ini digunakan untuk memperoleh pelabelan yang lebih objektif berdasarkan karakteristik bentuk lengkung longitudinal medial pada telapak kaki.

Penelitian mencakup tahapan **footprint extraction**, perhitungan **Staheli Arch Index**, validasi label oleh ahli, serta klasifikasi menggunakan beberapa arsitektur *deep learning*.

## Research Workflow

```text
Fluorescent Footprint Dataset
            │
            ▼
    Footprint Extraction
            │
            ├── Green Channel Extraction
            ├── Gaussian Blur
            ├── Otsu Thresholding
            ├── PCA Alignment
            └── Bounding Box Cropping
            │
            ▼
    Width Profile Computation
            │
            ▼
   Staheli Arch Index (SAI)
            │
            ▼
       Labeling
            │
            ▼
      Expert Validation
            │
            ▼
     Dataset Classification
            │
            ▼
       Image Pre-processing
            │
            ▼
     ┌──────┼──────────┐
     ▼      ▼          ▼
 ResNet50  VGG16+KNN  EfficientViT
     │      Stacking       │
     └──────┼──────────────┘
            ▼
        Evaluation
```

## Dataset

Dataset penelitian terdiri dari **879 citra *footprint* fluoresensi** yang diperoleh dari dataset publik **Flat Feet Detection Computer Vision Model** pada Roboflow.

Pelabelan ulang dilakukan menggunakan **Staheli Arch Index** sehingga diperoleh distribusi:

| Class     | Description             | Number of Images |
| --------- | ----------------------- | ---------------: |
| 0         | Non-Flatfoot            |              653 |
| 1         | Flatfoot (*Pes Planus*) |              226 |
| **Total** |                         |          **879** |

Label hasil perhitungan SAI kemudian melalui proses validasi manual oleh dokter spesialis kedokteran fisik dan rehabilitasi.

## Methodology

### 1. Footprint Extraction

Tahap *footprint extraction* digunakan untuk memperoleh area telapak kaki dari citra awal dan menghilangkan bagian latar belakang yang tidak diperlukan.

Tahapan yang digunakan:

1. **Green Channel Extraction**
   Mengekstraksi kanal hijau dari citra RGB untuk memperoleh representasi citra yang sesuai dengan karakteristik citra fluoresensi.

2. **Gaussian Blur**
   Mengurangi *noise* pada citra sebelum proses segmentasi.

3. **Otsu Thresholding**
   Melakukan segmentasi untuk memisahkan objek telapak kaki dari latar belakang secara otomatis.

4. **PCA Alignment**
   Menyeragamkan orientasi objek telapak kaki berdasarkan arah utama objek menggunakan *Principal Component Analysis* (PCA).

5. **Bounding Box Cropping**
   Memotong area citra berdasarkan *bounding box* sehingga hanya area telapak kaki yang digunakan pada tahap selanjutnya.

### 2. Width Profile Computation

Setelah proses ekstraksi *footprint*, dilakukan perhitungan profil lebar telapak kaki. Profil tersebut digunakan untuk memperoleh parameter yang dibutuhkan dalam perhitungan Staheli Arch Index.

### 3. Staheli Arch Index

Staheli Arch Index digunakan sebagai dasar penentuan kelas kondisi kaki.

SAI dihitung berdasarkan perbandingan antara lebar bagian tengah telapak kaki dengan lebar bagian belakang telapak kaki:

```text
SAI = Midfoot Width / Rearfoot Width
```

Nilai SAI kemudian digunakan untuk menentukan kelas **Non-Flatfoot** atau **Flatfoot (*Pes Planus*)** sesuai dengan kriteria yang digunakan dalam penelitian.

### 4. Expert Validation

Validasi label dilakukan untuk mengevaluasi kesesuaian hasil pelabelan berbasis SAI dengan penilaian klinis ahli.

Sebanyak **11 citra** dipilih secara *purposive* untuk divalidasi oleh dokter spesialis kedokteran fisik dan rehabilitasi. Hasil validasi menunjukkan bahwa **10 dari 11 citra** memiliki hasil klasifikasi yang sesuai dengan pelabelan berdasarkan SAI.

## Classification Models

Penelitian ini membandingkan tiga pendekatan klasifikasi.

### ResNet-50

ResNet-50 digunakan sebagai model *deep convolutional neural network* untuk melakukan klasifikasi citra *footprint*.

Konfigurasi utama:

* Architecture: ResNet-50
* Optimizer: AdamW
* Learning rate: `1e-4`
* Scheduler: ReduceLROnPlateau
* Early stopping: Applied
* Random seed: `42`

### VGG16 + KNN Stacking

Pendekatan *stacking ensemble* menggabungkan fitur citra yang diperoleh dari VGG16 dengan informasi numerik hasil pengukuran SAI.

Komponen utama:

* VGG16 sebagai *feature extractor*
* KNN sebagai model klasifikasi berbasis fitur numerik
* Features: `L`, `Q`, dan `R`
* Feature scaling: StandardScaler
* Ensemble strategy: Stacking

### EfficientViT

EfficientViT digunakan sebagai pendekatan berbasis *Vision Transformer* untuk mengevaluasi kemampuan arsitektur yang lebih efisien dalam melakukan klasifikasi citra *footprint*.

Model yang digunakan:

```text
efficientvit_m0
```

## Data Splitting

Dataset dibagi menjadi:

```text
100% Dataset
│
├── 80% Training
│      └── Stratified 5-Fold Cross-Validation
│
└── 20% Testing
```

**Stratified 5-Fold Cross-Validation** digunakan pada data pelatihan untuk mempertahankan proporsi kelas pada setiap *fold*.

Data pengujian dipisahkan dan tidak digunakan selama proses pelatihan maupun pemilihan model.

## Data Augmentation

Augmentasi diterapkan hanya pada data pelatihan dan difokuskan pada kelas minoritas.

Teknik augmentasi yang digunakan:

* Horizontal Flip

Augmentasi dilakukan untuk mengurangi ketidakseimbangan kelas antara **Non-Flatfoot** dan **Flatfoot** tanpa mengubah karakteristik utama bentuk telapak kaki.

## Class Weighting

Untuk mengatasi ketidakseimbangan kelas selama proses pelatihan, digunakan *class weight*:

```text
Non-Flatfoot : 0.8445
Flatfoot     : 1.2257
```

Bobot yang lebih tinggi diberikan kepada kelas Flatfoot untuk memberikan penalti yang lebih besar terhadap kesalahan klasifikasi pada kelas tersebut.

## Evaluation

Performa model dievaluasi menggunakan beberapa metrik:

* Accuracy
* Precision
* Recall
* F1-Score
* AUC-ROC
* Confusion Matrix
* Classification Report

Evaluasi dilakukan pada data pengujian untuk mengukur kemampuan generalisasi model terhadap data yang tidak digunakan selama proses pelatihan.

## Repository Structure

```text
.
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── images/
│
├── preprocessing/
│   ├── footprint_extraction.py
│   ├── width_profile.py
│   └── staheli_index.py
│
├── models/
│   ├── resnet50.py
│   ├── vgg16.py
│   ├── knn.py
│   ├── stacking.py
│   └── efficientvit.py
│
├── evaluation/
│   ├── evaluate.py
│   └── metrics.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

> Struktur direktori dapat disesuaikan dengan struktur aktual repository.

## Technologies

Penelitian ini menggunakan beberapa teknologi dan *framework* berikut:

* Python
* PyTorch
* Torchvision
* timm
* Scikit-learn
* OpenCV
* NumPy
* Pandas
* Matplotlib

## Reproducibility

Untuk menjalankan kode penelitian, instal dependensi yang tercantum pada `requirements.txt`.

```bash
git clone <repository-url>
cd <repository-folder>

pip install -r requirements.txt
```

Kemudian jalankan *pipeline* sesuai dengan struktur program pada repository.

Contoh:

```bash
python preprocessing/staheli_index.py
python preprocessing/augment.py
python preprocessing/split_data.py
python models/resnet50-Holdout.py
python models/resnet50-5fold.py
python models/EfficientVitM0-Holdout.py
python models/EfficientVitM0-5fold.py
python models/Stacking-Holdout.py
python models/Stacking-5fold.py
```

## Research Objectives

Penelitian ini berfokus pada:

1. Melakukan ekstraksi area *footprint* dari citra fluoresensi.
2. Menggunakan Staheli Arch Index sebagai dasar pelabelan kondisi kaki.
3. Memvalidasi hasil pelabelan menggunakan penilaian ahli.
4. Mengembangkan model klasifikasi *Non-Flatfoot* dan *Flatfoot*.
5. Membandingkan performa ResNet-50, VGG16 + KNN *stacking*, dan EfficientViT.
6. Mengevaluasi kemampuan model menggunakan metrik klasifikasi dan AUC-ROC.

## Citation

Jika menggunakan repository atau hasil penelitian ini, silakan mengutip penelitian sesuai format berikut:

```bibtex
@thesis{Retno Mentari_2026,
  author  = {Retno Mentari},
  title   = {Deteksi Pes Planus pada Citra Telapak Kaki Menggunakan Deep Learning dengan Pelabelan Berbasis Staheli Arch Index},
  school  = {Universitas Airlangga},
  year    = {2026},
  type    = {S1 Thesis}
}
```

## Author

**Retno Mentari**
Sistem Informasi
Universitas Airlangga
2026

