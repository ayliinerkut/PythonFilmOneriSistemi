# 🎬 İçerik Tabanlı Film Öneri Sistemi (Python)

Bu proje, kullanıcının seçtiği bir filme dayanarak, o filme **içeriksel olarak benzeyen** (tür, anahtar kelimeler vb. baz alınarak) diğer filmleri öneren basit bir tavsiye sistemidir.

## ✨ Proje Amacı

Makine öğrenimi ve doğal dil işleme (NLP) tekniklerinin temelini oluşturan **İçerik Tabanlı Filtreleme (Content-Based Filtering)** yöntemini uygulamak ve Python ile veri bilimi kütüphanelerini pratik etmektir.

## 🛠 Kullanılan Teknolojiler ve Kütüphaneler

Proje, temel olarak aşağıdaki Python kütüphanelerini kullanır:

* **Python:** Projenin temel dili.
* **`Pandas`:** Veri işleme ve analiz işlemleri için.
* **`Scikit-learn`:**
    * `TfidfVectorizer`: Film türü/özet metinlerini sayısal vektörlere dönüştürmek için (NLP).
    * `linear_kernel` (veya `cosine_similarity`): İki film arasındaki benzerliği (Kosinüs Benzerliği) hesaplamak için.

## 💾 Veri Seti

Bu projede, film verilerini içeren popüler **MovieLens Small Latest Dataset** kullanılmıştır.
* **Kullanılan Dosya:** `movies.csv`
* **Temel Veri Alanları:** `title` (başlık) ve `genres` (türler).
