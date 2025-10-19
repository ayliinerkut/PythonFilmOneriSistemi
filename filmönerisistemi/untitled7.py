# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 12:37:28 2025

@author: erkut
"""
# ====================================================================
# BÖLÜM 1: Kütüphaneleri ve Veriyi Yükleme
# ====================================================================

import sys 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. Veri setini tam yoluyla yükle
# Lütfen KENDİ KOPYALADIĞINIZ TAM YOLU buraya yapıştırın.
tam_yol = r'C:\Users\erkut\OneDrive\Masaüstü\filmönerisistemi\movies.csv'
df = pd.read_csv(tam_yol)

print("Veri seti başarıyla yüklendi. Toplam film sayısı:", len(df))

# 2. Gereksiz sütunları (movieId hariç) kaldırıp sadece işimize yarayacak sütunları tutalım.
# İçerik Tabanlı Öneride sadece film adlarına ve türlere ihtiyacımız var (genre: türler).
df = df[['movieId', 'title', 'genres']] 

# ... (Kodun geri kalanını değiştirmeyin)

try:
    
    sys.exit() # <--- Artık bu doğru şekilde programı durduracak.
    print("Veri seti başarıyla yüklendi. Toplam film sayısı:", len(df))
except FileNotFoundError:
    print("HATA: 'movies.csv' dosyası bulunamadı. Dosyayı aynı klasöre koyduğunuzdan emin olun.")
    exit()

# 2. Gereksiz sütunları (movieId hariç) kaldırıp sadece işimize yarayacak sütunları tutalım.
# İçerik Tabanlı Öneride sadece film adlarına ve türlere ihtiyacımız var (genre: türler).
df = df[['movieId', 'title', 'genres']]

# ====================================================================
# BÖLÜM 2: Veri Ön İşleme ve Sayısallaştırma (TF-IDF)
# ====================================================================

# 3. Türler (genres) sütununu, kolay işlenmesi için temizleyelim (örnek: "Action|Adventure" -> "Action Adventure")
df['genres'] = df['genres'].apply(lambda x: str(x).replace('|', ' '))

# 4. TF-IDF Vektörleştiriciyi Oluşturma
# Her filmin 'genres' (türler) bilgisine göre sayısal bir vektörünü oluştururuz.
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# ====================================================================
# BÖLÜM 3: Benzerliği Hesaplama (Kosinüs Benzerliği)
# ====================================================================

# 5. Kosinüs Benzerliği Matrisini Hesaplama
# Bu matris, her filmin diğer tüm filmlerle olan benzerlik skorunu içerir.
# linear_kernel, kosinüs benzerliğini hesaplamanın hızlı bir yoludur.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# ====================================================================
# BÖLÜM 4: Öneri Fonksiyonu Oluşturma
# ====================================================================

# 6. Film başlıklarını DataFrame'deki indekslerle eşleştiren bir seri oluşturma
# Kullanıcı film adını girdiğinde, bu seri bize DataFrame'deki satır numarasını verecek.
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, df=df, indices=indices):
    """
    Belirli bir film başlığına (title) göre benzer filmleri önerir.
    """
    if title not in indices:
        print(f"\nHATA: '{title}' filmi veri setinde bulunamadı. Lütfen tam ve doğru başlığı girin.")
        return []
        
    # Filmin indeksini (satır numarasını) al
    idx = indices[title]

    # O filme ait tüm benzerlik puanlarını al
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Benzerlik puanlarını azalan sıraya göre sırala
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # En benzer 11 filmin skorlarını (ilk skor filmin kendisidir) al
    sim_scores = sim_scores[1:11] 

    # Film indekslerini (satır numaralarını) al
    movie_indices = [i[0] for i in sim_scores]

    # Filmlerin başlıklarını döndür
    return df['title'].iloc[movie_indices]

# ====================================================================
# BÖLÜM 5: Test Etme ve Kullanıcı Girdisi
# ====================================================================

print("\n--- Film Öneri Sistemi Hazır ---")
print("Örnek filmler: Toy Story (1995), Forrest Gump (1994), Pulp Fiction (1994)")

while True:
    film_adi = input("\nLütfen öneri almak istediğiniz filmin adını girin (Çıkmak için 'q'): ")

    if film_adi.lower() == 'q':
        break
    
    # Filmin tam başlığını bulmak için bir deneme yapalım (kullanıcı küçük/büyük harf hatası yapabilir)
    # Veri setindeki başlığı birebir eşleştirmek zor olduğu için yaklaşık eşleşme arayalım
    
    # Girdiyi kullanarak veri setindeki tam başlığı bulmaya çalışıyoruz
    try:
        # Veri setinde tam eşleşen veya en yakın eşleşen başlığı bulalım (örnek: 'Toy Story' girilirse 'Toy Story (1995)' bulunsun)
        full_title = df['title'][df['title'].str.contains(film_adi, case=False, na=False)].iloc[0]
    except IndexError:
        print(f"HATA: '{film_adi}' filmi veri setinde bulunamadı. Lütfen kontrol edin.")
        continue


    # Öneri fonksiyonunu çağır ve sonuçları yazdır
    print(f"\n>>>> '{full_title}' filmine benzer öneriler:")
    oneriler = get_recommendations(full_title)
    
    if len(oneriler) > 0:
        for i, film in enumerate(oneriler, 1):
            print(f"{i}. {film}")
