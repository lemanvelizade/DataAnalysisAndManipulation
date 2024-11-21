# Gerekli Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veri Setlerinin Yüklenmesi
satis_verisi = pd.read_csv("dataset/satis_verisi_5000.csv")
musteri_verisi = pd.read_csv("dataset/musteri_verisi_5000_utf8.csv")

# Görev 1: Veri Temizleme ve Manipülasyonu
# Eksik verileri doldurma
satis_verisi['fiyat'] = pd.to_numeric(satis_verisi['fiyat'], errors='coerce')
satis_verisi['toplam_satis'] = pd.to_numeric(satis_verisi['toplam_satis'], errors='coerce')
satis_verisi['fiyat'] = satis_verisi['fiyat'].fillna(satis_verisi['fiyat'].mean())
satis_verisi['toplam_satis'] = satis_verisi['toplam_satis'].fillna(satis_verisi['toplam_satis'].mean())

# Aykırı değerleri temizleme (IQR yöntemi)
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

satis_verisi = remove_outliers(satis_verisi, 'fiyat')
musteri_verisi = remove_outliers(musteri_verisi, 'harcama_miktari')

# Veri birleştirme
merged_data = pd.merge(satis_verisi, musteri_verisi, on='musteri_id')

# Görev 2: Zaman Serisi Analizi
merged_data['tarih'] = pd.to_datetime(merged_data['tarih'])
merged_data['hafta'] = merged_data['tarih'].dt.isocalendar().week
merged_data['ay'] = merged_data['tarih'].dt.month

# Haftalık ve aylık satış trendleri
haftalik_satis = merged_data.groupby('hafta')['toplam_satis'].sum()
aylik_satis = merged_data.groupby('ay')['toplam_satis'].sum()

plt.figure(figsize=(10, 6))
plt.plot(haftalik_satis, label="Haftalık Satışlar")
plt.plot(aylik_satis, label="Aylık Satışlar")
plt.title("Haftalık ve Aylık Satış Trendleri")
plt.xlabel("Zaman")
plt.ylabel("Satış Tutarı")
plt.legend()
plt.show()

# Görev 3: Kategorisel ve Sayısal Analiz
# Kategori bazında analiz
kategori_satış = merged_data.groupby('kategori')['toplam_satis'].sum()
kategori_oran = kategori_satış / kategori_satış.sum()
print("Kategori Bazında Satış Oranları:\n", kategori_oran)

# Yaş gruplarına göre satış
merged_data['yas_grubu'] = pd.cut(merged_data['yas'], bins=[18, 25, 35, 50, 100],
                                  labels=["18-25", "26-35", "36-50", "50+"])
yas_grubu_satis = merged_data.groupby('yas_grubu')['toplam_satis'].sum()

plt.figure(figsize=(8, 4))
yas_grubu_satis.plot(kind='bar', color='skyblue')
plt.title("Yaş Gruplarına Göre Satış")
plt.xlabel("Yaş Grupları")
plt.ylabel("Satış Tutarı")
plt.show()

# Cinsiyet bazında harcama
cinsiyet_harcama = merged_data.groupby('cinsiyet')['harcama_miktari'].mean()
print("Cinsiyete Göre Ortalama Harcama:\n", cinsiyet_harcama)

# Görev 4: İleri Düzey Veri Manipülasyonu
# Şehir bazında harcama
sehir_harcama = merged_data.groupby('sehir')['harcama_miktari'].sum().sort_values(ascending=False)
print("Şehir Bazında Harcama:\n", sehir_harcama)

# Aylık kategori bazında satış trendleri
kategori_aylik_satis = merged_data.groupby(['kategori', 'ay'])['toplam_satis'].sum().unstack()
kategori_aylik_satis.plot(figsize=(10, 6))
plt.title("Aylık Kategori Bazında Satış Trendleri")
plt.xlabel("Ay")
plt.ylabel("Toplam Satış")
plt.legend(title="Kategori")
plt.show()

# Görev 5: Pareto Analizi
kategori_toplam_satis = merged_data.groupby('kategori')['toplam_satis'].sum()
kategori_pareto = kategori_toplam_satis.sort_values(ascending=False).cumsum() / kategori_toplam_satis.sum()

plt.figure(figsize=(8, 4))
kategori_pareto.plot(marker='o', color='green')
plt.axhline(y=0.8, color='red', linestyle='--')
plt.title("Pareto Analizi (80/20 Kuralı)")
plt.xlabel("Kategori")
plt.ylabel("Kümülatif Oran")
plt.show()

print("Veri analizi ve manipülasyonu tamamlandı.")
