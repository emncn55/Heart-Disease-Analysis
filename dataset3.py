import pandas as pd

# 1. Veri Setini Yükleme
try:
    df = pd.read_csv('heart.csv')
    print("Veri seti başarıyla yüklendi. Boyut:", df.shape)
except FileNotFoundError:
    print("Hata: heart.csv dosyası bulunamadı!")
    exit()

# 2. Veri Önizleme
print("\nVeri setinin ilk 5 satırı:")
print(df.head())

print("\nVeri tipleri ve eksik değerler:")
print(df.info())

# 3. Eksik Veri Kontrolü (Daha doğru bir yaklaşım)
missing_values = df.isnull().sum()
print("\nEksik veri analizi:")
print(missing_values[missing_values > 0])  # Sadece eksik olanları göster

# Eksik veri yoksa işlem yapma
if missing_values.sum() == 0:
    print("\nVeri setinde eksik değer bulunmamaktadır.")
else:
    # Sayısal sütunlar için medyan ile doldur
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Kategorik sütunlar için mod ile doldur (eğer varsa)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    print("\nEksik veriler dolduruldu. Son durum:")
    print(df.isnull().sum())

# 4. Temel İstatistikler (Daha kapsamlı)
print("\nSayısal değişkenlerin istatistikleri:")
print(df.describe().T)  # Transpoz alarak daha okunabilir hale getir

# Kategorik değişken analizi (eğer varsa)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
if not categorical_cols.empty:
    print("\nKategorik değişken istatistikleri:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True) * 100)

# 5. Hedef Değişken Analizi
if 'target' in df.columns:
    print("\nHedef değişken dağılımı:")
    target_dist = df['target'].value_counts(normalize=True) * 100
    print(target_dist)
    
    # Görselleştirme için (opsiyonel)
    import matplotlib.pyplot as plt
    target_dist.plot(kind='bar')
    plt.title('Hedef Değişken Dağılımı')
    plt.ylabel('Yüzde')
    plt.show()

# 6. Korelasyon Analizi (Sadece sayısal sütunlar için)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_cols) > 1:
    print("\nKorelasyon matrisi:")
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix)
    
    # Isı haritası (opsiyonel)
    import seaborn as sns
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Korelasyon Isı Haritası')
    plt.show()

# 7. Yaş ve Cinsiyet Analizi (Örnek spesifik analizler)
if 'age' in df.columns:
    print("\nYaş istatistikleri:")
    print(df['age'].describe())
    
    # Yaş grupları
    age_bins = [20, 30, 40, 50, 60, 70, 80]
    age_labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    print("\nYaş gruplarına göre dağılım:")
    print(df['age_group'].value_counts().sort_index())

if 'sex' in df.columns:
    print("\nCinsiyet dağılımı:")
    # Veri setine göre 1: erkek, 0: kadın olabilir, kontrol edelim
    if set(df['sex'].unique()) == {0, 1}:
        sex_mapping = {1: 'Erkek', 0: 'Kadın'}
        sex_dist = df['sex'].replace(sex_mapping).value_counts(normalize=True) * 100
    else:
        sex_dist = df['sex'].value_counts(normalize=True) * 100
    print(sex_dist)

# 8. Diğer Önemli Sütunların Analizi
important_cols = ['cp', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in important_cols:
    if col in df.columns:
        print(f"\n{col} istatistikleri:")
        print(df[col].describe())