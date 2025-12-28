import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# 1. Veriyi Yükle ve Hazırla
file_name = r"C:\Users\emine\OneDrive\Masaüstü\YL- 2025 GUZ\Proje - Yapay sinir\Phishing_Websites_Data (1).csv"
df = pd.read_csv(file_name)

# -1 ve 1 etiketlerini 0 ve 1 yapalım (Derin öğrenme formatı ile uyumlu olsun)
df['Result'] = df['Result'].apply(lambda x: 1 if x == -1 else 0)

X = df.drop('Result', axis=1).values
y = df['Result'].values

# Veriyi 0-1 arasına sıkıştır (Naive Bayes negatif sayı sevmez)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim/Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\n=== NAIVE BAYES TÜR KARŞILAŞTIRMASI ===")

# 1. GAUSSIAN NB (Eski Denediğimiz - Düşük Çıkması Beklenen)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
acc_gnb = accuracy_score(y_test, gnb.predict(X_test))
print(f"Gaussian NB (Normal Dağılım Varsayımı) Başarısı:  %{acc_gnb*100:.2f}")

# 2. MULTINOMIAL NB (Kategorik Veriye Daha Uygun)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
acc_mnb = accuracy_score(y_test, mnb.predict(X_test))
print(f"Multinomial NB (Kategorik Yaklaşım) Başarısı:     %{acc_mnb*100:.2f}")

# 3. BERNOULLI NB (İkili 0/1 Veriye En Uygun)
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
acc_bnb = accuracy_score(y_test, bnb.predict(X_test))
print(f"Bernoulli NB (Binary Yaklaşım) Başarısı:          %{acc_bnb*100:.2f}")