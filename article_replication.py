# =============================================================================
# MAKALE DOĞRULAMA ÇALIŞMASI (REPLICATION STUDY)
# Referans Makale: Koşand, Yıldız, Karacan (2018)
# Amaç: Makaledeki deney ortamını Python ile simüle edip sonuçları kıyaslamak.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Scikit-learn Kütüphaneleri
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Makalede Kullanılan Algoritmalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Rastgeleliği sabitle (Makaledeki gibi kararlı sonuçlar için)
np.random.seed(42)

# --- 1. VERİ YÜKLEME ---
print(">> Veri Seti Yükleniyor...")
file_name = r"C:\Users\emine\OneDrive\Masaüstü\YL- 2025 GUZ\Proje - Yapay sinir\Phishing_Websites_Data (1).csv"

try:
    df = pd.read_csv(file_name)
    print(f"   Veri Boyutu: {df.shape}")
except FileNotFoundError:
    print("HATA: Dosya bulunamadı.")
    exit()

# Etiket Düzeltme (Makaledeki gibi -1 ve 1 olarak bırakabiliriz ama sklearn 0-1 sever)
# Phishing (-1) -> 1 (Pozitif Sınıf)
# Legitimate (1) -> 0 (Negatif Sınıf)
df['Result'] = df['Result'].apply(lambda x: 1 if x == -1 else 0)

X = df.drop('Result', axis=1).values
y = df['Result'].values

# --- 2. MAKALEDEKİ ALGORİTMALARIN HAZIRLANMASI ---
# Makalede belirtilen parametrelere sadık kalmaya çalışıyoruz.

models = {
    # 1. Random Forest (Makalede: 100 Ağaç)
    "Random Forest (RF)": RandomForestClassifier(n_estimators=100, random_state=42),
    
    # 2. KNN (Makalede: k=3 olarak belirtilmiş)
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    
    # 3. Naive Bayes (Makalede en düşük sonucu vermişti)
    "Naive Bayes": GaussianNB(),
    
    # 4. C4.5 / ID3 Benzeri (Karar Ağacı - Entropy kullanılarak)
    "Decision Tree (C4.5/ID3)": DecisionTreeClassifier(criterion='entropy', random_state=42)
}

# --- 3. DENEYSEL SÜREÇ (10-KATLI ÇAPRAZ DOĞRULAMA) ---
# Makalede "10-Fold Cross Validation" kullanıldığı yazıyor. Biz de aynısını yapıyoruz.
print("\n>> 10-Katlı Çapraz Doğrulama (10-Fold CV) Başlatılıyor...\n")

results_data = []

# Metrikler
scoring = {'accuracy': 'accuracy', 'f1': 'f1', 'auc': 'roc_auc'}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    start_time = time.time()
    
    # Cross Validate işlemi (Eğitim + Testi 10 kere yapar ortalamasını alır)
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    elapsed_time = time.time() - start_time
    
    # Ortalamaları al
    mean_acc = np.mean(scores['test_accuracy'])
    mean_f1 = np.mean(scores['test_f1'])
    mean_auc = np.mean(scores['test_auc'])
    
    results_data.append({
        "Algoritma": name,
        "Doğruluk (Accuracy)": mean_acc,
        "F1-Skoru": mean_f1,
        "ROC Alanı (AUC)": mean_auc,
        "Süre (sn)": elapsed_time
    })
    
    print(f"   --> {name} tamamlandı. Ortalama Doğruluk: %{mean_acc*100:.2f}")

# --- 4. SONUÇLARIN KARŞILAŞTIRILMASI ---
print("\n" + "="*60)
print("   MAKALEDEKİ SONUÇLARLA KIYASLAMA TABLOSU")
print("="*60)

df_results = pd.DataFrame(results_data).sort_values(by="Doğruluk (Accuracy)", ascending=False)
print(df_results.to_string(index=False))

# --- 5. GÖRSELLEŞTİRME ---
plt.figure(figsize=(10, 6))
sns.barplot(x="Doğruluk (Accuracy)", y="Algoritma", data=df_results, palette="magma")
plt.xlim(0.90, 1.0)
plt.title("Makale Replikasyonu: Algoritma Performansları")
plt.axvline(x=0.973, color='r', linestyle='--', label="Makaledeki RF Skoru (%97.3)")
plt.legend()
plt.tight_layout()
plt.show()

print("\nNOT: Kırmızı kesikli çizgi, makaledeki (Koşand, 2018) en yüksek skoru temsil eder.")