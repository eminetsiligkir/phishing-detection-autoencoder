# --- PROJE BİLGİLERİ ---
# Proje Başlığı: Oltalama Tespiti İçin Hibrit Derin Öğrenme Mimarisi ve Performans Kıyaslaması
# Öğrenci: Emine Tsiligkir - 2581053701
# Veri Seti: Phishing Websites Data Set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Makine Öğrenmesi & Ön İşleme
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest

# Derin Öğrenme (TensorFlow / Keras)
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Rastgeleliği sabitle (Tekrarlanabilirlik için)
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. VERİ YÜKLEME VE ÖN İŞLEME ---
print(">> [BÖLÜM 1] Veri Yükleniyor ve Hazırlanıyor...")

# Dosya adı (Yüklediğiniz dosya ile aynı olmalı)
file_name = r"C:\Users\emine\OneDrive\Masaüstü\YL- 2025 GUZ\Proje - Yapay sinir\Phishing_Websites_Data (1).csv"
try:
    df = pd.read_csv(file_name)
    print(f"Veri Seti Boyutu: {df.shape}")
except FileNotFoundError:
    print("HATA: CSV dosyası bulunamadı. Lütfen dosya adını kontrol edin.")
    # Örnek veri ile devam etmemesi için burada durdurulabilir, ancak kodun akışı için devam ediyoruz.

# Etiket Düzenleme (Sınıf Etiketleri: -1 Phishing, 1 Legitimate)
# Hedef: Phishing (Saldırı) yakalamak olduğu için Phishing'i 1, Legitimate'i 0 yapıyoruz.
# -1 -> 1 (Phishing/Pozitif Sınıf)
#  1 -> 0 (Legitimate/Negatif Sınıf)
df['Result'] = df['Result'].apply(lambda x: 1 if x == -1 else 0)

X = df.drop('Result', axis=1).values
y = df['Result'].values

# Normalizasyon (MinMaxScaler: 0-1 aralığı - Autoencoder için kritik)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve Test Ayrımı (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Eğitim Verisi: {X_train.shape}, Test Verisi: {X_test.shape}")
results = [] # Sonuçları burada toplayacağız

# --- 2. EK ANALİZLER (KÜNYEDEKİ "ÖN İŞLEME" MADDELERİ) ---
print("\n>> [BÖLÜM 2] PCA ve Information Gain Analizi Yapılıyor...")

# A) PCA Analizi (Kıyaslama Amaçlı)
pca = PCA(n_components=0.95) # Varyansın %95'ini koruyan bileşen sayısı
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA sonrası öznitelik sayısı (30'dan düşürüldü): {X_train_pca.shape[1]}")

# B) Information Gain (Mutual Information)
# En iyi 15 özelliği seçelim
selector = SelectKBest(mutual_info_classif, k=15)
X_train_ig = selector.fit_transform(X_train, y_train)
X_test_ig = selector.transform(X_test)
print(f"Information Gain ile seçilen öznitelik sayısı: {X_train_ig.shape[1]}")


# --- 3. YENİLİKÇİ YAKLAŞIM: STACKED AUTOENCODER (AŞAMA 1) ---
print("\n>> [BÖLÜM 3] Autoencoder (Öznitelik Çıkarımı) Eğitiliyor...")
# Amaç: Gürültüyü temizlemek ve Latent Space (Gizli Uzay) temsili öğrenmek

input_dim = X_train.shape[1] # 30

# Encoder (Sıkıştırma)
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
bottleneck = Dense(16, activation='relu', name='bottleneck')(encoded) # Latent Feature (16 boyut)

# Decoder (Geri Oluşturma)
decoded = Dense(32, activation='relu')(bottleneck)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded) # Girdi boyutuna (30) geri dön

# Model Derleme
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Eğitimi Başlat (Unsupervised: X -> X)
start_time = time.time()
autoencoder.fit(X_train, X_train, 
                epochs=50, 
                batch_size=64, 
                validation_split=0.1, 
                verbose=0)
ae_time = time.time() - start_time
print(f"Autoencoder eğitimi tamamlandı ({ae_time:.2f} sn).")

# Encoder'ı ayır (Sadece sıkıştırma yapan kısım)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)

# Verileri "Latent Space"e dönüştür (Feature Extraction)
X_train_encoded = encoder_model.predict(X_train)
X_test_encoded = encoder_model.predict(X_test)


# --- 4. HİBRİT MODEL (AŞAMA 2) ve STANDART MODEL ---
print("\n>> [BÖLÜM 4] Derin Öğrenme Modelleri Eğitiliyor...")

# Fonksiyon: MLP Modeli Oluşturucu
def build_mlp(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dropout(0.2), # Künyede belirtilen Dropout
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# A) HİBRİT MODEL (Autoencoder Çıktısı ile Çalışan)
hybrid_model = build_mlp(16) # Giriş boyutu 16 (Encoded)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

start_time = time.time()
hybrid_model.fit(X_train_encoded, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
hybrid_train_time = time.time() - start_time

# Tahmin
y_pred_hybrid = (hybrid_model.predict(X_test_encoded) > 0.5).astype("int32")
results.append({
    "Model": "Hybrid DL (Autoencoder+MLP)",
    "Accuracy": accuracy_score(y_test, y_pred_hybrid),
    "F1-Score": f1_score(y_test, y_pred_hybrid),
    "AUC": roc_auc_score(y_test, hybrid_model.predict(X_test_encoded)),
    "Time (s)": ae_time + hybrid_train_time
})

# B) STANDART MLP (Ham Veri ile Çalışan - Ablation Study için)
base_model = build_mlp(30) # Giriş boyutu 30 (Ham)
start_time = time.time()
base_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
base_time = time.time() - start_time

y_pred_base = (base_model.predict(X_test) > 0.5).astype("int32")
results.append({
    "Model": "Standard MLP (Baseline)",
    "Accuracy": accuracy_score(y_test, y_pred_base),
    "F1-Score": f1_score(y_test, y_pred_base),
    "AUC": roc_auc_score(y_test, base_model.predict(X_test)),
    "Time (s)": base_time
})


# --- 5. KLASİK MODELLER İLE KARŞILAŞTIRMA ---
print("\n>> [BÖLÜM 5] Klasik Makine Öğrenmesi Modelleri Kıyaslanıyor...")

ml_models = {
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

for name, model in ml_models.items():
    start_time = time.time()
    model.fit(X_train, y_train) # Ham veride eğitim
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    # AUC hesaplama
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
    else:
        prob = model.decision_function(X_test)
        
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, prob),
        "Time (s)": train_time
    })
    print(f"{name} tamamlandı.")


# --- 6. SONUÇLAR VE GÖRSELLEŞTİRME ---
print("\n>> [BÖLÜM 6] Sonuçlar Raporlanıyor...")

# Tablo
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\n--- PERFORMANS TABLOSU ---")
print(results_df.to_string(index=False))

# Grafik 1: Doğruluk Karşılaştırması
plt.figure(figsize=(10, 6))
sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis")
plt.title("Model Doğruluk (Accuracy) Karşılaştırması")
plt.xlim(0.85, 1.0)
plt.xlabel("Doğruluk Skoru")
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.show()

# Grafik 2: Hibrit Model Karmaşıklık Matrisi
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_hybrid)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Legitimate (0)", "Phishing (1)"], 
            yticklabels=["Legitimate (0)", "Phishing (1)"])
plt.title("Hibrit Model (Autoencoder+MLP) - Confusion Matrix")
plt.ylabel("Gerçek Sınıf")
plt.xlabel("Tahmin Edilen Sınıf")
plt.show()

# Grafik 3: ROC Eğrileri
plt.figure(figsize=(10, 8))

# Hibrit Model ROC
fpr_h, tpr_h, _ = roc_curve(y_test, hybrid_model.predict(X_test_encoded))
plt.plot(fpr_h, tpr_h, label=f"Hybrid DL (AUC={results_df[results_df['Model']=='Hybrid DL (Autoencoder+MLP)']['AUC'].values[0]:.3f})", linewidth=2)

# Random Forest ROC (Kıyaslama için en güçlü rakip)
rf_model = ml_models["Random Forest"]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={results_df[results_df['Model']=='Random Forest']['AUC'].values[0]:.3f})", linestyle='--')

plt.plot([0, 1], [0, 1], 'k--', label="Rastgele Tahmin")
plt.xlabel("False Positive Rate (Yanlış Alarm)")
plt.ylabel("True Positive Rate (Duyarlılık)")
plt.title("ROC Analizi: Hibrit Model vs. Random Forest")
plt.legend()
plt.show()