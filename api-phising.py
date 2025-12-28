# =============================================================================
# PROJE: Siber Güvenlikte Hibrit Derin Öğrenme Yaklaşımı (Phishing Detection)
# YAZAR: Emine Tsiligkir
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib  # Modeli kaydetmek için

# Makine Öğrenmesi Kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

# Derin Öğrenme Kütüphaneleri (TensorFlow / Keras)
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Tekrarlanabilirlik için rastgelelikleri sabitle
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. VERİ YÜKLEME VE ÖN İŞLEME ---
print("\n>> [ADIM 1] Veri Seti Yükleniyor...")

file_name = r"C:\Users\emine\OneDrive\Masaüstü\YL- 2025 GUZ\Proje - Yapay sinir\Phishing_Websites_Data (1).csv"

try:
    df = pd.read_csv(file_name)
    print(f"   Veri başarıyla yüklendi. Boyut: {df.shape}")
except FileNotFoundError:
    print(f"HATA: '{file_name}' dosyası bulunamadı! Lütfen dosya adını kontrol edin.")
    exit()

# Etiket Düzeltme: -1 (Phishing) -> 1, 1 (Legitimate) -> 0
# Amacımız saldırıyı (1) tespit etmek.
df['Result'] = df['Result'].apply(lambda x: 1 if x == -1 else 0)

X = df.drop('Result', axis=1).values
y = df['Result'].values

# Normalizasyon (0-1 Aralığına Çekme) - Derin Öğrenme için Kritik
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve Test Ayrımı (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"   Eğitim Verisi: {X_train.shape}, Test Verisi: {X_test.shape}")

results = [] # Sonuçları burada toplayacağız

# =============================================================================
# --- 2. MODEL: RANDOM FOREST (REFERANS NOKTASI) ---
# =============================================================================
print("\n>> [ADIM 2] Random Forest Modeli Eğitiliyor...")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

start_time = time.time()
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

# Tahminler
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Performans Kaydı
acc_rf = accuracy_score(y_test, y_pred_rf)
results.append({
    "Model": "Random Forest",
    "Accuracy": acc_rf,
    "F1-Score": f1_score(y_test, y_pred_rf),
    "AUC": roc_auc_score(y_test, y_prob_rf),
    "Time (s)": rf_time
})
print(f"   Random Forest Başarısı: %{acc_rf*100:.2f}")

# Gelecek Çalışmalar (API) için en iyi modeli kaydet
joblib.dump(rf_model, 'phishing_rf_model.pkl')


# =============================================================================
# --- 3. MODEL: OPTİMİZE EDİLMİŞ SAF MLP (YÜKSEK PERFORMANS) ---
# =============================================================================
print("\n>> [ADIM 3] Optimize Edilmiş Derin MLP Eğitiliyor...")

mlp_model = Sequential([
    Input(shape=(30,)), # 30 Giriş Özelliği
    Dense(128, activation='relu'), 
    Dropout(0.3), # Aşırı öğrenmeyi önlemek için
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Çıktı: 0 ile 1 arası olasılık
])

mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

start_time = time.time()
history_mlp = mlp_model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.1, 
    callbacks=[early_stop],
    verbose=0 # Ekrana çok yazı basmasın
)
mlp_time = time.time() - start_time

# Tahminler
y_prob_mlp = mlp_model.predict(X_test)
y_pred_mlp = (y_prob_mlp > 0.5).astype("int32")

# Performans Kaydı
acc_mlp = accuracy_score(y_test, y_pred_mlp)
results.append({
    "Model": "Deep MLP (Optimized)",
    "Accuracy": acc_mlp,
    "F1-Score": f1_score(y_test, y_pred_mlp),
    "AUC": roc_auc_score(y_test, y_prob_mlp),
    "Time (s)": mlp_time
})
print(f"   Derin MLP Başarısı: %{acc_mlp*100:.2f}")


# =============================================================================
# --- 4. MODEL: AUTOENCODER HİBRİT MİMARİ (YENİLİKÇİ YAKLAŞIM) ---
# =============================================================================
print("\n>> [ADIM 4] Autoencoder (Otokodlayıcı) Hibrit Model Eğitiliyor...")

# A) Autoencoder Eğitimi (Gürültü Temizleme & Özellik Çıkarımı)
input_dim = 30
encoding_dim = 16 # Veriyi 30'dan 16 boyuta sıkıştıracağız

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
bottleneck = Dense(encoding_dim, activation='relu', name='bottleneck')(encoded) # Latent Space

# Decoder
decoded = Dense(32, activation='relu')(bottleneck)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

start_time = time.time()
autoencoder.fit(X_train, X_train, epochs=40, batch_size=64, validation_split=0.1, verbose=0)
ae_train_time = time.time() - start_time

# Encoder'ı ayırıp özellik çıkarıcı olarak kullanma
encoder = Model(inputs=input_layer, outputs=bottleneck)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# B) Sınıflandırıcı Eğitimi (Sıkıştırılmış 16 özellik ile)
hybrid_classifier = Sequential([
    Input(shape=(encoding_dim,)),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

hybrid_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hybrid_classifier.fit(
    X_train_encoded, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.1, 
    callbacks=[early_stop], 
    verbose=0
)
total_hybrid_time = ae_train_time + (time.time() - start_time)

# Tahminler
y_prob_hybrid = hybrid_classifier.predict(X_test_encoded)
y_pred_hybrid = (y_prob_hybrid > 0.5).astype("int32")

# Performans Kaydı
acc_hybrid = accuracy_score(y_test, y_pred_hybrid)
results.append({
    "Model": "Hybrid Autoencoder",
    "Accuracy": acc_hybrid,
    "F1-Score": f1_score(y_test, y_pred_hybrid),
    "AUC": roc_auc_score(y_test, y_prob_hybrid),
    "Time (s)": total_hybrid_time
})
print(f"   Hibrit Model Başarısı: %{acc_hybrid*100:.2f}")


# =============================================================================
# --- 5. SONUÇLARIN KARŞILAŞTIRILMASI VE GÖRSELLEŞTİRME ---
# =============================================================================
print("\n>> [ADIM 5] Sonuçlar Raporlanıyor...")

# Tablo Oluşturma
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\n--- PERFORMANS TABLOSU ---")
print(results_df.to_string(index=False))

# Grafik 1: Doğruluk Karşılaştırması (Bar Plot)
plt.figure(figsize=(10, 6))
sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis")
plt.xlim(0.90, 1.0)
plt.title("Modellerin Doğruluk (Accuracy) Karşılaştırması")
plt.xlabel("Doğruluk Oranı")
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Grafik 2: ROC Eğrileri (Kıyaslamalı)
plt.figure(figsize=(9, 7))
# RF
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y_test, y_prob_rf):.3f})", linestyle='--')
# MLP
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
plt.plot(fpr_mlp, tpr_mlp, label=f"Deep MLP (AUC={roc_auc_score(y_test, y_prob_mlp):.3f})", linewidth=2)
# Hybrid
fpr_hyb, tpr_hyb, _ = roc_curve(y_test, y_prob_hybrid)
plt.plot(fpr_hyb, tpr_hyb, label=f"Hybrid AE (AUC={roc_auc_score(y_test, y_prob_hybrid):.3f})")

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5) # Rastgele çizgi
plt.xlabel("Yanlış Pozitif Oranı (FPR)")
plt.ylabel("Doğru Pozitif Oranı (TPR)")
plt.title("ROC Eğrisi Analizi")
plt.legend()
plt.show()

# Grafik 3: Confusion Matrix (En İyi Derin Öğrenme Modeli İçin - MLP)
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_mlp)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Güvenli", "Oltalama"], 
            yticklabels=["Güvenli", "Oltalama"])
plt.title("Saf MLP Modeli - Karmaşıklık Matrisi")
plt.ylabel("Gerçek Durum")
plt.xlabel("Tahmin Edilen")
plt.show()

print("\n>> İŞLEM TAMAMLANDI. Grafikleri kaydedebilirsiniz.")