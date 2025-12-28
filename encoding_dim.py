# =============================================================================
# PROJE: Siber Güvenlik (Latent Space = 22 Denemesi)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib

# Makine Öğrenmesi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

# Derin Öğrenme
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Sonuçların aynı çıkması için (Tekrarlanabilirlik)
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. VERİ YÜKLEME ---
print("\n>> [ADIM 1] Veri Seti Yükleniyor...")
file_name = r"C:\Users\emine\OneDrive\Masaüstü\YL- 2025 GUZ\Proje - Yapay sinir\Phishing_Websites_Data (1).csv"

try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"HATA: '{file_name}' dosyası bulunamadı!")
    exit()

# Etiket Düzeltme
df['Result'] = df['Result'].apply(lambda x: 1 if x == -1 else 0)

X = df.drop('Result', axis=1).values
y = df['Result'].values

# Normalizasyon
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim/Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
results = []

# =============================================================================
# --- 2. RANDOM FOREST (Kıyaslama İçin Tekrar Çalıştırıyoruz) ---
# =============================================================================
print("\n>> [ADIM 2] Random Forest (Referans) Çalışıyor...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

acc_rf = accuracy_score(y_test, rf_model.predict(X_test))
results.append({"Model": "Random Forest", "Accuracy": acc_rf, "Time": rf_time})
print(f"   Random Forest: %{acc_rf*100:.2f}")

# =============================================================================
# --- 3. OPTİMİZE SAF MLP (Kıyaslama İçin) ---
# =============================================================================
print("\n>> [ADIM 3] Saf MLP Modeli Çalışıyor...")
mlp_model = Sequential([
    Input(shape=(30,)),
    Dense(128, activation='relu'), Dropout(0.3),
    Dense(64, activation='relu'), Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

start_time = time.time()
mlp_model.fit(X_train, y_train, epochs=80, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
mlp_time = time.time() - start_time

acc_mlp = accuracy_score(y_test, (mlp_model.predict(X_test) > 0.5).astype("int32"))
results.append({"Model": "Saf MLP (Deep)", "Accuracy": acc_mlp, "Time": mlp_time})
print(f"   Saf MLP: %{acc_mlp*100:.2f}")

# =============================================================================
# --- 4. HİBRİT AUTOENCODER (DEĞİŞİKLİK BURADA: 22 BOYUT) ---
# =============================================================================
print("\n>> [ADIM 4] Autoencoder (Latent Space = 22) Eğitiliyor...")

input_dim = 30
encoding_dim = 22  # <--- BURAYI DEĞİŞTİRDİK (Eskiden 16 idi)

# Encoder Mimarisi
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
# encoded = Dense(32, activation='relu')(encoded) # Ara katmanı kaldırdım, doğrudan 22'ye insin
bottleneck = Dense(encoding_dim, activation='relu', name='bottleneck')(encoded)

# Decoder Mimarisi
# decoded = Dense(32, activation='relu')(bottleneck)
decoded = Dense(64, activation='relu')(bottleneck)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Autoencoder Eğitimi
start_time = time.time()
autoencoder.fit(X_train, X_train, epochs=50, batch_size=64, validation_split=0.1, verbose=0)
ae_time = time.time() - start_time

# Özellik Çıkarımı (30 -> 22)
encoder = Model(inputs=input_layer, outputs=bottleneck)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Sınıflandırıcı Eğitimi (22 Özellik Üzerinden)
hybrid_classifier = Sequential([
    Input(shape=(encoding_dim,)), # Artık 22 giriş bekliyor
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
hybrid_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hybrid_classifier.fit(X_train_encoded, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
hybrid_total_time = ae_time + (time.time() - start_time)

# Sonuç
y_pred_hybrid = (hybrid_classifier.predict(X_test_encoded) > 0.5).astype("int32")
acc_hybrid = accuracy_score(y_test, y_pred_hybrid)
results.append({"Model": "Hybrid Autoencoder (22-Dim)", "Accuracy": acc_hybrid, "Time": hybrid_total_time})
print(f"   Hibrit Model (22 Boyut): %{acc_hybrid*100:.2f}")

# =============================================================================
# --- 5. SONUÇ TABLOSU ---
# =============================================================================
print("\n" + "="*50)
print("   YENİ SONUÇLAR (LATENT DIM = 22)")
print("="*50)
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print(results_df.to_string(index=False))