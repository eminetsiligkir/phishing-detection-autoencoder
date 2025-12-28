# --- PROJE: SAF DERİN ÖĞRENME (MLP) İLE OLTALAMA TESPİTİ ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Kütüphaneler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Tekrarlanabilirlik için
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. VERİ YÜKLEME ---
print(">> Veri yükleniyor...")
file_name = r"C:\Users\emine\OneDrive\Masaüstü\YL- 2025 GUZ\Proje - Yapay sinir\Phishing_Websites_Data (1).csv"

try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print("HATA: Dosya bulunamadı! Dosya yolunu veya adını kontrol et.")
    exit()

# Etiket Düzeltme (-1: Phishing -> 1, 1: Legitimate -> 0)
df['Result'] = df['Result'].apply(lambda x: 1 if x == -1 else 0)

X = df.drop('Result', axis=1).values
y = df['Result'].values

# --- 2. ÖN İŞLEME ---
# Normalizasyon (MLP için çok önemlidir)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim/Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

results = []

# --- 3. MODEL A: OPTİMİZE EDİLMİŞ MLP (DERİN ÖĞRENME) ---
print("\n>> Derin Öğrenme (MLP) Modeli Eğitiliyor...")

# Model Mimarisi (Autoencoder yok, doğrudan derin ağ)
model = Sequential([
    Input(shape=(30,)), # 30 Giriş Özelliği
    
    # 1. Gizli Katman (Geniş tutuldu)
    Dense(128, activation='relu'), 
    Dropout(0.3), # %30 nöron kapatma (Overfitting önlemi)
    
    # 2. Gizli Katman
    Dense(64, activation='relu'),
    Dropout(0.3),
    
    # 3. Gizli Katman
    Dense(32, activation='relu'),
    
    # Çıkış Katmanı
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks (Modeli en iyi yerde durdurur ve kaydeder)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

start_time = time.time()
history = model.fit(
    X_train, y_train, 
    epochs=150, # Epoch sayısını artırdık
    batch_size=32, 
    validation_split=0.1, 
    callbacks=[early_stop],
    verbose=0 # Ekrana kalabalık yazı basmasın
)
mlp_time = time.time() - start_time

# Tahminler
y_pred_mlp = (model.predict(X_test) > 0.5).astype("int32")
y_prob_mlp = model.predict(X_test)

# Sonuçları Kaydet
acc_mlp = accuracy_score(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp)
auc_mlp = roc_auc_score(y_test, y_prob_mlp)

results.append({"Model": "Deep MLP (Optimized)", "Accuracy": acc_mlp, "F1-Score": f1_mlp, "AUC": auc_mlp, "Time (s)": mlp_time})
print(f"MLP Tamamlandı. Accuracy: %{acc_mlp*100:.2f}")


# --- 4. MODEL B: RANDOM FOREST (BENCHMARK) ---
print("\n>> Random Forest (Kıyaslama) Eğitiliyor...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

start_time = time.time()
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
results.append({"Model": "Random Forest", "Accuracy": acc_rf, "F1-Score": f1_score(y_test, y_pred_rf), "AUC": roc_auc_score(y_test, y_prob_rf), "Time (s)": rf_time})
print(f"Random Forest Tamamlandı. Accuracy: %{acc_rf*100:.2f}")


# --- 5. RAPORLAMA ---
print("\n--- KARŞILAŞTIRMA TABLOSU ---")
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print(results_df)

# Grafik 1: Doğruluk Kıyaslaması
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=results_df, palette="viridis")
plt.title("Model Doğruluk (Accuracy) Karşılaştırması")
plt.ylim(0.90, 1.0)
plt.show()

# Grafik 2: Eğitim Kayıp Grafiği (Loss) - MLP Nasıl Öğrendi?
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Eğitim Kaybı (Train Loss)')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Val Loss)')
plt.title("Derin Öğrenme Modeli - Eğitim Süreci")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Grafik 3: Confusion Matrix (MLP İçin)
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_mlp)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
plt.title("MLP Model - Confusion Matrix")
plt.show()