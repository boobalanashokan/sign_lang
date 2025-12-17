import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- 1. DOWNLOAD DATA DIRECTLY (Robust Method) ---
# We use raw CSV links from a reliable public GitHub repo
TRAIN_URL = "https://raw.githubusercontent.com/radenmuaz/Sign-Language-MNIST/master/sign_mnist_train.csv"
TEST_URL  = "https://raw.githubusercontent.com/radenmuaz/Sign-Language-MNIST/master/sign_mnist_test.csv"

print("⏳ Downloading data directly from GitHub (this may take 10-20 seconds)...")
try:
    df_train = pd.read_csv(TRAIN_URL)
    df_test = pd.read_csv(TEST_URL)
    print("✅ Data Downloaded Successfully!")
except Exception as e:
    print(f"❌ Error downloading data: {e}")
    exit()

# --- 2. PREPROCESS ---
# The CSV has 'label' as the first column, and pixel1...pixel784 as the rest
y_train = df_train['label'].values
X_train = df_train.drop('label', axis=1).values

y_test = df_test['label'].values
X_test = df_test.drop('label', axis=1).values

# Normalize pixel values (0-255 -> 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# --- 3. TRAIN MODEL ---
print("🧠 Training Random Forest Model...")
# n_jobs=-1 uses all CPU cores in your cloud instance for speed
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# --- 4. TEST ACCURACY ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Trained! Accuracy on Test Set: {acc * 100:.2f}%")

# --- 5. SAVE MODEL ---
with open('model.p', 'wb') as f:
    pickle.dump(model, f)

print("💾 Model saved to 'model.p'. You can now run 'streamlit run app.py'")