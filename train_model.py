import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("⏳ Downloading Sign Language MNIST from Hugging Face...")
# Load dataset from Hugging Face (Open Source)
dataset = load_dataset("nielsr/sign-language-mnist", split="train")

print("✅ Download Complete. Processing...")

# Convert to Pandas DataFrame for easier handling
df = pd.DataFrame(dataset)

# The dataset has 'image' (PIL Image) and 'label' (Int)
# We need to flatten the 28x28 images into 1D arrays (784 pixels)
X = []
y = []

for i in range(len(df)):
    # Get image as numpy array (28x28)
    img_array = np.array(df['image'][i])
    # Flatten to 1D array
    X.append(img_array.flatten())
    y.append(df['label'][i])

X = np.array(X)
y = np.array(y)

# Train Random Forest
print("🧠 Training Model (This might take a minute)...")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
model.fit(X, y)

# Accuracy Check (Optional, using a subset of training for speed)
y_pred = model.predict(X[:1000])
acc = accuracy_score(y[:1000], y_pred)
print(f"✅ Model Trained! Training Accuracy: {acc * 100:.2f}%")

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump(model, f)

print("💾 Model saved to 'model.p'")