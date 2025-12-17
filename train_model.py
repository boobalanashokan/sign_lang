import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = './data'

data = []
labels = []

# Load data
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_path = os.path.join(DATA_DIR, dir_, img_path)
        with open(data_path, 'rb') as f:
            data_aux = pickle.load(f)
        
        # Ensure data consistency (sometimes mediapipe fails, filter those out)
        if len(data_aux) == 42: # 21 points * 2 coordinates (x, y)
            data.append(data_aux)
            labels.append(int(dir_))

data = np.asarray(data)
labels = np.asarray(labels)

# Split Data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize Model
model = RandomForestClassifier()

# Train
model.fit(x_train, y_train)

# Test
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100}% of samples were classified correctly !')

# Save the model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()