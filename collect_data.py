import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Directory to save data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the classes (signs) you want to collect
# Start small for the hackathon: 3 classes
classes = ['Hello', 'Yes', 'No']
dataset_size = 100 # Number of samples per class

for j, class_name in enumerate(classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    cap = cv2.VideoCapture(0)
    print(f'Collecting data for class: {class_name}')

    # Wait for user to get ready
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks (x, y) - 21 points per hand
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize the data (relative to min x and y)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Save raw data
                save_path = os.path.join(DATA_DIR, str(j), f'{counter}.pickle')
                with open(save_path, 'wb') as f:
                    pickle.dump(data_aux, f)
                
                counter += 1
                
        cv2.putText(frame, f'Collecting {class_name}: {counter}/{dataset_size}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()