import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from gtts import gTTS
import base64

st.set_page_config(page_title="SignBridge: Open Source Edition")

# --- MAPPING ---
# MNIST Label (0-24) to Letter, then to Hospital Word
# J (9) and Z (25) are excluded in MNIST
alphabet = "ABCDEFGHIKLMNOPQRSTUVWXY" # J and Z skipped by dataset structure usually
label_map = {i: letter for i, letter in enumerate(alphabet)}

# Custom Hackathon Mapping (Letter -> Word)
word_map = {
    'H': 'Hello',
    'Y': 'Yes',
    'N': 'No',
    'D': 'Doctor',
    'P': 'Pain',
    'L': 'Later',
    'T': 'Thanks'
}

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        with open('model.p', 'rb') as f:
            return pickle.load(f)
    except:
        return None

model = load_model()

if not model:
    st.error("⚠️ Model not found! Run 'python train_model.py' first.")
    st.stop()

# --- SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

st.title("🏥 SignBridge: Open Source AI")
st.markdown("Translating **Sign Language MNIST** (Open Source Data) into Medical Commands.")

col1, col2 = st.columns([1, 1])

with col1:
    st.info("📷 Input: Web Browser Camera")
    img_file_buffer = st.camera_input("Take a Snapshot of your Sign")

detected_word = ""

if img_file_buffer is not None:
    # 1. Read Image
    bytes_data = img_file_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # 2. Find Hand with MediaPipe (To get the bounding box)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates
            h, w, c = frame.shape
            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            
            min_x, max_x = int(min(x_vals) * w), int(max(x_vals) * w)
            min_y, max_y = int(min(y_vals) * h), int(max(y_vals) * h)
            
            # Add padding
            padding = 20
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(w, max_x + padding)
            max_y = min(h, max_y + padding)
            
            # 3. Crop the Hand
            hand_crop = frame[min_y:max_y, min_x:max_x]
            
            if hand_crop.size > 0:
                # 4. Preprocess for MNIST Model (Gray -> Resize 28x28 -> Flatten)
                gray_hand = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                resized_hand = cv2.resize(gray_hand, (28, 28))
                
                # Visualize what the AI sees
                st.image(resized_hand, caption="AI Input (Processed)", width=100)
                
                flat_hand = resized_hand.flatten().reshape(1, -1)
                
                # 5. Predict
                prediction = model.predict(flat_hand)
                predicted_idx = prediction[0]
                
                # Handle label mapping safely
                if predicted_idx < len(alphabet):
                    letter = alphabet[predicted_idx]
                    detected_word = word_map.get(letter, f"Letter {letter}")
                else:
                    detected_word = "Unknown"
    else:
        st.warning("No hand detected in the frame.")

with col2:
    st.subheader("prediction")
    if detected_word:
        st.success(f"**{detected_word}**")
        
        # Text to Speech in Browser (Cloud Compatible)
        try:
            tts = gTTS(text=detected_word, lang='en')
            tts.save("temp_audio.mp3")
            st.audio("temp_audio.mp3", format="audio/mp3", autoplay=True)
        except:
            st.error("Audio generation failed")
    else:
        st.write("Waiting for snapshot...")

st.markdown("---")
st.markdown("### 🗺️ Cheat Sheet (Sign -> Meaning)")
st.code("""
Sign 'H' -> Hello
Sign 'Y' -> Yes
Sign 'N' -> No
Sign 'D' -> Doctor
Sign 'P' -> Pain
""")