import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import speech_recognition as sr
import threading
import time

# --- CONFIGURATION & SETUP ---
st.set_page_config(layout="wide", page_title="SignBridge: Medical Access")

# Load Model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except:
    st.error("Model not found! Please run train_model.py first.")
    st.stop()

# Maps (Ensure these match your training data folders 0, 1, 2, 3, 4)
labels_dict = {0: 'Hello', 1: 'Yes', 2: 'No', 3: 'Doctor', 4: 'Pain'}

# Initialize components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Text to Speech Engine
engine = pyttsx3.init()

# Speech to Text Recognizer
recognizer = sr.Recognizer()

# Session State for storing conversation
if 'sign_sentence' not in st.session_state:
    st.session_state['sign_sentence'] = []
if 'last_pred' not in st.session_state:
    st.session_state['last_pred'] = ""
if 'hearing_response' not in st.session_state:
    st.session_state['hearing_response'] = "Waiting for doctor to speak..."

# --- FUNCTIONS ---

def speak_text(text):
    """Runs TTS in a thread to avoid blocking UI"""
    def _speak():
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=_speak).start()

def listen_to_speech():
    """Listens to the microphone and converts to text"""
    with sr.Microphone() as source:
        st.toast("Listening...", icon="🎤")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.session_state['hearing_response'] = text
        except sr.UnknownValueError:
            st.session_state['hearing_response'] = "Could not understand audio."
        except sr.RequestError:
            st.session_state['hearing_response'] = "API unavailable."
        except:
            st.session_state['hearing_response'] = "No audio detected."

# --- UI LAYOUT ---
st.title("🏥 SignBridge: Hospital Communication Interface")
st.markdown("Bridging the gap between Deaf patients and Medical Staff.")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Patient View (Sign Language)")
    run = st.checkbox('Start Camera', value=True)
    FRAME_WINDOW = st.image([])
    
    # Logic to capture video and predict
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Camera Error")
            break
            
        H, W, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        predicted_character = ""
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                if len(data_aux) == 42:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                    
                    # Draw on frame
                    cv2.rectangle(frame, (0, 0), (W, 60), (0, 0, 0), -1)
                    cv2.putText(frame, f"Sign: {predicted_character}", (20, 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert back to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        
        # Stability mechanism: Only update if the sign is held for a moment 
        # (Simplified for hackathon: Using a button to 'Add Word' is more reliable)
        st.session_state['last_pred'] = predicted_character
        
        # Streamlit loop delay
        time.sleep(0.05)

    cap.release()

with col2:
    st.header("Communication Log")
    
    # 1. Sign to Text
    st.subheader("🗣️ Patient Says:")
    
    # Display current prediction
    current_sign = st.session_state.get('last_pred', '...')
    st.info(f"Detected Sign: **{current_sign}**")
    
    # Button to add word to sentence
    if st.button("➕ Add Word to Sentence"):
        if current_sign:
            st.session_state['sign_sentence'].append(current_sign)
    
    # Button to Clear
    if st.button("❌ Clear Sentence"):
        st.session_state['sign_sentence'] = []

    # Display full sentence
    full_sentence = " ".join(st.session_state['sign_sentence'])
    st.success(f"Sentence: {full_sentence}")
    
    if st.button("🔊 Speak Sentence"):
        speak_text(full_sentence)

    st.divider()

    # 2. Speech to Text (Hearing Person)
    st.subheader("👂 Doctor Says:")
    if st.button("🎤 Doctor: Click to Speak"):
        listen_to_speech()
        
    st.warning(f"Transcriped Text: {st.session_state['hearing_response']}")