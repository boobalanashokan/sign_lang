# Sign Language Recognition & Translation Tool 🤟🗣️

Bridging the communication gap for the deaf and hard-of-hearing community using Computer Vision and AI.

## 📌 Problem Statement
The deaf community often faces challenges communicating with hearing individuals who do not understand Sign Language. This project provides a real-time translator that converts hand gestures into text and speech.

## 🚀 Features
- **Real-time Detection:** Uses MediaPipe to track hand landmarks with high precision.
- **Machine Learning Classification:** Random Forest classifier for identifying signs.
- **Text-to-Speech:** Converts recognized gestures into spoken audio instantly.
- **Lightweight:** Runs on CPU, no heavy GPU required.

## 🛠️ Tech Stack
- **Language:** Python
- **CV:** OpenCV, MediaPipe
- **ML:** Scikit-Learn
- **Audio:** Pyttsx3

## ⚙️ How to Run
1. **Install Dependencies:**
   `pip install -r requirements.txt`

2. **Collect Data (Optional - Data already included for Hello, Yes, No):**
   Run `python collect_data.py` to record your own custom signs.

3. **Train Model:**
   Run `python train_model.py` to generate the classifier model.

4. **Run Application:**
   Run `python inference.py`
   - Show your hand to the camera.
   - Press **'S'** to hear the translation.
   - Press **'Q'** to quit.

## 🔮 Future Scope
- Integration with mobile apps (Android/iOS).
- Support for Sentence formation (LSTM/RNN).
- Two-way communication (Speech-to-Sign animation).