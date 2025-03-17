from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import multiprocessing  # Use multiprocessing instead of threading

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Define gesture mapping
GESTURES = {
    "Hello": [1, 1, 1, 1, 1],
    "Good Morning": [1, 1, 0, 0, 0],
    "How are you?": [1, 1, 1, 0, 0],
    "I am fine": [1, 1, 1, 1, 0],
    "Thank You": [0, 1, 1, 0, 0],
    "OK": [1, 0, 0, 0, 0],
    "See you soon": [1, 0, 1, 1, 1],
    "Bye": [1, 0, 0, 0, 1],
    "Speak": [0, 0, 0, 0, 0],  # Fist (Trigger speech)
}

sentence = []
last_gesture = ""

def speak(text):
    """Use multiprocessing to speak the detected sentence without blocking Flask."""
    p = multiprocessing.Process(target=speak_thread, args=(text,))
    p.daemon = True  # ✅ Prevents blocking the main thread
    p.start()

def speak_thread(text):
    """Function to handle text-to-speech in a separate process."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def recognize_gesture(fingers):
    """Match detected fingers to predefined gestures."""
    for gesture, pattern in GESTURES.items():
        if fingers == pattern:
            return gesture
    return ""

def generate_frames():
    """Capture video frames and process hand gestures."""
    global last_gesture, sentence
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        detected_text = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                fingers = []
                tips = [4, 8, 12, 16, 20]

                for i, tip in enumerate(tips):
                    if i == 0:
                        fingers.append(int(landmarks[tip][0] > landmarks[tip - 1][0]))
                    else:
                        fingers.append(int(landmarks[tip][1] < landmarks[tip - 2][1]))

                detected_text = recognize_gesture(fingers)

                if detected_text and detected_text != last_gesture:
                    if detected_text == "Speak":
                        full_sentence = " ".join(sentence)
                        if full_sentence:
                            speak(full_sentence)  # ✅ Speech works without freezing screen!
                            sentence = []  # Clear sentence after speaking
                    else:
                        sentence.append(detected_text)

                    last_gesture = detected_text

        full_sentence = " ".join(sentence)
        cv2.putText(frame, full_sentence, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
