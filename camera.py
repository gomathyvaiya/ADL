import cv2
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

IMG_SIZE = 100
THRESHOLD = 0.5  # Adjust based on performance

# Load base model
base_model = load_model('saved_models/base_model.h5')

# Load known embeddings
with open('known_embeddings/embeddings.pkl', 'rb') as f:
    known_embeddings = pickle.load(f)

def preprocess_face_from_frame(frame, box):
    x, y, w, h = box
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)

def recognize_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = preprocess_face_from_frame(frame, (x, y, w, h))
        emb = base_model.predict(face_img)[0]

        name = "Unknown"
        min_dist = float("inf")

        for known_name, known_emb in known_embeddings.items():
            dist = np.linalg.norm(emb - known_emb)
            if dist < min_dist:
                min_dist = dist
                name = known_name if dist < THRESHOLD else "Unknown"

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{name} ({min_dist:.2f})"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None

        frame = recognize_face(frame)
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
