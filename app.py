from flask import Flask, render_template, Response, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import pickle

# Import your camera module functions and variables
from camera import VideoCamera, known_embeddings, base_model, recognize_face

app = Flask(__name__)
app.secret_key = "secretkey"

# Configure folder for uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Video Camera for live streaming
camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

# Live streaming generator function
def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for registering a new face via image upload
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        image = request.files.get('image')

        if not name or not image:
            flash("Please provide both a name and an image.")
            return redirect(request.url)

        # Save uploaded file to UPLOAD_FOLDER
        filename = secure_filename(image.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(save_path)

        # Preprocess image and extract embedding with base_model
        img = cv2.imread(save_path)
        if img is None:
            flash("Error reading the image. Please try again.")
            return redirect(request.url)
        img = cv2.resize(img, (100, 100))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        embedding = base_model.predict(img)[0]

        # Update known_embeddings dictionary and save to file
        known_embeddings[name] = embedding
        os.makedirs("known_embeddings", exist_ok=True)
        with open("known_embeddings/embeddings.pkl", "wb") as f:
            pickle.dump(known_embeddings, f)

        flash(f"Successfully registered {name}!")
        return redirect(url_for('register'))

    return render_template('register.html')

# Route for video file upload and processing
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            cap = cv2.VideoCapture(save_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Process each frame for face recognition (using the same function from camera.py)
                frame = recognize_face(frame)
                # Optionally, you can also add a delay here for debugging
            cap.release()
            os.remove(save_path)
            flash("Video processed successfully!")
            return redirect(url_for('index'))
    return render_template('upload_video.html')

if __name__ == '__main__':
    app.run(debug=True)
