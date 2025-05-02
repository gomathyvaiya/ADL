import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

IMG_SIZE = 100

def preprocess_face(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return img

def generate_embeddings(base_model_path='saved_models/base_model.h5', known_faces_dir='known_faces', output_path='known_embeddings/embeddings.pkl'):
    base_model = load_model(base_model_path)

    # Load existing embeddings if available
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = {}

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            face = preprocess_face(img_path)
            if face is None:
                print(f"⚠️ Skipped invalid image: {img_path}")
                continue

            face = np.expand_dims(face, axis=0)
            emb = base_model.predict(face)[0]

            if person_name in embeddings:
                if isinstance(embeddings[person_name], list):
                    embeddings[person_name].append(emb)
                else:
                     # Convert existing numpy array to list
                     embeddings[person_name] = [embeddings[person_name], emb]
            else:
                embeddings[person_name] = [emb]


            print(f"✅ Processed: {img_name} for {person_name}")
            break  # One image per person; remove if you want to allow multiple

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"✅ Embeddings saved to {output_path}")

if __name__ == "__main__":
    generate_embeddings()
