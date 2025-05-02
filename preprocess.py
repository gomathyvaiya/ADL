import os
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 100

def load_images_from_folder(folder_path):
    images = {}
    for person_name in os.listdir(folder_path):
        person_dir = os.path.join(folder_path, person_name)
        if os.path.isdir(person_dir):
            person_imgs = []
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img.astype('float32') / 255.0
                    person_imgs.append(img)
            if len(person_imgs) >= 2:  # At least 2 images needed for pair
                images[person_name] = person_imgs
    return images

def create_pairs(images_dict):
    pairs = []
    labels = []

    people = list(images_dict.keys())

    for person in people:
        imgs = images_dict[person]
        # Positive pairs
        for i in range(len(imgs) - 1):
            pairs.append([imgs[i], imgs[i+1]])
            labels.append(1)

        # Negative pairs
        other_person = random.choice([p for p in people if p != person])
        img1 = random.choice(imgs)
        img2 = random.choice(images_dict[other_person])
        pairs.append([img1, img2])
        labels.append(0)

    return np.array(pairs), np.array(labels)

def get_data(data_path='known_faces'):
    images = load_images_from_folder(data_path)
    pairs, labels = create_pairs(images)

    X1 = np.array([pair[0] for pair in pairs])
    X2 = np.array([pair[1] for pair in pairs])

    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
        X1, X2, labels, test_size=0.2, random_state=42)

    return (X1_train, X2_train, y_train), (X1_test, X2_test, y_test)
