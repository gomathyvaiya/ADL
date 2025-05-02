from preprocess import get_data
from siamese_model import build_siamese_model
import tensorflow as tf
import os

# Step 1: Load preprocessed training and testing data
(X1_train, X2_train, y_train), (X1_test, X2_test, y_test) = get_data()

# Step 2: Build Siamese model
siamese_model, base_model = build_siamese_model()
siamese_model.summary()

# Step 3: Train
history = siamese_model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_test, X2_test], y_test),
    batch_size=32,
    epochs=10
)

# Step 4: Save the full Siamese model
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

siamese_model.save("saved_models/siamese_model.h5")
base_model.save("saved_models/base_model.h5")

print("✅ Model training complete and saved to 'saved_models/'")
