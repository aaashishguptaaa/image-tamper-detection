"""
CNN Model for Image Forgery Classification
-------------------------------------------
A lightweight CNN built with TensorFlow/Keras.
Input: ELA image (128x128x3)
Output: Binary classification — REAL (0) or FAKE (1)
"""

import os
import numpy as np
import cv2

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

IMG_SIZE = 128


def build_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)) -> "keras.Model":
    """
    Build the CNN architecture for forgery detection.

    Architecture:
        Conv → Pool → Conv → Pool → Conv → Pool
        → Flatten → Dense → Dropout → Output
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed. Run: pip install tensorflow")

    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")   # 0=Real, 1=Fake
    ], name="ForgeryDetectorCNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def preprocess_image_for_cnn(image_path: str) -> np.ndarray:
    """
    Preprocess image for CNN input.
    Converts to ELA representation at 128x128.
    """
    from techniques.ela import perform_ela

    ela = perform_ela(image_path)
    ela_resized = cv2.resize(ela, (IMG_SIZE, IMG_SIZE))
    ela_norm = ela_resized.astype(np.float32) / 255.0
    return np.expand_dims(ela_norm, axis=0)   # shape: (1, 128, 128, 3)


def load_or_create_model(model_path: str = "models/forgery_cnn.keras"):
    """
    Load existing model or create a new one.
    """
    if not TF_AVAILABLE:
        return None

    if os.path.exists(model_path):
        print(f"[CNN] Loading saved model from {model_path}")
        return keras.models.load_model(model_path)
    else:
        print("[CNN] No saved model found — creating new model (untrained)")
        return build_cnn_model()


def predict_with_cnn(image_path: str,
                     model_path: str = "models/forgery_cnn.keras") -> dict:
    """
    Run CNN prediction on an image.

    Returns:
        dict with:
            cnn_score (float 0-1): probability of being FAKE
            is_trained (bool): whether model is trained
    """
    if not TF_AVAILABLE:
        return {"cnn_score": None, "is_trained": False,
                "error": "TensorFlow not installed"}

    model = load_or_create_model(model_path)
    img_tensor = preprocess_image_for_cnn(image_path)

    is_trained = os.path.exists(model_path)
    prediction = float(model.predict(img_tensor, verbose=0)[0][0])

    return {
        "cnn_score": round(prediction, 4),
        "is_trained": is_trained
    }


def get_model_summary():
    """Return CNN architecture summary as string."""
    if not TF_AVAILABLE:
        return "TensorFlow not installed."
    model = build_cnn_model()
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
