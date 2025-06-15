# api/model.py
import os
import numpy as np
from tensorflow.keras.models import load_model

LABELS = {
    0: "dyskeratotic",
    1: "koilocytotic",
    2: "metaplastic",
    3: "parabasal",
    4: "superficial"
}

def load_keras_model():
    here = os.path.dirname(__file__)
    model_path = os.path.join(here, "..", "models", "densenet_ccu.h5")
    return load_model(model_path, compile=False)

def predict_class(model, img_array: np.ndarray) -> str:
    preds = model.predict(img_array, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])
    return LABELS.get(idx, "Unknown")
