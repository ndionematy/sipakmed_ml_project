# api/utils.py
import numpy as np
from PIL import Image
import io
from sklearn.pipeline import Pipeline

# 🟢 importe uniquement les étapes utiles
from notebooks.pipeline import ResizeAndPad, EnhanceQuality, Normalize

# 🔹 Pipeline SANS ImageReader et SANS DataAugmentation
test_pipeline = Pipeline([
    ('resize_pad', ResizeAndPad(target_size=224)),
    ('enhance',    EnhanceQuality(blur_thresh=100, bright_thresh=50)),
    ('normalize',  Normalize())
])

def preprocess_image(contents: bytes) -> np.ndarray:
    """
    Prend le contenu brut d'un fichier (bytes) → image prétraitée shape (1,224,224,3)
    """
    # 1. bytes -> PIL -> np.array (RGB)
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)

    # 2. Appliquer le pipeline (il attend une liste d'images np.array)
    img_proc = test_pipeline.transform([img_np])[0]

    # 3. Ajouter la dimension batch
    return np.expand_dims(img_proc, axis=0)   # (1,224,224,3)
