#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import cv2
import numpy as np
import random


# # Image reader

# In[9]:


class ImageReader(BaseEstimator, TransformerMixin):
    def __init__(self, target_size=224, placeholder_color=(255, 255, 255)):
        self.target_size = target_size
        self.placeholder_color = placeholder_color
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        images = []
        self.status = []
        for path in X:
            img = cv2.imread(path)
            if img is None:
                print(f"Aucune image détectée : {path} (placeholder ajouté)")
                img = np.full((self.target_size, self.target_size, 3), self.placeholder_color, dtype=np.uint8)
                self.status.append("introuvable")
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.status.append("ok")
            images.append(img)
        return images


# # Image resizing

# In[ ]:


class ResizeAndPad(BaseEstimator, TransformerMixin):
    def __init__(self, target_size=224):
        self.target_size = target_size
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        result = []
        for img in X:
            h, w = img.shape[:2]
            scale = min(self.target_size / w, self.target_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_padded = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            x_start = (self.target_size - new_w) // 2
            y_start = (self.target_size - new_h) // 2
            img_padded[y_start:y_start+new_h, x_start:x_start+new_w] = img_resized
            result.append(img_padded)
        return result


# # Image quality

# In[11]:


class EnhanceQuality(BaseEstimator, TransformerMixin):
    def __init__(self, blur_thresh=100, bright_thresh=50):
        self.blur_thresh = blur_thresh
        self.bright_thresh = bright_thresh
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        result = []
        for img in X:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Détecte si l'image est floue
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < self.blur_thresh:
                print("Image floue: renforcement des contours appliqué")
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                img = cv2.filter2D(img, -1, kernel)

            # Détecte si l'image est sombre
            if np.mean(gray) < self.bright_thresh:
                print("💡 Sombre détecté : CLAHE appliqué")
                img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
                img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            result.append(img)
        return result


# # Data augmentation

# In[12]:


class DataAugmentation(BaseEstimator, TransformerMixin):
    def __init__(self, p_flip=0.5, max_rotation=15):
        self.p_flip = p_flip
        self.max_rotation = max_rotation
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        result = []
        for img in X:
            if random.random() < self.p_flip:
                img = cv2.flip(img, 1)
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            result.append(img)
        return result


# # Normalisation

# In[ ]:


class Normalize(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array(X) / 255.0


# # Pipeline final

# In[ ]:


# image_pipeline = Pipeline([
#     ('reader', ImageReader(target_size=224)),
#     ('resize_pad', ResizeAndPad(target_size=224)),
#     ('enhance', EnhanceQuality(blur_thresh=100, bright_thresh=50)),
#     ('augment', DataAugmentation(p_flip=0.5, max_rotation=15)),
#     ('normalize', Normalize())
# ])

# image_pipeline


# In[ ]:


# 🔹 Exemple d'utilisation
# image_paths = ["path/to/img1.bmp", "path/to/img2.bmp"]
# processed_images = image_pipeline.fit_transform(image_paths)

# Pour visualiser :
# import matplotlib.pyplot as plt
# plt.imshow(processed_images[0])
# plt.axis('off')
# plt.show()

