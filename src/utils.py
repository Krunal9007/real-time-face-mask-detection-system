import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

def load_image_for_model(face_roi, target_size=(128,128)):
    """Prepare a detected face ROI for prediction."""
    img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def ensure_dir(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
