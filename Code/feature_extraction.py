import numpy as np
import tensorflow.keras.applications.nasnet as nasnet

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.nasnet import preprocess_input

def extract_frame_features(frame):
    img = image.img_to_array(frame)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    base_model = nasnet.NASNetLarge(weights='imagenet')
    features = base_model.predict(img)

    return features.flatten()