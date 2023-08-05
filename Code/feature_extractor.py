import numpy as np

from keras.applications.nasnet import NASNetLarge
from keras.models import Model
from typing import List


class NASNetFeatureExtractor:
    def __init__(self):
        """
        Initialize the NASNet feature extractor.
        """
        # Load the full pre-trained NASNetLarge model with weights from ImageNet
        full_model = NASNetLarge(weights="imagenet", include_top=True)

        # Create a new model that includes all layers of the full model up to the last convolutional layer
        self.model = Model(
            inputs=full_model.input,
            outputs=full_model.get_layer("normal_concat_18").output,
        )

    def extract_from_frame(self, frame: np.ndarray) -> List[float]:
        """
        Extract features from an image frame.

        Args:
            frame (ndarray): Input image frame as a NumPy array.

        Returns:
            List[float]: Feature representation of the image as a 1D list.
        """
        # Expand dimensions of the frame to make it suitable for the model input
        x = np.expand_dims(frame, axis=0)

        # Extract features using the NASNetLarge model
        features = self.model.predict(x)

        # Flatten the features and convert them to a 1D list
        return features.flatten().tolist()
