import numpy as np

from keras.applications.nasnet import NASNetLarge
from keras.models import Model

class NASNetFeatureExtractor:
    def __init__(self):
        """
        Initialize the NASNet feature extractor.

        """
        # Load the full pre-trained model
        full_model = NASNetLarge(weights='imagenet', include_top=True)

        # Create a new model that includes all layers of the full model up to the last convolutional layer
        self.model = Model(inputs=full_model.input, outputs=full_model.get_layer('normal_concat_18').output)

    def extract_from_frame(self, frame):
        """
        Extract features from an frame.

        Args:
            frame (str): Frame.

        Returns:
            ndarray: Feature representation of the image.
        """
        x = np.expand_dims(frame, axis=0)
        features = self.model.predict(x)
        return features.flatten().tolist()
