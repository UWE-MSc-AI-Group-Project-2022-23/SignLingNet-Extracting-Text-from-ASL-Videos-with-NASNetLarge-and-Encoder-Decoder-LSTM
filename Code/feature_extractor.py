import numpy as np

from keras.applications.nasnet import NASNetLarge, NASNetMobile

class NASNetFeatureExtractor:
    def __init__(self, model_type='large'):
        """
        Initialize the NASNet feature extractor.

        Args:
            model_type (str): Type of NASNet model to use. Options: 'large' or 'mobile'.
        """
        if model_type.lower() == 'large':
            self.model = NASNetLarge(weights='imagenet')
        elif model_type.lower() == 'mobile':
            self.model = NASNetMobile(weights='imagenet')
        else:
            raise ValueError("Invalid model_type. Options are 'large' or 'mobile'.")

    def extract_features_from_frame(self, frame):
        """
        Extract features from an frame.

        Args:
            frame (str): Frame.

        Returns:
            ndarray: Feature representation of the image.
        """
        x = np.expand_dims(frame, axis=0)
        features = self.model.predict(x)
        return features.flatten()
