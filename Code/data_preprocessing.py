import cv2
import numpy as np

class DataPreprocessing:
    """
    A class for performing various data preprocessing operations.
    """

    def __init__(self):
        pass

    def resize(self, frame, width, height):
        """
        Resize the input frame.

        Input:
        - frame: The frame to be resized.
        - width: The desired width of the resized frame.
        - height: The desired height of the resized frame.

        Output:
        - Resized frame.
        """
        resized_frame = cv2.resize(frame, (width, height))
        return resized_frame

    def crop(self, frame, x, y, width, height):
        """
        Crop a rectangular region from the input frame and save it as an image file.
        
        Args:
            frame (ndarray): Input frame as a numpy array.
            x (int): X-coordinate of the top-left corner of the region.
            y (int): Y-coordinate of the top-left corner of the region.
            width (int): Width of the region.
            height (int): Height of the region.
        """
        cropped_frame = frame[y:y+height, x:x+width, :]
        return cropped_frame

    def reduce_noise(self, frame, h, templateWindowSize, searchWindowSize):
        """
        Apply non-local means denoising technique to the input frame.

        Input:
        - frame: The frame to have noise reduced.
        - h: The parameter regulating filter strength. Higher h value preserves more details, but may result in less noise reduction.
        - templateWindowSize: Size in pixels of the template patch that is used to compute weights. Should be an odd value.
        - searchWindowSize: Size in pixels of the window that is used to compute weighted average for given pixel. Should be an odd value.

        Output:
        - Frame with reduced noise.
        """
        denoised_frame = cv2.fastNlMeansDenoising(frame, None, h, templateWindowSize, searchWindowSize)
        return denoised_frame

    def add_noise(self, frame, noise_type='gaussian', mean=0, std=1):
        """
        Add noise to the input frame.

        Args:
            frame (ndarray): The frame to which noise will be added.
            noise_type (str): Type of noise to be added. Options: 'gaussian', 'salt-and-pepper', 'poisson'.
            mean (float): Mean of the noise distribution (used for 'gaussian' noise only).
            std (float): Standard deviation of the noise distribution (used for 'gaussian' noise only).

        Returns:
            ndarray: Frame with added noise.
        """
        noisy_frame = np.copy(frame)

        if noise_type == 'gaussian':
            noise = np.random.normal(mean, std, frame.shape).astype(np.uint8)
            noisy_frame = cv2.add(frame, noise)

        elif noise_type == 'salt-and-pepper':
            prob = 0.05
            mask = np.random.random(frame.shape[:2]) < prob / 2
            noisy_frame[mask] = 0

            mask = np.random.random(frame.shape[:2]) < prob / 2
            noisy_frame[mask] = 255

        elif noise_type == 'poisson':
            noisy_frame = np.random.poisson(frame.astype(np.float32))

        return noisy_frame

    def spatial_transformation(self, frame, rotation_angle, flip_horizontal, flip_vertical):
        """
        Apply spatial transformations to the input frame.
        
        Args:
            frame (ndarray): Input frame as a numpy array.
            rotation_angle (float): Rotation angle in degrees.
            flip_horizontal (bool): Flag indicating horizontal flipping.
            flip_vertical (bool): Flag indicating vertical flipping.
            
        Returns:
            ndarray: Transformed frame.
        """
        transformed_frame = frame

        if rotation_angle != 0:
            rows, cols, _ = frame.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
            transformed_frame = cv2.warpAffine(frame, M, (cols, rows))
        
        if flip_horizontal:
            transformed_frame = cv2.flip(transformed_frame, 1)
        
        if flip_vertical:
            transformed_frame = cv2.flip(transformed_frame, 0)
        
        return transformed_frame
    
    def color_jitter(self, video, brightness=0, contrast=0, saturation=0, hue=0):
        """
        Apply color jittering to the video frames.
        
        Args:
            video (ndarray): Input video frames.
            brightness (float): Brightness adjustment factor.
            contrast (float): Contrast adjustment factor.
            saturation (float): Saturation adjustment factor.
            hue (float): Hue adjustment factor.
            
        Returns:
            ndarray: Color jittered video frames.
        """
        num_frames = video.shape[0]
        adjusted_frames = []
        
        for i in range(num_frames):
            frame = video[i].astype(np.float32) / 255.0
            
            if brightness != 0:
                frame = cv2.add(frame, brightness)
            
            if contrast != 0:
                frame = cv2.multiply(frame, contrast)
            
            if saturation != 0:
                frame = cv2.multiply(frame, saturation)
            
            if hue != 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                frame[:, :, 0] += hue
                frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
            
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            adjusted_frames.append(frame)
        
        return np.array(adjusted_frames)


    def grey_scale(self, frame):
        """
        Convert the input frame to grayscale.

        Input:
        - frame: The frame to be converted to grayscale.

        Output:
        - Grayscale frame.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray_frame