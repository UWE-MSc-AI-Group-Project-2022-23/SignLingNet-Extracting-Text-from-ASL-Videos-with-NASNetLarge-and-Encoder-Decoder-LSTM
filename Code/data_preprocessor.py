import cv2
import numpy as np

class DataPreprocessor:
    """
    A class for performing various data preprocessing operations on video frames.
    """

    def __init__(self, resize_width, resize_height):
        """
        Initialize the DataPreprocessing object.

        Args:
            resize_width (int): The desired width for frame resizing.
            resize_height (int): The desired height for frame resizing.
        """
        self.resize_width = resize_width
        self.resize_height = resize_height

    def process_frame(self, frame, step):
        """
        Process a video frame based on the specified preprocessing step.

        Args:
            frame (ndarray): The video frame to be processed.
            step (str): The name of the preprocessing step to apply.

        Returns:
            ndarray: The processed video frame.
        """
        try:
            # Resize frame
            processed_frame = self.resize(frame, self.resize_width, self.resize_height)

            # Perform the respective processing step on frames
            if step == "crop":
                processed_frame = self.crop(frame, 300, 0, 680, 720)
                processed_frame = self.resize(processed_frame, self.resize_width, self.resize_height)

            elif step == "reduce_noise":
                processed_frame = self.reduce_noise(processed_frame, h=10, templateWindowSize=7,
                                                            searchWindowSize=21)

            elif step == "add_noise":
                processed_frame = self.add_noise(processed_frame, noise_type='gaussian', mean=0, std=1)

            elif step == "rotate":
                processed_frame = self.spatial_transformation(processed_frame, rotation_angle=90,
                                                                    flip_horizontal=False,
                                                                    flip_vertical=False)

            elif step == "flip_horizontal":
                processed_frame = self.spatial_transformation(processed_frame, rotation_angle=0,
                                                                    flip_horizontal=True,
                                                                    flip_vertical=False)

            elif step == "flip_vertical":
                processed_frame = self.spatial_transformation(processed_frame, rotation_angle=0,
                                                                    flip_horizontal=False,
                                                                    flip_vertical=True)

            elif step == "brightness":
                processed_frame = self.color_jitter(processed_frame, brightness=0.5, contrast=1.0,
                                                            saturation=1.0, hue=0.0)

            elif step == "contrast":
                processed_frame = self.color_jitter(processed_frame, brightness=0.0, contrast=1.5,
                                                            saturation=1.0, hue=0.0)

            elif step == "saturation":
                processed_frame = self.color_jitter(processed_frame, brightness=0.0, contrast=1.0,
                                                            saturation=1.5, hue=0.0)

            elif step == "greyscale":
                processed_frame = self.grey_scale(processed_frame)
            
            return processed_frame
        except Exception as e:
            print(f"Error occurred during {step} preprocessing: {str(e)}")
            return None

    def resize(self, frame, width, height):
        """
        Resize the input frame.

        Args:
            frame (ndarray): The frame to be resized.
            width (int): The desired width of the resized frame.
            height (int): The desired height of the resized frame.

        Returns:
            ndarray: The resized frame.
        """
        try:
            resized_frame = cv2.resize(frame, (width, height))
            return resized_frame
        except Exception as e:
            print("Error occurred during resizing: ", str(e))
            return None

    def crop(self, frame, x, y, width, height):
        """
        Crop a rectangular region from the input frame.

        Args:
            frame (ndarray): Input frame as a numpy array.
            x (int): X-coordinate of the top-left corner of the region.
            y (int): Y-coordinate of the top-left corner of the region.
            width (int): Width of the region.
            height (int): Height of the region.

        Returns:
            ndarray: Cropped frame.
        """
        try:
            cropped_frame = frame[y:y+height, x:x+width, :]
            return cropped_frame
        except Exception as e:
            print("Error occurred during cropping: ", str(e))
            return None

    def reduce_noise(self, frame, h, templateWindowSize, searchWindowSize):
        """
        Apply non-local means denoising technique to the input frame.

        Args:
            frame (ndarray): The frame to have noise reduced.
            h (int): The parameter regulating filter strength. Higher h value preserves more details, but may result in less noise reduction.
            templateWindowSize (int): Size in pixels of the template patch that is used to compute weights. Should be an odd value.
            searchWindowSize (int): Size in pixels of the window that is used to compute weighted average for a given pixel. Should be an odd value.

        Returns:
            ndarray: Frame with reduced noise.
        """
        try:
            denoised_frame = cv2.fastNlMeansDenoising(frame, None, h, templateWindowSize, searchWindowSize)
            return denoised_frame
        except Exception as e:
            print("Error occurred during noise reduction: ", str(e))
            return None

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
        try:
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
        except Exception as e:
            print("Error occurred during noise addition: ", str(e))
            return None

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
        try:
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
        except Exception as e:
            print("Error occurred during spatial transformation: ", str(e))
            return None
    
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
        try:
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
        except Exception as e:
            print("Error occurred during color jittering: ", str(e))
            return None


    def grey_scale(self, frame):
        """
        Convert the input frame to grayscale.

        Args:
            frame (ndarray): The frame to be converted to grayscale.

        Returns:
            ndarray: Grayscale frame.
        """
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return gray_frame
        except Exception as e:
            print("Error occurred during grayscale conversion: ", str(e))
            return None
