import cv2
import numpy as np

class DataAugmentation:
    """
    A class for performing various data augmentation techniques.
    """

    def __init__(self):
        self.random_state = np.random.RandomState(seed=42)

    def clip_crop(self, video, x, y, width, height):
        """
        Crop a rectangular region from the video frames.
        
        Args:
            video (ndarray): Input video frames.
            x (int): X-coordinate of the top-left corner of the region.
            y (int): Y-coordinate of the top-left corner of the region.
            width (int): Width of the region.
            height (int): Height of the region.
            
        Returns:
            ndarray: Cropped video frames.
        """
        return video[:, y:y+height, x:x+width, :]

    def temporal_subsampling(self, video, interval):
        """
        Perform temporal subsampling by selecting frames at regular intervals.
        
        Args:
            video (ndarray): Input video frames.
            interval (int): Interval between selected frames.
            
        Returns:
            ndarray: Subsampled video frames.
        """
        return video[::interval]

    def time_jittering(self, video, max_shift):
        """
        Apply time jittering by shifting the temporal order of frames.
        
        Args:
            video (ndarray): Input video frames.
            max_shift (int): Maximum number of frames to shift.
            
        Returns:
            ndarray: Time jittered video frames.
        """
        num_frames = video.shape[0]
        shift = self.random_state.randint(-max_shift, max_shift + 1)
        shifted_frames = np.roll(video, shift, axis=0)
        
        if shift > 0:
            shifted_frames[:shift] = video[num_frames-shift:]
        elif shift < 0:
            shifted_frames[shift:] = video[:num_frames+shift]
            
        return shifted_frames

    def speed_perturbation(self, video, speed_factor):
        """
        Apply speed perturbation by adjusting the playback rate of the video.
        
        Args:
            video (ndarray): Input video frames.
            speed_factor (float): Speed factor (>1 for faster, <1 for slower).
            
        Returns:
            ndarray: Speed perturbed video frames.
        """
        return cv2.resize(video, None, fx=speed_factor, fy=1.0, interpolation=cv2.INTER_LINEAR)

    def spatial_transformation(self, video, rotation_angle, flip_horizontal, flip_vertical):
        """
        Apply spatial transformations to each frame of the video.
        
        Args:
            video (ndarray): Input video frames.
            rotation_angle (float): Rotation angle in degrees.
            flip_horizontal (bool): Flag indicating horizontal flipping.
            flip_vertical (bool): Flag indicating vertical flipping.
            
        Returns:
            ndarray: Transformed video frames.
        """
        num_frames = video.shape[0]
        transformed_frames = []
        
        for i in range(num_frames):
            frame = video[i]
            
            if rotation_angle != 0:
                rows, cols, _ = frame.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
                frame = cv2.warpAffine(frame, M, (cols, rows))
            
            if flip_horizontal:
                frame = cv2.flip(frame, 1)
            
            if flip_vertical:
                frame = cv2.flip(frame, 0)
            
            transformed_frames.append(frame)
        
        return np.array(transformed_frames)

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
