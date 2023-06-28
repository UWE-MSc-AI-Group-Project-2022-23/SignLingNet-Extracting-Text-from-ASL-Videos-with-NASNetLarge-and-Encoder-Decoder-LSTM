import cv2

class PreprocessData:
    """
    A class for performing various data preprocessing operations.
    """

    def __init__(self):
        # Initialize any required variables or resources here
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
    
    def segment(self, input_video):

        desired_frame_interval = 0.5  # Set the desired time interval between frames in seconds
        frame_rate = input_video.get(cv2.CAP_PROP_FPS)
        desired_frame_rate = 1 / desired_frame_interval

        # Calculate the frame skip count to achieve the desired frame rate
        frame_skip_count = int(frame_rate / desired_frame_rate)

        frame_count = 0

        while input_video.isOpened():
            ret, frame = input_video.read()
            
            if not ret:
                break
            
            # Save the frame or segment as an image or segment
            cv2.imwrite(f'frame_{frame_count}.jpg', frame)
            
            frame_count += 1
            
            # Skip frames based on the frame skip count
            for _ in range(frame_skip_count - 1):
                input_video.read()

        input_video.release()
        cv2.destroyAllWindows()

    # def detect_object(self, frame, object_classifier):

    #     # Convert the frame to the HSV color space
    #     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #     # Define the lower and upper bounds of the skin color in HSV
    #     lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    #     upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    #     # Create a binary mask of the skin color regions
    #     skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    #     # Perform morphological operations to refine the mask
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #     skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    #     # Apply the mask to the original frame
    #     masked_frame = cv2.bitwise_and(frame, frame, mask=skin_mask)