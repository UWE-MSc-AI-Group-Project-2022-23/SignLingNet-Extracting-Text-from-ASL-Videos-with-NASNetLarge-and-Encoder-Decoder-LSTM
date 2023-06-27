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

    def segment(self, frame):
        """
        Segment the input frame into distinct regions.

        Input:
        - frame: The frame to be segmented.

        Output:
        - Segmented regions of the frame.
        """
        # Perform segmentation algorithm of your choice
        # Example: Apply thresholding to create binary regions
        _, binary_frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

        # Apply any post-processing or analysis on the segmented regions

        return binary_frame

    def annotate_roi(self, frame, regions):
        """
        Annotate the regions of interest in the input frame using the provided regions.

        Input:
        - frame: The frame to be annotated.
        - regions: List of regions of interest.

        Output:
        - Annotated frame with regions of interest.
        """
        annotated_frame = frame.copy()

        # Draw rectangles around the regions of interest
        for region in regions:
            x, y, w, h = region
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return annotated_frame
