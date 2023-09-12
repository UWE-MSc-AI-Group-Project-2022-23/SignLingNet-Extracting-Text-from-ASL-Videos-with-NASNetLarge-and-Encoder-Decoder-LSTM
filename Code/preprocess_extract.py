import cv2
import os
import math
import json
import random

from data_preprocessor import DataPreprocessor
from feature_extractor import NASNetFeatureExtractor


class PreprocessorAndFeatureExtractor:
    """
    A class for processing videos, applying data preprocessing, and extracting features.
    """

    def __init__(
        self,
        input_json_file: str,
        output_folder: str,
        resize_width: int = 331,
        resize_height: int = 331,
        processing_steps: list = None,
    ):
        """
        Initialize the MLFlow object.

        Args:
            input_json_file (str): Path to the JSON file containing video information.
            output_folder (str): Folder where the extracted features will be saved.
            resize_width (int, optional): Width to which the video frames will be resized during preprocessing. Default is 331.
            resize_height (int, optional): Height to which the video frames will be resized during preprocessing. Default is 331.
            processing_steps (list, optional): List of data preprocessing steps. Default is None.
        """
        # Store the input parameters in instance variables
        self.input_json_file = input_json_file
        self.output_folder = output_folder
        self.processing_steps = processing_steps

        # Create a DataPreprocessor object for frame preprocessing
        self.data_preprocessor = DataPreprocessor(resize_width, resize_height)

        # Create a NASNetFeatureExtractor object for feature extraction
        self.feature_extractor = NASNetFeatureExtractor()

        # Initialize an empty dictionary to store the output directories for each processing step
        self.output_dirs = {}

    def process_video_and_extract_features(self):
        """
        Process videos, apply data preprocessing, and extract features from frames.
        """
        try:
            # Create output directories
            self._create_output_directories()

            # Load video metadata from the JSON file
            with open(self.input_json_file, "r") as json_file:
                data = json.load(json_file)

            # Calculate the number of videos per step (if processing_steps provided)
            if self.processing_steps:
                num_videos = len(data)
                num_steps = len(self.processing_steps)
                videos_per_step = math.ceil(num_videos / num_steps)

                # Shuffle the processing steps to randomize video processing order
                random.shuffle(self.processing_steps)

            sentences = []
            video_names = []

            desired_keyframes_per_video = 20

            # Process videos in the input folder
            for i, file in enumerate(data):
                video_path = file["SENTENCE_FILE_PATH"]
                video_name = file["SENTENCE_NAME"]

                # Determine the processing step and output directory for the video
                if self.processing_steps is None:
                    step = None
                    output_dir = self.output_folder
                else:
                    step_idx = i // videos_per_step
                    step = self.processing_steps[step_idx]
                    output_dir = self.output_dirs[step_idx]

                # Open the input video using OpenCV
                cap = cv2.VideoCapture(video_path)

                frame_count = 0
                keyframe_count = 0
                previous_frame = None

                threshold = 1000000

                step_size = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // desired_keyframes_per_video, 1)

                path = os.path.join(output_dir, 'features', f'{video_name}.json')

                # Process frames in the video
                video_features = {}
                processed_frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
                    if previous_frame is not None and frame_count % step_size == 0:
                        frame_diff = cv2.absdiff(gray_frame, previous_frame)
                        
                        diff_sum = frame_diff.sum()
                    
                        if diff_sum > threshold and keyframe_count < desired_keyframes_per_video:

                            keyframe_count += 1

                            processed_frame = self.data_preprocessor.process_frame(frame, step)

                            processed_frames.append(processed_frame)
            
                    previous_frame = gray_frame.copy()

                    frame_count += 1

                # Release the video capture
                cap.release()

                for i in range(len(processed_frames)):

                    features = self.feature_extractor.extract_from_frame(processed_frames[i])

                    video_features[f"frame_{i}"] = features

                with open(path, 'w') as f:
                    json.dump(video_features, f)

                sentences.append(file["SENTENCE_DESCRIPTION"])

                video_names.append(video_name)

            # Video processing completed
            print("Video processing completed.")

            # Preprocess sentences related to the videos
            sequences = self.process_labels(sentences)

            preprocessed_data = [
                {
                    "SENTENCE_NAME": item,
                    "SENTENCE_DESCRIPTION": sentence,
                    "PREPROCESSED_SENTENCE": sequence,
                }
                for item, sentence, sequence in zip(video_names, sentences, sequences)
            ]
            preprocessed_path = os.path.join(output_dir, "preprocessed_sentences.json")
            with open(preprocessed_path, "w") as json_file:
                json.dump(preprocessed_data, json_file, indent=4)

        except Exception as e:
            # Handle any errors during video processing
            print(f"Error occurred during video processing: {str(e)}")
            raise

    def process_labels(self, sentences: list) -> dict:
        """
        Preprocesses a list of sentences.
        """
        try:
            # Preprocess sentences using the data_preprocessor object
            sequences = self.data_preprocessor.process_labels(
                sentences, self.output_folder
            )

            sequences = [list(map(int, seq)) for seq in sequences]

            return sequences

        except Exception as e:
            # Handle any other unexpected exception
            raise RuntimeError(
                "An error occurred during sentence preprocessing."
            ) from e

    def _create_output_directories(self):
        """
        Create output directories for each processing step or the main output folder.

        This method creates directories for each processing step in 'self.processing_steps' and stores their paths
        in the 'self.output_dirs' dictionary. If 'self.processing_steps' is empty, it creates the main output folder.
        """
        try:
            if self.processing_steps:
                # Create directories for each processing step
                for step in self.processing_steps:
                    step_dir = os.path.join(self.output_folder, step)
                    os.makedirs(step_dir, exist_ok=True)
                    self.output_dirs[step] = step_dir
            else:
                # Create the main output folder if no processing steps are specified
                os.makedirs(self.output_folder, exist_ok=True)
                features_path = os.path.join(self.output_folder, 'features')
                os.makedirs(features_path, exist_ok=True)
        except OSError as e:
            # Handle OSError (e.g., file exists, permission denied) during directory creation
            raise OSError(
                f"Error occurred while creating output directories: {str(e)}"
            ) from e
