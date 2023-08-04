import cv2
import os
import math
import json
import random

from data_preprocessor import DataPreprocessor
from feature_extractor import NASNetFeatureExtractor

class MLFlow:
    """
    A class for processing videos, applying data preprocessing, and extracting features.
    """

    def __init__(self, input_json_file, output_folder, processing_steps=None):
        """
        Initialize the MLFlow object.

        Args:
            input_json_file (str): Path to the JSON file containing video information.
            output_folder (str): Folder where the extracted features will be saved.
            processing_steps (list, optional): List of data preprocessing steps. Default is None.
        """
        self.input_json_file = input_json_file
        self.output_folder = output_folder
        self.processing_steps = processing_steps
        self.data_preprocessor = DataPreprocessor(resize_height=331, resize_width=331)
        self.feature_extractor = NASNetFeatureExtractor()
        self.output_dirs = {}

    def process_video_and_extract_features(self):
        """
        Process videos, apply data preprocessing, and extract features from frames.
        """
        try:
            self._create_output_directories()

            # Get the list of videos in the JSON file
            with open(self.input_json_file, 'r') as json_file:
                data = json.load(json_file)

            preprocessed_sentences = self.process_sentences(data)

            if self.processing_steps:
                num_videos = len(data)
                num_steps = len(self.processing_steps)
                videos_per_step = math.ceil(num_videos / num_steps)

                # Shuffle the processing steps
                random.shuffle(self.processing_steps)

            # Process videos in the input folder
            for i, file in enumerate(data):
                video_path = file['SENTENCE_FILE_PATH']
                video_name = file['SENTENCE_NAME']
                video_description = preprocessed_sentences[video_name]

                # Determine the processing step for the video
                if self.processing_steps is None:
                    step = None
                    output_dir = self.output_folder
                else:
                    step_idx = i // videos_per_step
                    step = self.processing_steps[step_idx]
                    output_dir = self.output_dirs[step]

                # Create a subdirectory for the current video within the processing step directory
                video_dir = os.path.join(output_dir, video_name)

                # Open the input video
                cap = cv2.VideoCapture(video_path)

                # Process frames in the video
                output_json = []
                video_features = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process the frame and extract features
                    try:
                        processed_frame = self.data_preprocessor.process_frame(frame, step)
                        features = self.feature_extractor.extract_from_frame(processed_frame)

                        # Add the features to the list of video features
                        video_features.append(features)

                    except Exception as e:
                        print(f"Error occurred during frame processing: {str(e)}")

                # Release the video capture
                cap.release()

                output_json = {"video_name": video_name, "video_description": video_description, "video_features": video_features}
                # Open the JSON file in write mode
                with open(video_dir + '.json', 'w') as f:
                    # Write to the JSON file
                    json.dump(output_json, f)

            print("Video processing completed.")

        except Exception as e:
            print(f"Error occurred during video processing: {str(e)}")
            raise

    def process_sentences(self, data):
        """
        Preprocesses a list of sentences and returns a dictionary mapping sentence names to their preprocessed sequences.

        Args:
            data (list): A list of dictionaries, where each dictionary contains a 'SENTENCE_DESCRIPTION' and 'SENTENCE_NAME' key-value pair.

        Returns:
            dict: A dictionary where sentence names are keys, and their corresponding preprocessed sequences (as integer lists) are values.
        """
        try:
            # Extract 'SENTENCE_DESCRIPTION' and 'SENTENCE_NAME' lists from the data
            sentences = [item['SENTENCE_DESCRIPTION'] for item in data]
            sentence_names = [item['SENTENCE_NAME'] for item in data]

            # Preprocess sentences using the data_preprocessor object
            sequences = self.data_preprocessor.process_sentences(sentences, self.output_folder)

            # Convert sequences to lists of integers
            preprocessed_sentences = [list(map(int, seq)) for seq in sequences]

            # Create a dictionary to store the preprocessed data
            preprocessed_data = {}
            for sentence_name, pre_seq in zip(sentence_names, preprocessed_sentences):
                preprocessed_data[sentence_name] = pre_seq

            return preprocessed_data

        except KeyError as e:
            # Handle KeyError if 'SENTENCE_DESCRIPTION' or 'SENTENCE_NAME' keys are missing
            raise ValueError("Invalid data format. Each dictionary in 'data' must have 'SENTENCE_DESCRIPTION' and 'SENTENCE_NAME' keys.") from e

        except AttributeError as e:
            # Handle AttributeError if 'data_preprocessor' object doesn't have 'process_sentences' method
            raise AttributeError("'data_preprocessor' object must have 'process_sentences' method.") from e

        except Exception as e:
            # Handle any other unexpected exception
            raise RuntimeError("An error occurred during sentence preprocessing.") from e

    def _create_output_directories(self):
        """
        Create output directories for each processing step or the main output folder.

        This method creates directories for each processing step in 'self.processing_steps' and stores their paths
        in the 'self.output_dirs' dictionary. If 'self.processing_steps' is empty, it creates the main output folder.

        Raises:
            OSError: If an error occurs while creating directories.
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
        except OSError as e:
            # Handle OSError (e.g., file exists, permission denied) during directory creation
            raise OSError(f"Error occurred while creating output directories: {str(e)}") from e