import json
import os
import cv2
import pickle

import numpy as np

from sklearn.model_selection import train_test_split
from preprocess_extract import PreprocessorAndFeatureExtractor
from encoder_decoder_lstm import EncoderDecoderLSTM


class MLFlow:
    def __init__(self, output_folder, train_data_file_path, videos_file_path):
        self.output_folder = output_folder
        self.train_data_file_path = train_data_file_path
        self.videos_file_path = videos_file_path
        self.encoder_model = None
        self.decoder_model = None
        self.tokenizer = None
        self.max_seq_length = None
        self.preprocessed_data = None

    def get_video_fps(self, video_file_path):

        cap = cv2.VideoCapture(video_file_path)

        fps = cap.get(cv2.CAP_PROP_FPS) 

        cap.release()

        return fps
    
    def prepare_data(self):
        with open(self.train_data_file_path, 'r') as json_file:
            data = json.load(json_file)

        filtered_data = []

        for i, item in enumerate(data):
            if 2.0 <= item['SENTENCE_DURATION'] < 2.1:
                fps = self.get_video_fps(os.path.join(self.videos_file_path, item['SENTENCE_NAME'] + '.mp4'))
                if fps == 24:
                    filtered_data.append(item)

        filtered_data = filtered_data[:20]

        print(len(filtered_data))

        json_output_file_path = os.path.join(self.output_folder, 'train_data_to_be_extracted.json')

        with open(json_output_file_path, 'w') as labels_file:
            json.dump(filtered_data, labels_file, indent=4)

        processor = PreprocessorAndFeatureExtractor(json_output_file_path, self.output_folder)
        processor.process_video_and_extract_features()

        preprocessed_file_path = os.path.join(self.output_folder, 'preprocessed_sentences.json')
        with open(preprocessed_file_path, 'r') as f:
            preprocessed_data = json.load(f)

        # Load the saved tokenizer to use its vocabulary size
        tokenizer_file_path = os.path.join(self.output_folder, 'tokenizer.pickle')
        with open(tokenizer_file_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        vocab_size = len(self.tokenizer.word_index) + 1

        # Load max_seq_length
        max_seq_length_file_path = os.path.join(self.output_folder, 'max_seq_length.pkl')
        with open(max_seq_length_file_path, 'rb') as f:
            self.max_seq_length = pickle.load(f)

        # Extract the preprocessed sequences directly
        Y_data = [item['PREPROCESSED_SENTENCE'] for item in preprocessed_data]

        # Adjust for decoder input and target
        decoder_input_data = np.array(Y_data)[:, :-1]
        decoder_target_data = np.array(Y_data)[:, 1:]

        # Load video features from the directory
        feature_dir = os.path.join(self.output_folder, 'features')

        video_features = []
        for item in preprocessed_data:
            file_path = os.path.join(feature_dir, f"{item['SENTENCE_NAME']}.json")
            try:
                with open(file_path, 'r') as f:
                    video_features.append(json.load(f))
                print(len(video_features))
            except Exception as e:
                print(f"[ERROR] Failed to load: {file_path}. Error: {e}")

        max_frames = max([len(features) for features in video_features])
        feature_length = len(video_features[0]['frame_0'][0][0])

        # Prepare training data
        X_data = np.zeros((len(video_features), max_frames, feature_length))
        for i, features in enumerate(video_features):
            sorted_frames = sorted(features.items(), key=lambda x: x[0])
            for j, (_, frame_data) in enumerate(sorted_frames):
                X_data[i, j, :] = frame_data[0][0]

        X_train, X_val, decoder_input_train, decoder_input_val, decoder_target_train, decoder_target_val = train_test_split(X_data, decoder_input_data, decoder_target_data, test_size=0.2, random_state=42)

        return preprocessed_data, feature_length, vocab_size, X_train, decoder_input_train, decoder_target_train, X_val, decoder_input_val, decoder_target_val
    
    def train(self):

        preprocessed_data, feature_length, vocab_size, X_train, decoder_input_train, decoder_target_train, X_val, decoder_input_val, decoder_target_val = self.prepare_data()

        encoder_decoder_lstm = EncoderDecoderLSTM(feature_length=feature_length, vocab_size=vocab_size, embedding_dim=128, units=128, dropout_rate=0.5)
        encoder_decoder_lstm.construct_model()
        encoder_decoder_lstm.compile(learning_rate=0.003)
        encoder_decoder_lstm.train(X_train, decoder_input_train, decoder_target_train, X_val, decoder_input_val, decoder_target_val, batch_size=32, epochs=1000, patience=100)

        self.encoder_model = encoder_decoder_lstm.inference_encoder_model()
        self.decoder_model = encoder_decoder_lstm.inference_decoder_model()

        return X_train, X_val, decoder_input_val, decoder_target_val, preprocessed_data

    # Decoding sequence function
    def decode_sequence(self, input_seq, video_name):
        # Encode the input sequence to get the internal state vectors.
        states_value = self.encoder_model.predict(input_seq)
        
        # Generate an empty target sequence of length 1 with only the start token.
        target_seq = np.array(self.tokenizer.texts_to_sequences(['start'])).reshape(1, 1)
        
        # Display the video name
        print(f"Video: {video_name}")
        
        # Sampling loop for a batch of sequences
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            
            # Sample the next token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.tokenizer.index_word.get(sampled_token_index, '')
            decoded_sentence += ' ' + sampled_word
            
            # Exit condition: either hit max length or find the end token.
            if (sampled_word == 'end' or len(decoded_sentence.split()) > self.max_seq_length):
                stop_condition = True

            # Update the target sequence to the last predicted token.
            target_seq = np.array([[sampled_token_index]])
            
            # Update states for the next time step
            states_value = [h, c]

        return video_name, f"{decoded_sentence.replace(' end', '')}" 
    
if __name__=='__main__':
    mlflow = MLFlow('../Data/train', '../Data/train.json', '../../Dataset/train/videos')

    X_train, X_val, decoder_input_val, decoder_target_val, preprocessed_data = mlflow.train()

    # Test
    video_names = []
    actual_sentences = []
    predicted_sentences = []
    for i in range(len(X_train)):
        sample_index = i
        input_seq = X_train[sample_index:sample_index+1]
        video_name = preprocessed_data[sample_index]['SENTENCE_NAME']
        actual_sentence = preprocessed_data[sample_index]['SENTENCE_DESCRIPTION']
        video_name, predicted_sentence = mlflow.decode_sequence(input_seq, video_name)
        video_names.append(video_name)
        actual_sentences.append(actual_sentence)
        predicted_sentences.append(predicted_sentence.replace('start', '').strip())
    
    with open('output.txt', 'w') as f:
        for i in range(len(X_train)):
            f.write('Video: ' + str(video_names[i]) + '\n')
            f.write('Actual: ' + str(actual_sentences[i]) + '\n')
            f.write('Predicted: ' + str(predicted_sentences[i]) + '\n')
            f.write('\n')