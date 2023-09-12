from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers.legacy import Adam


class EncoderDecoderLSTM:
    def __init__(self, feature_length, vocab_size, embedding_dim=128, units=128, dropout_rate=0.5):
        self.feature_length = feature_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.dropout_rate = dropout_rate
        self.model = None

    def construct_model(self):
        # Encoder
        encoder_inputs = Input(shape=(None, self.feature_length), name="encoder_input")
        encoder_lstm = LSTM(self.units, return_state=True, name='encoder_lstm')
        _, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(None,), name="decoder_input")
        decoder_embedding = Embedding(self.vocab_size, self.embedding_dim, name='embedding', mask_zero=True)
        decoder_inputs_embedded = decoder_embedding(decoder_inputs)
        decoder_lstm = LSTM(self.units, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_lstm_out, _, _ = decoder_lstm(decoder_inputs_embedded, initial_state=encoder_states)
        decoder_lstm_out = Dropout(self.dropout_rate)(decoder_lstm_out)
        decoder_dense = Dense(self.vocab_size, activation='softmax', name='dense')
        decoder_outputs = decoder_dense(decoder_lstm_out)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    def compile(self, learning_rate=0.003):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, decoder_input_train, decoder_target_train, X_val, decoder_input_val, decoder_target_val, batch_size=32, epochs=1000, patience=100):
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.model.fit([X_train, decoder_input_train], decoder_target_train,
                       validation_data=([X_val, decoder_input_val], decoder_target_val),
                       batch_size=batch_size, epochs=epochs, callbacks=[early_stopping])

    def inference_encoder_model(self):
        encoder_inputs = self.model.get_layer('encoder_input').input
        encoder_lstm = self.model.get_layer('encoder_lstm')
        _, state_h, state_c = encoder_lstm.output
        encoder_states = [state_h, state_c]
        encoder_model = Model(encoder_inputs, encoder_states)
        return encoder_model

    def inference_decoder_model(self):
        decoder_inputs = self.model.get_layer('decoder_input').input
        decoder_state_input_h = Input(shape=(None,), name='inf_decoder_input_h')
        decoder_state_input_c = Input(shape=(None,), name='inf_decoder_input_c')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_embedding = self.model.get_layer('embedding')
        decoder_lstm = self.model.get_layer('decoder_lstm')
        decoder_lstm_out, state_h_dec, state_c_dec = decoder_lstm(decoder_embedding(decoder_inputs),
                                                                 initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = self.model.get_layer('dense')
        decoder_outputs = decoder_dense(decoder_lstm_out)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        return decoder_model
