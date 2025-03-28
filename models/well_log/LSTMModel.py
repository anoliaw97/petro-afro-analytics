import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

class LSTMModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            tf.keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        return self.model
