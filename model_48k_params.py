"""
Recurrent Neural Networks (RNNs) handle sequential data, where past information influences future predictions.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
import numpy as np
from tensorflow.keras.models import Sequential

def rnn_model(vocab_size=500, seq_length=10, num_units=32):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=16, input_length=seq_length),  # Embedding layer
        LSTM(num_units, return_sequences=False),  # LSTM layer
        Dense(64, activation='relu'),  # Dense layer
        Dense(vocab_size, activation='softmax')  # Output layer
    ])
    
    # build the model 
    model.build((None, seq_length))
    return model

rnn_model_1 = rnn_model()

rnn_model_1.summary()
