# step 1: data preparation
""" We create a small dataset with text examples, also we have added an label list where the 1 label is for positive
    ones like 'I love machine learning', and the 0 one is for positive statements like 'I hate bugs'"""
texts = [
    'I love machine learning',
    'Deep learning is amazing',
    'I hate bugs',
    'Programming is fun',
    'I love programming',
    'Bugs are terrible'
]

labels = [1,1,0,1,1,0]

# step 2: tokenization of text
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)


# step 3: padding sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences

X = pad_sequences(sequences, padding = 'post', maxlen = 10)

# step 4: splitting data into Training and Testing Sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)

# define the rnn model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

def model_rnn(vocab_size = 14, seq_length = 10, num_units = 32):
    model = Sequential([

        Embedding(input_dim = vocab_size, output_dim = 16, input_length = seq_length),
        LSTM(num_units, return_sequences = False),
        Dense(64, activation = 'relu'),
        Dense(1, activation = 'sigmoid')

    ])

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

# train the model
model = model_rnn()
model.summary()

import numpy as np
model.fit(X_train, np.array(y_train), epochs = 10, batch_size = 2, validation_data = (X_test, np.array(y_test)))

# evaluate the model
loss, accuracy = model.evaluate(X_test, np.array(y_test))
print(f"Test Accuracy: {accuracy * 100:.2f}%")


