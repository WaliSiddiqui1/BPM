import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# generate sinusoidal positional encoding for sequence input
def get_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis] # shape is (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :] # shape is (1, d_model)

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    # apply sin to even indices and cos to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32) # shape is (1, seq_len, d_model)

# fit a tokenizer on the dataset and limit vocab size
def create_tokenizer(captions, vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(captions)
    return tokenizer

# convert list of captions to padded token sequences
def tokenize_captions(tokenizer, captions, maxlen):
    sequences = tokenizer.texts_to_sequences(captions)
    return pad_sequences(sequences, maxlen=maxlen, padding='post')