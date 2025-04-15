import tensorflow as tf
from tensorflow.keras import layers

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.1)

    def call(self, x, training):
        attn_out = self.att(x, x)
        x = self.norm1(x + self.dropout(attn_out, training=training))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out, training=training))

class VideoCaptionModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.input_proj = layers.Dense(config.EMBED_DIM)
        self.encoder = [TransformerEncoderBlock(config.EMBED_DIM, config.NUM_HEADS, config.FF_DIM) 
                        for _ in range(config.NUM_LAYERS)]
        self.decoder_lstm = layers.LSTM(512, return_sequences=True, return_state=True)
        self.output_layer = layers.Dense(config.VOCAB_SIZE)

    def call(self, frames, captions):
        x = self.input_proj(frames)
        for block in self.encoder:
            x = block(x)
        x = tf.reduce_mean(x, axis=1)  # Global pooled context
        repeated = tf.repeat(tf.expand_dims(x, 1), tf.shape(captions)[1], axis=1)
        decoder_input = tf.concat([repeated, captions], axis=-1)
        lstm_out, _, _ = self.decoder_lstm(decoder_input)
        return self.output_layer(lstm_out)
