import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Embedding
from tensorflow.keras.layers.experimental import LayerNormalization


class UniformAttentionSublayer(Layer):
    def __init__(self, dimension, heads, **kwargs):
        super(UniformAttentionSublayer, self).__init__(**kwargs)

        self.dimension = dimension
        self.heads = heads
        self.scale = tf.constant(math.sqrt(dimension / heads))

        self.input_transform = Dense(3*dimension, activation=None)
        self.output_transform = Dense(dimension, activation=None)

    def call(self, input, mask, training=None):
        QKV = self.input_transform(input)
        QKV = tf.reshape(QKV, (tf.shape(QKV)[0], -1, self.heads, 3 * self.dimension // self.heads))
        QKV = tf.transpose(QKV, perm=[0, 2, 1, 3])
        Q, K, V = tf.split(QKV, 3, axis=3)

        indices = tf.linalg.matmul(Q, K, transpose_b=True) / self.scale
        if mask is not None: indices += (mask * -1e9)
        indices = tf.nn.softmax(indices)
        combined = tf.linalg.matmul(indices, V)

        heads_concat = tf.transpose(combined, perm=[0, 2, 1, 3])
        heads_concat = tf.reshape(heads_concat, (tf.shape(heads_concat)[0], -1, self.dimension))

        return self.output_transform(heads_concat)


class DividedAttentionSublayer(Layer):
    def __init__(self, dimension, heads, **kwargs):
        super(DividedAttentionSublayer, self).__init__(**kwargs)

        self.dimension = dimension
        self.heads = heads
        self.scale = tf.constant(math.sqrt(dimension / heads))

        self.input_transform_q = Dense(dimension, activation=None)
        self.input_transform_k = Dense(dimension, activation=None)
        self.input_transform_v = Dense(dimension, activation=None)
        self.output_transform = Dense(dimension, activation=None)

    def _split_heads(self, x):
        x = tf.reshape(x, (tf.shape(x)[0], -1, self.heads, self.dimension // self.heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x_q, x_k, x_v, mask, training=None):
        Q = self._split_heads(self.input_transform_q(x_q))
        K = self._split_heads(self.input_transform_k(x_k))
        V = self._split_heads(self.input_transform_v(x_v))

        indices = tf.linalg.matmul(Q, K, transpose_b=True) / self.scale
        if mask is not None: indices += (mask * -1e9)
        indices = tf.nn.softmax(indices)
        combined = tf.linalg.matmul(indices, V)

        heads_concat = tf.transpose(combined, perm=[0, 2, 1, 3])
        heads_concat = tf.reshape(heads_concat, (tf.shape(heads_concat)[0], -1, self.dimension))

        return self.output_transform(heads_concat)


class EncoderLayer(Layer):
    def __init__(self, dimension, heads, dropout, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.dimension = dimension
        self.heads = heads
        self.dropout = dropout

        self.attention = UniformAttentionSublayer(dimension, heads)
        self.dropout_1 = Dropout(dropout)
        self.layer_norm_1 = LayerNormalization(epsilon=1e-6)

        self.nonlinear_sublayer = tf.keras.Sequential([
            Dense(4*dimension, activation=tf.nn.relu),
            Dense(1*dimension, activation=None),
            Dropout(dropout)
        ])
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6)

    def call(self, x, mask, training=None):
        attention = self.attention(x, mask)
        attention = self.dropout_1(attention, training=training)
        x = self.layer_norm_1(attention + x)

        nonlinear = self.nonlinear_sublayer(x, training=training)
        return self.layer_norm_2(nonlinear + x)


class DecoderLayer(Layer):
    def __init__(self, dimension, heads, dropout, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        self.dimension = dimension
        self.heads = heads
        self.dropout = dropout

        self.attention_1 = UniformAttentionSublayer(dimension, heads)
        self.dropout_1 = Dropout(dropout)
        self.layer_norm_1 = LayerNormalization(epsilon=1e-6)

        self.attention_2 = DividedAttentionSublayer(dimension, heads)
        self.dropout_2 = Dropout(dropout)
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6)

        self.nonlinear_sublayer = tf.keras.Sequential([
            Dense(4*dimension, activation=tf.nn.relu),
            Dense(1*dimension, activation=None),
            Dropout(dropout)
        ])
        self.layer_norm_3 = LayerNormalization(epsilon=1e-6)

    def call(self, encoder_output, input, look_ahead_mask, padding_mask, training=None):
        attention = self.attention_1(input, look_ahead_mask)
        attention = self.dropout_1(attention, training=training)
        x = self.layer_norm_1(attention + input)

        attention = self.attention_2(x, encoder_output, encoder_output, padding_mask)
        attention = self.dropout_2(attention, training=training)
        x = self.layer_norm_2(attention + x)

        nonlinear = self.nonlinear_sublayer(x, training=training)
        return self.layer_norm_3(nonlinear + x)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(1000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return pos_encoding[np.newaxis, ...]


class Encoder(Layer):
    def __init__(self, num_chars, dimension, heads, layers, dropout):
        super(Encoder, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.dropout = dropout
        self.scale = tf.constant(math.sqrt(dimension))

        self.embedding = Embedding(num_chars, dimension)
        self.positional_encoding = positional_encoding(1000, dimension)
        self.dropout = Dropout(dropout)

        self.encoding = [EncoderLayer(dimension, heads, dropout) for _ in range(layers)]

    def call(self, x, mask, training=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) * self.scale + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for encoding in self.encoding:
            x = encoding(x, mask, training=training)

        return x


class Decoder(Layer):
    def __init__(self, num_chars, dimension, heads, layers, dropout):
        super(Decoder, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.dropout = dropout
        self.scale = tf.constant(math.sqrt(dimension))

        self.embedding = Embedding(num_chars, dimension)
        self.positional_encoding = positional_encoding(1000, dimension)
        self.dropout = Dropout(dropout)

        self.decoding = [DecoderLayer(dimension, heads, dropout) for _ in range(layers)]

    def call(self, encoder_output, x, look_ahead_mask, padding_mask, training=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) * self.scale + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for decoding in self.decoding:
            x = decoding(encoder_output, x, look_ahead_mask, padding_mask, training=training)

        return x