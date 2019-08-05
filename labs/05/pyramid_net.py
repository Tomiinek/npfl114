#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import Activation, Concatenate, UpSampling2D, Layer, BatchNormalization, ReLU, Dense, Conv2D, GlobalAveragePooling2D, Input, add

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

from mnist_augmented import MNIST
import math


class ShakeDrop(Layer):

    def __init__(self, level, total_levels, **kwargs):
        super(ShakeDrop, self).__init__(dynamic=True, **kwargs)
        self.pl = 1 - 0.5 * level / total_levels

        self.alpha_bounds = [-1, 1]
        self.beta_bounds = [0, 1]

    def call(self, inputs, training=None):

        x = inputs[0]
        r = inputs[1]

        if training:

            shake_shape = [tf.shape(x)[0], 1, 1, 1]
            bl = tf.math.floor(self.pl + tf.random.uniform(shape=shake_shape))
            alpha = tf.random.uniform(shape=shake_shape, minval=-1, maxval=1)
            beta = tf.random.uniform(shape=shake_shape, minval=0, maxval=1)

            fwd = bl + alpha * (1 - bl)
            bwd = bl + beta * (1 - bl)
            x = x * bwd + tf.stop_gradient(x * fwd - x * bwd)

        else:
            x = self.pl * x

        return r + x

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class PyramidNet(tf.keras.Model):
    def __init__(self, depth, alpha, weight_decay):

        self.input_filters = 16
        self.current_filters = self.input_filters
        # Number of layers in group (minus first and last layer, three groups with three layers)
        self.group_depth = int((depth - 2) / 9)
        self.alpha = alpha
        # Add rate per layer. Three groups and self.n layers (bottlenecks) in a group.
        self.add_rate = self.alpha / (3.0 * self.group_depth)
        self.weight_decay = weight_decay

        inputs, outputs = self._build()
        super().__init__(inputs, outputs)

    def train(self, checkpoint_path, data, batch_size, num_epochs, label_smoothing, learning_rate):
        self.compile(
            optimizer=tf.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True),
            loss=self._loss(label_smoothing),
            metrics=[self._metric(label_smoothing)],
        )
        self.fit_generator(
            generator=data.train.batches(batch_size),
            steps_per_epoch=math.ceil(data.train.size / batch_size),
            epochs=num_epochs,
            validation_data=data.dev.batches(batch_size),
            validation_steps=math.ceil(data.dev.size / batch_size),
            callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True,
                                                          save_weights_only=True)]
    )

    def _bottleneck_block(self, filters, strides, x, level, downsample=None):

        block = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters, 1, strides=1, padding='same', use_bias=False,
                            kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False,
                            kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters * 4, 1, strides=1, padding='same', use_bias=False,
                            kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization()
        ])

        shortcut = downsample(x) if downsample else x
        x = block(x)

        padding = tf.constant([[0,0], [0,0], [0,0], [0, x.shape[3] - shortcut.shape[3]]])
        shortcut = tf.pad(shortcut, padding)

        return ShakeDrop(level, self.group_depth * 3)([x, shortcut])

    def _pyramidal_group(self, x, strides, level):

        downsample = tf.keras.layers.AveragePooling2D(2) if strides != 1 else None

        self.current_filters += self.add_rate
        x = self._bottleneck_block(int(round(self.current_filters)), strides, x, level[0], downsample)
        level[0] += 1
        for i in range(1, self.group_depth):
            self.current_filters += self.add_rate
            x = self._bottleneck_block(int(round(self.current_filters)), 1, x, level[0])
            level[0] += 1
        self.input_filters = int(round(self.current_filters)) * 4

        return x

    def _build(self):

        x = inputs = tf.keras.layers.Input(shape=(MNIST.H, MNIST.W, MNIST.C), dtype=tf.float32)
        x = tf.keras.layers.Conv2D(self.input_filters, 3, strides=1, padding='same', use_bias=False,
                            kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        level = [0]
        x = self._pyramidal_group(x, strides=1, level=level)  # No pooling
        x = self._pyramidal_group(x, strides=2, level=level)  # Puling ve dvi, should be 16x16
        x = self._pyramidal_group(x, strides=2, level=level)  # Puling ve dvi, should be 8x8

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(x)

        return inputs, outputs

    def _loss(self, label_smoothing):
        if label_smoothing == 0: return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)
        
    def _metric(self, label_smoothing):
        if label_smoothing == 0: return tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        return tf.keras.metrics.CategoricalAccuracy(name="accuracy")

