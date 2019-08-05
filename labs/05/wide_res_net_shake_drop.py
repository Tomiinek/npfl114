# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf
import math

from tensorflow.keras.layers import Layer, BatchNormalization, ReLU, Dense, Conv2D, GlobalAveragePooling2D, Input, add
from tensorflow.keras.regularizers import l2

from mnist_augmented import MNIST


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


class WideResNet(tf.keras.Model):
    def __init__(self, depth, width_factor, weight_decay):
        self.depth = depth
        self.width_factor = width_factor
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

    def predict_augmented(self, input, augmentation_loops):
        labels = []
        for image in input:
            ensamble = [MNIST.horizontal_flip(MNIST.translate(image, amount=4)) for _ in range(augmentation_loops)]
            predictions = tf.nn.softmax(self.predict(np.array(ensamble)))
            labels.append(np.sum(predictions, axis=0))

        return np.array(labels)

    def _build(self):
        filters = [16, 1*16 * self.width_factor, 2*16 * self.width_factor, 4*16 * self.width_factor]
        block_depth = (self.depth - 4) // (3 * 2)
        total_block_depth = block_depth * 3

        x = inputs = Input(shape=(MNIST.H, MNIST.W, MNIST.C), dtype=tf.float32)
        x = Conv2D(filters[0], kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal')(x)

        level = [0]
        x = self._block(x, stride=1, depth=block_depth, filters=filters[1], level=level, total_levels=total_block_depth)
        x = self._block(x, stride=2, depth=block_depth, filters=filters[2], level=level, total_levels=total_block_depth)
        x = self._block(x, stride=2, depth=block_depth, filters=filters[3], level=level, total_levels=total_block_depth)

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(MNIST.LABELS, activation=None)(x)

        return inputs, outputs

    def _block(self, x, stride, depth, filters, level, total_levels):
        x = self._downsample_layer(x, filters, stride, level[0], total_levels)
        level[0] += 1
        for _ in range(depth - 1):
            x = self._basic_layer(x, filters, level[0], total_levels)
            level[0] += 1
        return x

    def _basic_layer(self, x, filters, level, total_levels):
        block = tf.keras.Sequential([
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
            BatchNormalization()
        ])
        return ShakeDrop(level, total_levels)([block(x), x])

    def _downsample_layer(self, x, filters, stride, level, total_levels):
        x = BatchNormalization()(x)
        x = ReLU()(x)

        block = tf.keras.Sequential([
            Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
            BatchNormalization()
        ])
        downsample = Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False,
                            kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal')

        return ShakeDrop(level, total_levels)([block(x), downsample(x)])

    def _loss(self, label_smoothing):
        if label_smoothing == 0: return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)

    def _metric(self, label_smoothing):
        if label_smoothing == 0: return tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        return tf.keras.metrics.CategoricalAccuracy(name="accuracy")