# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf
import math

from tensorflow.keras.layers import BatchNormalization, ReLU, Dense, Conv2D, GlobalAveragePooling2D, Input, add
from tensorflow.keras.regularizers import l2

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

from mnist_augmented import MNIST


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

        x = inputs = Input(shape=(MNIST.H, MNIST.W, MNIST.C), dtype=tf.float32)
        x = Conv2D(filters[0], kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal')(x)

        x = self._block(x, stride=1, depth=block_depth, filters=filters[1])  # No pooling
        x = self._block(x, stride=2, depth=block_depth, filters=filters[2])  # Puling ve dvi, should be 16x16
        x = self._block(x, stride=2, depth=block_depth, filters=filters[3])  # Puling ve dvi, should be 8x8

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(MNIST.LABELS, activation=None)(x)

        return inputs, outputs

    def _block(self, x, stride, depth, filters):
        x = self._downsample_layer(x, filters, stride)
        for _ in range(depth - 1):
            x = self._basic_layer(x, filters)
        return x

    def _basic_layer(self, x, filters):
        block = tf.keras.Sequential([
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
        ])
        return add([block(x), x])

    def _downsample_layer(self, x, filters, stride):
        x = BatchNormalization()(x)
        x = ReLU()(x)

        block = tf.keras.Sequential([
            Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal'),
        ])
        downsample = Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False,
                            kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal')

        return add([block(x), downsample(x)])

    def _loss(self, label_smoothing):
        if label_smoothing == 0: return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)

    def _metric(self, label_smoothing):
        if label_smoothing == 0: return tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        return tf.keras.metrics.CategoricalAccuracy(name="accuracy")