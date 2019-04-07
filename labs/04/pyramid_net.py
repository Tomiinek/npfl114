#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

from cifar10_augmented import CIFAR10


class Network(tf.keras.Model):
    def __init__(self, args):

        self.input_filters = 16
        self.current_filters = self.input_filters
        # Number of layers in group (minus first and last layer, three groups with three layers)
        self.group_depth = int((args.depth - 2) / 9)
        self.alpha = args.alpha
        # Add rate per layer. Three groups and self.n layers (bottlenecks) in a group.
        self.add_rate = self.alpha / (3.0 * self.group_depth)

        inputs, outputs = self._build()
        super().__init__(inputs, outputs)

        train_steps = (cifar.train.size // args.batch_size) * args.epochs
        learning_rate = tf.optimizers.schedules.PiecewiseConstantDecay(
            [train_steps / 5],
            [0.01, 0.001]
        )
        self.compile(
            optimizer=tf.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True),
            loss=self._loss(),
            metrics=[self._metric()],
        )

    def _bottleneck_block(self, filters, strides, x, downsample=None):

        block = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters, 1, strides=1, padding='same', use_bias=False,
                            kernel_regularizer=l2(args.weight_decay), kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False,
                            kernel_regularizer=l2(args.weight_decay), kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters * 4, 1, strides=1, padding='same', use_bias=False,
                            kernel_regularizer=l2(args.weight_decay), kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization()
        ])

        shortcut = downsample(x) if downsample else x
        x = block(x)

        padding = tf.constant([[0,0], [0,0], [0,0], [0, x.shape[3] - shortcut.shape[3]]])
        shortcut = tf.pad(shortcut, padding)

        return x + shortcut

    def _pyramidal_group(self, x, strides):

        downsample = tf.keras.layers.AveragePooling2D(2) if strides != 1 else None

        self.current_filters += self.add_rate
        x = self._bottleneck_block(int(round(self.current_filters)), strides, x, downsample)
        for i in range(1, self.group_depth):
            self.current_filters += self.add_rate
            x = self._bottleneck_block(int(round(self.current_filters)), 1, x)
        self.input_filters = int(round(self.current_filters)) * 4

        return x

    def _build(self):

        x = inputs = tf.keras.layers.Input(shape=(CIFAR10.H, CIFAR10.W, CIFAR10.C), dtype=tf.float32)
        x = tf.keras.layers.Conv2D(self.input_filters, 3, strides=1, padding='same', use_bias=False,
                            kernel_regularizer=l2(args.weight_decay), kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = self._pyramidal_group(x, strides=1)  # No pooling
        x = self._pyramidal_group(x, strides=2)  # Puling ve dvi, should be 16x16
        x = self._pyramidal_group(x, strides=2)  # Puling ve dvi, should be 8x8

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(x)

        return inputs, outputs

    def train(self, checkpoint_path, args, cifar):
        self.fit_generator(
            generator=cifar.train.batches(args.batch_size),
            steps_per_epoch=cifar.train.size // args.batch_size,
            epochs=args.epochs,
            validation_data=cifar.dev.batches(args.batch_size),
            validation_steps=cifar.dev.size // args.batch_size,
            callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True)]
        )
        
    def _loss(self):
        if args.label_smoothing == 0: return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing)
        
    def _metric(self):
        if args.label_smoothing == 0: return tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        return tf.keras.metrics.CategoricalAccuracy(name="accuracy")

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=98, type=int, help="Depth.")
    parser.add_argument("--alpha", default=73, type=int, help="Alpha.")
    parser.add_argument("--weight_decay", default=0.0005 / 2, type=int, help="L2 regularization parameter.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0 for no label smoothing")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Evaluation batch size.")
    parser.add_argument("--epochs", default=65, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10(sparse_labels=args.label_smoothing == 0)

    # Create the network and train
    network = Network(args)
    network.load_weights('pyramid_models/modelson_acc=0.9209')
    checkpoint_path = os.path.join("pyramid_models", "modelson_{}".format("acc={val_accuracy:.4f}"))
    network.train(checkpoint_path, args, cifar)