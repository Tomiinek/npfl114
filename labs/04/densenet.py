#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from tensorflow.keras.regularizers import l2

# TF2.0
# tf.config.gpu.set_per_process_memory_growth(True)
# tf.config.threading.set_inter_op_parallelism_threads(4)
# tf.config.threading.set_intra_op_parallelism_threads(4)

# TF1.13
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.intra_op_parallelism_threads = 5 
config.inter_op_parallelism_threads = 5 

sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

from cifar10_augmented import CIFAR10


class Network(tf.keras.Model):
    def __init__(self, args):
        self.depth = args.depth
        self.k = args.k
        self.bottlenecks = args.bottlenecks
        self.compression = args.compression
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
        
        self.block_count = 3
        layers_per_block = ((self.depth - 1) // self.block_count) - 1
        self.block_depth = layers_per_block // (2 if self.bottlenecks else 1)
        assert ((2 if self.bottlenecks else 1) * self.block_depth) == layers_per_block
        assert ((layers_per_block + 1) * self.block_count + 1) == self.depth
        self.init_filters = (2 * self.k) if (self.bottlenecks and self.compression) else 16
        
        inputs, outputs = self._build()
        super().__init__(inputs, outputs)

        train_steps = args.epochs * cifar.train.size / args.batch_size
        global_step = tf.train.create_global_step()
        boundaries = [int(train_steps / 2), int(train_steps * 3 / 4)]
        values = [0.1, 0.01, 0.001]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        
        self.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True),
            loss=tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=args.label_smoothing),
            metrics=[self._metric()],
        )

    def train(self, checkpoint_path, args, cifar):
        self.fit_generator(
            generator=cifar.train.batches(args.batch_size),
            steps_per_epoch=cifar.train.size // args.batch_size,
            epochs=args.epochs,
            validation_data=cifar.dev.batches(args.batch_size),
            validation_steps=cifar.dev.size // args.batch_size,
            callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True)]
        )

    def _composite_fnc(self, x, filters, kernel_size):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='same', use_bias=False,
            kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal')(x)
        if self.dropout is not None:
            x = tf.keras.layers.Dropout(self.dropout)(x)
        return x
    
    def _transition_layer(self, x, is_last):
        if is_last:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
        else:
            compressed_depth = (x.shape[-1] // 2) if self.compression else x.shape[-1]
            # x = self._composite_fnc(x, compressed_depth, 1)
            x = self._composite_fnc(x, int(compressed_depth), 1)
            x = tf.keras.layers.AveragePooling2D()(x)
        return x
    
    def _dense_block(self, x):
        for _ in range(self.block_depth):
            shortcut = x
            if self.bottlenecks:
                x = self._composite_fnc(x, 4 * self.k, 1)
            x = self._composite_fnc(x, self.k, 3)
            x = tf.keras.layers.Concatenate()([shortcut, x])
        return x
    
    def _build(self):
        x = inputs = tf.keras.layers.Input(shape=(None, None, 3), dtype=tf.float32)
        x = tf.keras.layers.Conv2D(
            self.init_filters, 3, padding='same', use_bias=False,
            kernel_regularizer=l2(self.weight_decay), kernel_initializer='he_normal')(x)
        for i in range(self.block_count):
            x = self._dense_block(x)
            x = self._transition_layer(x, i == (self.block_count - 1))
        x = tf.keras.layers.Dense(CIFAR10.LABELS, activation=None)(x)
        
        return inputs, x

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
    parser.add_argument("--depth", default=190, type=int, help="Depth of the network.")
    parser.add_argument("--k", default=40, type=int, help="Growth rate.")
    parser.add_argument("--bottlenecks", action="store_true", help="Use bottleneck layers")
    parser.add_argument("--compression", action="store_true", help="Compress channels on transition layers")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate to use.")
    # the weight decay is divided by two because: https://bbabenko.github.io/weight-decay/
    parser.add_argument("--weight_decay", default=0.0001 / 2, type=int, help="L2 regularization parameter.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0 for no label smoothing")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Evaluation batch size.")
    parser.add_argument("--epochs", default=300, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=5, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
#     tf.random.set_seed(42)
#     tf.config.threading.set_inter_op_parallelism_threads(args.threads)
#     tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load data
    cifar = CIFAR10(sparse_labels=(args.label_smoothing == 0))

    # Create the network and train
    network = Network(args)
    checkpoint_path = os.path.join("densenet_models", "d={}_k={}_{}{}_{}".format(
        args.depth, args.k, "B" if args.bottlenecks else "", "C" if args.compression else "",
        "acc={val_accuracy:.4f}"))
    network.train(checkpoint_path, args, cifar)