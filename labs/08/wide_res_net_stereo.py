#!/usr/bin/env python3
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras.layers import BatchNormalization, ReLU, Dense, Conv2D, GlobalAveragePooling2D, Input
from tensorflow.keras.regularizers import l2

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

from modelnet2D import ModelNet


class WideResNet(tf.keras.Model):
    def __init__(self):
        inputs, outputs = self._build()
        super().__init__(inputs, outputs)

        train_step = cifar.train.size / args.batch_size
        learning_rate = tf.optimizers.schedules.PiecewiseConstantDecay(
            [train_step * 60, train_step * 120, train_step * 160],
            [0.1*(0.2**i) for i in range(4)]
        )
        self.compile(
            optimizer=tf.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True),
            loss=self._loss(),
            metrics=[self._metric()],
        )

    def train(self, checkpoint_path, args, cifar):
        self.fit_generator(
            generator=cifar.train.batches(args.batch_size, repeat=True),
            steps_per_epoch=cifar.train.size // args.batch_size,
            epochs=args.epochs,
            validation_data=cifar.dev.batches(args.batch_size, repeat=True),
            validation_steps=(cifar.dev.size + args.batch_size - 1) // args.batch_size,
            callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True)]
        )

    def evaluate(self, ds, args):
        probas = np.zeros(shape=(ds.size, 10), dtype=np.float32)
        for epoch_i in range(args.eval_epochs):
            for batch_i, (batch, _) in tqdm(
                enumerate(ds.batches(args.batch_size, do_augment=(args.eval_epochs != 1 and epoch_i != 0))),
                leave=False, desc=f"{epoch_i + 1}/{args.eval_epochs}"
            ):
                begin = (batch_i * args.batch_size)
                end = begin + len(batch[0])
                probas[begin:end] += self.predict_on_batch(batch)
        
        return probas / args.eval_epochs

    def _build(self):
        filters = [32, 1*16 * args.spek_factor, 2*16 * args.spek_factor, 4*16 * args.spek_factor]
        depth = (args.depth - 4) // (3*2)

        input_x = Input(shape=(32, 32, 3), dtype=tf.float32)
        input_y = Input(shape=(32, 32, 3), dtype=tf.float32)
        input_z = Input(shape=(32, 32, 3), dtype=tf.float32)
        x = tf.keras.layers.Concatenate()([input_x, input_y, input_z])
        x = Conv2D(filters[0], kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(args.weight_decay), kernel_initializer='he_normal')(x)

        x = self._block(x, stride=1, depth=depth, filters=filters[1])  # No pooling
        x = self._block(x, stride=2, depth=depth, filters=filters[2])  # Puling ve dvi, should be 16x16
        x = self._block(x, stride=2, depth=depth, filters=filters[3])  # Puling ve dvi, should be 8x8

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(10, activation=None)(x) # dont use regularizer here (for some strange reason)

        return [input_x, input_y, input_z], outputs

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
                   kernel_regularizer=l2(args.weight_decay), kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(args.weight_decay), kernel_initializer='he_normal'),
        ])
        return block(x) + x

    def _downsample_layer(self, x, filters, stride):
        x = BatchNormalization()(x)
        x = ReLU()(x)

        block = tf.keras.Sequential([
            Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False,
                   kernel_regularizer=l2(args.weight_decay), kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                   kernel_regularizer=l2(args.weight_decay), kernel_initializer='he_normal'),
        ])
        downsample = Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False,
                            kernel_regularizer=l2(args.weight_decay), kernel_initializer='he_normal')

        return block(x) + downsample(x)

    def _loss(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def _metric(self):
        return tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train or just predict?")
    parser.add_argument("--eval_epochs", default=1, type=int, help="Number of samples to use in prediction")
    parser.add_argument("--model_path", default='wideresnet_models/tlustoprd_28-10_acc=0.9702', type=str, help="Path to weights of a model.")
    parser.add_argument("--output_path", default='dev_out.txt', type=str, help="Path to test predictions.")
    parser.add_argument("--depth", default=28, type=int, help="Depth of the network.")
    parser.add_argument("--spek_factor", default=10, type=int, help="Widening factor over classical resnet.")
    # the weight decay is divided by two because: https://bbabenko.github.io/weight-decay/
    parser.add_argument("--weight_decay", default=0.0005 / 2, type=int, help="L2 regularization parameter.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Evaluation batch size.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load data
    cifar = ModelNet(32)

    # Create the network and train or evaluate
    network = WideResNet()
    
    if args.train:
        checkpoint_path = os.path.join("wideresnet_models_stereo", "tlustoprd_28-10_{}".format("acc={val_accuracy:.4f}"))
        network.train(checkpoint_path, args, cifar)

    network.load_weights(args.model_path)
    dev_probas = network.evaluate(cifar.dev, args)

    expectation = np.mean(np.equal(cifar.dev.data['labels'], np.argmax(dev_probas, axis=-1)))
    print(expectation)
    
    test_probas = network.evaluate(cifar.test, args)
    
    np.save(args.model_path + f'_exp={expectation:.4f}_dev.npy', dev_probas)
    np.save(args.model_path + f'_exp={expectation:.4f}_test.npy', test_probas)

    with open(args.model_path + f'_exp={expectation:.4f}_out.txt', "w", encoding="utf-8") as out_file:
        print(*list(np.argmax(test_probas, axis=-1)), file=out_file, sep='\n')
    
        