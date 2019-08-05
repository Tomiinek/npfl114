#!/usr/bin/env python3

# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import contextlib

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# from timit_mfcc_ordering import TimitMFCC
from timit_mfcc import TimitMFCC
from sgdr_learning_rate import SGDRLearningRate

session = tf.Session(
    config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        inter_op_parallelism_threads=4,
        intra_op_parallelism_threads=4
    ))
tf.keras.backend.set_session(session)


def edit_distance(y_true, y_pred):
    return tf.edit_distance(y_true, y_pred, normalize=True)


def data_generator(dataset, batch_size):
    while True:
        for batch in dataset.batches(batch_size):
            yield (
                [batch["mfcc"], batch["mfcc_len"], batch["letters"]],
                {'ctc': batch["mfcc_len"], 'edit_distance': batch["mfcc_len"]})


class Network:
    @staticmethod
    def residual_block(x, dilation, args):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('tanh')(x)
        x = tf.keras.layers.SpatialDropout1D(args.dropout)(x)

        f = tf.keras.layers.Conv1D(
            args.filters, 7, padding='same', dilation_rate=dilation, kernel_initializer='he_normal', use_bias=False)(x)
        f = tf.keras.layers.BatchNormalization()(f)
        f = tf.keras.layers.Activation('sigmoid')(f)

        g = tf.keras.layers.Conv1D(
            args.filters, 7, padding='same', dilation_rate=dilation, kernel_initializer='he_normal', use_bias=False)(x)
        g = tf.keras.layers.BatchNormalization()(g)
        g = tf.keras.layers.Activation('tanh')(g)

        x = tf.keras.layers.Multiply()([f, g])
        x = tf.keras.layers.SpatialDropout1D(args.dropout)(x)

        x = tf.keras.layers.Conv1D(
            args.filters, 1, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        return x

    def __init__(self, args):
        self.best_distance = 1.0

        # TODO: Define a suitable model, given already masked `mfcc` with shape
        # `[None, TimitMFCC.MFCC_DIM]`. The last layer should be a Dense layer
        # without an activation and with `len(TimitMFCC.LETTERS) + 1` outputs,
        # where the `+ 1` is for the CTC blank symbol.
        #
        # Store the results in `self.model`.
        x = inputs = tf.keras.Input(shape=(None, TimitMFCC.MFCC_DIM), dtype=tf.float32)
        x = shortcut = tf.keras.layers.Conv1D(args.filters, 1, kernel_initializer='he_normal', use_bias=False)(x)

        global_shortcut = None
        for block_i in range(args.blocks):
            for dilation in [1, 2, 4, 8, 16]:
                x = self.residual_block(x, dilation, args)
                global_shortcut = x if (global_shortcut is None) else tf.keras.layers.Add()([global_shortcut, x])
                x = shortcut = tf.keras.layers.Add()([shortcut, x])

        x = tf.keras.layers.BatchNormalization()(global_shortcut)
        x = tf.keras.layers.Activation('tanh')(x)
        x = tf.keras.layers.SpatialDropout1D(args.dropout)(x)
        x = tf.keras.layers.Conv1D(args.filters, 1, kernel_initializer='he_normal', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('tanh')(x)
        x = tf.keras.layers.SpatialDropout1D(args.dropout)(x)
        dense = tf.keras.layers.Dense(len(TimitMFCC.LETTERS) + 1, activation=None)(x)
        logits = tf.keras.layers.Lambda(lambda x: tf.transpose(x, [1, 0, 2]))(dense)

        inputs_len = tf.keras.Input(shape=tuple(), dtype=tf.int32)
        predictions = tf.keras.layers.Lambda(
            lambda inps: tf.cast(tf.nn.ctc_beam_search_decoder_v2(inps[0], inps[1], beam_width=args.ctc_beam)[0][0],
                                 tf.int32),
            name="predictions")([logits, inputs_len])
        x = tf.keras.layers.Lambda(lambda x: tf.sparse.to_dense(x))(predictions)

        self.pred_model = tf.keras.Model(inputs=[inputs, inputs_len], outputs=[x, dense])

        targets = tf.keras.Input(shape=(None,), dtype=tf.int32)
        indices = tf.keras.layers.Lambda(
            lambda x: tf.where(tf.not_equal(x, 0)))(targets)
        sparse_targets = tf.keras.layers.Lambda(
            lambda inps: tf.sparse.SparseTensor(inps[0], tf.gather_nd(inps[1], inps[0]),
                                                tf.shape(inps[1], out_type=tf.int64))
        )([indices, targets])
        loss = tf.keras.layers.Lambda(
            lambda inps: tf.nn.ctc_loss_v2(inps[0], inps[1], None, inps[2], blank_index=len(TimitMFCC.LETTERS)),
            name='ctc'
        )([sparse_targets, logits, inputs_len])
        edit_distance = tf.keras.layers.Lambda(
            lambda inps: tf.stop_gradient(
                tf.reshape(tf.edit_distance(inps[1], inps[0], normalize=True), tf.shape(inps[0])[0:1])),
            name='edit_distance'
        )([sparse_targets, predictions])
        self.train_model = tf.keras.Model(inputs=[inputs, inputs_len, targets], outputs=[loss, edit_distance])

    def predict(self, dataset, args):
        max_sentence_len = max(sentence.shape[0] for sentence in dataset.data["mfcc"])
        output = np.zeros((dataset.size, max_sentence_len, len(TimitMFCC.LETTERS) + 1))

        begin = 0
        sentences = []
        for batch in dataset.batches(args.batch_size):
            predictions, logits = self.pred_model.predict_on_batch(
                [batch["mfcc"], batch["mfcc_len"]])

            for prediction in predictions:
                prediction_len = (prediction != 0).sum()
                sentences.append(prediction[:prediction_len])

            end = begin + batch["mfcc"].shape[0]
            output[begin:end, :np.max(batch["mfcc_len"]), :] = logits
            begin = end

        return sentences, output

    def evaluate(self, dataset, args):
        self.train_model.compile(
            loss={'ctc': lambda y_true, y_pred: y_pred,
                  'edit_distance': lambda y_true, y_pred: y_pred},
            optimizer=tf.keras.optimizers.Adam(),
            loss_weights={'ctc': 1.0, 'edit_distance': 0.0})

        val_steps = (dataset.size + args.batch_size - 1) // args.batch_size
        val_generator = data_generator(dataset, args.batch_size)
        self.train_model.evaluate_generator(val_generator, val_steps, verbose=1)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--ctc_beam", default=16, type=int, help="CTC beam.")
    parser.add_argument("--dropout", default=0.42, type=float, help="Cell input dropout")  # or drop by layers?
    parser.add_argument("--epochs", default=64, type=int, help="Number of epochs.")
    parser.add_argument("--blocks", default=3, type=int, help="Number of dilated blocks.")
    parser.add_argument("--filters", default=128, type=int, help="Number of filters in CNNs.")
    parser.add_argument("--threads", default=5, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--save_logits", action="store_true")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)

    # Load the data
    timit = TimitMFCC()

    # Create the network and train
    network = Network(args)
    network.train_model.load_weights(args.checkpoint)
    network.evaluate(timit.dev, args)

    # Generate test set annotations, but to allow parallel execution, create it
    # in in args.logdir if it exists.
    out_path = args.checkpoint + "_speech_recognition_test.txt"
    with open(out_path, "w", encoding="utf-8") as out_file:
        sentences, logits = network.predict(timit.test, args)

        if args.save_logits:
            np.save(args.checkpoint + "_logits_test", logits)

        for sentence in sentences:
            print(" ".join(timit.LETTERS[letters] for letters in sentence), file=out_file)

    # Generate test set annotations, but to allow parallel execution, create it
    # in in args.logdir if it exists.
    out_path = args.checkpoint + "_speech_recognition_dev.txt"
    with open(out_path, "w", encoding="utf-8") as out_file:
        sentences, logits = network.predict(timit.dev, args)

        if args.save_logits:
            np.save(args.checkpoint + "_logits_dev", logits)

        for sentence in sentences:
            print(" ".join(timit.LETTERS[letters] for letters in sentence), file=out_file)