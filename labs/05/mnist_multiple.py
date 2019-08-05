#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Flatten, Dense, Conv2D, Input
from tensorflow.keras.models import Model

from mnist import MNIST

# The neural network model
class Network:
    def __init__(self, args):

        image_a = Input(shape=(MNIST.H, MNIST.W, MNIST.C), dtype=tf.float32)
        image_b = Input(shape=(MNIST.H, MNIST.W, MNIST.C), dtype=tf.float32)

        embedding_network = self._build()
        embedding_a = embedding_network(image_a)
        embedding_b = embedding_network(image_b)

        class_layer = Dense(10, activation='softmax')
        class_prediction_a = class_layer(embedding_a)
        class_prediction_b = class_layer(embedding_b)
        a_b_concat = Concatenate()([embedding_a, embedding_b])
        embedding_a_b = Dense(200, activation='relu')(a_b_concat)
        order_prediction = Dense(1, activation='sigmoid')(embedding_a_b)

        self.model = Model(inputs=[image_a, image_b], outputs=[class_prediction_a, class_prediction_b, order_prediction])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy'],
            metrics=['sparse_categorical_accuracy', 'sparse_categorical_accuracy', 'binary_accuracy'],
        )

    @staticmethod
    def _build():
        return tf.keras.Sequential([
            Conv2D(10, kernel_size=3, strides=2, padding='valid', activation='relu'),
            Conv2D(20, kernel_size=3, strides=2, padding='valid', activation='relu'),
            Flatten(),
            Dense(200, activation='relu'),
        ])

    @staticmethod
    def _prepare_batches(batches_generator):
        batches = []
        for batch in batches_generator:
            batches.append(batch)
            if len(batches) >= 2:
                yield ([batches[0]["images"], batches[1]["images"]],
                       [batches[0]["labels"], batches[1]["labels"], batches[0]["labels"] > batches[1]["labels"]])
                batches.clear()

    def train(self, mnist, args):
        for epoch in range(args.epochs):
            for x, y in self._prepare_batches(mnist.train.batches(args.batch_size)):
                self.model.train_on_batch(x, y)
            # Print development evaluation
            print("Dev {}: directly predicting: {:.4f}, comparing digits: {:.4f}".format(epoch + 1, *self.evaluate(mnist.dev, args)))

    def evaluate(self, dataset, args):
        da = ia = 0
        for inputs, targets in self._prepare_batches(dataset.batches(args.batch_size)):
            predictions = self.model.predict_on_batch(inputs)
            direct_predictions = np.squeeze(predictions[2] > 0.5)
            indirect_predictions = np.argmax(predictions[0], axis=1) > np.argmax(predictions[1], axis=1)
            da += np.sum(direct_predictions == targets[2])
            ia += np.sum(indirect_predictions == targets[2])
        return da / (dataset.size // 2), ia / (dataset.size // 2)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)
    with open("mnist_multiple.out", "w") as out_file:
        direct, indirect = network.evaluate(mnist.test, args)
        print("{:.2f} {:.2f}".format(100 * direct, 100 * indirect), file=out_file)
