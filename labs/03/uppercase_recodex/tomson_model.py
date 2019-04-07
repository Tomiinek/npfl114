#!/usr/bin/env python3
import datetime
import os
import re

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from uppercase_data import UppercaseData

from bayes_opt import BayesianOptimization
from bayes_opt.observer import ScreenLogger, JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events


class Network:
    def __init__(self, seed=42):
        graph = tf.Graph()
        graph.seed = seed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=graph, config=config)

    def construct(self, window, alphabet_size, dropout, learning_rate, learning_rate_final, label_smoothing, epochs, decay_steps):
        with self.session.graph.as_default():
            self.windows = tf.placeholder(tf.int32, [None, 2 * window + 1], name="windows")
            self.labels = tf.placeholder(tf.int32, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            embeddings = tf.get_variable("s_embeddings", [alphabet_size, alphabet_size // 3],
                                         dtype=tf.float32)
            embedded = tf.nn.embedding_lookup(embeddings, self.windows)
            flattened = tf.layers.flatten(embedded, name="flatten")

            hidden_layer_1 = tf.layers.dense(flattened, 2048, activation=tf.nn.relu, name="hidden_layer")
            tmp = tf.layers.dropout(hidden_layer_1, dropout, training=self.is_training)

            output_layer = tf.layers.dense(tmp, 2, activation=None, name="output_layer")

            loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.labels, 2), output_layer, label_smoothing=label_smoothing, scope="loss")
            self.predictions = tf.argmax(output_layer, axis=1, name="activations", output_type=tf.int32)

            global_step = tf.train.create_global_step()
            if learning_rate_final:
                decay_rate = (learning_rate_final / learning_rate) ** (1 / (float(epochs) - 1))
                learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps,
                                                           decay_rate, staircase=True)
            else:
                learning_rate = learning_rate

            self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                           name="training")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            self.session.run(tf.global_variables_initializer())

    def train(self, windows, labels):
        self.session.run([self.training], {self.is_training: True, self.windows: windows, self.labels: labels})

    def evaluate(self, windows, labels):
        a, p = self.session.run([self.accuracy, self.predictions], {self.is_training: False, self.windows: windows, self.labels: labels})
        return a, p

    def close(self):
        self.session.close()


def black_box_function(window, alphabet_size, dropout, learning_rate, learning_rate_final, label_smoothing):

    window = int(window)
    alphabet_size = int(alphabet_size)
    # embedding_size = int(embedding_size)
    # layers_size = int(layers_size)
    if learning_rate < learning_rate_final: learning_rate_final = learning_rate
    learning_rate = 10**learning_rate
    learning_rate_final = 10**learning_rate_final

    epochs = 8
    batch_size = 1024
    batch_size_dev = 2048

    uppercase_data = UppercaseData(window, alphabet_size)

    network = Network()
    network.construct(window, alphabet_size, dropout, learning_rate, learning_rate_final, label_smoothing, epochs, uppercase_data.train.size // batch_size)

    for i in range(epochs):
        for batch in uppercase_data.train.batches(batch_size):
            a = network.train(batch["windows"], batch["labels"])
            # print("Training {}: {:.2f}".format(i + 1, 100 * a))

    accuracy = 0.0
    batch_count = 0
    for batch in uppercase_data.dev.batches(batch_size_dev):
        a, _ = network.evaluate(batch["windows"], batch["labels"])
        accuracy += a * len(batch["windows"])
        batch_count += len(batch["windows"])
    accuracy /= batch_count

    network.close()

    return accuracy


if __name__ == "__main__":

    np.random.seed(42)

    params = {
        'alphabet_size': (40, 120),
        'dropout': (0.1, 0.5),
        # 'embedding_size': (75, 130),
        'label_smoothing': (0.05, 0.5),
        # 'layers_size': (512, 1024),
        'learning_rate': (-4, -2),
        'learning_rate_final': (-6, -3),
        'window': (7, 11),
    }

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=params,
        random_state=1,
    )

    load_logs(optimizer, logs=["./logs2.json"])

    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
    optimizer.subscribe(Events.OPTMIZATION_STEP, ScreenLogger())

    optimizer.maximize(
        init_points=max(0, 8 - len(optimizer.space)),
        n_iter=max(0, 64 - len(optimizer.space))
    )
