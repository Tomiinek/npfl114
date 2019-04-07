#!/usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.observer import JSONLogger, ScreenLogger
from bayes_opt.util import load_logs

from uppercase_data_diakritika import UppercaseDataDiakritika

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--activation", default="relu", type=str, help="Activation function.")
parser.add_argument("--batch_size", default=1024, type=int, help="Batch size.")
parser.add_argument("--decay", default='exponential', type=str, help="Learning decay rate type")
parser.add_argument("--epochs", default=8, type=int, help="Number of epochs.")
parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer to use.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()

# Helper methods
activation_dict = {
    "none": None,
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
    "sigmoid": tf.nn.sigmoid,
    "elu": tf.nn.elu,
    "leaky_relu": tf.nn.leaky_relu
}


def learning_rate(decay, train_size, lr, lr_final):
    decay_steps = args.epochs * train_size // args.batch_size
    if decay is None:
        return lr
    elif decay == 'polynomial':
        return tf.optimizers.schedules.PolynomialDecay(lr, decay_steps, lr_final)
    elif decay == 'exponential':
        return tf.optimizers.schedules.ExponentialDecay(lr, decay_steps, lr_final / lr)


def sgd_optimizer(type, train_size, lr, lr_final):
    if type == 'SGD':
        return tf.optimizers.SGD(learning_rate=learning_rate(args.decay, train_size, lr, lr_final),
                                 momentum=0.0 if args.momentum is None else args.momentum)
    elif type == 'Adam':
        return tf.optimizers.Adam(learning_rate=learning_rate(args.decay, train_size, lr, lr_final))
    elif type == 'Nadam':
        return tf.optimizers.Nadam(learning_rate=learning_rate(args.decay, train_size, lr, lr_final))


def sparse_to_distribution(labels, label_smoothing):
    if label_smoothing == 0: return labels
    return np.eye(2)[labels]


def loss(label_smoothing):
    if label_smoothing == 0: return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)

def metric(label_smoothing):
    if label_smoothing == 0: return tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    return tf.keras.metrics.CategoricalAccuracy(name="accuracy")


# Black-box function to be optimized by Mr. Bayes
def model_accuracy(alphabet_size, dropout, embedding_size, label_smoothing, layer_size, learning_rate, learning_rate_final, window):

    # Scale the parameters correctly
    alphabet_size = int(alphabet_size + 0.5)
    embedding_size = min(alphabet_size, int(embedding_size + 0.5))
    layer_size = int(layer_size + 0.5)
    learning_rate = 10 ** learning_rate
    learning_rate_final = min(learning_rate, 10 ** learning_rate_final)
    window = int(window + 0.5)

    # Load data
    uppercase_data = UppercaseDataDiakritika(window, alphabet_size)

    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(alphabet_size, embedding_size, input_length=2*window + 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(layer_size, activation=activation_dict[args.activation]),
        tf.keras.layers.Dropout(dropout),
        # tf.keras.layers.Dense(layer_size, activation=activation_dict[args.activation]),
        # tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(uppercase_data.LABELS)
    ])

    model.compile(
        optimizer=sgd_optimizer(args.optimizer, uppercase_data.train.size, learning_rate, learning_rate_final),
        loss=loss(label_smoothing),
        metrics=[metric(label_smoothing)],
    )

    filename = os.path.join("models", "{},{},{},{},{},{},{},{},{},{},{}".format(
        "acc={val_accuracy:.4f}",
        "a_s={}".format(alphabet_size),
        "d={:.2f}".format(dropout),
        "e_s={}".format(embedding_size),
        "h_s={}".format(layer_size),
        "l_s={:.4f}".format(label_smoothing),
        "l_r={:.6f}".format(learning_rate),
        "l_r_f={:.8f}".format(learning_rate_final),
        "w={}".format(window),
        "act={}".format(args.activation),
        "diac=True"
    ))
    model.save(filename)

    model.fit(
        uppercase_data.train.data["windows"],
        sparse_to_distribution(uppercase_data.train.data["labels"], label_smoothing),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=0,
        validation_data=(
            uppercase_data.dev.data["windows"],
            sparse_to_distribution(uppercase_data.dev.data["labels"], label_smoothing)),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy'),
            tf.keras.callbacks.ModelCheckpoint(filename, monitor='val_accuracy', save_best_only=True)],
    )

    test_logs = model.evaluate(
        uppercase_data.dev.data["windows"],
        sparse_to_distribution(uppercase_data.dev.data["labels"], label_smoothing),
        batch_size=args.batch_size,
        verbose=0
    )

    accuracy = test_logs[model.metrics_names.index("accuracy")]
    return accuracy


pbounds = {
    'alphabet_size': (64.0, 128.0),
    'dropout': (0.0, 0.4),
    'embedding_size': (32.0, 64.0),
    'label_smoothing': (0.0, 0.2),
    'layer_size': (1280.0, 2048.0),
    'learning_rate': (-4.0, -2.0),
    'learning_rate_final': (-5.0, -3.0),
    'window': (6.0, 11.0)
}

optimizer = BayesianOptimization(
    f=model_accuracy,
    pbounds=pbounds,
    verbose=2,
    random_state=1
)
if os.path.isfile("./parameters_log.json"):
    load_logs(optimizer, logs=["./parameters_log.json"])
    print("Loaded {} model evaluations".format(len(optimizer.space)))

logger = JSONLogger(path="./parameters_log_new.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
optimizer.subscribe(Events.OPTMIZATION_STEP, ScreenLogger())

optimizer.maximize(
    init_points=max(0, 20-len(optimizer.space)),
    n_iter=40-max(len(optimizer.space)-20, 0),
)
