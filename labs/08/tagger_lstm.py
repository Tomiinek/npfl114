# !/usr/bin/env python3
import os
import re
import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Embedding, Dense, Lambda, concatenate, Conv1D, GlobalMaxPool1D

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


def log_train_progress(epoch, loss, accuracy, progress, out_file):
    if progress < 100:
        print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.2f} % ║ done: {progress} %', end='', flush=True)
    else:
        print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.2f} % ║ ', end='', flush=True)
        print(f'epoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.2f} ║ ', end='', flush=True, file=out_file)


def log_dev_progress(loss, accuracy, learning_rate, out_file):
    print(f'eval loss: {loss:1.4f} │ acc: {accuracy:2.2f} % ║ lr: {learning_rate}', flush=True)
    print(f'eval loss: {loss:1.4f} │ acc: {accuracy:2.2f} % ║ lr: {learning_rate}', flush=True, file=out_file)


class Network:
    def __init__(self, args, num_words, num_tags, num_chars):
        self.best_accuracy = 0.0
        self.not_improved_epochs = 0
        self.learning_rate = args.learning_rate

        self.build(args, num_words, num_tags, num_chars)
        self._optimizer = tf.optimizers.Adam()
        self._loss = tf.losses.SparseCategoricalCrossentropy()
        self._metrics = {"loss": tf.metrics.Mean(), "accuracy": tf.metrics.SparseCategoricalAccuracy()}

    def build(self, args, num_words, num_tags, num_chars):
        word_ids = tf.keras.Input(shape=[None], dtype=tf.int32)
        charseqs = tf.keras.Input(shape=[None], dtype=tf.int32)
        charseq_ids = tf.keras.Input(shape=[None], dtype=tf.int32)

        chars_embedded = Embedding(input_dim=num_chars, output_dim=args.cle_dim, mask_zero=False)(charseqs)
        convoluted = []
        for width in range(2, args.cnn_max_width + 1):
            hidden = Conv1D(args.cnn_filters, kernel_size=width, strides=1, padding='valid', activation=tf.nn.relu)(chars_embedded)
            convoluted.append(GlobalMaxPool1D()(hidden))
        chars_hidden = concatenate(convoluted, axis=1)
        chars_hidden = Dense(args.we_dim, activation=tf.nn.relu)(chars_hidden)

        words_embedded_1 = Lambda(lambda args: tf.gather(*args))([chars_hidden, charseq_ids])
        words_embedded_2 = Embedding(input_dim=num_words, output_dim=args.we_dim, mask_zero=True)(word_ids)
        embedded = concatenate([words_embedded_2, words_embedded_1], axis=2)
        embedded = tf.keras.layers.SpatialDropout1D(args.input_dropout)(embedded)

        x = embedded
        for _ in range(args.rnn_layers):
            if args.rnn_cell == 'LSTM':
                hidden = Bidirectional(LSTM(units=args.rnn_cell_dim, return_sequences=True))(x)
            else:
                hidden = Bidirectional(GRU(units=args.rnn_cell_dim, return_sequences=True))(x)
            hidden = tf.keras.layers.SpatialDropout1D(args.hidden_dropout)(hidden)
            x = tf.keras.layers.add([x, hidden])

        predictions = Dense(units=num_tags, activation=tf.nn.softmax)(x)

        self.model = tf.keras.Model(inputs=[word_ids, charseq_ids, charseqs], outputs=predictions)

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3,
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def train_batch(self, inputs, tags):
        mask = tf.not_equal(tags, 0)

        with tf.GradientTape() as tape:
            probabilities = self.model(inputs, training=True)
            loss = self._loss(tags, probabilities, mask)

        gradients = tape.gradient(loss, self.model.variables)
        if args.clip_gradient is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, args.clip_gradient)
        self._optimizer.apply_gradients(zip(gradients, self.model.variables))

        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            else:
                metric.update_state(tags, probabilities, mask)

    def train_epoch(self, epoch, dataset, log_file, args):
        if self.not_improved_epochs > 3:
            self.learning_rate /= 10
            self._optimizer.learning_rate = self.learning_rate
            self.not_improved_epochs = 0

        for name, metric in self._metrics.items():
            metric.reset_states()

        batches_num = dataset.size() / args.batch_size

        for i, batch in enumerate(dataset.batches(args.batch_size)):
            self.train_batch(
                [batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs],
                batch[dataset.TAGS].word_ids)

            log_train_progress(epoch, self._metrics["loss"].result(), 100 * self._metrics["accuracy"].result(), int(i / batches_num * 100), log_file)
        log_train_progress(epoch, self._metrics["loss"].result(), 100 * self._metrics["accuracy"].result(), 100, log_file)

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3,
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def evaluate_batch(self, inputs, tags):
        mask = tf.not_equal(tags, 0)

        probabilities = self.model(inputs, training=False)
        loss = self._loss(tags, probabilities, mask)

        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            else:
                metric.update_state(tags, probabilities, mask)

    def evaluate(self, dataset, log_file, args):
        for metric in self._metrics.values():
            metric.reset_states()

        for batch in dataset.batches(args.batch_size):
            self.evaluate_batch(
                [batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs], batch[dataset.TAGS].word_ids)

        log_dev_progress(self._metrics["loss"].result(), 100 * self._metrics["accuracy"].result(), self.learning_rate, log_file)

        if self._metrics["accuracy"].result() > self.best_accuracy:
            self.best_accuracy = self._metrics["accuracy"].result()
            self.model.save_weights(f"{args.directory}/acc-{100 * self.best_accuracy:2.2f}")
        else:
            self.not_improved_epochs += 1


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=350, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
    parser.add_argument("--learning_rate", default=0.003, type=int, help="Initial learning rate.")
    parser.add_argument("--clip_gradient", default=1.0, type=int, help="Clip the gradient norm.")
    parser.add_argument("--input_dropout", default=0.2, type=int, help="Dropout to the input embedding.")
    parser.add_argument("--hidden_dropout", default=0.5, type=int, help="Dropout between layers.")
    parser.add_argument("--cnn_filters", default=64, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnn_max_width", default=6, type=int, help="Maximum CNN filter width.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--rnn_layers", default=4, type=int, help="RNN layers.")
    parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=128, type=int, help="RNN cell dimension.")
    parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
    parser.add_argument("--directory", default="drive/My Drive/Colab Notebooks/Tagger", type=str, help="Directory for the outputs.")
    parser.add_argument("--checkpoint", default="acc_94.28", type=str, help="Checkpoint of a model to load weights from.")

    args = parser.parse_args()

    architecture = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if key not in ["directory", "epochs", "batch_size", "clip_gradient", "checkpoint"]))
    args.directory = f"{args.directory}/models/{architecture}"
    # if not os.path.exists(args.directory):
    #    os.makedirs(args.directory)

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    tags = sorted(morpho.train.data[morpho.train.TAGS].words)

    forms = {tag[:2]: list(tag) for tag in tags if tag not in ["<unk>", "<pad>"]}
    for tag in tags:
        if tag in ["<unk>", "<pad>"]: continue

        p = tag[:2]
        form = forms[p]
        for i, c in enumerate(tag):
            if form[i] == c: continue
            if c == '-' or form[i] == '-': form[i] = '*'; continue
            form[i] = 'X'
        forms[p] = form

    with open("tags.txt", "w", encoding="utf-8") as out_file:
        for form in sorted(forms.values()):
            print(''.join(form[:12] + [form[14]]), file=out_file)
        print(file=out_file)
        for tag in tags:
            if tag in ["<unk>", "<pad>"]: continue
            print(tag[:12] + tag[14], file=out_file)

    exit()

    # Create the network and train
    network = Network(args,
                      num_words=len(morpho.train.data[morpho.train.FORMS].words),
                      num_tags=len(morpho.train.data[morpho.train.TAGS].words),
                      num_chars=len(morpho.train.data[morpho.train.FORMS].alphabet))
    if args.checkpoint is not None:
        network.model.load_weights(f"{args.directory}/{args.checkpoint}")
        network.learning_rate = 0.003

    for epoch in range(8, args.epochs):
        with open(f"{args.directory}/log.txt", "a") as log_file:
            network.train_epoch(epoch, morpho.train, log_file, args)
            metrics = network.evaluate(morpho.dev, log_file, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = f"{args.directory}/test.txt"
    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, sentence in enumerate(network.predict(morpho.test, args)):
            for j in range(len(morpho.test.data[morpho.test.FORMS].word_strings[i])):
                print(morpho.test.data[morpho.test.FORMS].word_strings[i][j],
                      morpho.test.data[morpho.test.LEMMAS].word_strings[i][j],
                      morpho.test.data[morpho.test.TAGS].words[sentence[j]],
                      sep="\t", file=out_file)
            print(file=out_file)