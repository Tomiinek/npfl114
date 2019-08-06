#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from morpho_dataset_enhanced import MorphoDataset
from attention import Encoder, Decoder

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


def log_train_progress(epoch, loss, accuracy, progress, example):
    print('\r' + 140*' ', end='') # clear line
    print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.3f} % ║ done: {progress:2d} % ║ {example}', end='', flush=True)

def log_train(epoch, loss, accuracy, out_file):
    print('\r' + 140*' ', end='') # clear line
    print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.3f} % ║ ', end='', flush=True)
    print(f'epoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.3f} % ║ ', end='', flush=True, file=out_file)

def log_dev(accuracy, learning_rate, out_file):
    print(f'acc: {accuracy:2.3f} % ║ lr: {learning_rate:1.6f}', flush=True)
    print(f'acc: {accuracy:2.3f} % ║ lr: {learning_rate:1.6f}', flush=True, file=out_file)

def log_mistakes(mistakes, out_file):
    max_before_len = max(len(mistake[0]) for mistake in mistakes)
    max_after_len = max(len(mistake[4]) for mistake in mistakes)
    max_original_len = max(len(mistake[1]) for mistake in mistakes)
    max_lemma_len = max(len(mistake[2]) for mistake in mistakes)
    for mistake in mistakes:
        print(f'{mistake[0].rjust(max_before_len)}\t║\t{mistake[1].ljust(max_original_len)}\t║\t{mistake[2].ljust(max_lemma_len)}\t║\t{mistake[3].ljust(max_lemma_len)}\t║\t{mistake[4].ljust(max_after_len)}', file=out_file)

class Model(tf.keras.Model):
    def __init__(self, args, num_source_chars, num_target_chars):
        super().__init__()

        self._best_accuracy = -1.0
        self._steps = 0
        self._learning_rate = args.learning_rate

        self._build(args, num_source_chars, num_target_chars)

        self._optimizer = tf.optimizers.Adam(learning_rate=self._learning_rate)
        self._loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._metrics_training = {"loss": tf.metrics.Mean(name="train_loss"), "accuracy": tf.metrics.SparseCategoricalAccuracy()}
        self._metrics_evaluation = {"loss": tf.metrics.Mean(name="dev_loss"), "accuracy": tf.metrics.Mean(name="dev_accuracy")}

    def _build(self, args, num_source_chars, num_target_chars):
        self._encoder = Encoder(num_source_chars, args.dim, args.heads, args.layers, args.dropout)
        self._decoder = Decoder(num_target_chars, args.dim, args.heads, args.layers, args.dropout)
        self._classifier = tf.keras.layers.Dense(num_target_chars, activation=None)

    def _lr_decay(self):
        self._steps += 1

        if self._steps < 2000: self._learning_rate = args.dim ** (-0.5) * self._steps * 2000 ** (-1.5) * args.learning_rate
        else: self._learning_rate = args.dim ** (-0.5) * self._steps ** (-0.5) * args.learning_rate

        self._optimizer.learning_rate = self._learning_rate

    def _create_look_ahead_mask(self, target):
        size = tf.shape(target)[1]

        look_ahead =  1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        padding = self._create_padding_mask(target)
        return tf.maximum(padding, look_ahead)

    def _create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    #@tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 4, autograph=False)
    #@tf.function
    def train_batch(self, source_charseq_ids, source_charseqs, target_charseq_ids, target_charseqs):
        source_mask = tf.not_equal(source_charseq_ids, 0)
        targets = tf.boolean_mask(tf.gather(target_charseqs, target_charseq_ids), source_mask)

        targets_in = targets[:, :-1]
        targets_out = targets[:, 1:]

        encoder_mask = self._create_padding_mask(source_charseqs)
        decoder_combined_mask = self._create_look_ahead_mask(targets_in)

        with tf.GradientTape() as tape:
            encoded_chars = self._encoder(source_charseqs, encoder_mask, training=True)
            encoded_chars = tf.boolean_mask(tf.gather(encoded_chars, source_charseq_ids), source_mask)
            encoder_mask = tf.boolean_mask(tf.gather(encoder_mask, source_charseq_ids), source_mask)

            decoded_chars = self._decoder(encoded_chars, targets_in, decoder_combined_mask, encoder_mask, training=True)
            prediction = self._classifier(decoded_chars)

            loss = self._loss(targets_out, prediction, tf.not_equal(targets_out, 0))

        gradients = tape.gradient(loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._metrics_training["loss"](loss)
        self._metrics_training["accuracy"](targets_out, prediction, tf.not_equal(targets_out, 0))

        return tf.math.argmax(prediction, axis=2), targets

    def train_epoch(self, epoch, dataset, out_file, args):
        for name, metric in self._metrics_training.items():
            metric.reset_states()

        batches_num = dataset.size() / args.batch_size

        for b, batch in enumerate(dataset.batches(args.batch_size, args.max_length)):
            self._lr_decay()
            predictions, targets = self.train_batch(batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs, batch[dataset.LEMMAS].charseq_ids, batch[dataset.LEMMAS].charseqs)

            if b % 10 == 0:
                source_mask = tf.not_equal(batch[dataset.FORMS].charseq_ids, 0)
                sources = tf.boolean_mask(tf.gather(batch[dataset.FORMS].charseqs, batch[dataset.FORMS].charseq_ids), source_mask)
                example = ""
                for i in range(tf.shape(targets)[0]):
                    is_mistake = False
                    form, gold_lemma, system_lemma = "", "", ""
                    for j in range(tf.shape(targets)[1]):
                        if targets[i][j].numpy() != MorphoDataset.Factor.EOW:
                            gold_character = dataset.data[dataset.LEMMAS].alphabet[targets[i][j]]
                            system_character = dataset.data[dataset.LEMMAS].alphabet[predictions[i][j-1]]
                            if gold_character != system_character: is_mistake = True
                            gold_lemma += gold_character
                            system_lemma += system_character
                        else:
                            break
                    if not is_mistake: continue
                    for j in range(1, tf.shape(sources)[1]):
                        if sources[i][j].numpy() != MorphoDataset.Factor.EOW:
                            form += dataset.data[dataset.FORMS].alphabet[sources[i][j]]
                        else:
                            break
                    example = f"{form} {gold_lemma} {system_lemma}"
                    break

                accuracy = 100 * float(self._metrics_training["accuracy"].result())
                loss = float(self._metrics_training["loss"].result())
                progress = int(b / batches_num * 100)

                log_train_progress(epoch, loss, accuracy, progress, example)

        log_train(epoch, loss, accuracy, out_file)

    #@tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 2, autograph=False)
    #@tf.function
    def predict_batch(self, source_charseq_ids, source_charseqs):
        maximum_iterations = tf.shape(source_charseqs)[1] + 10

        source_mask = tf.not_equal(source_charseq_ids, 0)

        encoder_mask = self._create_padding_mask(source_charseqs)
        encoded_chars = self._encoder(source_charseqs, encoder_mask, training=False)
        encoded_chars = tf.boolean_mask(tf.gather(encoded_chars, source_charseq_ids), source_mask)
        encoder_mask = tf.boolean_mask(tf.gather(encoder_mask, source_charseq_ids), source_mask)

        output = tf.fill([tf.shape(encoded_chars)[0], 1], MorphoDataset.Factor.BOW)
        finished = tf.fill([tf.shape(encoded_chars)[0]], False)

        for _ in range(maximum_iterations):
            decoder_combined_mask = self._create_look_ahead_mask(output)
            decoded_chars = self._decoder(encoded_chars, output, decoder_combined_mask, encoder_mask, training=False)
            predictions = self._classifier(decoded_chars)

            next_prediction = predictions[:, -1, :]
            next_char = tf.cast(tf.argmax(next_prediction, -1), tf.int32)

            output = tf.concat([output,tf.expand_dims(next_char, axis=-1)], axis=-1)

            finished = tf.math.logical_or(finished, tf.equal(next_char, MorphoDataset.Factor.EOW))
            if tf.reduce_all(finished): break

        return output[:,1:]

    #@tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 4, autograph=False)
    #@tf.function
    def evaluate_batch(self, source_charseq_ids, source_charseqs, target_charseq_ids, target_charseqs):
        # Predict
        predictions = self.predict_batch(source_charseq_ids, source_charseqs)

        source_mask = tf.not_equal(source_charseq_ids, 0)
        targets = tf.boolean_mask(tf.gather(target_charseqs[:, 1:], target_charseq_ids), source_mask)
        sources = tf.boolean_mask(tf.gather(source_charseqs, source_charseq_ids), source_mask)

        # Compute accuracy, but on the whole sequences
        mask = tf.cast(tf.not_equal(targets, 0), tf.int32)
        resized_predictions = tf.concat([predictions, tf.zeros_like(targets)], axis=1)[:, :tf.shape(targets)[1]]
        equals = tf.reduce_all(tf.equal(resized_predictions * mask, targets * mask), axis=1)
        self._metrics_evaluation["accuracy"](equals)

        return resized_predictions, equals, targets, sources

    def evaluate(self, epoch, dataset, dataset_name, log_file, args):
        for metric in self._metrics_evaluation.values():
            metric.reset_states()

        mistakes = []
        for batch in dataset.batches(16, 1000):
            predictions, equals, targets, sources = self.evaluate_batch(batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs,
                                                                        batch[dataset.LEMMAS].charseq_ids, batch[dataset.LEMMAS].charseqs)
            if len(mistakes) > 1024: continue

            for i in range(tf.shape(targets)[0]):
                if equals[i].numpy(): continue
                before, after, form, gold_lemma, system_lemma = "", "", "", "", ""
                for j in range(tf.shape(targets)[1]):
                    if targets[i][j].numpy() != MorphoDataset.Factor.EOW:
                        gold_lemma += dataset.data[dataset.LEMMAS].alphabet[targets[i][j]]
                        system_lemma += dataset.data[dataset.LEMMAS].alphabet[predictions[i][j]]
                    else: break
                for j in range(1, tf.shape(sources)[1]):
                    if sources[i][j].numpy() != MorphoDataset.Factor.EOW:
                        form += dataset.data[dataset.FORMS].alphabet[sources[i][j]]
                    else: break
                for k in range(1, 4):
                    if i - (4 - k) >= 0:
                        form_before = sources[i - (4 - k)]
                        for j in range(1, tf.shape(sources)[1]):
                            if form_before[j].numpy() != MorphoDataset.Factor.EOW:
                                before += dataset.data[dataset.FORMS].alphabet[form_before[j]]
                            else: break
                    before += " "
                    if i + k < tf.shape(targets)[0]:
                        form_after = sources[i + k]
                        for j in range(1, tf.shape(sources)[1]):
                            if form_after[j].numpy() != MorphoDataset.Factor.EOW:
                                after += dataset.data[dataset.FORMS].alphabet[form_after[j]]
                            else: break
                    after += " "
                mistakes.append([before, form, gold_lemma, system_lemma, after])

        accuracy = 100 * float(self._metrics_evaluation["accuracy"].result())
        log_dev(accuracy, self._learning_rate, log_file)
        with open(f"{args.directory}/mistakes-{epoch}.txt", "w", encoding="utf-8") as mistakes_file:
            log_mistakes(mistakes, mistakes_file)

        if accuracy > self._best_accuracy:
            self.save_weights(f"{args.directory}/acc-{accuracy:2.3f}.h5")
            self._best_accuracy = accuracy


if __name__ == "__main__":
    import argparse
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", default=".", type=str, help="Directory for the outputs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--dim", default=320, type=int, help="Dimension of hidden layers.")
    parser.add_argument("--heads", default=4, type=int, help="Number of attention heads.")
    parser.add_argument("--layers", default=4, type=int, help="Number of attention layers.")
    parser.add_argument("--dropout", default=0.42, type=float, help="Dropout rate.")
    parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
    parser.add_argument("--max_length", default=60, type=int, help="Max length of sentence in training.")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="Initial learning rate multiplier.")
    args = parser.parse_args()

    architecture = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if key not in ["directory", "base_directory", "epochs", "batch_size", "clip_gradient", "checkpoint"]))
    args.directory = f"{args.base_directory}/models/attention_{architecture}"
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load the data
    morpho = MorphoDataset("czech_pdt", args.base_directory, add_bow_eow=True)

    # Create the network and train
    num_source_chars = len(morpho.train.data[morpho.train.FORMS].alphabet)
    num_target_chars = len(morpho.train.data[morpho.train.LEMMAS].alphabet)

    network = Model(args, num_source_chars, num_target_chars)

    for epoch in iter(int, 1):
        with open(f"{args.directory}/log.txt", "a", encoding="utf-8") as log_file:
            #network.train_epoch(epoch, morpho.train, log_file, args)
            network.evaluate(epoch, morpho.dev, "dev", log_file, args)
