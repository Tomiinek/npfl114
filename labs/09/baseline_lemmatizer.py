#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, GRUCell, Dense

import decoder
from morpho_dataset_enhanced import MorphoDataset

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


def log_train_progress(epoch, loss, accuracy, progress, example):
    print('\r' + 140*' ', end='') # clear line
    print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.3f} % ║ done: {progress:2d} % │ {example}', end='', flush=True)

def log_train(epoch, loss, accuracy, out_file):
    print('\r' + 140*' ', end='') # clear line
    print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.3f} % ║ ', end='', flush=True)
    print(f'epoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.3f} % ║ ', end='', flush=True, file=out_file)

def log_dev(accuracy, learning_rate, out_file):
    print(f'acc: {accuracy:2.3f} % ║ lr: {learning_rate:1.6f}', flush=True)
    print(f'acc: {accuracy:2.3f} % ║ lr: {learning_rate:1.6f}', flush=True, file=out_file)

def log_mistakes(mistakes, out_file):
    for mistake in mistakes:
        print(f'{mistake[0]}\t║\t{mistake[1]}\t║\t{mistake[2]}\t║\t{mistake[3]}\t║\t{mistake[4]}', file=out_file)

class Network:
    def __init__(self, args, num_source_chars, num_target_chars):
        class Model(tf.keras.Model):
            def __init__(self):
                super().__init__()

                self.source_embeddings = Embedding(num_source_chars, args.cle_dim, mask_zero=True)
                self.source_rnn = Bidirectional(GRU(args.rnn_dim, return_sequences=True), merge_mode='sum')

                self.target_embeddings = Embedding(num_target_chars, args.cle_dim)
                self.target_rnn_cell = GRUCell(args.rnn_dim)
                self.target_output_layer = Dense(num_target_chars, activation=None)

                self.attention_source_layer = Dense(args.rnn_dim)
                self.attention_state_layer = Dense(args.rnn_dim)
                self.attention_weight_layer = Dense(1)

        self._model = Model()

        self._best_accuracy = 0.0
        self._steps = 0
        self._learning_rate = args.learning_rate

        self._optimizer = tf.optimizers.Adam(learning_rate=self._learning_rate)
        self._loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._metrics_training = {"loss": tf.metrics.Mean(), "accuracy": tf.metrics.SparseCategoricalAccuracy()}
        self._metrics_evaluation = {"loss": tf.metrics.Mean(), "accuracy": tf.metrics.Mean()}

    def lr_decay(self):
        self._steps += 1

        if self._steps < 2000: self._learning_rate = args.rnn_dim ** (-0.5) * self._steps * 2000 ** (-1.5) * args.learning_rate
        else: self._learning_rate = args.rnn_dim ** (-0.5) * self._steps ** (-0.5) * args.learning_rate

        self._optimizer.learning_rate = self._learning_rate

    def _append_eow(self, sequences):
        """Append EOW character after end every given sequence."""
        sequences_rev = tf.reverse_sequence(sequences, tf.reduce_sum(tf.cast(tf.not_equal(sequences, 0), tf.int32), axis=1), 1)
        sequences_rev_eow = tf.pad(sequences_rev, [[0, 0], [1, 0]], constant_values=MorphoDataset.Factor.EOW)
        return tf.reverse_sequence(sequences_rev_eow, tf.reduce_sum(tf.cast(tf.not_equal(sequences_rev_eow, 0), tf.int32), axis=1), 1)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 4, autograph=False)
    def train_batch(self, source_charseq_ids, source_charseqs, target_charseq_ids, target_charseqs):
        target_charseqs = self._append_eow(target_charseqs)

        with tf.GradientTape() as tape:
            source_embeddings = self._model.source_embeddings(source_charseqs)
            source_encoded = self._model.source_rnn(source_embeddings)

            # Copy the source_encoded to corresponding batch places, and then flatten it
            source_mask = tf.not_equal(source_charseq_ids, 0)
            source_encoded = tf.boolean_mask(tf.gather(source_encoded, source_charseq_ids), source_mask)
            targets = tf.boolean_mask(tf.gather(target_charseqs, target_charseq_ids), source_mask)

            class DecoderTraining(decoder.BaseDecoder):
                @property
                def batch_size(self): return tf.shape(self._source_encoded)[0]
                @property
                def output_size(self): return tf.shape(self._targets)[1]
                @property
                def output_dtype(self): return tf.float32

                def _with_attention(self, inputs, states):
                    attention_source = self._model.attention_source_layer(self._source_encoded) # slovo X char X rnn_dim
                    attention_states = self._model.attention_state_layer(states) # slovo X rnn_dim

                    weights = attention_source + tf.expand_dims(attention_states, 1) # slovo X char X rnn_dim
                    weights = tf.tanh(weights) # slovo X char X rnn_dim
                    weights = self._model.attention_weight_layer(weights) # slovo X char X 1
                    weights = tf.nn.softmax(weights, axis=1) # slovo X char X 1

                    attention = tf.math.multiply(self._source_encoded, weights) # slovo X char X rnn_dim
                    attention = tf.reduce_sum(attention, axis=1) # slovo X rnn_dim
                    return tf.concat([inputs, attention], axis=1)

                def initialize(self, layer_inputs, initial_state=None):
                    self._model, self._source_encoded, self._targets = layer_inputs

                    finished = tf.fill([self.batch_size], False)
                    inputs = self._model.target_embeddings(tf.fill([self.batch_size], MorphoDataset.Factor.BOW))

                    states = self._source_encoded[:, -1, :]
                    inputs = self._with_attention(inputs, states)
                    return finished, inputs, states

                def step(self, time, inputs, states):
                    outputs, [states] = self._model.target_rnn_cell(inputs, [states])
                    outputs = self._model.target_output_layer(outputs)
                    next_inputs = self._model.target_embeddings(self._targets[:, time])
                    next_inputs = self._with_attention(next_inputs, states)
                    finished = tf.equal(self._targets[:, time], MorphoDataset.Factor.EOW)

                    return outputs, states, next_inputs, finished

            output_layer, _, _ = DecoderTraining()([self._model, source_encoded, targets])

            loss = self._loss(targets, output_layer, tf.not_equal(targets, 0))

        gradients = tape.gradient(loss, self._model.variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.variables))

        self._metrics_training["loss"](loss)
        self._metrics_training["accuracy"](targets, output_layer, tf.not_equal(targets, 0))

        return tf.math.argmax(output_layer, axis=2), targets

    def train_epoch(self, epoch, dataset, out_file, args):
        for name, metric in self._metrics_training.items():
            metric.reset_states()

        batches_num = dataset.size() / args.batch_size

        for b, batch in enumerate(dataset.batches(args.batch_size, args.max_length)):
            self.lr_decay()
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
                            system_character = dataset.data[dataset.LEMMAS].alphabet[predictions[i][j]]
                            if gold_character != system_character: is_mistake = True
                            gold_lemma += gold_character
                            system_lemma += system_character
                        else: break
                    if not is_mistake: continue
                    for j in range(tf.shape(sources)[1]):
                        if sources[i][j].numpy() != MorphoDataset.Factor.PAD:
                            form += dataset.data[dataset.FORMS].alphabet[sources[i][j]]
                        else: break
                    example = f"{form} {gold_lemma} {system_lemma}"
                    break

                accuracy = 100*float(self._metrics_training["accuracy"].result())
                loss = float(self._metrics_training["loss"].result())
                progress = int(b / batches_num * 100)

                log_train_progress(epoch, loss, accuracy, progress, example)

        log_train(epoch, loss, accuracy, out_file)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 2, autograph=False)
    def predict_batch(self, source_charseq_ids, source_charseqs):
        source_embeddings = self._model.source_embeddings(source_charseqs)
        source_encoded = self._model.source_rnn(source_embeddings)

        # Copy the source_encoded to corresponding batch places, and then flatten it
        source_mask = tf.not_equal(source_charseq_ids, 0)
        source_encoded = tf.boolean_mask(tf.gather(source_encoded, source_charseq_ids), source_mask)

        class DecoderPrediction(decoder.BaseDecoder):
            @property
            def batch_size(self): return tf.shape(self._source_encoded)[0]
            @property
            def output_size(self): return 1
            @property
            def output_dtype(self): return tf.int32

            def _with_attention(self, inputs, states):
                attention_source = self._model.attention_source_layer(self._source_encoded)  # slovo X char X rnn_dim
                attention_states = self._model.attention_state_layer(states)  # slovo X rnn_dim

                weights = attention_source + tf.expand_dims(attention_states, 1)  # slovo X char X rnn_dim
                weights = tf.tanh(weights)  # slovo X char X rnn_dim
                weights = self._model.attention_weight_layer(weights)  # slovo X char X 1
                weights = tf.nn.softmax(weights, axis=1)  # slovo X char X 1

                attention = tf.math.multiply(self._source_encoded, weights)  # slovo X char X rnn_dim
                attention = tf.reduce_sum(attention, axis=1)  # slovo X rnn_dim

                return tf.concat([inputs, attention], axis=1)

            def initialize(self, layer_inputs, initial_state=None):
                self._model, self._source_encoded = layer_inputs

                finished = tf.fill([self.batch_size], False)
                inputs = self._model.target_embeddings(tf.fill([self.batch_size], MorphoDataset.Factor.BOW))

                states = self._source_encoded[:, -1, :]
                inputs = self._with_attention(inputs, states)
                return finished, inputs, states

            def step(self, time, inputs, states):
                outputs, [states] = self._model.target_rnn_cell(inputs, [states])
                outputs = self._model.target_output_layer(outputs)
                outputs = tf.argmax(outputs, axis=1, output_type=tf.int32)
                next_inputs = self._model.target_embeddings(outputs)
                next_inputs = self._with_attention(next_inputs, states)
                finished = tf.equal(outputs, MorphoDataset.Factor.EOW)

                return outputs, states, next_inputs, finished

        predictions, _, _ = DecoderPrediction(maximum_iterations=tf.shape(source_charseqs)[1] + 10)([self._model, source_encoded])
        return predictions

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 4, autograph=False)
    def evaluate_batch(self, source_charseq_ids, source_charseqs, target_charseq_ids, target_charseqs):
        # Predict
        predictions = self.predict_batch(source_charseq_ids, source_charseqs)

        # Append EOW to target_charseqs and copy them to corresponding places and flatten it
        target_charseqs = self._append_eow(target_charseqs)
        source_mask = tf.not_equal(source_charseq_ids, 0)
        targets = tf.boolean_mask(tf.gather(target_charseqs, target_charseq_ids), source_mask)
        sources = tf.boolean_mask(tf.gather(source_charseqs, source_charseq_ids), source_mask)

        # Compute accuracy, but on the whole sequences
        mask = tf.cast(tf.not_equal(targets, 0), tf.int32)
        resized_predictions = tf.concat([predictions, tf.zeros_like(targets)], axis=1)[:, :tf.shape(targets)[1]]
        equals = tf.reduce_all(tf.equal(resized_predictions * mask, targets * mask), axis=1)
        self._metrics_evaluation["accuracy"](equals)

        return resized_predictions, equals, targets, sources

    def evaluate(self, dataset, dataset_name, log_file, epoch_num, args):
        for metric in self._metrics_evaluation.values():
            metric.reset_states()

        mistakes = []
        for batch in dataset.batches(args.batch_size * args.max_length // 1000, 1000):
            predictions, equals, targets, sources = self.evaluate_batch(batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs,
                                                                        batch[dataset.LEMMAS].charseq_ids, batch[dataset.LEMMAS].charseqs)
            for i in range(tf.shape(targets)[0]):
                if equals[i] == 1: continue
                before, after, form, gold_lemma, system_lemma = "", "", "", "", ""
                for j in range(tf.shape(targets)[1]):
                    if targets[i][j].numpy() != MorphoDataset.Factor.EOW:
                        gold_lemma += dataset.data[dataset.LEMMAS].alphabet[targets[i][j]]
                        system_lemma += dataset.data[dataset.LEMMAS].alphabet[predictions[i][j]]
                    else: break
                for j in range(tf.shape(sources)[1]):
                    if sources[i][j].numpy() != MorphoDataset.Factor.PAD:
                        form += dataset.data[dataset.FORMS].alphabet[sources[i][j]]
                    else: break
                for k in range(1,4):
                    if i-k >= 0:
                        form_before = sources[i-k]
                        for j in range(tf.shape(sources)[1]):
                            if form_before[j].numpy() != MorphoDataset.Factor.PAD:
                                before += dataset.data[dataset.FORMS].alphabet[form_before[j]]
                            else: break
                        before += " "
                    if i+k < tf.shape(targets)[0]:
                        form_after = sources[i+k]
                        for j in range(tf.shape(sources)[1]):
                            if form_after[j].numpy() != MorphoDataset.Factor.PAD:
                                after += dataset.data[dataset.FORMS].alphabet[form_after[j]]
                            else: break
                        after += " "
                mistakes.append([before, form, gold_lemma, system_lemma, after])

        accuracy = 100*float(self._metrics_evaluation["accuracy"].result())
        log_dev(accuracy, self._learning_rate, log_file)
        with open(f"{args.directory}/mistakes-{epoch_num}.txt", "w", encoding="utf-8") as mistakes_file:
            log_mistakes(mistakes, mistakes_file)

        if accuracy > self._best_accuracy:
            self._model.save_weights(f"{args.directory}/acc-{accuracy:2.3f}.h5")
            self._best_accuracy = accuracy


if __name__ == "__main__":
    import argparse
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", default=".", type=str, help="Directory for the outputs.")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--max_length", default=60, type=int, help="Max length of sentence in training.")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="Initial learning rate.")
    parser.add_argument("--rnn_dim", default=32, type=int, help="RNN cell dimension.")
    args = parser.parse_args()

    architecture = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if key not in ["directory", "base_directory", "epochs", "batch_size", "clip_gradient", "checkpoint"]))
    args.directory = f"{args.base_directory}/models/base_{architecture}"
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load the data
    morpho = MorphoDataset("czech_pdt", args.base_directory)

    # Create the network and train
    network = Network(args,
                      num_source_chars=len(morpho.train.data[morpho.train.FORMS].alphabet),
                      num_target_chars=len(morpho.train.data[morpho.train.LEMMAS].alphabet))

    for epoch in range(args.epochs):
        with open(f"{args.directory}/log.txt", "a", encoding="utf-8") as log_file:
            network.train_epoch(epoch, morpho.train, log_file, args)
            network.evaluate(morpho.dev, "dev", log_file, epoch, args)
