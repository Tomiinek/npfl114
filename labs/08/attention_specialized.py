#!/usr/bin/env python3
import os
import re
import math
import numpy as np
import tensorflow as tf

from morpho_positional_analyzer import MorphoAnalyzer
from attention_dataset2 import MorphoDataset
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Embedding, Dense, Lambda, concatenate, Conv1D, \
    GlobalMaxPool1D, SpatialDropout1D, add, Dropout
from tensorflow.keras.layers.experimental import LayerNormalization

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


def log_train_progress(epoch, loss, accuracy, progress, out_file):
    if progress is not None:
        print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.3f} % ║ done: {progress} %', end='',
              flush=True)
    else:
        print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.3f} % ║ ', end='', flush=True)
        print(f'epoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.3f} % ║ ', end='', flush=True,
              file=out_file)


def log_dev_progress(loss, accuracy, learning_rate, out_file):
    print(f'eval loss: {loss:1.4f} │ acc: {accuracy:2.2f} % ║ lr: {learning_rate:1.6f}', flush=True)
    print(f'eval loss: {loss:1.4f} │ acc: {accuracy:2.2f} % ║ lr: {learning_rate:1.6f}', flush=True, file=out_file)


class Network:
    def __init__(self, args, num_chars):
        self.best_accuracy = 0.0
        self.not_improved_epochs = 0
        self.learning_rate = args.learning_rate

        self.build(args, num_chars)
        self._optimizer = tf.optimizers.Adam(args.learning_rate)
        self._loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._metrics = {"loss": tf.metrics.Mean(), "accuracy": tf.metrics.SparseCategoricalAccuracy()}
        self._steps = 0

    def build(self, args, num_chars):
        self.scale = tf.constant(math.sqrt(args.attention_dim / args.heads))

        word_embeddings = tf.keras.Input(shape=[None, MorphoDataset.Dataset.EMBEDDING_SIZE], dtype=tf.float32)
        charseqs = tf.keras.Input(shape=[None], dtype=tf.int32)
        charseq_ids = tf.keras.Input(shape=[None], dtype=tf.int32)
        positional_encoding = tf.keras.Input(shape=[None, args.attention_dim], dtype=tf.float32)

        chars_embedded = Embedding(input_dim=num_chars, output_dim=args.cle_dim, mask_zero=False)(charseqs)
        convoluted = []
        for width in range(2, args.cnn_max_width + 1):
            hidden = chars_embedded
            for _ in range(args.cle_layers):
                hidden = Conv1D(args.cnn_filters, kernel_size=width, strides=1, padding='valid', activation=tf.nn.relu)(
                    hidden)
            convoluted.append(GlobalMaxPool1D()(hidden))
        chars_hidden = concatenate(convoluted, axis=1)
        chars_hidden = Dense(args.we_dim, activation=tf.nn.tanh)(chars_hidden)
        char_embedding = Lambda(lambda args: tf.gather(*args))([chars_hidden, charseq_ids])

        embedded = concatenate([word_embeddings, char_embedding], axis=2)
        embedded = Dense(args.attention_dim, activation=tf.nn.tanh)(embedded)
        embedded = add([embedded, positional_encoding])
        embedded = SpatialDropout1D(args.input_dropout)(embedded)

        x = embedded
        for _ in range(args.layers):
            x = self.attention_layer(x)

        predictions = []
        for tag in range(MorphoDataset.TAGS):
            bias_init = tf.constant_initializer(self.bias(tag))
            tag_prediction = Dense(units=MorphoDataset.TAG_SIZES[tag] - 1, activation=None, bias_initializer=bias_init)(
                x)
            predictions.append(tag_prediction)

        self.model = tf.keras.Model(inputs=[word_embeddings, charseq_ids, charseqs, positional_encoding],
                                    outputs=predictions)

    def lr_decay(self):
        self._steps += 1

        if self._steps < 3000:
            self.learning_rate = args.attention_dim ** (-0.5) * self._steps * 3000 ** (-1.5) * args.learning_rate
        else:
            self.learning_rate = args.attention_dim ** (-0.5) * self._steps ** (-0.5) * args.learning_rate

        self._optimizer.learning_rate = self.learning_rate

    def bias(self, tag):
        prob = MorphoDataset.TAG_RATIOS[tag]
        logits = np.log(prob)
        return logits - np.mean(logits)

    def attention_layer(self, x):
        attentioned = self.fast_attention(x)
        attentioned = SpatialDropout1D(args.hidden_dropout)(attentioned)
        x = LayerNormalization(epsilon=1e-6)(add([x, attentioned]))

        forwarded = Dense(4 * args.attention_dim, activation=tf.nn.relu)(x)
        forwarded = Dense(args.attention_dim, activation=None)(forwarded)
        forwarded = SpatialDropout1D(args.hidden_dropout)(forwarded)
        return LayerNormalization(epsilon=1e-6)(add([x, forwarded]))

    def fast_attention(self, x):
        values = Dense(3 * args.attention_dim, activation=None)(x)
        values = Lambda(
            lambda x: tf.reshape(x, (tf.shape(x)[0], -1, args.heads, 3 * args.attention_dim // args.heads)))(values)
        values = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(values)
        Q, K, V = Lambda(lambda x: tf.split(x, 3, axis=3))(values)

        indices = Lambda(lambda qk: tf.linalg.matmul(qk[0], qk[1], transpose_b=True) / self.scale)((Q, K))
        indices = Lambda(lambda x: tf.nn.softmax(x))(indices)
        combined = Lambda(lambda iv: tf.linalg.matmul(iv[0], iv[1]))((indices, V))

        concat = Lambda(lambda c: tf.transpose(c, perm=[0, 2, 1, 3]))(combined)
        concat = Lambda(lambda c: tf.reshape(c, (tf.shape(c)[0], -1, args.attention_dim)))(concat)

        return Dense(args.attention_dim, activation=None)(concat)

    def slow_attention(self, x):
        heads = []
        for _ in range(args.heads):
            heads.append(self.attention_head(x))
        concatenated = concatenate(heads, axis=2)
        return Dense(args.attention_dim, activation=None)(concatenated)

    def attention_head(self, x):
        Q = Dense(args.attention_dim // args.heads)(x)
        K = Dense(args.attention_dim // args.heads)(x)
        V = Dense(args.attention_dim // args.heads)(x)

        indices = tf.linalg.matmul(Q, K, transpose_b=True)
        indices = tf.nn.softmax(indices / self.scale)
        return tf.linalg.matmul(indices, V)

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None, 256], dtype=tf.float32)] + [
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 2 + [
                                      tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)],
                                  [tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * MorphoDataset.TAGS])
    def train_batch(self, inputs, input_tags):
        masks, tags = [], []
        for input_tag in input_tags:
            mask = tf.not_equal(input_tag, 0)
            masks.append(mask)
            tags.append(input_tag - tf.cast(mask, tf.int32))

        with tf.GradientTape() as tape:
            probabilities = self.model(inputs, training=True)

            # gold = tf.one_hot(tags[0], MorphoDataset.TAG_SIZES[0]-1)
            loss = self._loss(tags[0], probabilities[0], masks[0])
            for tag in range(1, MorphoDataset.TAGS):
                # gold = tf.one_hot(tags[tag], MorphoDataset.TAG_SIZES[tag]-1)
                # gold = tags[tag]
                loss += self._loss(tags[tag], probabilities[tag], masks[tag]) / (MorphoDataset.TAGS - 1)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        if args.clip_gradient is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, args.clip_gradient)
        self._optimizer.apply_gradients(zip(gradients, self.model.variables))

        self._metrics["loss"](loss)
        self._metrics["accuracy"].update_state(tags[0], probabilities[0], masks[0])

    def train_epoch(self, epoch, dataset, log_file, args):
        #         if self.not_improved_epochs > 3:
        #             self.learning_rate /= 10
        #             self._optimizer.learning_rate = self.learning_rate
        #             self.not_improved_epochs = 0

        for name, metric in self._metrics.items():
            metric.reset_states()

        batches_num = dataset.size() / args.batch_size

        for i, batch in enumerate(dataset.batches(args.batch_size, args.max_length)):
            self.lr_decay()
            self.train_batch(
                [batch[dataset.FORMS].word_embeddings, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs,
                 batch[-1]],
                [batch[tag].word_ids for tag in
                 range(MorphoDataset.Dataset.TAGS_BEGIN, MorphoDataset.Dataset.TAGS_END)])

            log_train_progress(epoch, self._metrics["loss"].result(), 100 * self._metrics["accuracy"].result(),
                               int(i / batches_num * 100), log_file)
        log_train_progress(epoch, self._metrics["loss"].result(), 100 * self._metrics["accuracy"].result(), None,
                           log_file)

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None, 256], dtype=tf.float32)] + [
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 2 + [
                                      tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)],
                                  [tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * MorphoDataset.TAGS])
    def evaluate_batch(self, inputs, input_tags):
        masks, tags = [], []
        for input_tag in input_tags:
            mask = tf.not_equal(input_tag, 0)
            masks.append(mask)
            tags.append(input_tag - tf.cast(mask, tf.int32))

        probabilities = self.model(inputs, training=False)

        # gold = tf.one_hot(tags[0], MorphoDataset.TAG_SIZES[0]-1)
        loss = self._loss(tags[0], probabilities[0], masks[0])
        self._metrics["accuracy"].update_state(tags[0], probabilities[0], masks[0])

        for tag in range(1, MorphoDataset.TAGS):
            # gold = tf.one_hot(tags[tag], MorphoDataset.TAG_SIZES[tag]-1)
            loss += self._loss(tags[tag], probabilities[tag], masks[tag]) / (MorphoDataset.TAGS - 1)
        self._metrics["loss"](loss)

    def evaluate_analyzed_batch(self, inputs, input_tags, dataset, analyses):
        mask = tf.not_equal(input_tags[0], 0)
        tags = [tag - tf.cast(mask, tf.int32) for tag in input_tags]

        probabilities = self.model(inputs, training=False)

        loss = 0
        for tag in range(MorphoDataset.TAGS):
            # gold = tf.one_hot(tags[tag], MorphoDataset.TAG_SIZES[tag]-1)
            gold = tags[tag]
            loss += self._loss(gold, probabilities[tag], mask)
            self._metrics["accuracy"].update_state(tags[tag], probabilities[tag], mask)
        self._metrics["loss"](loss / MorphoDataset.TAGS)

        correct, total = 0, 1

        for b in range(inputs[0].shape[0]):
            for w in range(inputs[0].shape[1]):
                if mask[b, w] == False: break

                input = dataset.data[dataset.FORMS].words[inputs[0][b, w]]
                suggestions = analyses.get_tag_ids(input)
                gold = np.array([tags[t][b, w] for t in range(MorphoDataset.TAGS)])

                if len(suggestions) == 0:
                    prediction = np.array([tf.argmax(probabilities[t][b, w], axis=-1, output_type=tf.int32) for t in
                                           range(MorphoDataset.TAGS)])
                else:
                    best = None
                    for s in suggestions:
                        prob = np.product(np.array([probabilities[i][b, w, t] for i, t in enumerate(s)]))
                        if best is None or prob > best[1]:
                            best = (s, prob)
                    prediction = np.array(best[0])

                if np.array_equal(gold, prediction): correct += 1
                total += 1

        self._metrics["real accuracy"](tf.reduce_mean(correct / total))

    def evaluate(self, dataset, log_file, args, analyses=None):
        for metric in self._metrics.values():
            metric.reset_states()

        for i, batch in enumerate(dataset.batches(128, 1000)):
            if analyses is None:
                self.evaluate_batch(
                    [batch[dataset.FORMS].word_embeddings, batch[dataset.FORMS].charseq_ids,
                     batch[dataset.FORMS].charseqs, batch[-1]],
                    [batch[tag].word_ids for tag in
                     range(MorphoDataset.Dataset.TAGS_BEGIN, MorphoDataset.Dataset.TAGS_END)])
            else:
                self.evaluate_analyzed_batch(
                    [batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs],
                    [batch[tag].word_ids for tag in
                     range(MorphoDataset.Dataset.TAGS_BEGIN, MorphoDataset.Dataset.TAGS_END)],
                    dataset, analyses)

        log_dev_progress(self._metrics["loss"].result(), 100 * self._metrics["accuracy"].result(), self.learning_rate,
                         log_file)

        if self._metrics["accuracy"].result() > self.best_accuracy:
            # self.model.save_weights(f"{args.directory}/acc-{100 * self.best_accuracy:2.3f}.h5")
            self.best_accuracy = self._metrics["accuracy"].result()
        else:
            self.not_improved_epochs += 1

        if self._metrics["accuracy"].result() > 0.975:
            accuracy = self._metrics["accuracy"].result()
            self.model.save_weights(f"{args.directory}/acc-{100 * accuracy:2.3f}.h5")

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None, 256], dtype=tf.float32)] + [
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 2 + [
                                      tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)]])
    def predict_batch(self, inputs):
        probabilities = self.model(inputs, training=False)
        return np.argmax(probabilities, axis=2)

    def predict(self, input):
        pass


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=96, type=int, help="CLE embedding dimension.")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="Initial learning rate.")
    parser.add_argument("--clip_gradient", default=1.0, type=float, help="Clip the gradient norm.")
    parser.add_argument("--input_dropout", default=0.42, type=float, help="Dropout to the input embedding.")
    parser.add_argument("--hidden_dropout", default=0.2, type=float, help="Dropout between layers.")
    parser.add_argument("--cle_layers", default=2, type=int, help="CLE embedding layers.")
    parser.add_argument("--cnn_filters", default=64, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnn_max_width", default=5, type=int, help="Maximum CNN filter width.")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.")
    parser.add_argument("--layers", default=4, type=int, help="RNN layers.")
    parser.add_argument("--max_length", default=60, type=int, help="Max number of words in a sentence")
    parser.add_argument("--attention_dim", default=256, type=int, help="RNN cell dimension.")
    parser.add_argument("--heads", default=8, type=int, help="RNN cell dimension.")
    parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
    parser.add_argument("--base_directory", default="drive/My Drive/Colab Notebooks/Tagger", type=str,
                        help="Directory for the outputs.")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint of a model to load weights from.")
    args = parser.parse_args()

    architecture = ",".join(
        ("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if
         key not in ["directory", "base_directory", "epochs", "batch_size", "clip_gradient", "checkpoint"]))
    args.directory = f"{args.base_directory}/models/attention_{architecture}"
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset(args.base_directory, "czech_pdt3_sem_divided", args.attention_dim)
    # analyses = MorphoAnalyzer("czech_pdt_analyses", morpho.train)

    # Create the network and train
    network = Network(args, num_chars=len(morpho.train.data[morpho.train.FORMS].alphabet))
    if args.checkpoint is not None:
        network.model.load_weights(f"{args.directory}/{args.checkpoint}")
        network.learning_rate = 0.003

    for epoch in range(args.epochs):
        with open(f"{args.directory}/log.txt", "a", encoding="utf-8") as log_file:
            network.train_epoch(epoch, morpho.train, log_file, args)
            network.evaluate(morpho.dev, log_file, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
#     out_path = f"{args.directory}/test.txt"
#     with open(out_path, "w", encoding="utf-8") as out_file:
#         for i, sentence in enumerate(network.predict(morpho.test, args)):
#             for j in range(len(morpho.test.data[morpho.test.FORMS].word_strings[i])):
#                 print(morpho.test.data[morpho.test.FORMS].word_strings[i][j],
#                       morpho.test.data[morpho.test.LEMMAS].word_strings[i][j],
#                       morpho.test.data[morpho.test.TAGS].words[sentence[j]],
#                       sep="\t", file=out_file)
#             print(file=out_file)