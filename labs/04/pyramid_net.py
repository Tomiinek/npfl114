#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from cifar10_augmented import CIFAR10


def get_l2(scale):
    # return tf.contrib.layers.l2_regularizer(scale)
    return None


class Network():
    LABELS = 10
    INPUT_FILTERS = 16

    def __init__(self, args, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.inter_op_parallelism_threads = threads
        self.session = tf.Session(graph=graph, config=config)
        # Number of layers in group (minus first and last layer, three groups with three layers)
        self.group_depth = int((args.depth - 2) / 9)
        self.alpha = args.alpha
        # Add rate per layer. Three groups and self.n layers (bottlenecks) in a group.
        self.add_rate = self.alpha / (3.0 * self.group_depth)
        self.current_filters = self.INPUT_FILTERS
        self.batch_size = args.batch_size
        self.l2_scale = args.weight_decay

    def train(self, images, labels):
        a, _ = self.session.run([self.accuracy, self.training],
                                {self.is_training: True, self.images: images, self.labels: labels})
        return a

    def evaluate(self, images, labels):
        a, p = self.session.run([self.accuracy, self.predictions],
                                {self.is_training: False, self.images: images, self.labels: labels})
        return a, p

    def construct(self, args, train_steps):
        with self.session.graph.as_default():
            self.images = tf.placeholder(tf.float32, [None, CIFAR10.H, CIFAR10.W, CIFAR10.C], name="images")
            self.labels = tf.placeholder(tf.int32, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            x = tf.layers.conv2d(self.images, filters=self.INPUT_FILTERS, kernel_size=[3, 3],
                                 strides=1, padding='same', kernel_regularizer=get_l2(self.l2_scale),
                                 kernel_initializer='he_normal', use_bias=False)
            x = tf.layers.batch_normalization(x, training=self.is_training)

            x = self._pyramidal_group(x, downsample=False)  # No pooling
            x = self._pyramidal_group(x, downsample=True)  # Puling ve dvi, should be 16x16
            x = self._pyramidal_group(x, downsample=True)  # Puling ve dvi, should be 8x8

            x = tf.layers.batch_normalization(x, training=self.is_training)
            x = tf.nn.relu(x)
            x = tf.reduce_mean(x, axis=[1, 2])  # global average pooling
            outputs = tf.layers.dense(x, self.LABELS, activation=tf.nn.softmax, name="output_layer")

            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, outputs, scope="loss")
            self.predictions = tf.argmax(outputs, axis=1, name="activations", output_type=tf.int32)

            global_step = tf.train.create_global_step()
            boundaries = [int(train_steps / 2), int(train_steps * 3 / 4)]
            values = [0.1, 0.01, 0.001]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, True)
            self.training = optimizer.minimize(loss, global_step=global_step, name="training")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            self.session.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()

    def save(self, path):
        self.saver.save(self.session, path, write_meta_graph=False, write_state=False)

    def _pyramidal_group(self, x, downsample):

        self.current_filters += self.add_rate
        x = self._bottleneck_block(int(round(self.current_filters)), 2 if downsample else 1, x, downsample)

        for i in range(1, self.group_depth):
            self.current_filters += self.add_rate
            x = self._bottleneck_block(int(round(self.current_filters)), 1, x)
        self.input_filters = int(round(self.current_filters)) * 4

        return x

    def _bottleneck_block(self, filters, strides, x, downsample=False):

        if downsample:
            shortcut = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
        else:
            shortcut = x

        # alphas = tf.random_uniform([self.batch_size, 1, 1, 1], dtype=tf.float32)
        # betas = tf.random_uniform([self.batch_size, 1, 1, 1], dtype=tf.float32)

        x = tf.layers.batch_normalization(x, training=self.is_training)
        x = tf.layers.conv2d(x, filters=filters, kernel_size=1, strides=1, padding='same',
                             kernel_regularizer=get_l2(self.l2_scale), kernel_initializer='he_normal', use_bias=False)
        x = tf.layers.batch_normalization(x, training=self.is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters=filters, kernel_size=3, strides=strides, padding='same',
                             kernel_regularizer=get_l2(self.l2_scale), kernel_initializer='he_normal', use_bias=False)
        x = tf.layers.batch_normalization(x, training=self.is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters=filters * 4, kernel_size=1, strides=1, padding='same',
                             kernel_regularizer=get_l2(self.l2_scale), kernel_initializer='he_normal', use_bias=False)
        x = tf.layers.batch_normalization(x, training=self.is_training)

        padding = tf.constant([[0, 0], [0, 0], [0, 0], [0, (x.shape[3] - shortcut.shape[3]).value]])
        shortcut = tf.pad(shortcut, padding)

        return x + shortcut


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=15, type=int, help="Depth.")
    parser.add_argument("--alpha", default=5, type=int, help="Alpha.")
    parser.add_argument("--weight_decay", default=0.0005 / 2, type=int, help="L2 regularization parameter.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0 for no label smoothing")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Evaluation batch size.")
    parser.add_argument("--epochs", default=300, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)

    # Load data
    cifar = CIFAR10(True)

    # Create the network and train
    network = Network(args, threads=args.threads)
    network.construct(args, (cifar.train.size // args.batch_size) * args.epochs)

    # Train
    best = 0
    train_gen = cifar.train.batches(args.batch_size)
    dev_gen = cifar.dev.batches(args.eval_batch_size)
    for i in range(args.epochs):
        print("Epoch {} acc:".format(i), end="")
        for j in range(cifar.train.size // args.batch_size):
            batch = next(train_gen)
            a = network.train(batch[0], np.squeeze(batch[1]))
            if j % 64 == 0: print(" {:.2f}".format(100 * a), end="")
        print("")

        accuracy = 0.0
        batch_count = 0
        for _ in range(cifar.dev.size // args.eval_batch_size):
            batch = next(dev_gen)
            a, _ = network.evaluate(batch[0], np.squeeze(batch[1]))
            accuracy += a * len(batch[0])
            batch_count += len(batch[0])
        accuracy /= batch_count
        if accuracy > best:
            best = accuracy
            network.save("models/acc={}_depth={}_alpha={}".format(accuracy, args.depth, args.alpha))
        print("--------------- Eval acc: {:.2f} --------------".format(100 * accuracy))
