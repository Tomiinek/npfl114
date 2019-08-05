# !/usr/bin/env python3
import sys
import math
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

from caltech42_crossvalidation import Caltech42
import caltech42_augmentor as aug

from tensorflow.keras.applications import MobileNetV2

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


class SpickaNet(tf.keras.Sequential):
    def __init__(self):
        super(SpickaNet, self).__init__([
            tf.keras.layers.GlobalAveragePooling2D(),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(210, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(Caltech42.LABELS, activation=None)
        ])




class Network:
    def __init__(self, args):

        self.models = []
        self.spickas = []

        for _ in range(args.folds):
            # x = inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
            base_model = MobileNetV2(weights='imagenet', include_top=False)
            # MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

            # for layer in base_model.layers[:144]: layer.trainable = False
            # for layer in base_model.layers[144:]: layer.trainable = True
            for layer in base_model.layers[:]: layer.trainable = False

            x = base_model.output
            spicka = SpickaNet()
            x = spicka(x)

            model = tf.keras.Model(inputs=base_model.inputs, outputs=x)
            self.models.append(model)
            self.spickas.append(spicka)

    def train(self, caltech42, args):
        for i, model in enumerate(self.models):
            train_step = caltech42.folds[i].train.batched_size(args.batch_size)
            learning_rate = tf.optimizers.schedules.PiecewiseConstantDecay(
                [60.0 * train_step, 120.0 * train_step, 180.0 * train_step],
                [0.001, 0.0003, 0.0001, 0.00001]
            )
            model.compile(
                # tf.optimizers.SGD(learning_rate=0.1),
                tf.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
            )

        best_acc = 0
        for i in range(args.epochs):
            accuracy_train = accuracy_test = 0
            for idx, dataset in enumerate(caltech42.folds):
                train_logs = self.models[idx].fit_generator(
                    generator=dataset.train.batches(args.batch_size, repeat=True),
                    steps_per_epoch=dataset.train.batched_size(args.batch_size),
                    epochs=1,
                    validation_data=dataset.dev.batches(args.batch_size, repeat=True),
                    validation_steps=dataset.dev.batched_size(args.batch_size),
                    # callbacks=[tf.keras.callbacks.ModelCheckpoint(filename, monitor='val_accuracy', save_best_only=True)],
                    # callbacks=[tb_callback, checkpoint_callback],
                    verbose=0)

                accuracy_train += train_logs.history.get('accuracy')[-1]

                test_logs = self.models[idx].evaluate_generator(
                    generator=dataset.dev.batches(args.batch_size, repeat=False),
                    steps=dataset.dev.batched_size(args.batch_size),
                    verbose=0
                )

                accuracy_test += test_logs[self.models[idx].metrics_names.index("accuracy")]
                print("\repoch {:2d} | fold {:2d}/{} | train acc: {:.3f} % | test acc: {:.3f} %".format(i + 1, idx + 1,
                                                                                                        args.folds,
                                                                                                        100 * accuracy_train / (
                                                                                                                    idx + 1),
                                                                                                        100 * accuracy_test / (
                                                                                                                    idx + 1)),
                      end='', flush=True)

            if accuracy_test / args.folds > best_acc:
                best_acc = accuracy_test / args.folds
                # for i, spicka in enumerate(self.spickas):
                #    spicka.save_weights("models/acc-{}_fold-{}".format(best_acc, i))

            print(" | best acc: {:.3f}".format(100 * best_acc), flush=True)

    def predict(self, caltech42, args):
        return self.model.predict_generator(
            generator=caltech42.train.data.batches(args.batch_size),
            steps=caltech42.train.batched_size(args.batch_size))


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=5, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--folds", default=10, type=int, help="Number of crossvalidation folds.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))
    checkpoint_dir = os.path.join(args.logdir, "weights")
    os.makedirs(checkpoint_dir)
    args.checkpoint_path = os.path.join(checkpoint_dir, "{val_acc:.4f}")

    # Load data
    caltech42 = Caltech42(aug.augment, aug.center_crop, sparse_labels=False, folds=args.folds)

    # Create the network and train
    network = Network(args)
    network.train(caltech42, args)

#     # Generate test set annotations, but in args.logdir to allow parallel execution.
#     with open(os.path.join(args.logdir, "caltech42_competition_test.txt"), "w", encoding="utf-8") as out_file:
#         for probs in network.predict(caltech42.test, args):
#             print(np.argmax(probs), file=out_file)
