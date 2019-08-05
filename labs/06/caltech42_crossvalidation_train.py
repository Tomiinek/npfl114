#!/usr/bin/env python3

# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub # Note: you need to install tensorflow_hub

from caltech42_crossvalidation import Caltech42, center_crop

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


class Network:
    def __init__(self, args):
        self.models = []
        for _ in range(args.folds):
            x = inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
            x = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)(x, training=False)
            x = tf.keras.layers.Dense(Caltech42.LABELS, activation=None)(x)
        
            model = tf.keras.Model(inputs=inputs, outputs=x)
            model.compile(
                "adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
            self.models.append(model)
        

    def train(self, caltech42, args): 
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
                print("\repoch {:2d} | fold {:2d}/{} | train acc: {:.3f} % | test acc: {:.3f} %".format(i+1, idx+1, args.folds, 100*accuracy_train/(idx+1), 100*accuracy_test/(idx+1)), end='')
            print()

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
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
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
    caltech42 = Caltech42(center_crop, center_crop, sparse_labels=True, folds=args.folds)

    # Create the network and train
    network = Network(args)
    network.train(caltech42, args)

#     # Generate test set annotations, but in args.logdir to allow parallel execution.
#     with open(os.path.join(args.logdir, "caltech42_competition_test.txt"), "w", encoding="utf-8") as out_file:
#         for probs in network.predict(caltech42.test, args):
#             print(np.argmax(probs), file=out_file)
