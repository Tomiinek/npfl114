#!/usr/bin/env python3

# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub # Note: you need to install tensorflow_hub

from caltech42 import Caltech42, center_crop

# The neural network model
class Network:
    def __init__(self, args):
        x = inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
        x = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)(x, training=False)
        x = tf.keras.layers.Dense(Caltech42.LABELS, activation="softmax")(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=x)
        
        # TODO: You should define `self.model`. You should use the following layer:
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280])
        # The layer:
        # - if given `trainable=True/False` to KerasLayer constructor, the layer weights
        #   either are marked or not marked as updatable by an optimizer;
        # - however, batch normalization regime is set independently, by `training=True/False`
        #   passed during layer execution.
        #
        # Therefore, to not train the layer at all, you should use
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)
        #   features = mobilenet(inputs, training=False)
        # On the other hand, to fully train it, you should use
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=True)
        #   features = mobilenet(inputs)
        # where the `training` argument to `mobilenet` is passed automatically in that case.
        #
        # Note that a model with KerasLayer can currently be saved only using
        #   tf.keras.experimental.export_saved_model(model, path, serving_only=True/False)
        # where `serving_only` controls whether only prediction, or also training/evaluation
        # graphs are saved. To again load the model, use
        #   model = tf.keras.experimental.load_from_saved_model(path, {"KerasLayer": tfhub.KerasLayer})

    def train(self, caltech42, args):
        self.model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
        
        tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        tb_callback.on_train_end = lambda *_: None
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            args.checkpoint_path, monitor="val_acc", save_best_only=True, save_weights_only=True)
        
        self.model.fit_generator(
            generator=caltech42.train.batches(args.batch_size, repeat=True),
            steps_per_epoch=caltech42.train.batched_size(args.batch_size),
            epochs=args.epochs,
            validation_data=caltech42.dev.batches(args.batch_size, repeat=True),
            validation_steps=caltech42.dev.batched_size(args.batch_size),
            callbacks=[tb_callback, checkpoint_callback]),
        verbose=2

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
    caltech42 = Caltech42(center_crop, center_crop)

    # Create the network and train
    network = Network(args)
    network.train(caltech42, args)

#     # Generate test set annotations, but in args.logdir to allow parallel execution.
#     with open(os.path.join(args.logdir, "caltech42_competition_test.txt"), "w", encoding="utf-8") as out_file:
#         for probs in network.predict(caltech42.test, args):
#             print(np.argmax(probs), file=out_file)
