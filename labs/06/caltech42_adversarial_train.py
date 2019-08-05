#!/usr/bin/env python3

# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub # Note: you need to install tensorflow_hub
from tqdm import tqdm

from caltech42 import Caltech42, center_crop

# see https://arxiv.org/pdf/1412.6572.pdf for more on adversarial training

# The neural network model
class Network:
    def __init__(self, args):
        x = inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
        x = bottlenecks = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)(x, training=False)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(Caltech42.LABELS, activation="softmax")(x)
        
        self.bottlenecks = tf.keras.Model(inputs=inputs, outputs=bottlenecks)
        self.model = tf.keras.Model(inputs=inputs, outputs=x)
        self.classifier = tf.keras.Model(inputs=bottlenecks, outputs=x)
        
    @tf.function
    def train_on_batch(self, batch_X, batch_y, epsilon):
        
        with tf.GradientTape() as tape:
            tape.watch(batch_X)
            probas = self.model(batch_X, training=True)
            loss = self.loss(batch_y, probas)
        input_gradients, batch_gradients = tape.gradient(loss, (batch_X, self.model.trainable_variables))
        
        self.train_acc(batch_y, probas)
        
        self.optimizer.apply_gradients(zip(batch_gradients, self.model.trainable_variables))

        # TODO try apply gradients here - Nesterov-ish way
        
        adversarial_X = batch_X + epsilon * tf.sign(input_gradients)
        with tf.GradientTape() as tape:
            probas = self.model(adversarial_X, training=True)
            loss = self.loss(batch_y, probas)
        adversarial_gradients = tape.gradient(loss, self.model.trainable_variables)

        self.adversarial_acc(batch_y, probas)
        
        self.optimizer.apply_gradients(zip(adversarial_gradients, self.model.trainable_variables))
        
#         with tf.GradientTape() as tape:
#             tape.watch(batch_X)
#             probas = self.model(batch_X, training=True)
#             loss = self.loss(batch_y, probas)
#         input_gradients, batch_gradients = tape.gradient(loss, (batch_X, self.model.trainable_variables))
        
#         self.train_acc(batch_y, probas)
        
#         self.optimizer.apply_gradients(zip(batch_gradients, self.model.trainable_variables))

#         # TODO try apply gradients here - Nesterov-ish way
        
#         adversarial_X = batch_X + epsilon * tf.sign(input_gradients)
#         with tf.GradientTape() as tape:
#             probas = self.model(adversarial_X, training=True)
#             loss = self.loss(batch_y, probas)
#         adversarial_gradients = tape.gradient(loss, self.model.trainable_variables)

#         self.adversarial_acc(batch_y, probas)
        
#         self.optimizer.apply_gradients(zip(adversarial_gradients, self.model.trainable_variables))
    
    def train(self, caltech42, args):
        self.optimizer = tf.optimizers.Adam()
        if args.label_smoothing:
            self.loss = tf.losses.CategoricalCrossentropy()
            self.train_acc = tf.metrics.CategoricalAccuracy()
            self.adversarial_acc = tf.metrics.CategoricalAccuracy()
            self.val_acc = tf.metrics.CategoricalAccuracy()
        else:
            self.loss = tf.losses.SparseCategoricalCrossentropy()
            self.train_acc = tf.metrics.SparseCategoricalAccuracy()
            self.adversarial_acc = tf.metrics.SparseCategoricalAccuracy()
            self.val_acc = tf.metrics.SparseCategoricalAccuracy()
        
        for epoch_i in range(args.epochs):
            self.train_acc.reset_states()
            self.adversarial_acc.reset_states()
            self.val_acc.reset_states()
            
            # train
            for (X, y) in tqdm(caltech42.train.batches(args.batch_size), leave=False, total=caltech42.train.batched_size(args.batch_size)):
                self.train_on_batch(X, y, args.epsilon)
                
            # validate
            for (X, y) in caltech42.dev.batches(args.batch_size):
                probas = self.model(X, training=False)
                self.val_acc(y, probas)
            
            print('{:02d} - train {:.4f} - adv {:.4f} - val {:.4f}'.format(
                epoch_i, self.train_acc.result(),
                self.adversarial_acc.result(), self.val_acc.result()))
            
            self.model.save_weights(args.checkpoint_path.format(val_acc=self.val_acc.result()))
            
    def predict(self, caltech42, args):
        return self.model.predict_generator(
            generator=caltech42.train.data.batches(args.batch_size),
            steps=caltech42.train.batched_size(args.batch_size))


def augment(image):
    # random crop
    y_margin, x_margin = np.asarray(image.shape[:2]) - Caltech42.MIN_SIZE
    t, l = np.random.randint(y_margin + 1), np.random.randint(x_margin + 1)
    
    image = image[t:(t + Caltech42.MIN_SIZE), l:(l + Caltech42.MIN_SIZE)]
    
    # random flip
    if np.random.random() > 0.5:
        image = image[:, ::-1, :]
        
    return image
    

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing intensity.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Adversarial loss parameter.")
    parser.add_argument("--threads", default=5, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.gpu.set_per_process_memory_growth(True)

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
    caltech42 = Caltech42(augment, center_crop)

    # Create the network and train
    network = Network(args)
    network.train(caltech42, args)

#     # Generate test set annotations, but in args.logdir to allow parallel execution.
#     with open(os.path.join(args.logdir, "caltech42_competition_test.txt"), "w", encoding="utf-8") as out_file:
#         for probs in network.predict(caltech42.test, args):
#             print(np.argmax(probs), file=out_file)
