#!/usr/bin/env python3

# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import sys
import math
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

from caltech42_crossvalidation import Caltech42
import caltech42_augmentor as aug

tf.config.gpu.set_per_process_memory_growth(True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


class SpickaNet(tf.keras.Sequential):
    def __init__(self):
        super(SpickaNet, self).__init__([
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(Caltech42.LABELS, activation=None)
        ])


class Network:
    def __init__(self, args):
        
        self.models = []
        self.spickas = []

        for fold in range(args.folds):
            x = inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
            x = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)(x, training=False)
            
            spicka = SpickaNet()
            spicka.load_weights(args.model_path.format(fold))
            x = spicka(x)
        
            model = tf.keras.Model(inputs=inputs, outputs=x)
            self.models.append(model)
            self.spickas.append(spicka)        

    def test(self, caltech42, args):
        for i, model in enumerate(self.models):
            train_step = caltech42.folds[i].train.batched_size(args.batch_size)
            learning_rate = tf.optimizers.schedules.PiecewiseConstantDecay(
                [60.0*train_step, 120.0*train_step, 180.0*train_step],
                [0.003, 0.0003, 0.0001, 0.00001]
            )
            model.compile(
                #tf.optimizers.SGD(learning_rate=0.1),
                tf.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
            )

        accuracy = 0.0
        m = np.zeros((42,42))
        false_labels = 42*[0]
        for idx, dataset in enumerate(caltech42.folds):
            total, correct = 0, 0
            for i in range(dataset.dev.size):
                image, label = dataset.dev.data["images"][i], np.argmax(dataset.dev.data["labels"][i])
                batch, weights = aug.create_inference_augmented_batch(image)
                
                probabilities = tf.nn.softmax(self.models[idx].predict(batch))
                probabilities = np.sum(probabilities*tf.expand_dims(weights, 1), axis=0)
                prediction = np.argmax(probabilities)
                
                total += 1
                if prediction == label: correct += 1
                m[prediction, label] += 1

            accuracy += correct / total
            print(accuracy / (idx + 1))

        print(accuracy / args.folds)
        
        np.savetxt("matrix.csv", m, delimiter=",")

    def predict(self, caltech42, args):
        return self.model.predict_generator(
            generator=caltech42.train.data.batches(args.batch_size),
            steps=caltech42.train.batched_size(args.batch_size))


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--model_path", default='models/acc-0.9635594666004181_fold-{}', type=str)
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=5, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--folds", default=10, type=int, help="Number of crossvalidation folds.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load data
    caltech42 = Caltech42(aug.augment, aug.center_crop, sparse_labels=False, folds=args.folds)

    # Create the network and train
    network = Network(args)
    network.test(caltech42, args)

#     # Generate test set annotations, but in args.logdir to allow parallel execution.
#     with open(os.path.join(args.logdir, "caltech42_competition_test.txt"), "w", encoding="utf-8") as out_file:
#         for probs in network.predict(caltech42.test, args):
#             print(np.argmax(probs), file=out_file)
