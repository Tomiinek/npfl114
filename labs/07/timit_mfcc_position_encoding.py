import os
import sys
import pickle

import numpy as np
import tensorflow as tf

class TimitMFCC:
    LETTERS = [
        "<pad>", "_", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    ]

    MFCC_DIM = 26

    MEAN = np.array([[
        -7.3310425e+01,  8.7198648e+00, -2.5698996e+00,  2.1943305e-01,
        -3.2655041e+00, -1.8872676e+00, -1.4643098e+00, -1.4495075e+00,
        -3.2618607e-03, -5.5006921e-01, -1.1491005e-01, -3.1593916e-01,
        -3.2570928e-01,  7.0352770e-02,  2.2284938e-02,  5.4601543e-03,
        -4.8744259e-04, -3.4759252e-04,  6.8471916e-03,  1.1420489e-04,
         2.9241682e-03, -1.4091476e-03, -2.8760815e-03, -6.8956008e-04,
        -4.0535578e-03, -2.9590982e-03]])
    STD = np.array([[
        33.11649,   12.926924,   6.6708994,  5.6361804,  4.5045495,  3.5361245,
         3.2106228,  3.0031688,  2.460201,   2.4700053,  2.1056616,  2.1891782,
         1.8670087, 22.710585,   8.855508,   5.1357937,  3.6156619,  3.145585,
         2.6496606,  2.3845098,  2.3035953,  1.9821684,  1.9326628,  1.7434945,
         1.6861254,  1.5864328]])

    class Dataset:
        def __init__(self, data, shuffle_batches, args, seed=42):
            self._data = {}
            self._data["mfcc"] = [np.tanh((d - TimitMFCC.MEAN) / TimitMFCC.STD) for d in data["mfcc"]]
            self._data["letters"] = [letters + 1 for letters in data["letters"]]
            self._size = len(self._data["mfcc"])
            self._dim = args.dim

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None
            self._positions = self.positional_encoding(4096, self._dim)

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None):
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {}
                for key, values in self._data.items():
                    max_length = max(len(values[i]) for i in batch_perm)
                    batch[key] = np.zeros([batch_size, max_length, *values[batch_perm[0]].shape[1:]], values[batch_perm[0]].dtype)
                    batch[key + "_len"] = np.zeros([batch_size], dtype=np.int32)

                    for i, index in enumerate(batch_perm):
                        batch[key][i][:len(values[index])] = values[index]
                        batch[key + "_len"][i] = len(values[index])

                max_length = max(len(self._data["mfcc"][i]) for i in batch_perm)
                batch["positional_encoding"] = np.zeros([batch_size, max_length, self._dim])
                for i, index in enumerate(batch_perm):
                    length = len(self._data["mfcc"][index])
                    batch["positional_encoding"][i, :length, :] = self._positions[0, :length, :]

                yield batch

        def get_angles(self, pos, i, d_model):
            angle_rates = 1 / np.power(1000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        def positional_encoding(self, position, d_model):
            angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                         np.arange(d_model)[np.newaxis, :],
                                         d_model)
            # apply sin to even indices in the array; 2i
            sines = np.sin(angle_rads[:, 0::2])
            # apply cos to odd indices in the array; 2i+1
            cosines = np.cos(angle_rads[:, 1::2])
            pos_encoding = np.concatenate([sines, cosines], axis=-1)
            return pos_encoding[np.newaxis, ...]

    def __init__(self, args, path="timit_mfcc.pickle"):
        if not os.path.exists(path):
            print("The Timit dataset is not public, you need to manually download\n" +
                  "timit_mfcc.pickle file from ReCodEx.", file=sys.stderr)
            sys.exit(1)

        with open(path, "rb") as timit_mfcc_file:
            data = pickle.load(timit_mfcc_file)

        for dataset in ["train", "dev", "test"]:
            setattr(self, dataset, self.Dataset(data[dataset], shuffle_batches=dataset == "train", args=args))