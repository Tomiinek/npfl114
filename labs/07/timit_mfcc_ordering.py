import os
import sys
import pickle

import numpy as np


class TimitMFCC:
    LETTERS = [
        "<pad>", "_", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    ]

    MFCC_DIM = 26

    MEAN = np.array([[
        -7.3310425e+01, 8.7198648e+00, -2.5698996e+00, 2.1943305e-01,
        -3.2655041e+00, -1.8872676e+00, -1.4643098e+00, -1.4495075e+00,
        -3.2618607e-03, -5.5006921e-01, -1.1491005e-01, -3.1593916e-01,
        -3.2570928e-01, 7.0352770e-02, 2.2284938e-02, 5.4601543e-03,
        -4.8744259e-04, -3.4759252e-04, 6.8471916e-03, 1.1420489e-04,
        2.9241682e-03, -1.4091476e-03, -2.8760815e-03, -6.8956008e-04,
        -4.0535578e-03, -2.9590982e-03]])
    STD = np.array([[
        33.11649, 12.926924, 6.6708994, 5.6361804, 4.5045495, 3.5361245,
        3.2106228, 3.0031688, 2.460201, 2.4700053, 2.1056616, 2.1891782,
        1.8670087, 22.710585, 8.855508, 5.1357937, 3.6156619, 3.145585,
        2.6496606, 2.3845098, 2.3035953, 1.9821684, 1.9326628, 1.7434945,
        1.6861254, 1.5864328]])

    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42):
            self._data = {}
            self._data["mfcc"] = [np.tanh((d - TimitMFCC.MEAN) / TimitMFCC.STD) for d in data["mfcc"]]
            self._data["letters"] = [letters + 1 for letters in data["letters"]]
            self._size = len(self._data["mfcc"])

            index_order = [i for i, _ in sorted(enumerate(self._data["mfcc"]), key=lambda x: len(x[1]))]
            self._data = {k: [v[i] for i in index_order] for k, v in self._data.items()}

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None, prioritize_short=0.0):

            buckets_num = int(self._size * prioritize_short)
            bucket_indices = np.array(list(self.equally_spaced(self._size, buckets_num)))
            bucket_indices = np.concatenate(([0], np.cumsum(bucket_indices)))

            partial_permutations = []
            for i in range(len(bucket_indices) - 1):
                bucket_size = bucket_indices[i + 1] - bucket_indices[i]
                partial_permutations.append(
                    bucket_indices[i] + self._shuffler.permutation(bucket_size) if self._shuffler else np.arange(
                        bucket_size))

            permutation = np.concatenate(partial_permutations)

            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {}
                for key, values in self._data.items():
                    max_length = max(len(values[i]) for i in batch_perm)
                    batch[key] = np.zeros([batch_size, max_length, *values[batch_perm[0]].shape[1:]],
                                          values[batch_perm[0]].dtype)
                    batch[key + "_len"] = np.zeros([batch_size], dtype=np.int32)

                    for i, index in enumerate(batch_perm):
                        batch[key][i][:len(values[index])] = values[index]
                        batch[key + "_len"][i] = len(values[index])

                yield batch

        def equally_spaced(self, total, groups):
            if groups == 0:
                yield total
            else:
                d, s = total / groups, 0
                for _ in range(groups):
                    yield int(s + d) - int(s)
                    s += d

    def __init__(self, path="timit_mfcc.pickle"):
        if not os.path.exists(path):
            print("The Timit dataset is not public, you need to manually download\n" +
                  "timit_mfcc.pickle file from ReCodEx.", file=sys.stderr)
            sys.exit(1)

        with open(path, "rb") as timit_mfcc_file:
            data = pickle.load(timit_mfcc_file)

        for dataset in ["train", "dev", "test"]:
            setattr(self, dataset, self.Dataset(data[dataset], shuffle_batches=dataset == "train"))