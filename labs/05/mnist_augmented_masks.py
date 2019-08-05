# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import os
import sys
import urllib.request

import numpy as np
import tensorflow as tf

class MNIST:
    H, W, C = 28, 28, 1
    LABELS = 10

    MEAN = 0.6432103514671326
    STD = 0.26544612646102905

    CUTOUT_PROB = 0.5
    CUTOUT_SIZE = 14

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1819/datasets/fashion_masks_data.npz"

    class Dataset:
        def __init__(self, data, shuffle_batches, augment, sparse_labels=True, seed=42):
            self._data = data
            self._data["images"] = self._data["images"].astype(np.float32) / 255
            self._size = len(self._data["images"])
            self._use_augmentation = augment

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

            self._data["images"] = MNIST.normalize(self._data["images"])
            if not sparse_labels:
                self._data["labels"][self._data["labels"] == 255] = 0
                self._data["labels"] = tf.keras.utils.to_categorical(self._data["labels"], num_classes=MNIST.LABELS)

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None):
            while True:
                permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
                while len(permutation):
                    batch_size = min(size or np.inf, len(permutation))
                    batch_perm = permutation[:batch_size]
                    permutation = permutation[batch_size:]

                    labels = self._data["labels"][batch_perm]
                    if self._use_augmentation:
                        images = []
                        masks = []
                        for t in zip(self._data["images"][batch_perm], self._data["masks"][batch_perm]):
                            ai, am = MNIST.augment(*t)
                            images.append(ai)
                            masks.append(am)
                    else:
                        images = self._data["images"][batch_perm]
                        masks = self._data["masks"][batch_perm]

                    yield (np.array(images), {'class_output': labels, 'mask_output': np.array(masks)})


    def __init__(self, sparse_labels=True):
        path = os.path.basename(self._URL)
        if not os.path.exists(path):
            print("Downloading FashionMasks dataset...", file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename=path)

        fashion_masks = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = dict((key[len(dataset) + 1:], fashion_masks[key]) for key in fashion_masks if key.startswith(dataset))
            setattr(self, dataset, self.Dataset(data, shuffle_batches=dataset == "train", augment=dataset == "train", sparse_labels=sparse_labels))

    @staticmethod
    def augment(image, mask):
        return MNIST.horizontal_flip(*MNIST.cutout(*MNIST.translate(image, mask)))

    @staticmethod
    def cutout(image, mask):
        if np.random.uniform() > MNIST.CUTOUT_PROB: return (image, mask)
        half_size = MNIST.CUTOUT_SIZE // 2

        left = np.random.randint(-half_size, image.shape[0] - half_size)
        top = np.random.randint(-half_size, image.shape[1] - half_size)

        image[max(0, left):min(image.shape[0], left + MNIST.CUTOUT_SIZE),
        max(0, top):min(image.shape[1], top + MNIST.CUTOUT_SIZE), :] = 0
        return (image, mask)

    @staticmethod
    def horizontal_flip(image, mask):
        if np.random.uniform() > 0.5: return (image, mask)
        return (np.fliplr(image), np.fliplr(mask))

    @staticmethod
    def translate(image, mask, amount=4):
        top, left = np.random.randint(-amount, amount), np.random.randint(-amount, amount)
        image = MNIST._translate_image(image, top, left)
        mask = MNIST._translate_image(mask, top, left)
        return (image, mask)

    @staticmethod
    def _translate_image(image, top, left):
        clamp = lambda n: max(0, min(n, MNIST.H))
        translated = np.zeros((MNIST.H, MNIST.W, MNIST.C))
        translated[clamp(-top):clamp(-top + MNIST.H), clamp(-left):clamp(-left + MNIST.W), :] = \
             image[clamp(top):clamp(top + MNIST.H), clamp(left):clamp(left + MNIST.W), :]
        return translated

    @staticmethod
    def normalize(image):
        return (image - MNIST.MEAN) / MNIST.STD