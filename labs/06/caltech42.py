import os
import sys
import urllib.request
import zipfile

import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.color import gray2rgb


# Note: Because images have different size, the user
# - can specify `image_processing` method to dataset construction, which
#   is applied to every image during loading;
# - and/or can specify `image_processing` method to `batches` call, which is
#   applied to an image during batch construction.
#
# In any way, the batch images must be Numpy arrays with shape (224, 224, 3)
# and type np.float32. (In order to convert tf.Tensor to Numpty array
# use `tf.Tensor.numpy()` method.)
#
# If all images are of the above datatype after dataset construction
# (i.e., `image_processing` passed to `Caltech42` already generates such images),
# then `data["images"]` is a Numpy array with the images. Otherwise, it is
# a Python list of images, and the Numpy array is constructed only in `batches` call.


class Caltech42:
    labels = [
        "airplanes", "bonsai", "brain", "buddha", "butterfly",
        "car_side", "chair", "chandelier", "cougar_face", "crab",
        "crayfish", "dalmatian", "dragonfly", "elephant", "ewer",
        "faces", "flamingo", "grand_piano", "hawksbill", "helicopter",
        "ibis", "joshua_tree", "kangaroo", "ketch", "lamp", "laptop",
        "llama", "lotus", "menorah", "minaret", "motorbikes", "schooner",
        "scorpion", "soccer_ball", "starfish", "stop_sign", "sunflower",
        "trilobite", "umbrella", "watch", "wheelchair", "yin_yang",
    ]
    MIN_SIZE, C, LABELS = 224, 3, len(labels)

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1819/datasets/caltech42.zip"

    def get_folds(self):
        return self

    class Dataset:
        def __init__(self, data, shuffle_batches, transformation, sparse_labels=True, seed=42):
            self._data = data
            self._size = len(self._data["images"])

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None
            self._transformation = transformation

            if not sparse_labels:
                self._data["labels"][self._data["labels"] == 255] = 0
                self._data["labels"] = tf.keras.utils.to_categorical(self._data["labels"], num_classes=Caltech42.LABELS)

        @property
        def data(self):
            return self._data

        @property
        def images_transformed(self):
            images = np.zeros([self.size, Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C], dtype=np.float32)
            for i, image in enumerate(self._data["images"]):
                images[i] = self._transformation(image.copy())
            return images

        @property
        def size(self):
            return self._size

        def batched_size(self, batch_size):
            return int(np.ceil(self._size / batch_size))

        def batches(self, size=None, repeat=False):
            while True:
                permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
                while len(permutation):
                    batch_size = min(size or np.inf, len(permutation))
                    batch_perm = permutation[:batch_size]
                    permutation = permutation[batch_size:]

                    X = np.zeros([batch_size, Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C], dtype=np.float32)
                    for i, index in enumerate(batch_perm):
                        data = self._transformation(self._data["images"][index].copy())

                        if type(data) != np.ndarray:
                            raise ValueError("Caltech42: Expecting images after `transformation` to be Numpy `ndarray`")
                        if data.dtype != np.float32 or data.shape != (
                        Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C):
                            raise ValueError(
                                "Caltech42: Expecting images after `transformation` to have shape {} and dtype {}".format(
                                    (Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C), np.float32))
                        X[i] = data

                    y = self._data["labels"][batch_perm]

                    yield X, y

                if not repeat:
                    break

    def __init__(self, augmentation, preprocessing, sparse_labels=True):
        """
        Parameters
        ----------
        augmentation : Callable[[np.ndarray], np.ndarray]
            function called on each image in training split during batch preprocessing
            input: array (writable) [0, 1]^(H, W, 3) with H, W at least MIN_SIZE
            output: array [0, 1]^(MIN_SIZE, MIN_SIZE, 3)
        preprocessing : Callable[[np.ndarray], np.ndarray]
            function called on each image in dev or test split during batch preprocessing
            input: array (writable) [0, 1]^(H, W, 3) with H, W at least MIN_SIZE
            output: array [0, 1]^(MIN_SIZE, MIN_SIZE, 3)
        """
        path = os.path.basename(self._URL)
        if not os.path.exists(path):
            print("Downloading Caltech42 dataset...", file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename=path)

        with zipfile.ZipFile(path, "r") as caltech42_zip:
            for dataset in ["train", "dev", "test"]:
                data = {"images": [], "labels": []}
                for name in sorted(caltech42_zip.namelist()):
                    if not name.startswith(dataset) or not name.endswith(".jpg"): continue

                    with caltech42_zip.open(name, "r") as image_file:
                        jpeg_bytes = image_file.read()
                        image_arr = imread(jpeg_bytes, plugin="imageio")
                        if image_arr.ndim == 2:  # grayscale
                            image_arr = gray2rgb(image_arr)
                        data["images"].append(np.asarray(image_arr, dtype=np.float32) / 255)

                    if "_" in name:
                        data["labels"].append(self.labels.index(name[name.index("_") + 1:-4]))
                    else:
                        data["labels"].append(-1)

                data["labels"] = np.array(data["labels"], dtype=np.uint8)
                setattr(self, dataset, self.Dataset(
                    data, shuffle_batches=(dataset == "train"),
                    transformation=(augmentation if (dataset == "train") else preprocessing),
                    sparse_labels=sparse_labels
                ))


def random_crop(image):
    t, l = np.round(np.random.uniform([0, 0], np.asarray(image.shape[:2]) - Caltech42.MIN_SIZE)).astype(int)
    cropped = image[t:(t + Caltech42.MIN_SIZE), l:(l + Caltech42.MIN_SIZE), :Caltech42.C]

    if np.random.uniform() > 0.5: return np.fliplr(cropped)
    return cropped


def center_crop(image):
    t, l = (np.asarray(image.shape[:2]) - Caltech42.MIN_SIZE) // 2
    return image[t:(t + Caltech42.MIN_SIZE), l:(l + Caltech42.MIN_SIZE), :Caltech42.C]