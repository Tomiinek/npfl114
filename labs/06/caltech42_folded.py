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

    class Dataset:
        def __init__(self, data, shuffle_batches, augmentation, preprocessing, sparse_labels=True, folds=0, seed=42):
            self._data = data
            self._size = len(self._data["images"])
            self._folds = folds

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None
            self._augmentation = augmentation
            self._preprocessing = preprocessing

            if not sparse_labels:
                self._data["labels"][self._data["labels"] == 255] = 0
                self._data["labels"] = tf.keras.utils.to_categorical(self._data["labels"], num_classes=Caltech42.LABELS)

        @property
        def size(self):
            return self._size

        def data_images(self):
            images = np.zeros([self.size, Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C], dtype=np.float32)
            for i, image in enumerate(self._data["images"]):
                images[i] = self._transformation(image.copy())
            return images

        def folds(self):
            fold_size = self.size // self._folds
            for i in range(self._folds):
                test_fold_begin = i * fold_size
                test_fold_end = test_fold_begin + fold_size

                train_x = np.zeros([self.size - fold_size, Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C], dtype=np.float32)
                test_x = np.zeros([fold_size, Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C], dtype=np.float32)
                a = b = 0

                for j, image in enumerate(self._data["images"]):
                    if j >= test_fold_begin or j < test_fold_end:
                        test_x[b] = self._augmentation(self._data["images"][j].copy())
                        b += 1
                    else:
                        train_x[a] = self._preprocessing(self._data["images"][j].copy())
                        a += 1

                train_y = np.concatenate((self._data["labels"][:test_fold_begin], self._data["labels"][test_fold_end:]), axis=0)
                test_y = self._data["labels"][test_fold_begin:test_fold_end]

                yield (train_x, train_y), (test_x, test_y)

    def __init__(self, augmentation, preprocessing, folds_num, sparse_labels=True):
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
            self._load_dataset_images(caltech42_zip, augmentation, preprocessing, sparse_labels, ["train", "dev"], True, folds_num)
            self._load_dataset_images(caltech42_zip, augmentation, preprocessing, sparse_labels, ["test"], False)

    def _load_dataset_images(self, caltech42_zip, augmentation, preprocessing, sparse_labels, dataset, dev, folds=1):
        data = {"images": [], "labels": []}
        for name in sorted(caltech42_zip.namelist()):
            if not name.startswith(tuple(dataset)) or not name.endswith(".jpg"): continue
            with caltech42_zip.open(name, "r") as image_file:
                jpeg_bytes = image_file.read()
                image_arr = imread(jpeg_bytes, plugin="imageio")
                if image_arr.ndim == 2:  # grayscale
                    image_arr = gray2rgb(image_arr)
                data["images"].append(np.asarray(image_arr, dtype=np.float32) / 255)

            if "_" in name: data["labels"].append(self.labels.index(name[name.index("_") + 1:-4]))
            else: data["labels"].append(-1)

        data["labels"] = np.array(data["labels"], dtype=np.uint8)
        setattr(self, "dev" if dev else "test", self.Dataset(
            data, transformation=(augmentation if (dataset == "train") else preprocessing),
            sparse_labels=sparse_labels,
            folds_num=folds
        ))


def random_crop(image):
    t, l = np.round(np.random.uniform([0, 0], np.asarray(image.shape[:2]) - Caltech42.MIN_SIZE)).astype(int)
    cropped = image[t:(t + Caltech42.MIN_SIZE), l:(l + Caltech42.MIN_SIZE), :Caltech42.C]

    if np.random.uniform() > 0.5: return np.fliplr(cropped)
    return cropped


def center_crop(image):
    t, l = (np.asarray(image.shape[:2]) - Caltech42.MIN_SIZE) // 2
    return image[t:(t + Caltech42.MIN_SIZE), l:(l + Caltech42.MIN_SIZE), :Caltech42.C]