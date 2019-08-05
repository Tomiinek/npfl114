import os
import sys
import urllib.request

import numpy as np

class ModelNet:
    # The D, H, W are set in the constructor depending
    # on requested resolution and are only instance variables.
    D, H, W, C = None, None, None, 1
    LABELS = [
        "bathtub", "bed", "chair", "desk", "dresser",
        "monitor", "night_stand", "sofa", "table", "toilet",
    ]

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1819/datasets/modelnet{}.npz"

    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42, do_augment=False):
            self._data = data
            self._size = len(self._data["projections"])
            self._do_augment = do_augment

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None, repeat=False, do_augment=False):
            while True:
                permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
                while len(permutation):
                    batch_size = min(size or np.inf, len(permutation))
                    batch_perm = permutation[:batch_size]
                    permutation = permutation[batch_size:]

                    batch = {}
                    for key in self._data:
                        batch[key] = self._data[key][batch_perm]

                    projections = batch['projections']
                    
                    if self._do_augment or do_augment:
                        for ex_i in range(len(projections)):
                            ex = flip_augment(projections[ex_i:ex_i+1], np.random.random() > 0.5, np.random.random() > 0.5)
                            ex = shift_augment(ex)
                            projections[ex_i] = ex
                        
                    x, y, z = projections[..., :3], projections[..., 3:6], projections[..., 6:]
                        
                    yield (
                        (x, y, z),
                        batch['labels'])
                
                if not repeat:
                    break
        
        def mixed_batches(self, directions=[0, 1, 2], size=None, repeat=False, do_augment=False):
            while True:
                permutation = np.concatenate(
                    [self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
                     for _ in directions])

                if len(directions) == 1:
                    projection_types = [directions]
                elif len(directions) == 2:
                    projection_types = [directions, directions[::-1]]
                else:
                    projection_types = [
                        [0, 1, 2],
                        [0, 2, 1],
                        [1, 0, 2],
                        [1, 2, 0],
                        [2, 0, 1],
                        [2, 1, 0]
                    ]
                    
                projection_types = np.asarray(projection_types)[np.random.choice(len(projection_types), self._size)].T.ravel()

                while len(permutation):
                    batch_size = min(size or np.inf, len(permutation))
                    batch_perm = permutation[:batch_size]
                    batch_types = projection_types[:batch_size]
                    permutation = permutation[batch_size:]
                    projection_types = projection_types[batch_size:]

                    labels = self._data['labels'][batch_perm]
                    
                    projections = np.empty(
                        shape=((batch_size,) + self._data['projections'].shape[1:3] + (3,)), dtype=np.float32)
                    for b_i, (ex_i, ex_type) in enumerate(zip(batch_perm, batch_types)):
                        ex = self._data['projections'][ex_i:ex_i+1]
                        if self._do_augment or do_augment:
                            ex = flip_augment(ex, np.random.random() > 0.5, np.random.random() > 0.5)
                            ex = shift_augment(ex)
                        projections[b_i] = ex[0, ..., ex_type*3:(ex_type+1)*3]

                    yield projections, labels
                
                if not repeat:
                    break

    # The resolution parameter can be either 20 or 32.
    def __init__(self, resolution):
        assert resolution in [20, 32], "Only 20 or 32 resolution is supported"

        self.D = self.H = self.W = resolution
        url = self._URL.format(resolution)

        path = os.path.basename(url)
        if not os.path.exists(path):
            print("Downloading {} dataset...".format(path), file=sys.stderr)
            urllib.request.urlretrieve(url, filename=path)

        mnist = np.load(path)
        for dataset in ["train", "dev", "test"]:
            voxels = mnist[dataset + '_voxels']
            projections = np.empty(shape=(voxels.shape[:3] + (9,)), dtype=np.float32)
            for v_i, voxel_map in enumerate(voxels):
                projections[v_i] = example_to_2D(voxel_map)

            data = {
                'projections': projections,
                'labels': mnist[dataset + '_labels']
            }
            setattr(self, dataset, self.Dataset(data, shuffle_batches=dataset == "train", do_augment=dataset == "train"))


def project(example, axis=0, inverted=False):
    example = np.moveaxis(example[..., 0], axis, 0)
    
    if inverted:
        example = example[::-1]

    mask = (np.sum(example, axis=0) > 0)
    depth_map = example.shape[axis] - np.argmax(example, axis=0)
    depth_map[~mask] = 0
    return depth_map / example.shape[0]   


def xray(example, axis=0):
    example = example[..., 0]
    return np.sum(example, axis=axis) / example.shape[axis]            


def example_to_2D(example):
    return np.stack(
        sum((
            [project(example, axis=ax_i, inverted=False),
             xray(example, axis=ax_i),
             project(example, axis=ax_i, inverted=True)]
            for ax_i in range(3)), []),
        axis=-1
    )


def flip_augment(examples, flip_lr, flip_fb):
    flipped = examples.copy()
    
    if flip_lr:
        flipped[..., 0] = examples[..., 0][:, :, ::-1]
        flipped[..., 1] = examples[..., 1][:, :, ::-1]
        flipped[..., 2] = examples[..., 2][:, :, ::-1]
        flipped[..., 3] = examples[..., 3][:, :, ::-1]
        flipped[..., 4] = examples[..., 4][:, :, ::-1]
        flipped[..., 5] = examples[..., 5][:, :, ::-1]
        flipped[..., 6] = examples[..., 8]
        flipped[..., 8] = examples[..., 6]
    
    examples = flipped.copy()
    
    if flip_fb:
        flipped[..., 0] = examples[..., 0][:, ::-1]
        flipped[..., 1] = examples[..., 1][:, ::-1]
        flipped[..., 2] = examples[..., 2][:, ::-1]
        flipped[..., 3] = examples[..., 5]
        flipped[..., 5] = examples[..., 3]
        flipped[..., 6] = examples[..., 6][:, :, ::-1]
        flipped[..., 7] = examples[..., 7][:, :, ::-1]
        flipped[..., 8] = examples[..., 8][:, :, ::-1]
        
    return flipped


def shift_augment(examples):
    examples = examples.copy()
    
    for axis in range(3):
        n, p = - (32 - 32 * examples[..., axis*3].max()), (32 - 32 * examples[..., axis*3 + 2].max()) + 1
    
        shift = np.clip(np.random.randint(int(n), int(p), dtype=np.int32), -3, 3)
        
        examples[..., axis*3] += (examples[..., axis*3] != 0) * (shift / 32)
        examples[..., axis*3 + 2] -= (examples[..., axis*3 + 2] != 0) * (shift / 32)
    
        if axis == 0:
            for channel in range(3, 9):
                examples[..., channel] = np.roll(examples[..., channel], shift, axis=1)
        elif axis == 1:
            for channel in range(3):
                examples[..., channel] = np.roll(examples[..., channel], shift, axis=1)
            for channel in range(6, 9):
                examples[..., channel] = np.roll(examples[..., channel], shift, axis=2)
        else:  # axis == 2
            for channel in range(3):
                examples[..., channel] = np.roll(examples[..., channel], shift, axis=2)
            for channel in range(3, 6):
                examples[..., channel] = np.roll(examples[..., channel], shift, axis=2)
    
    return examples