#!/usr/bin/env python3

# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import random
import numpy as np
import tensorflow as tf
import scipy

from skimage.io import imread
from skimage.color import gray2rgb

from caltech42 import Caltech42


def create_inference_augmented_batch(image):
    images = np.zeros((18, Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C))
    weights = tf.constant([1] + 8*[0.25] + [1] + 8*[0.25], dtype='float32')
    
    h, w = (image.shape[0] - Caltech42.MIN_SIZE) // 2, (image.shape[1] - Caltech42.MIN_SIZE) // 2
        
    images[0,:,:,:] = np.copy(center_crop(image))
    images[1,:,:,:] = np.copy(image[: Caltech42.MIN_SIZE , : Caltech42.MIN_SIZE , :Caltech42.C])
    images[2,:,:,:] = np.copy(image[ -Caltech42.MIN_SIZE:, : Caltech42.MIN_SIZE , :Caltech42.C])
    images[3,:,:,:] = np.copy(image[: Caltech42.MIN_SIZE ,  -Caltech42.MIN_SIZE:, :Caltech42.C])
    images[4,:,:,:] = np.copy(image[ -Caltech42.MIN_SIZE:,  -Caltech42.MIN_SIZE:, :Caltech42.C])    
    images[5,:,:,:] = np.copy(image[h: h+Caltech42.MIN_SIZE ,  :   Caltech42.MIN_SIZE , :Caltech42.C])
    images[6,:,:,:] = np.copy(image[ :   Caltech42.MIN_SIZE , w: w+Caltech42.MIN_SIZE , :Caltech42.C])
    images[7,:,:,:] = np.copy(image[h: h+Caltech42.MIN_SIZE ,     -Caltech42.MIN_SIZE:, :Caltech42.C])
    images[8,:,:,:] = np.copy(image[    -Caltech42.MIN_SIZE:, w: w+Caltech42.MIN_SIZE , :Caltech42.C])
    
    images[9,:,:,:] = np.fliplr(np.copy(center_crop(image)))
    images[10,:,:,:] = np.fliplr(np.copy(image[: Caltech42.MIN_SIZE , : Caltech42.MIN_SIZE , :Caltech42.C]))
    images[11,:,:,:] = np.fliplr(np.copy(image[ -Caltech42.MIN_SIZE:, : Caltech42.MIN_SIZE , :Caltech42.C]))
    images[12,:,:,:] = np.fliplr(np.copy(image[: Caltech42.MIN_SIZE ,  -Caltech42.MIN_SIZE:, :Caltech42.C]))
    images[13,:,:,:] = np.fliplr(np.copy(image[ -Caltech42.MIN_SIZE:,  -Caltech42.MIN_SIZE:, :Caltech42.C]))
    images[14,:,:,:] = np.fliplr(np.copy(image[h: h+Caltech42.MIN_SIZE ,  :   Caltech42.MIN_SIZE , :Caltech42.C]))
    images[15,:,:,:] = np.fliplr(np.copy(image[ :   Caltech42.MIN_SIZE , w: w+Caltech42.MIN_SIZE , :Caltech42.C]))
    images[16,:,:,:] = np.fliplr(np.copy(image[h: h+Caltech42.MIN_SIZE ,     -Caltech42.MIN_SIZE:, :Caltech42.C]))
    images[17,:,:,:] = np.fliplr(np.copy(image[    -Caltech42.MIN_SIZE:, w: w+Caltech42.MIN_SIZE , :Caltech42.C]))
    
    return images, weights

    
def augment(image):
    return horizontal_flip(random_crop(image))

def rotate(image):
    k = random.randint(0, 4)
    if k % 4 != 0: return tf.image.rot90(image, k)
    return image

def horizontal_flip(image):
    return tf.image.random_flip_left_right(image)

def random_crop(image):
    return tf.image.random_crop(image, [Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C])
#     t, l = np.round(np.random.uniform([0, 0], np.asarray(image.shape[:2]) - Caltech42.MIN_SIZE)).astype(int)
#     return image[t:(t + Caltech42.MIN_SIZE), l:(l + Caltech42.MIN_SIZE), :Caltech42.C]

def center_crop(image):
    t, l = (np.asarray(image.shape[:2]) - Caltech42.MIN_SIZE) // 2
    return image[t:(t + Caltech42.MIN_SIZE), l:(l + Caltech42.MIN_SIZE), :Caltech42.C]
