#!/usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData
from uppercase_data_diakritika import UppercaseDataDiakritika


parser = argparse.ArgumentParser()
parser.add_argument("--models_path", default="./ensamble_models", type=str)
parser.add_argument("--output", default="test_out.txt", type=str)
args = parser.parse_args()

original_data = UppercaseData(0, 128)

def process_stepic(filename):
    stepic_p = np.load(filename)
    stepic_p[:, :, 1] = stepic_p[:, :, 1] + stepic_p[:, :, 2]
    return np.mean(stepic_p[:, :, 0:2], axis=1)

stepic_p = process_stepic('test_y_aligned.npy') + process_stepic('v4_test_y_aligned.npy')

p = 0.5*stepic_p[:,::-1]

for filename in os.listdir(args.models_path):
    parameters = dict([pair.split('=') for pair in filename.split(',')])

    alphabet_size = int(parameters['a_s'])
    window_size = int(parameters['w'])
    if "diac" in parameters: uppercase_data = UppercaseDataDiakritika(window_size, alphabet_size, "digits" in parameters)
    else: uppercase_data = UppercaseData(window_size, alphabet_size)

    model = tf.keras.models.load_model(os.path.join(args.models_path, filename), compile=False)
    print("Loaded: " + filename)
    model_p = model.predict(uppercase_data.test.data["windows"])

    for i, c in enumerate(original_data.test.text):
        if uppercase_data.test.transform_char(c) not in uppercase_data.test.alphabet: continue
        p[i] += model_p[i]


with open(args.output, "w", encoding="utf-8") as file:
    for i, c in enumerate(original_data.test.text):
        c = c.lower()
        file.write(c.upper() if p[i, 0] < p[i, 1] else c)
