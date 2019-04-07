#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import json
import os
import sys
import urllib
import zipfile

import numpy as np
import scipy.stats as stats
import tensorflow as tf


# In[ ]:

parser = argparse.ArgumentParser()
parser.add_argument("run_id", type=str)
parser.add_argument("--window", default=11, type=int)
parser.add_argument("--alphabet_size", default=80, type=int)
parser.add_argument("--embeddings_dim", default=32, type=int)
parser.add_argument("--resnet", action="store_true")
parser.add_argument("--hidden_layers", default="1024", type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--sub_epochs", default=4, type=int)
parser.add_argument("--epochs", default=4, type=int)
parser.add_argument("--dropout_rate", default=0.25, type=int)
args = parser.parse_args(sys.argv[1:])

run_id = args.run_id
window = args.window
alphabet_size = args.alphabet_size
embeddings_dim = args.embeddings_dim
resnet = args.resnet
hidden_layers = list(map(int, args.hidden_layers.split(':')))
batch_size = args.batch_size
sub_epochs = args.sub_epochs
epochs = args.epochs
dropout_rate = args.dropout_rate

settings_path = '{}_settings.json'.format(run_id)
assert not os.path.exists(settings_path), "Use unique run id, {} already present".format(run_id)

with open(settings_path, 'w') as f:
    json.dump(vars(args), f)

# In[ ]:


def get_texts():
    URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1819/datasets/uppercase_data.zip"

    path = os.path.basename(URL)
    if not os.path.exists(path):
        print("Downloading dataset {}...".format(path), file=sys.stderr)
        urllib.request.urlretrieve(URL, filename=path)

    texts = {}
    with zipfile.ZipFile(path, "r") as zip_file:
        for dataset in ["train", "dev", "test"]:
            with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                texts[dataset] = dataset_file.read().decode("utf-8")
    
    return texts


# In[ ]:


texts = get_texts()


# In[ ]:


def preprocess_text(text, alphabet):
    chars = np.array(list(text))
    low_chars = np.char.lower(chars)
    up_chars = np.char.upper(chars)
        
    neutral_mask = (low_chars == up_chars)
    up_mask = np.logical_and((chars == up_chars), ~neutral_mask)
    low_mask = np.logical_and((chars == low_chars), ~neutral_mask)
    
    if isinstance(alphabet, int):
        symbols, frequencies = np.unique(low_chars, return_counts=True)
        clipped_symbols = ['<pad>', '<unk>'] + list(symbols[np.argsort(frequencies)[::-1]][:alphabet-2])
        alphabet = dict((c, i) for i, c in enumerate(clipped_symbols))
    
    pad = lambda arr, val: np.pad(arr, window - 1, 'constant', constant_values=val)
    
    x = pad(np.array([alphabet.get(c, alphabet['<unk>']) for c in low_chars]), alphabet.get('<pad>'))
    y = pad(neutral_mask + (2 * low_mask), 1)
    
    return x, y, alphabet


# In[ ]:


train_x_np, train_y_np, alphabet = preprocess_text(texts['train'], alphabet_size)
val_x_np, val_y_np, _ = preprocess_text(texts['dev'], alphabet)


# In[ ]:


to_windows = lambda data: np.stack([data[i:(len(data) - window + i + 1)] for i in range(window)], axis=-1)
train_x = to_windows(train_x_np)
train_y = to_windows(train_y_np)

val_x = to_windows(val_x_np)
val_y = to_windows(val_y_np)


# In[ ]:


def regularized_dense(x, units):
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units, use_bias=False)(x)
    return x


# In[ ]:


def build_model(x):
    x = tf.keras.layers.Embedding(alphabet_size, embeddings_dim, input_length=window)(x)
    x = tf.keras.layers.Reshape((embeddings_dim * window,))(x)
    x = tf.keras.layers.Dense(hidden_layers[0])(x)
    for units in hidden_layers[1:]:
        if resnet:
            shortcut = x
            x = regularized_dense(x, units // 4)
            x = regularized_dense(x, units)
            x = tf.keras.layers.Add()([x, shortcut])
        else:
            x = regularized_dense(x, units)
            
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(3 * window)(x)
    x = tf.keras.layers.Reshape((window, 3))(x)
    x = tf.keras.layers.Activation('softmax')(x)
    outputs = tf.keras.layers.Lambda(lambda x: tf.unstack(x, axis=-2), name='pos')(x)

    return outputs


# In[ ]:


inputs = tf.keras.Input(shape=(window,), dtype=tf.uint8)
outputs = build_model(inputs)
model = tf.keras.Model(inputs, outputs)


# In[ ]:


model.compile(tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['acc'])


# In[ ]:


class BatchGenerator:
    def __init__(self, x, y, batch_size, repeat=False):
        assert len(x) == len(y)
        
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.repeat = repeat
        self.size = len(x) // batch_size
    
    def __len__(self):
        return self.size
    
    def __iter__(self):
        while True:
            order = np.random.permutation(len(self.x))
            batches = order[:self.size*batch_size].reshape(self.size, self.batch_size)
            for batch in batches:
                yield self.x[batch], np.split(self.y[batch], window, axis=-1)

            if not self.repeat:
                break


def evaluate(text=None):
    if text is None:
        text = texts['dev']
        x = val_x
        gold_y = val_y_np[window - 1:1 - window]
    else:
        x_np, gold_y_np, _ = preprocess_text(text, alphabet)
        x = to_windows(x_np)
        gold_y = gold_y_np[window - 1:1 - window]
    
    y = model.predict(x, batch_size, verbose=1)
    y_aligned = np.stack(
        [pos_y[window - 1 - i:len(pos_y) - i] for i, pos_y in enumerate(y)], axis=1)
    labels = np.argmax(y_aligned, axis=-1)
    ensemble_labels = stats.mode(labels, axis=1)[0].squeeze()
    accuracy = np.mean(gold_y == ensemble_labels)
    soft_mask = np.logical_or(gold_y == 0, ensemble_labels == 0)
    soft_accuracy = 1 - (gold_y != ensemble_labels)[soft_mask].sum() / len(gold_y)
    
    predicted_text = ''.join(
        c.upper() if (label == 0) else c
        for c, label in zip(text, ensemble_labels)
    )
    
    return predicted_text, accuracy, soft_accuracy
                
# In[ ]:


train_gen = BatchGenerator(train_x, train_y, batch_size, repeat=True)
val_gen = BatchGenerator(val_x, val_y, batch_size, repeat=True)


# In[ ]:


checkpoint_path = run_id + '_uppercase_{val_loss:.4f}.h5'
best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, save_best_only=True, save_weights_only=True)
evaluate_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda *_: print('[Expected acc: {:.4f}]'.format(evaluate()[2])))

# In[ ]:


model.fit_generator(iter(train_gen), len(train_gen) // sub_epochs, epochs * sub_epochs,
                    validation_data=iter(val_gen), validation_steps=len(val_gen),
                    callbacks=[best_checkpoint, evaluate_callback])


model.load_weights(checkpoint_path.format(best_checkpoint.best))

predicted_val, accuracy, soft_accuracy = evaluate(texts['dev'])
print('Expected accuracy: {:.4f} (strict {:.4f})'.format(soft_accuracy, accuracy))
with open(run_id + '_val.txt', 'w') as f:
    f.write(predicted_val)

predicted_test, _, _ = evaluate(texts['test'])
with open(run_id + '_test.txt', 'w') as f:
    f.write(predicted_test)
