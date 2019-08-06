#!/usr/bin/env python3
# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

# Transformer architektura s relative positional representation a dvěma enkodéry -- jeden zpracovává písmena ve slově, druhý slova ve větě


import numpy as np
import torch
import torch.nn as nn
from itertools import count

from morpho_dataset import MorphoDataset
from torch_relative_attention_context import Model


if __name__ == "__main__":
    import argparse
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", default=".", type=str, help="Directory for the outputs.")
    parser.add_argument("--dim", default=128, type=int, help="Dimension of hidden layers.")
    parser.add_argument("--heads", default=8, type=int, help="Number of attention heads.")
    parser.add_argument("--layers", default=2, type=int, help="Number of attention layers.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
    parser.add_argument("--cle_layers", default=2, type=int, help="CLE embedding layers.")
    parser.add_argument("--cnn_filters", default=64, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnn_max_width", default=5, type=int, help="Maximum CNN filter width.")
    parser.add_argument("--max_length", default=60, type=int, help="Max length of sentence in training.")
    parser.add_argument("--max_pos_len", default=8, type=int, help="Maximal length of the relative positional representation.")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="Initial learning rate multiplier.")
    parser.add_argument("--warmup_steps", default=4000, type=int, help="Learning rate warmup.")
    parser.add_argument("--checkpoint", default="checkpoint_acc-97.84")
    args = parser.parse_args()

    architecture = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if key not in ["directory", "base_directory", "epochs", "batch_size", "clip_gradient", "checkpoint"]))
    args.directory = f"{args.base_directory}/models/rel_context_attention_{architecture}"
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # Fix random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Load the data
    morpho = MorphoDataset("czech_pdt", args.base_directory, add_bow_eow=True)

    # Create the network and train
    num_source_chars = len(morpho.train.data[morpho.train.FORMS].alphabet)
    num_target_chars = len(morpho.train.data[morpho.train.LEMMAS].alphabet)

    network = Model(args, num_source_chars, num_target_chars).cuda()

    state = torch.load(f"{args.checkpoint}")
    network.load_state_dict(state['state_dict'])


    #
    # PREDICT
    #

    network.eval()
    data = morpho.test

    with torch.no_grad():
        sentences = []
        size = data.size()
        for b, batch in enumerate(data.batches(180, 1000)):
            sentences += network.predict_to_list(batch, data)
            print(f"{b / (size / 160) * 100:3.2f} %")

    print("INFERED")

    out_path = "lemmatizer_competition_test.txt"
    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, sentence in enumerate(sentences):
            for j in range(len(data.data[data.FORMS].word_strings[i])):
                lemma = []
                for c in map(int, sentence[j]):
                    if c == MorphoDataset.Factor.EOW: break
                    lemma.append(data.data[data.LEMMAS].alphabet[c])

                print(data.data[data.FORMS].word_strings[i][j],
                      "".join(lemma),
                      data.data[data.TAGS].word_strings[i][j],
                      sep="\t", file=out_file)
            print(file=out_file)