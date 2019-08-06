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


def log_train_progress(epoch, loss, accuracy, progress, example):
    print('\r' + 140 * ' ', end='')  # clear line
    print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.3f} % ║ done: {progress:2d} % ║ {example}', end='', flush=True)


def log_train(epoch, loss, accuracy, out_file):
    print('\r' + 140 * ' ', end='')  # clear line
    print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.3f} % ║ ', end='', flush=True)
    print(f'epoch: {epoch:3d} ║ train loss: {loss:1.4f} │ acc: {accuracy:2.3f} % ║ ', end='', flush=True, file=out_file)


def log_dev(accuracy, learning_rate, out_file):
    print(f'acc: {accuracy:2.3f} % ║ lr: {learning_rate:1.6f}', flush=True)
    print(f'acc: {accuracy:2.3f} % ║ lr: {learning_rate:1.6f}', flush=True, file=out_file)


def log_mistakes(mistakes, out_file):
    max_before_len = max(len(mistake[0]) for mistake in mistakes)
    max_after_len = max(len(mistake[4]) for mistake in mistakes)
    max_original_len = max(len(mistake[1]) for mistake in mistakes)
    max_lemma_len = max(len(mistake[2]) for mistake in mistakes)
    for mistake in mistakes:
        print(f'{mistake[0].rjust(max_before_len)}\t║\t{mistake[1].ljust(max_original_len)}\t║\t{mistake[2].ljust(max_lemma_len)}\t║\t{mistake[3].ljust(max_lemma_len)}\t║\t{mistake[4].ljust(max_after_len)}', file=out_file)


class LRDecay:
    def __init__(self, optimizer, args):
        self.optimizer = optimizer
        self.dim = args.dim
        self.base_learning_rate = args.learning_rate
        self.warmup = args.warmup_steps
        self.step = 0

    def __call__(self):
        self.step += 1
        if self.step < self.warmup:
            learning_rate = self.dim ** (-0.5) * self.step * self.warmup ** (-1.5) * self.base_learning_rate
        else:
            learning_rate = self.dim ** (-0.5) * self.step ** (-0.5) * self.base_learning_rate

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

        return learning_rate


def get_example(indices, alphabet):
    example = []
    for i in indices:
        if i == MorphoDataset.Factor.BOW: continue
        if i == MorphoDataset.Factor.EOW: break
        example.append(alphabet[i])
    return ''.join(example)


def get_bad_example(truth_mask, dataset, inputs, predictions, targets):
    min_value, index = torch.min(truth_mask.all(dim=-1), dim=0)
    if min_value.item() == 1:
        return "everything correct :)"

    form = get_example(inputs[index, :].numpy(), dataset.data[dataset.FORMS].alphabet)
    gold_lemma = get_example(targets[index, :].numpy(), dataset.data[dataset.LEMMAS].alphabet)
    system_lemma = get_example(predictions[index, :].numpy(), dataset.data[dataset.LEMMAS].alphabet)

    return f"{form} / {gold_lemma} / {system_lemma}"


def get_mistakes(truth_mask, dataset, inputs, predictions, targets):
    inputs, predictions, targets = inputs.numpy(), predictions.numpy(), targets.numpy()
    mistakes = []

    for i in range(inputs.shape[0]):
        if truth_mask[i].item() == 1: continue
        before = ""
        for k in range(max(0, i - 4), i):
            before += get_example(inputs[k, :], dataset.data[dataset.FORMS].alphabet) + ' '

        form = get_example(inputs[i, :], dataset.data[dataset.FORMS].alphabet)
        gold_lemma = get_example(targets[i, :], dataset.data[dataset.LEMMAS].alphabet)
        system_lemma = get_example(predictions[i, :], dataset.data[dataset.LEMMAS].alphabet)

        after = ""
        for k in range(min(inputs.shape[0] - 1, i + 1), min(inputs.shape[0] - 1, i + 5)):
            after += get_example(inputs[k, :], dataset.data[dataset.FORMS].alphabet) + ' '

        mistakes.append([before, form, gold_lemma, system_lemma, after])

    return mistakes


if __name__ == "__main__":
    import argparse
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", default=".", type=str, help="Directory for the outputs.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--dim", default=16, type=int, help="Dimension of hidden layers.")
    parser.add_argument("--heads", default=4, type=int, help="Number of attention heads.")
    parser.add_argument("--layers", default=1, type=int, help="Number of attention layers.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
    parser.add_argument("--cle_dim", default=16, type=int, help="CLE embedding dimension.")
    parser.add_argument("--cle_layers", default=2, type=int, help="CLE embedding layers.")
    parser.add_argument("--cnn_filters", default=13, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnn_max_width", default=5, type=int, help="Maximum CNN filter width.")
    parser.add_argument("--max_length", default=60, type=int, help="Max length of sentence in training.")
    parser.add_argument("--max_pos_len", default=4, type=int, help="Maximal length of the relative positional representation.")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="Initial learning rate multiplier.")
    parser.add_argument("--warmup_steps", default=4000, type=int, help="Learning rate warmup.")
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
    criterion = nn.CrossEntropyLoss(ignore_index=MorphoDataset.Factor.PAD)
    optimizer = torch.optim.Adam(network.parameters())
    lr_decay = LRDecay(optimizer, args)

    best_accuracy = 0.0

    for epoch in count():
        with open(f"{args.directory}/log.txt", "a", encoding="utf-8") as log_file:

            #
            # TRAIN EPOCH
            #

            network.train()
            data = morpho.train

            batches_num = morpho.train.size() / args.batch_size
            running_loss, total_images, correct = 0.0, 0, 0.0

            for b, batch in enumerate(data.batches(args.batch_size, args.max_length)):
                learning_rate = lr_decay()

                optimizer.zero_grad()
                predictions, sources, targets = network(batch, data)

                loss = criterion(predictions.contiguous().view(-1, predictions.size(2)), targets.contiguous().view(-1))
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    targets, sources = targets.cpu(), sources.cpu()
                    targets_mask = targets != 0
                    predicted_labels = torch.argmax(predictions.data, 2).cpu()
                    truth_mask = predicted_labels == targets
                    size = targets_mask.sum().item()
                    correct += (truth_mask & targets_mask).sum().item()
                    total_images += size
                    running_loss += loss.item() * size

                    if b % 10 == 9:
                        bad_example = get_bad_example(truth_mask | ~targets_mask, data, sources, predicted_labels, targets)
                        log_train_progress(epoch, running_loss / total_images, correct / total_images * 100, int(b / batches_num * 100), bad_example)

            log_train(epoch, running_loss / total_images, correct / total_images * 100, log_file)

            #
            # EVALUATE EPOCH
            #

            network.eval()
            data = morpho.dev
            total_images, correct, mistakes = 0, 0.0, []
            with torch.no_grad():
                for b, batch in enumerate(data.batches(32, 240)):
                    predictions, sources, targets, _ = network.predict(batch, data)
                    predictions, sources, targets = predictions.cpu(), sources.cpu(), targets[:, 1:].cpu()

                    mask = (targets != 0).to(torch.long)
                    resized_predictions = torch.cat((predictions, torch.zeros_like(targets)), dim=1)[:, :targets.size(1)]
                    truth_mask = (resized_predictions * mask == targets * mask).all(dim=1)

                    total_images += targets.size(0)
                    correct += truth_mask.sum().item()

                    if len(mistakes) < 2500:
                        mistakes += get_mistakes(truth_mask, data, sources, predictions, targets)

                accuracy = correct / total_images * 100
                log_dev(accuracy, learning_rate, log_file)

                with open(f"{args.directory}/mistakes_{epoch:03d}.txt", "w", encoding="utf-8") as mistakes_file:
                    log_mistakes(mistakes, mistakes_file)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy

                    state = {
                        'epoch': epoch,
                        'state_dict': network.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(state, f'{args.directory}/checkpoint_acc-{accuracy:2.2f}')