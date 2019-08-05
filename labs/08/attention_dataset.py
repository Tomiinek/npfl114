import os
import sys
import urllib.request
import zipfile

import numpy as np
from collections import Counter


# Loads a morphological dataset in a vertical format.
# - The data consists of three Datasets
#   - train
#   - dev
#   - test
# - Each dataset is composed of factors (FORMS, LEMMAS, TAGS), each an
#   object containing the following fields:
#   - word_strings: Strings of the original words.
#   - word_ids: Word ids of the original words (uses <unk> and <pad>).
#   - words_map: String -> word_id map.
#   - words: Word_id -> string list.
#   - alphabet_map: Character -> char_id map.
#   - alphabet: Char_id -> character list.
#   - charseq_ids: Character_sequence ids of the original words.
#   - charseqs_map: String -> character_sequence_id map.
#   - charseqs: Character_sequence_id -> [characters], where character is an index
#       to the dataset alphabet.
class MorphoDataset:
    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1819/datasets/"
    TAGS = 13
    TAG_SIZES = [0] * TAGS
    TAG_RATIOS = [[]] * TAGS

    class Factor:
        PAD = 0
        UNK = 1
        BOW = 2
        EOW = 3

        def __init__(self, characters, train=None, include_unk=True):
            self.words_map = train.words_map if train else {"<pad>": self.PAD, "<unk>": self.UNK}
            self.words = train.words if train else (["<pad>", "<unk>"] if include_unk else ["<pad>"])
            self.word_ids = []
            self.word_strings = []
            self.word_embeddings = []
            self.characters = characters
            if characters:
                self.alphabet_map = train.alphabet_map if train else {
                    "<pad>": self.PAD, "<unk>": self.UNK, "<bow>": self.BOW, "<eow>": self.EOW}
                self.alphabet = train.alphabet if train else ["<pad>", "<unk>", "<bow>", "<eow>"]
                self.charseqs_map = {"<pad>": self.PAD, "<unk>": self.UNK}
                self.charseqs = [[self.PAD], [self.UNK]]
                self.charseq_ids = []

    class FactorBatch:
        def __init__(self, word_ids, word_embeddings, charseq_ids=None, charseqs=None):
            self.word_ids = word_ids
            self.word_embeddings = word_embeddings
            self.charseq_ids = charseq_ids
            self.charseqs = charseqs

    class Dataset:
        FORMS = 0
        LEMMAS = 1
        TAGS_BEGIN = 2
        TAGS_END = 15
        FACTORS = 15
        EMBEDDING_SIZE = 256

        def __init__(self, data_file, dim, embedding_file, train=None, shuffle_batches=True, add_bow_eow=False,
                     max_sentences=None, seed=42):

            # Create factors
            self._data = []
            for f in range(self.FACTORS):
                self._data.append(
                    MorphoDataset.Factor(f in [self.FORMS, self.LEMMAS], train._data[f] if train else None,
                                         f in [self.FORMS, self.LEMMAS]))

            self.process_factors(data_file, embedding_file, train, add_bow_eow, max_sentences)

            self._dim = dim
            self._positions = self.positional_encoding(1000, self._dim)
            self._size = len(self._data[self.FORMS].word_ids)
            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        def process_factors(self, data_file, embedding_file, train=None, add_bow_eow=False, max_sentences=None):

            if embedding_file:
                n = 0
                mean_embedding = np.zeros((self.EMBEDDING_SIZE), dtype=np.float32)
                words_embedding_map = {}
                for word_embedding in embedding_file:
                    n += 1
                    tokens = word_embedding.split(' ')
                    embedding = np.array(tokens[1:-1], dtype=np.float32)
                    mean_embedding += embedding
                    words_embedding_map[tokens[0]] = embedding
                mean_embedding /= n

            in_sentence = False
            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                if line:
                    columns = line.split("\t")
                    for f in range(self.FACTORS):
                        factor = self._data[f]
                        if not in_sentence:
                            if len(factor.word_ids): factor.word_ids[-1] = np.array(factor.word_ids[-1], np.int32)
                            factor.word_ids.append([])
                            factor.word_strings.append([])
                            if embedding_file and f == self.FORMS: factor.word_embeddings.append([])
                            if factor.characters: factor.charseq_ids.append([])
                        word = columns[f]
                        factor.word_strings[-1].append(word)
                        # Character-level information
                        if factor.characters:
                            if word not in factor.charseqs_map:
                                factor.charseqs_map[word] = len(factor.charseqs)
                                factor.charseqs.append([])
                                if add_bow_eow:
                                    factor.charseqs[-1].append(MorphoDataset.Factor.BOW)
                                for c in word:
                                    if c not in factor.alphabet_map:
                                        if train:
                                            c = "<unk>"
                                        else:
                                            factor.alphabet_map[c] = len(factor.alphabet)
                                            factor.alphabet.append(c)
                                    factor.charseqs[-1].append(factor.alphabet_map[c])
                                if add_bow_eow:
                                    factor.charseqs[-1].append(MorphoDataset.Factor.EOW)
                            factor.charseq_ids[-1].append(factor.charseqs_map[word])
                        # Word-level information
                        if word not in factor.words_map:
                            if train and f != self.FORMS:
                                word = "<unk>"
                            else:
                                factor.words_map[word] = len(factor.words)
                                factor.words.append(word)
                        factor.word_ids[-1].append(factor.words_map[word])
                        if f == self.FORMS and embedding_file:
                            lower_word = word.lower()
                            factor.word_embeddings[-1].append(words_embedding_map[
                                                                  lower_word] if lower_word in words_embedding_map else mean_embedding)
                    in_sentence = True
                else:
                    in_sentence = False
                    if max_sentences is not None and len(self._data[self.FORMS].word_ids) >= max_sentences:
                        break

        @property
        def data(self):
            return self._data

        def size(self):
            return self._size

        def batches(self, size, max_length):
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = []
                max_sentence_len = min(max_length, max(len(self._data[self.FORMS].word_ids[i]) for i in batch_perm))

                batch.append(MorphoDataset.FactorBatch(np.zeros([batch_size, max_sentence_len], np.int32),
                                                       np.zeros([batch_size, max_sentence_len, self.EMBEDDING_SIZE],
                                                                np.float32)))
                for i in range(batch_size):
                    length = min(max_sentence_len, len(self._data[self.FORMS].word_embeddings[batch_perm[i]]))
                    batch[0].word_embeddings[i, :length, :] = self._data[self.FORMS].word_embeddings[batch_perm[i]][
                                                              :length]
                    batch[0].word_ids[i, :length] = self._data[self.FORMS].word_ids[batch_perm[i]][:length]

                # Word-level data
                for factor in self._data[1:]:
                    batch.append(MorphoDataset.FactorBatch(np.zeros([batch_size, max_sentence_len], np.int32), None))
                    for i in range(batch_size):
                        length = min(max_sentence_len, len(factor.word_ids[batch_perm[i]]))
                        batch[-1].word_ids[i, :length] = factor.word_ids[batch_perm[i]][:length]

                # Character-level data
                for f, factor in enumerate(self._data):
                    if not factor.characters: continue

                    batch[f].charseq_ids = np.zeros([batch_size, max_sentence_len], np.int32)
                    charseqs_map = {"<pad>": factor.PAD}
                    charseqs = [factor.charseqs[factor.PAD]]
                    for i in range(batch_size):
                        for j, charseq_id in enumerate(factor.charseq_ids[batch_perm[i]]):
                            if j >= max_sentence_len: break
                            if charseq_id not in charseqs_map:
                                charseqs_map[charseq_id] = len(charseqs)
                                charseqs.append(factor.charseqs[charseq_id])
                            batch[f].charseq_ids[i, j] = charseqs_map[charseq_id]

                    max_charseq_len = max(len(charseq) for charseq in charseqs)
                    batch[f].charseqs = np.zeros([len(charseqs), max_charseq_len], np.int32)
                    for i in range(len(charseqs)):
                        batch[f].charseqs[i, :len(charseqs[i])] = charseqs[i]

                batch.append(np.zeros([batch_size, max_sentence_len, self._dim]))
                batch[-1][:] = self._positions[:, :max_sentence_len, :]

                yield batch

        def get_angles(self, pos, i, d_model):
            angle_rates = 1 / np.power(1000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        def positional_encoding(self, position, d_model):
            angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                         np.arange(d_model)[np.newaxis, :],
                                         d_model)
            # apply sin to even indices in the array; 2i
            sines = np.sin(angle_rads[:, 0::2])
            # apply cos to odd indices in the array; 2i+1
            cosines = np.cos(angle_rads[:, 1::2])
            pos_encoding = np.concatenate([sines, cosines], axis=-1)
            return pos_encoding[np.newaxis, ...]

    def __init__(self, directory, dataset, dim, add_bow_eow=False, max_sentences=None):
        path = "{}.zip".format(dataset)

        with zipfile.ZipFile(f"{directory}/{path}", "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open(f"{os.path.splitext(path)[0]}_{dataset}.txt", "r") as dataset_file, open(
                        f"{directory}/{dataset}_words_embedded.txt", "r", encoding="utf-8") as embedding_file:
                    setattr(self, dataset, self.Dataset(dataset_file, dim, embedding_file,
                                                        train=self.train if dataset != "train" else None,
                                                        shuffle_batches=dataset == "train",
                                                        add_bow_eow=add_bow_eow,
                                                        max_sentences=max_sentences))

        for tag in range(self.TAGS):
            self.TAG_SIZES[tag] = len(self.train.data[self.Dataset.TAGS_BEGIN + tag].words)

            everything = [t for sentence in self.train.data[self.Dataset.TAGS_BEGIN + tag].word_ids for t in sentence if
                          t != 0]
            counter = Counter(everything)
            for t in range(1, self.TAG_SIZES[tag]):
                assert counter[t] > 0
            self.TAG_RATIOS[tag] = np.array([counter[t] / len(everything) for t in range(1, self.TAG_SIZES[tag])])