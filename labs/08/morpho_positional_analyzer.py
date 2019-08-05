import os
import sys
import urllib.request
import zipfile

from morpho_positional_dataset import MorphoDataset
import numpy as np

class MorphoAnalyzer:
    """ Loads a morphological analyses in a vertical format.

    The analyzer provides only a method `get(word:str)` returning a list
    of analyses, each containing two fields `lemma` and `tag`.
    If an analysis of the word is not found, empty list is returned.
    """

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1819/datasets/"

    class LemmaTag:
        def __init__(self, lemma, tag):
            self.lemma = lemma
            self.tag = tag

        def __repr__(self):
            return "(lemma: {}, tag: {})".format(self.lemma, self.tag)

    def __init__(self, dataset, trainset):
        path = "{}.zip".format(dataset)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

        self.analyses = {}
        with zipfile.ZipFile(path, "r") as zip_file:
            with zip_file.open("{}.txt".format(dataset), "r") as analyses_file:
                for line in analyses_file:
                    line = line.decode("utf-8").rstrip("\n")
                    columns = line.split("\t")

                    analyses = []
                    for i in range(1, len(columns) - 1, 2):
                        analyses.append(self.LemmaTag(columns[i], columns[i + 1]))
                    self.analyses[columns[0]] = analyses

        self.tag_indices = {}
        tag_map = [trainset.data[MorphoDataset.Dataset.TAGS_BEGIN + i].words_map for i in range(MorphoDataset.TAGS)]
        word_map = trainset.data[0].words_map

        for word, suggestions in self.analyses.items():
            if word not in word_map: continue

            tags = []
            for suggestion in suggestions:
                s = [tag for i, tag in enumerate(suggestion.tag) if i not in [12, 13]]
                for i, tag in enumerate(s):
                    if tag not in tag_map[i]: break
                    s[i] = tag_map[i][tag] - 1
                else:
                    continue
                tags.append(s)

            self.tag_indices[word_map[word]] = tags

    def get(self, word):
        return self.analyses.get(word, [])

    def get_tag_ids(self, word_id):
        return self.tag_indices.get(word_id, [])