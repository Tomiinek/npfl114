#!/usr/bin/env python3
# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

# Transformer architektura s relative positional representation a dvěma enkodéry -- jeden zpracovává písmena ve slově, druhý slova ve větě


from morpho_analyzer import MorphoAnalyzer


def edit_distance(x, y):
    a = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(len(x) + 1): a[i][0] = i
    for j in range(len(y) + 1): a[0][j] = j
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            a[i][j] = min(
                a[i][j - 1] + 1,
                a[i - 1][j] + 1,
                a[i - 1][j - 1] + (x[i - 1] != y[j - 1])
            )
    return a[-1][-1]


input_file = 'lemmatizer_competition_dev.txt'
analyses = MorphoAnalyzer("czech_pdt_analyses")


with open(input_file, mode='r', encoding='utf-8') as in_file:
    lines = [line.rstrip("\n") for line in in_file]
processed_lines = []


for line in lines:
    if not line:
        processed_lines.append(line)
        continue

    form, lemma, tag = line.split('\t')

    options = analyses.get(form)

    if len(options) == 0:
        processed_lines.append(line)
        continue

    best = None
    for option in options:
        l = option.lemma
        distance = edit_distance(l, lemma)

        if best is None or distance < best["distance"]:
            best = {"distance": distance, "lemma": l}

    processed_lines.append(f"{form}\t{best['lemma']}\t{tag}")


with open(f"processed_{input_file}", mode='w', encoding='utf-8') as in_file:
    in_file.write("\n".join(processed_lines) + "\n")