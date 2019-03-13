#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    distribution = {}
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            if line in distribution:
                distribution[line] += 1
            else:
                distribution[line] = 1

        d = np.array([value for (key, value) in sorted(distribution.items())])
        d = d / d.sum(axis=0)

        # Load model distribution, each line `word \t probability`, creating
        # a NumPy array containing the model distribution
        model_distribution = {key: 0 for key in distribution.keys()}
        with open("numpy_entropy_model.txt", "r") as model:
            for line in model:
                line = line.rstrip("\n")
                key, value = line.split("\t")
                if key in model_distribution:
                    model_distribution[key] = float(value)

        m = np.array([value for (key, value) in sorted(model_distribution.items())])

        entropy = (-d * np.log(d)).sum(axis=0)
        print("{:.2f}".format(entropy))

        cross_entropy = (-d * np.log(m)).sum(axis=0)
        print("{:.2f}".format(cross_entropy))

        kl_div = (d * (np.log(d) - np.log(m))).sum(axis=0)
        print("{:.2f}".format(kl_div))
