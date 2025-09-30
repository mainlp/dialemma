import numpy as np
import sys

from split_dataset import load_data
from collections import defaultdict
from collections import Counter


# print frequency distribution by POS tag
def freq_distr(task, instances):
    pos_tags = ["NOUN", "ADJ", "ADV", "VERB", "PROPN", "ADP", "NUM", "SCONJ", "DET", "AUX", "PRON", "CCONJ", "X", "INTJ", "PART"]
    if task == "recognition":
        no_instances = get_class_instances(instances, "no")
        yes_instances = get_class_instances(instances, "yes")
        inflected_instances = get_class_instances(instances, "inflected")
        print("POS;yes;inflected;no")
        for pos in pos_tags:
            print(f"{pos};{yes_instances[pos]};{inflected_instances[pos]};{no_instances[pos]}")
        print(f"ALL;{sum(yes_instances.values())};{sum(inflected_instances.values())};{sum(no_instances.values())}")
    else:
        all_instances = get_class_instances(instances, label="all")
        print("POS;n_instances")
        for pos in pos_tags:
            print(f"{pos};{all_instances[pos]}")
        print(f"ALL;{sum(all_instances.values())}")


# filter instances and keep only those tagged with a specified label (yes, no, inflected, all)
def get_class_instances(instances, label):
    examples = [r["pos"] for r in instances if r["label"] == label or label == "all"]
    instances = Counter(examples)
    return instances


# German lemmas with direct Bavarian translations have 2.61 ± 1.88 spelling variations on average
def mean_std(dataset, label):
    lemma2instances = defaultdict(list)
    for r in dataset:
        if r["label"] == label:
            lemma2instances[r["de_term"]].append(r)
    
    lemma_n = []
    for lemma, rows in lemma2instances.items():
        lemma_n.append(len(rows))
    
    print(f"mean: {np.mean(lemma_n)}")
    print(f"stdev: {np.std(lemma_n)}")


def main():
    if len(sys.argv) != 2:
        raise AssertionError("Call script with: python statistics.py {recognition,translation}")
    
    task = sys.argv[1]
    
    if task == "translation":
        dev, test, both = load_data(task="translation")
        freq_distr(instances=test, task="translation")
        
    elif task == "recognition":
        dev, test, both = load_data(task="recognition")
        freq_distr(instances=test, task="recognition")
        # mean_std(both, "yes")
        # mean_std(both, "inflected")
        
    else:
        print(f"task not supported: {task}")


if __name__ == '__main__':
    main()
