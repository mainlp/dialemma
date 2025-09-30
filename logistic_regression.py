import csv
from collections import defaultdict

import tqdm
from nltk.util import ngrams
from functools import partial
from sklearn.linear_model import LogisticRegression
from utils import pos_tagset


def ngrams_jaccard(string_1, string_2, n=2):
    bigrams_s1 = set(ngrams(string_1, n))
    bigrams_s2 = set(ngrams(string_2, n))
    
    intersection = len(bigrams_s1.intersection(bigrams_s2))
    union = len(bigrams_s1.union(bigrams_s2))

    return intersection / union if union != 0 else 0


def featurize(instance):
  return [int(instance["ld"]), 
          ngrams_jaccard(instance["de_term"], instance["bar"], n=2), 
          ngrams_jaccard(instance["de_term"], instance["bar"], n=3)]


def get_data(dev_test, pos):
  x, y = [], []
  with open(f"data/recognition_{dev_test}.csv", "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
      if r["pos"] == pos or pos == "ALL":
        x.append(featurize(r))
        y.append(r["label"])
  return x, y


def evaluate(y_hat, y_test, label):
  TP = sum(prediction == gold for prediction, gold in zip(y_hat, y_test) if gold == label)
  FN = sum(prediction != gold for prediction, gold in zip(y_hat, y_test) if gold == label)
  FP = sum(prediction != gold for prediction, gold in zip(y_hat, y_test) if prediction == label)
  P = (TP / (TP + FP)) if TP + FP > 0 else 0
  R = (TP / (TP + FN)) if TP + FN > 0 else 0  
  denominator = (P + R)
  if denominator == 0:
    return 0 
  F1 = 2 * P * R / denominator
  return F1 


def main():
  x, y = get_data("dev", pos="ALL")
  clf = LogisticRegression(random_state=0, penalty=None).fit(x, y)
  clf.predict(x)
  pos2result = defaultdict()
  print("logreg")
  all_pos_tags = pos_tagset + ["ALL"]
  for pos in tqdm.tqdm(all_pos_tags): # ["NOUN", "ADJ", "ADV", "VERB", "PROPN", "ALL"]:
   x_test, y_test = get_data("test", pos)
   y_hat = clf.predict(x).tolist()
   eval_fn = partial(evaluate, y_test=y_test, y_hat=y_hat)
   macro_F1 = sum([eval_fn(label='yes'), eval_fn(label='no'), eval_fn(label='inflected')]) / 3
   if __name__ == '__main__':
    print(f"logreg;{pos};{round(macro_F1, 3)}")
   pos2result[pos] = macro_F1
  return pos2result


if __name__ == '__main__':
    main()
