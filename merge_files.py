import os 
import csv 

from collections import defaultdict


label_mapping = {"1": "yes", "2": "inflected", "3": "no"}

# The value of `annotations_dir` must be set to the path where the annotation files are located
annotations_dir = "/path/to/annotations"


def load_data(path):
  de2bar2sentences = defaultdict(lambda: defaultdict(list))
  i = 0
  for file in os.listdir(path):
    with open(os.path.join(path, file), "r") as csvfile:
      reader = csv.reader(csvfile)
      if i == 0:
        print(next(reader))
      else:
        next(reader)
      for row in reader:
        i += 1
        if row[0]:
          lemma_id = row[0]
          de_term = row[1]
          de_freq = row[2]
          pos = row[3][:row[3].index("(") - 1]
          pos_perc = row[3][row[3].index("(") + 1: row[3].index(")")-2]
          de_instance = (lemma_id, de_term, de_freq, pos, pos_perc)
  
        if row[4]:
          bar = row[4]
          bar_freq = row[5]
          ld = row[6]
          label_id = row[7]
          label = label_mapping[label_id]
          if not label_id.strip():
            print(de_term + "\t" + file)
            continue
          bar_instance = (bar, bar_freq, ld, label)
        
        context = row[8]
        de2bar2sentences[de_instance][bar_instance].append(context)
        
  return de2bar2sentences

# load chunked annotation files (csv files 1-10)
data = load_data(annotations_dir)
header = ["id", "lemma_id", "de_term", "de_freq", "pos", "pos_perc", "bar", "bar_freq", "ld", "label", "contexts"]
with open("data/annotations.csv", "w") as f:
  writer = csv.writer(f)
  writer.writerow(header)
  i = 0
  for de, bar_contexts in data.items():
    for bar, contexts in bar_contexts.items():
      all_contexts = "####".join(contexts)
      writer.writerow((str(i),) + de + bar + (all_contexts,))
      i += 1
