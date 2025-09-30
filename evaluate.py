import os 
import csv
import random
import tqdm

from argparse import ArgumentParser
from logistic_regression import main as run_logistic_regression
from metrics import f1, acc, PRECISION, confusion_matrix
from utils import load_prompts
from collections import defaultdict
from utils import pos_tagset

random.seed(0)

recognition_labels = ["yes", "inflected", "no"]


def parse_args():
  parser = ArgumentParser()
  # required
  parser.add_argument("--split", choices=["dev", "test"], required=True)
  parser.add_argument("--task", choices=["translation", "recognition"], required=True)
  # optional 
  parser.add_argument("--prompt_lang", choices=["de", "en"], default="en", required=False)
  parser.add_argument("--use_context", action='store_true', required=False)
  parser.add_argument("--confusion_matrix", action='store_true', required=False)
  parser.add_argument("--baselines", action='store_true', required=False)
  args = parser.parse_args()
  
  if sum([args.prompt_lang == 'de', args.use_context, args.confusion_matrix]) > 1:
    raise AssertionError("Only one of --prompt_lang 'de', --use_context, or --confusion_matrix can be used at a time.")
  
  if args.split == "dev" and any([args.confusion_matrix, args.baseline]):
    raise NotImplementedError("--confusion_matrix and --baselines require --split test.")
  
  return args


def extract(row):
  columns = ["id", "de_lemma_id", "de_lemma", "de_freq", "pos", "pos_prob", "bar", "bar_freq", "ld", "label", "contexts", "prediction"]
  return dict(zip(columns, row))


def get_counts(fPath, keep_pos, task, baseline=None):
  label_predictions = defaultdict(lambda: defaultdict(int))
  correct, total, fail_to_follow_instruction = 0, 0, 0
  lds = []
  freqs = []
  pos_tags = []
  outputs = []
  with open(fPath, "r") as f:
    reader = csv.reader(f)
    next(reader) # header
    for row in reader:
      pos = row[4]
      if not (keep_pos == "ALL" or pos == keep_pos):
        continue
      pos_tags.append(pos)
      
      _id = row[0]
      recognition_label = row[9]
      lemma = row[2]
      bar_term = row[6]
      prediction = row[11]

      if task == "recognition":
        # post-process prediction
        prediction = postprocess_recognition(prediction)
        if prediction not in recognition_labels:
          fail_to_follow_instruction += 1
          #prediction = "no"
          prediction = "ftfi"
        
        # Baseline results 
        if baseline == "random":
          prediction = recognition_labels[random.randint(0, 2)]
          fail_to_follow_instruction = 0
        elif baseline == "majority":
          prediction = "no"
        elif baseline == "levenshtein":
          ld = int(row[8])
          prediction = "yes" if ld <= 2 else "no"

        label_predictions[recognition_label][prediction] += 1
      else:
        prediction = postprocess_translation(prediction)
        outputs.append((lemma, bar_term, prediction))
        label_predictions[lemma][prediction] += 1

      ld = row[2]
      lds.append(ld)
      
      de_freq = row[3]
      freqs.append(de_freq)

      correct += int(recognition_label == prediction)
      total += 1

  if task == "recognition":
    return label_predictions, fail_to_follow_instruction, total
  else:
    return outputs


def postprocess_translation(prediction):
  prediction = prediction.strip()
  return prediction


def postprocess_recognition(prediction):
  prediction = prediction.replace("```", "").replace("plaintext", "").strip()
  prediction = prediction.lower()
  prediction = "".join([c if c.isalpha() else " " for c in prediction]).strip()
  prediction_toks = prediction.split()
  if prediction_toks:
    prediction = prediction_toks[0]
  return prediction


def evaluate_translation(input_filepath, pos="ALL"):
  outputs = get_counts(keep_pos=pos, fPath=input_filepath, task="translation")
  correct, total, ftfi = 0, 0, 0
  for lemma, bar_term, prediction in outputs:
    prediction = prediction.strip()
    if len(prediction.split()) > 1:
      ftfi += 1
    if lemma == prediction:
      correct += 1
    total += 1
  accuracy = correct / total if total > 0 else 0
  IFError_rate = ftfi / len(outputs)
  return accuracy, ftfi, IFError_rate, total


def evaluate_recognition(input_filepath, pos="ALL", baseline=None):
  label_predictions, ftfi, total = get_counts(keep_pos=pos, fPath=input_filepath, task="recognition", baseline=baseline)
  
  if total == 0:
    # micro_f1, macro_f1, accuracy, ftfi, ftfi_rate
    return 0, 0, 0, 0, 0, 0

  task_labels = ["yes", "inflected", "no"]

  # Instances where models fail to follow instructions are considered the same as predicting 'no'.
  for label in task_labels:
    label_predictions[label]["no"] += label_predictions[label]["ftfi"]
    del label_predictions[label]["ftfi"]

  # failed to follow instruction (ftfi)
  IFError_rate = round(ftfi/total, PRECISION)
  IF_rate = (1 - IFError_rate) * 100

  f1_yes, P_yes, R_yes = f1(label_predictions, labels=recognition_labels, positive_class="yes")
  n_yes = sum(v for v in label_predictions["yes"].values())

  f1_inflected, P_inflected, R_inflected = f1(label_predictions, labels=recognition_labels, positive_class="inflected")
  n_inflected = sum(v for v in label_predictions["inflected"].values())

  f1_no, P_no, R_no  = f1(label_predictions, labels=recognition_labels, positive_class="no")
  n_no = sum(v for v in label_predictions["no"].values())

  # print(input_filepath)
  # print()
  # print(";yes;inflected;no")
  # print(f"P;{P_yes};{P_inflected};{P_no}")
  # print(f"R;{R_yes};{R_inflected};{R_no}")
  # print(f"F1;{f1_yes};{f1_inflected};{f1_no}")
  # print()

  _sum = n_yes + n_no + n_inflected
  micro_f1 = n_yes/_sum * f1_yes + n_no/_sum * f1_no + n_inflected/_sum * f1_inflected
  macro_f1 = (f1_yes + f1_no + f1_inflected) / 3
  accuracy = acc(label_predictions, labels=recognition_labels)
  return micro_f1, macro_f1, accuracy, ftfi, IFError_rate, IF_rate, total


def main():
  llms = [
    "mistral:7b-instruct-fp16",
    "mistral-large",
    "llama3.1:8b-instruct-fp16",
    "llama3.3:70b-instruct-fp16",
    "llama4:scout",
    "aya-expanse:8b",
    "aya-expanse:32b-fp16",
    "gemma3:12b",
    "gemma3:27b",
  ]

  args = parse_args()
  dev_test = args.split
  task = args.task
  use_context = args.use_context
  prompt_lang = args.prompt_lang
  calculate_confusion_matrix = args.confusion_matrix
  calculate_baselines = args.baselines
  
  results_basedir = os.path.join("results", dev_test, f"{task}+context" if use_context else task)
  if prompt_lang == "de": results_basedir += "-with_de_prompts"

  # Evaluate development set results
  if dev_test == "dev":
    header = f"llm;lang;prompt-id;{'macro_f1' if task == 'recognition' else 'accuracy'};IFError_rate"
    print(header)
      
    prompt_templates = load_prompts(language=prompt_lang, task=task)
    for llm in llms:
      for i, temp in enumerate(prompt_templates):
        input_dir = os.path.join(results_basedir, f"{prompt_lang}_{i}")
        input_filepath = os.path.join(input_dir, f"{llm}.csv")
        if task == "recognition":
          micro_f1, macro_f1, accuracy, ftfi, IFError_rate, IF_rate, total = evaluate_recognition(input_filepath=input_filepath)
          print(f"{llm};{prompt_lang};{str(i)};{round(macro_f1, 3)};{round(IFError_rate, 3)}")
          
        elif task == "translation":
          accuracy, ftfi_n, IFError_rate, total = evaluate_translation(input_filepath=input_filepath)
          print(f"{llm};{prompt_lang};{str(i)};{round(accuracy, 3)};{ftfi_n};{round(IFError_rate, 3)};{total}")
  
  # Evaluate testset results
  else:
    
    # print baseline results (recognition)
    if calculate_baselines:
      header = "baseline;POS;macro_f1"
      print(header)
      baseline2pos2result = defaultdict(dict)
      baselines = ["levenshtein", "random", "majority"]
      all_pos_tags = pos_tagset + ["ALL"]
      
      for baseline in baselines:
        print(baseline)
        for pos in tqdm.tqdm(all_pos_tags):
          input_filepath = os.path.join(results_basedir, f"{llms[0]}.csv")
          micro_f1, macro_f1, accuracy, ftfi_n, IFError_rate, IF_rate, total = evaluate_recognition(
            input_filepath=input_filepath, pos=pos, baseline=baseline)
          baseline2pos2result[baseline][pos] = macro_f1
          
      baseline2pos2result["logreg"] = run_logistic_regression()

      all_baselines = [b for b in baselines] + ["logreg"]
      print(f"POS;" + ";".join(all_baselines))
      for pos in pos_tagset:
        l =  f"{pos};" + ";".join([str(round(baseline2pos2result[b][pos], 3)) for b in all_baselines])
        print(l)

    # Print confusion matrix (across all pos tags)
    elif calculate_confusion_matrix:
      print(task)
      for llm in llms:
        print(llm)
        input_filepath = os.path.join(results_basedir, f"{llm}.csv")
        label_predictions = get_counts(keep_pos="ALL", fPath=input_filepath, task="recognition")[0]
        confusion_matrix(label_predictions=label_predictions, labels=["yes", "inflected", "no"])

    # Print test set results
    else:
      header = f"llm;lang;{'macro_f1' if task == 'recognition' else 'accuracy'};IFError_rate"
      for pos in ["ALL"] + pos_tagset:
        print(pos)
        print(header)
        for llm in llms:
          input_filepath = os.path.join(results_basedir, f"{llm}.csv")
          if task == "recognition":
            micro_f1, macro_f1, accuracy, ftfi, IFError_rate, IF_rate, total = evaluate_recognition(input_filepath=input_filepath, pos=pos)
            print(f"{llm};{prompt_lang};{round(macro_f1, 3)};{round(IFError_rate, 3)}")
          else:
            accuracy, ftfi_n, IFError_rate, total = evaluate_translation(input_filepath=input_filepath, pos=pos)
            print(f"{llm};{prompt_lang};{round(accuracy, 3)};{IFError_rate}")
        print()


if __name__ == '__main__':
  main()
