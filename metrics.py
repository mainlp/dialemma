SEP = ";"
PRECISION = 3


def confusion_matrix(label_predictions, labels):
  table_data = []
  pos2ftfi = {}
  for row in labels:
    columns = {
      "pred\\actual": row,
      "yes": label_predictions["yes"][row],
      "inflected": label_predictions["inflected"][row],
      "no": label_predictions["no"][row],
    }
    table_data.append(columns)
    pos2ftfi[row] = label_predictions[row]["ftfi"]

  #print(tabulate(table_data, headers="keys", tablefmt="grid") + "\n")

  columns = table_data[0].keys()
  print(SEP.join(columns))
  for row in table_data:
    print(SEP.join(str(row[l]) for l in columns))
  print(SEP.join(["ftfi"] + [str(pos2ftfi[l]) for l in labels]))
  print()


def f1(label_predictions, positive_class, labels, verbose=False):
  negative_classes = [l for l in labels if l != positive_class]

  TP = label_predictions[positive_class][positive_class]
  FP = sum([label_predictions[neg_class][positive_class] for neg_class in negative_classes])
  FN = sum([label_predictions[positive_class][neg_class] for neg_class in negative_classes])

  if (TP + FP) == 0.0 or (TP + FN) == 0.0 or TP == 0.0:
    return 0, 0, 0

  P = round(TP / (TP + FP), PRECISION)
  R = round(TP / (TP + FN), PRECISION)
  F1 = round((2 * P * R) / (P + R), PRECISION)
  if verbose:
    print(f"label = {positive_class}")
    print(f"P({positive_class}):{SEP}{P}")
    print(f"R({positive_class}):{SEP}{R}")
    print(f"f1({positive_class}):{SEP}{F1}\n")

  return F1, P, R


def acc(label_predictions, labels, verbose=False):
  correct, total = 0, 0
  for a in labels:
    for b in labels:
      count = label_predictions[a][b]
      if a == b:
         correct += count
      total += count
  accuracy = round(correct / total, PRECISION)
  if verbose: print(f"Acc:{SEP}{accuracy}")
  return accuracy
