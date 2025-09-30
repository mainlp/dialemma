import csv
import os
from csv import DictWriter


def load_data(task):
    nrows = 300 if task == 'recognition' else 301
    with open("data/annotations.csv") as f:
        fieldnames = ['id', 'lemma_id', 'de_term', 'de_freq', 'pos', 'pos_perc', 'bar', 'bar_freq', 'ld', 'label',
                      'contexts']
        reader = csv.DictReader(f, fieldnames=fieldnames)
        next(reader) # skip header
        
        offset = nrows
        dev, test, full = [], [], []
        for i, row in enumerate(reader):
            if task == "translation" and row["label"] != "yes":
                continue
            if offset > 0:
                dev.append(row)
                offset -= 1
            else:
                test.append(row)
            full.append(row)
        pass
    
    return dev, test, full


def write_data(data, task, split, fieldnames):
    filename = f"data/{task}_{split}.csv"
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            writer = DictWriter(f,fieldnames=fieldnames)
            writer.writeheader()
            for r in data:
                if r["pos"] == "ADV" and r["label"] == "inflected":
                    r["label"] = "no"
                writer.writerow(r)
    else:
        print(f"Skipping, file exists {filename}")


def main():
    for task in ["recognition", "translation"]:
        dev, test, full = load_data(task=task)
        fieldnames = list(dev[0].keys())
        write_data(dev, task, "dev", fieldnames)
        write_data(test, task, "test", fieldnames)

if __name__ == '__main__':
    main()