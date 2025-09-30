"""
This script creates a jsonl dictionary file from the annotations file data/annotations.csv
{
  "id": "1702", 
  "pos": "NOUN", 
  "term": "Ortschaft", 
  "variants": [
     "Ortschft", 
     "Ortschoft", 
     "Ortschaoft", 
     "Oatschaft", 
     "Ortschåft", 
     "Ortsschoft"], 
  "inflected_variants": [
     "Ortschaftn", 
     "Ortschaftnn", 
     "Ortschafta"
  ]
}
"""

import json
from csv import DictReader

input_filepath = "data/annotations.csv"
output_filepath = input_filepath.replace("annotations.csv", "dictionary.jsonl")

variants_fieldname = "translations"
inflected_variants_fieldname = "inflected_variants"
term_fieldname = "term"

# aggregate lemma-variant_candidate pairs
dialect_variation_dict = {}
with open(input_filepath, "r") as f:
  reader = DictReader(f)
  for row in reader: 
    lemma_id = row["lemma_id"]
    if lemma_id not in dialect_variation_dict:
      dialect_variation_dict[lemma_id] = {
        "id": lemma_id,
        "pos": row["pos"],
        term_fieldname: row["de_term"],
        variants_fieldname: [],
        inflected_variants_fieldname: []
      }
      
    if row["label"] == "yes":
      dialect_variation_dict[lemma_id][variants_fieldname].append(row["bar"])
      
    elif row["label"] == "inflected":
      dialect_variation_dict[lemma_id][inflected_variants_fieldname].append(row["bar"])

# filter records and keep only entries for which we have at least either variants or inflected variants
tmp = {k: v for k, v in dialect_variation_dict.items() if v[variants_fieldname] or v[inflected_variants_fieldname]}

# remove empty lists for lemmas for which we did not annotate any dialect variant or inflected variant
for k, v in tmp.items():
  if not v[inflected_variants_fieldname]:
    del v[inflected_variants_fieldname]
  if not v[variants_fieldname]:
    del v[variants_fieldname]

# keep only entries for which we have at least either variants or inflected variants
records = [json.dumps(r, ensure_ascii=False) + "\n" for r in tmp.values()]
with open(output_filepath, "w") as f:
  f.writelines(records)
pass
