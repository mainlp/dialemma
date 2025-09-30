# Make Every Letter Count: Building Dialect Variation Dictionaries from Monolingual Corpora

This repository contains the code, data and evaluation scripts to reproduce the results of the paper [Make Every Letter Count: Building Dialect Variation Dictionaries from Monolingual Corpora](https://arxiv.org/pdf/2509.17855), which has been accepted at _EMNLP 2025 (Findings)_. 

## Overview

1. [Dialect Variation Dictionary](https://github.com/mainlp/dialemma?tab=readme-ov-file#-dialect-variation-dictionary)
2. [Dialect NLP Tasks](https://github.com/mainlp/dialemma?tab=readme-ov-file#-dialect-nlp-tasks)
3. [Reproduce Results](https://github.com/mainlp/dialemma?tab=readme-ov-file#-reproduce-results)
   1. [Setup](https://github.com/mainlp/dialemma?tab=readme-ov-file#1-setup)
   2. [Annotations](https://github.com/mainlp/dialemma?tab=readme-ov-file#2-annotations)
   3. [Experiments](https://github.com/mainlp/dialemma?tab=readme-ov-file#3-experiments)
4. [Citation](https://github.com/mainlp/dialemma?tab=readme-ov-file#citation)

## 📖 Dialect Variation Dictionary

Our Bavarian dialect variation dictionary contains 5,124 lemmas with dialect variants (i.e., translations) and inflected variants. The file can be found in [data/dictionary.jsonl](data/dictionary.jsonl).

**Example:**

```json lines
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
```

**Format:**
- `"id"`: Unique identifier for German lemmas.
- `"pos"`: Majority part-of-speech (POS) tag assigned by the [de_core_news_lg](https://spacy.io/models/de#de_core_news_lg) POS tagger in spaCy. 
- `"term"`: German lemma for which we collected (inflected) variants.
- `"variants"`: Bavarian terms that were annotated as **direct translations**.
- `"inflected_variants"`: Bavarian terms that were annotated **inflected translations**

The dictionary was created by running `python build_dictionary.py`.

## 📊 Dialect NLP Tasks

We created two dialect NLP task datasets based on [100K human-annotated German-Bavarian word pairs](data/annotations.csv): 
1. Judging Translation Candidates (`Recognition`)
2. Dialect-to-Standard Translation (`Translation`)

| Task         | Split | _n_ instances  | File                                              |
|--------------|-------|----------------|---------------------------------------------------|
| Recognition  | Dev   | 300            | [recognition_dev.csv](data/recognition_dev.csv)   |
| Recognition  | Test  | 97000          | [recognition_test.csv](data/recognition_test.csv) |
| Translation  | Dev   | 301            | [translation_dev.csv](data/translation_dev.csv)   |
| Translation  | Test  | 10775          | [translation_test.csv](data/translation_test.csv) |

The dev and test splits for both tasks are created with `python split_datasets.py`. 

To print the dataset statistics (Table 7), run `python statistics.py`.

## 📝 Reproduce Results

Below, we show the steps to reproduce the results in our paper.

### 1. Setup
Create a python environment and install the required packages:
```
conda create --name dvar python=3.10
conda activate dvar
pip install -r requirements.txt
```

### 2. Annotations

The following steps describe how we created the **annotation files**:

1. Download the Wikipedia dumps of a standard language and dialect ([link](https://dumps.wikimedia.org/other/cirrussearch/)). Example:
```
wget https://dumps.wikimedia.org/other/cirrussearch/20250310/barwiki-20250310-cirrussearch-content.json.gz
wget https://dumps.wikimedia.org/other/cirrussearch/20250310/dewiki-20250310-cirrussearch-content.json.gz
```
2. Run `python dialemma_pipeline.py` to create annotation files (.xls). The output is split into ten chunks to avoid large files.
3. Upload and share data with annotators (e.g., with google sheets).
4. Annotate pairs of German lemmas and Bavarian terms (see [annotation guidelines](data/guidelines.pdf)) and download annotated records as csv files.*
5. Run `python merge_files.py` to create one file with all records.

_*Note: Word pairs were annotated with respect to the POS tag of the lemma. We found rare cases (80 out of 99,700 instances) where words could be seen as "inflected" adjectives. Since those words were tagged as adverbs (which cannot be inflected) they received the label "no". We share a list of [ambiguous instances](data/ambiguous_instances.txt) for further uptake._

### 3. Experiments

#### A. Setup Local Endpoint

We use [Ollama](https://github.com/ollama/ollama/blob/main/docs/api.md) to set up a local endpoint that is [compatible](https://ollama.com/blog/openai-compatibility) with the Open AI python libray. We ran our experiments with v0.6.7 of Ollama.

1. Download LLM with `ollama pull llama3.1:8b-instruct-fp16`.
   1. The list of models used in our study can be found in [models.txt](models.txt).
2. Run LLM with `ollama serve`.
   1. If needed, change the port with `export OLLAMA_HOST=127.0.0.1:11435`.

#### B. Prompt Selection 

You can find the list of prompts and their German translations in the [prompt-templates/](prompt-templates/) folder. Note that in our study we only evaluated the German translation of the best-performing prompt. Run 
```
python prompt_llm.py --split dev --task {recognition,translation}
```
to prompt all LLMs on all prompts on instances of the development set. The results are written to [results/dev/recognition/](results/dev/recognition) and [results/dev/translation/](results/dev/translation). Folder names indicate the prompt language and index of the prompt that were used to generate the results. For example, the folder `en_0/` contains one csv files for each LLM and a text file with the first prompt (id: 0) written in English:

```
en_0/
├── aya-expanse:32b-fp16.csv
├── ...
└── prompt_template.txt
```

#### C. Get LLM Predictions
To obtain predictions of each LLM on the test dataseset using its best-performing prompt, run 
```
python prompt_llm.py --split test --task {recognition,translation} --prompt_lang {en,de} --use_context
```

The prompt ids of the best-performing prompts are hard-coded. To reproduce the results of our ablation experiments, use one of the both flags, respectively:
- `--prompt_lang de`: Uses German translations of best-performing prompts, default: en.
- `--use_context`: Flag that is used to run context ablation experiments, default: no context.

The output is written into [results/test/recognition/](results/test/recognition/) and [results/test/translation/](results/test/translation/).

```
test/
├── recognition
├── recognition+context
├── recognition-with_de_prompts
├── translation
├── translation+context
└── translation-with_de_prompts
```

#### D. Evaluate

- To reproduce the main results of our paper (Tables 1 and 2), run:

```
python evaluate.py --task {recognition,translation} --split {dev,test}
```

- To reproduce results for the **ablation experiments** (Figures 3 and 4), use `--prompt_lang de` or `--use_context` parameters.
- To reproduce the **confusion matrices** (Tables 5-6 and 8-14), use `--confusion_matrix`. Use this only with `--task recognition`.
- To reproduce **baseline results (Random, Levenshtein, Majority, Logistic Regression)** (Tables 1-2 and Tables 16-17), use `--baselines`. This applies only for the recognition task (test set).

# Citation

Please consider citing our paper if you use the code in this repository:

```bibtex
@misc{litschko2025make-every-letter-count,
      title={Make Every Letter Count: Building Dialect Variation Dictionaries from Monolingual Corpora}, 
      author={Litschko, Robert and Blaschke, Verena and Burkhardt, Diana and Plank, Barbara and Frassinelli, Diego},
      year={2025},
      eprint={2509.17855},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.17855}, 
}
```
