import os


def save_assert_prompt(output_dir, template, suffix=""):
  os.makedirs(output_dir, exist_ok=True)
  prompt_template_filepath = os.path.join(output_dir, f"{suffix}prompt_template.txt")

  if not os.path.exists(prompt_template_filepath):
    with open(prompt_template_filepath, "w") as f:
      f.write(template)
  else:
    with open(prompt_template_filepath, "r") as f:
        assert template == f.read()

  print(prompt_template_filepath)


def load_prompts(language, task, use_context=False):
  prompts = []
  if use_context:
    filepath = f"prompt-templates/{task}+context/{language}_prompts.txt"
  else:
    filepath = f"prompt-templates/{task}/{language}_prompts.txt"
  with open(filepath, "r") as f:
    cache = []
    for line in f:
      line
      if line.startswith("#"):
        cache[-1] = cache[-1].strip()
        prompts.append("".join(cache))
        cache = []
      else:
        cache.append(line)
  return prompts


pos_tagset = [
  "NOUN", 
  "ADJ", 
  "ADV", 
  "VERB",
  "PROPN",
  "ADP",
  "NUM",
  "SCONJ",
  "DET", 
  "AUX",
  "PRON",
  "CCONJ", 
  "X",
  "INTJ"
]
