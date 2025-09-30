import os.path
import csv
import tqdm
import time 
import tempfile

from copy import deepcopy
from argparse import ArgumentParser
from datetime import datetime
from openai import OpenAI
from ratelimiter import RateLimiter
from utils import save_assert_prompt, load_prompts


# constants
api_key = 'ollama' 
port = 11435 # 11436
base_url = f"http://localhost:{port}/v1"
temperature = 0
multiplier = 100
max_tokens = 20
print(base_url)


parser = ArgumentParser()
# required
parser.add_argument("--split", choices=["dev", "test"], required=True)
parser.add_argument("--task", choices=["translation", "recognition"], required=True)
# optional 
parser.add_argument("--prompt_lang", choices=["de", "en"], default="en", required=False)
parser.add_argument("--use_context", action='store_true', required=False)
args = parser.parse_args()


client = OpenAI(
      api_key = api_key,
      base_url = base_url
)


# CSV fieldnames
fieldnames_in = ['id', 'lemma_id', 'de_term', 'de_freq', 'pos', 'pos_perc', 'bar', 'bar_freq', 'ld', 'label', 'contexts']
fieldnames_out = fieldnames_in + ['prediction']


def get_already_processed(output_filepath):
  if os.path.exists(output_filepath):
    with open(output_filepath, "r") as f:
      reader = csv.DictReader(f, fieldnames=fieldnames_out)
      header = next(reader) # skip header
      assert header["id"] == "id" # verify file has a header
      return {r["id"] for r in reader}
  return set()


def get_row_count(input_filepath):
  with open(input_filepath, "r") as f:
    return sum(1 for _ in csv.reader(f))


@RateLimiter(max_calls=2*multiplier, period=1) # 2 /sec
@RateLimiter(max_calls=60*multiplier, period=60) # 60 /min
@RateLimiter(max_calls=3000*multiplier, period=60*60) # 3000 /h
def api_call(prompt, model):
  messages = [
    {"role":"system","content":"You are a helpful assistant"},
    {"role":"user","content": prompt}
  ]
  
  # Get response from GPT-oss (experimental, not used in our paper)
  if model.startswith("gpt-oss:"):
    # https://platform.openai.com/docs/guides/reasoning
    chat_completion = client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=256, # default recommended by open ai is 25K
      reasoning_effort="low",
      stream_options={ "include_usage": True } 
    )
    if chat_completion.choices[0].finish_reason == "length":
        llm_out = "max_tokens_exceeded"
    else:
      response = chat_completion.choices[0].message.content
      llm_out = response
      
  # Get response from instruction-tuned LLM
  else:
    chat_completion = client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
    )
    response = chat_completion.choices[0].message.content
    llm_out = response
    
  return llm_out


def run(input_filepath, output_filepath, template, model, use_context=False):
  already_processed = get_already_processed(output_filepath)
  project_root = os.path.dirname(__file__)
  input_filepath = os.path.join(project_root, input_filepath)
  out_file = lambda is_debug: tempfile.TemporaryFile(mode="w") if is_debug else open(output_filepath, "a")

  debug = True 

  # with out_file(debug=True) as f:
  #   f.write("test123")
  #   print(f.name)

  with open(input_filepath, "r") as f:
    reader = csv.DictReader(f, fieldnames=fieldnames_in)
    header = next(reader) # skip header
    assert header["id"] == "id" # verify file has a header
    
    n_processed = len(already_processed)
    n_missing = get_row_count(input_filepath) - n_processed - 1
    
    if debug:
      n_missing = 200
      n_processed = len(already_processed) - n_missing
    
    if n_missing > 0:
      # skip ahead until first record for which we need LLM prediction
      for i in range(n_processed):
        record_id = next(reader)["id"]
        assert record_id in already_processed
      
      print(f"Writing {n_missing} records in file {output_filepath}")
      with out_file(is_debug=debug) as g:
        writer = csv.DictWriter(g, fieldnames=fieldnames_out)
        writer.writeheader()
        
        for row in tqdm.tqdm(reader, total=n_missing):
          # record_id, lemma_id, de_lemma, de_freq, pos, pos_perc, bar, bar_freq, ld, label, contexts = row
          record_id = row["id"]
          
          if record_id in already_processed and not debug:
            # this should never occur, because records are processed in order
            raise AssertionError(f"{record_id} has already been enountered.")
    
          # fill in placeholders in prompt template
          prompt = template.replace("term_bar", row["bar"]).replace("term_de", row["de_term"])
          
          if use_context:
            # we use only the first example in our context ablation experiments
            prompt = prompt.replace("####", row["contexts"].split("####")[0])
          
          result = deepcopy(row)
          result['prediction'] = api_call(prompt, model)
          writer.writerow(result)
          
      loaded_model_on_gput = True
      
    else:
      print(f"Skipping file {output_filepath}")
      loaded_model_on_gput = False
      
    return loaded_model_on_gput


def maybe_wait(llm, loaded_LLM_on_gpu):
  # wait until ollama unloads model from GPU and continue after 15 min
  if llm in ["llama3.3:70b-instruct-fp16", "mistral-large"] and loaded_LLM_on_gpu:
    minutes = 60 * 15
    print(f"Waiting until {llm} gets unloaded due to timeout ({minutes} min)")
    time.sleep(minutes)
    

def main():
  llms = [
    "mistral-large",
    "mistral:7b-instruct-fp16",
    "llama3.3:70b-instruct-fp16",
    "llama3.1:8b-instruct-fp16",
    "llama4:scout",
    "aya-expanse:8b",
    "aya-expanse:32b-fp16",
    "gemma3:12b",
    "gemma3:27b",
  ]
  
  dev_test = args.split
  task = args.task
  use_context = args.use_context
  prompt_lang = args.prompt_lang

  project_dir = os.path.dirname(__file__)
  output_basedir = os.path.join(project_dir, "results", dev_test, task if not use_context else f"{task}+context")
  if prompt_lang == "de":
    output_basedir += "-with_de_prompts"
  os.makedirs(output_basedir, exist_ok=True)
  
  if dev_test == "dev":
    prompt_templates = load_prompts(language=prompt_lang, task=task)
    print(f"language={prompt_lang}")
    for llm in llms:
      for i, temp in enumerate(prompt_templates):
        output_dir = os.path.join(output_basedir, f"{prompt_lang}_{i}")
        save_assert_prompt(output_dir, template=temp)
        output_filepath = os.path.join(output_dir, f"{llm}.csv")
        input_filepath = f"data/{task}_{dev_test}.csv"
        print(output_filepath)
        loaded_LLM_on_gpu = run(input_filepath=input_filepath, output_filepath=output_filepath, template=temp, model=llm)
        maybe_wait(llm, loaded_LLM_on_gpu)
        
  else:
    # maps each LLM to the index of the best-performing prompt (recognition task)
    llm_to_prompt_recognition = {
      "mistral-large": 8,
      "mistral:7b-instruct-fp16": 2,
      "gemma3:12b": 8,
      "gemma3:27b": 6,
      "llama3.1:8b-instruct-fp16": 5,
      "llama3.3:70b-instruct-fp16": 5,
      "llama4:scout": 8,
      "aya-expanse:8b": 3,
      "aya-expanse:32b-fp16": 5,
    }

    # maps each LLM to the index of the best-performing prompt (recognition task)
    llm_to_prompt_translation = {
      "mistral-large": 20,
      "mistral:7b-instruct-fp16": 20,
      "gemma3:12b": 12,
      "gemma3:27b": 20,
      "llama3.1:8b-instruct-fp16": 20,
      "llama3.3:70b-instruct-fp16": 20,
      "llama4:scout": 20,
      "aya-expanse:8b": 7,
      "aya-expanse:32b-fp16": 20,
    }

    id_to_prompt = {
      i: prompt for i, prompt in enumerate(load_prompts(language=prompt_lang, task=task, use_context=use_context))
    }
    
    # sanity check for prompts with context
    if use_context and prompt_lang == "en":
      for llm in llms:
        temp = id_to_prompt[llm_to_prompt_translation[llm] if task == "translation" else llm_to_prompt_recognition[llm]]
        assert 'Usage example: "####"' in temp
    
    if use_context and prompt_lang == "de":
      raise NotImplementedError("Context and prompt language ablations cannot be run together.")
    
    for llm in llms:
      # output_dir = os.path.join(output_basedir, f"{lang}_{i}")
      output_dir = output_basedir
      temp = id_to_prompt[llm_to_prompt_translation[llm] if task == "translation" else llm_to_prompt_recognition[llm]]
      output_filepath = os.path.join(output_dir, f"{llm}.csv")
      input_filepath = f"data/{task}_{dev_test}.csv"
      print(output_filepath)
      save_assert_prompt(output_dir, template=temp, suffix=llm + "-")
      loaded_LLM_on_gpu = run(input_filepath=input_filepath, output_filepath=output_filepath, template=temp, model=llm, use_context=use_context)
      maybe_wait(llm, loaded_LLM_on_gpu)


if __name__ == '__main__':
    main()
