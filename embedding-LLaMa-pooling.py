import torch
from transformers import AutoTokenizer, LlamaModel, LlamaForCausalLM, AutoModelForCausalLM, pipeline, LlamaTokenizer
import csv
import json
import random
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F

def debug_print(msg):
    if DEBUG_MODE:
        log_file = os.path.join("logs/debug-{}.logs".format(time.strftime('%Y%m%d', time.gmtime())))
        with open(log_file, "a", encoding="utf-8") as fout:
            fout.write(msg + "\n")
        print(f"[DEBUG] {msg}")

def logprint(log):
    log_file = os.path.join("logs/llama-{}.logs".format(time.strftime('%Y%m%d', time.gmtime())))
    with open(log_file, "a", encoding="utf-8") as fout:
        fout.write(log + "\n")
    print(log)


DEBUG_MODE = True  # debug flag
SEED = 42
QUANTIZATION = False

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

debug_print(f"PyTorch CUDA availability: {torch.cuda.is_available()}")
debug_print(f"Available GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    debug_print(f"Current GPU: {torch.cuda.get_device_name(0)}")


overall_start_time = time.time()
logprint("Start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time)))

tvtropes_file = "data/tvtropes.clusters.txt"
category_to_characters = {}
all_categories = set()

if torch.cuda.is_available():
    logprint("CUDA is available. GPU will be used if there's enough memory.")
else:
    logprint("CUDA not available. The code will fall back to CPU or partial CPU/GPU usage.")


with open(tvtropes_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t', 1)
        if len(parts) != 2:
            logprint(f"Invalid line: {line}")
            continue
        category_name, char_info_str = parts
        char_info = json.loads(char_info_str)
        logprint(f"Category: {category_name}, Character: {char_info['char']}, Movie: {char_info['movie']}")

        if category_name not in category_to_characters:
            category_to_characters[category_name] = []
        category_to_characters[category_name].append(char_info)
        all_categories.add(category_name)

all_categories = sorted(all_categories)
logprint(f"Loaded {len(all_categories)} categories")

char_metadata_file = "data/character.metadata.tsv"
id_to_char_data = {}
map_id_to_char_data = {}

with open(char_metadata_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        w_movie_id = row[0]
        f_movie_id = row[1]
        map_id = row[10]
        freebase_char_id = row[11]
        character_name = row[3]
        id_to_char_data[freebase_char_id] = (w_movie_id, f_movie_id, character_name)
        map_id_to_char_data[map_id] = (w_movie_id, f_movie_id, character_name)

plot_summaries_file = "data/plot_summaries.txt"
movie_summaries = {}
summary_key_version = 0

with open(plot_summaries_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) == 2:
            w_movie_id, summary = row
            movie_summaries[w_movie_id] = summary
            summary_key_version = 2
        elif len(row) == 3:
            w_movie_id, c_name, short_summary = row
            movie_summaries[(w_movie_id, c_name.lower())] = short_summary
            summary_key_version = 3

logprint(f"Total categories: {len(all_categories)}")
categories_context_str = "The possible categories are: " + ", ".join(all_categories) + ". "

all_character_entries = []
for category_name, char_list in category_to_characters.items():
    for char_info in char_list:
        all_character_entries.append((category_name, char_info))

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

debug_print("Loading raw tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    legacy=False,
    use_fast=True
)

debug_print("Loading generation model...")

load_kwargs = {
    "torch_dtype": torch.bfloat16,
    "device_map": "auto"
}

if QUANTIZATION:
    load_kwargs.update({
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.bfloat16
    })

gen_model = LlamaForCausalLM.from_pretrained(model_id, **load_kwargs)
embed_model = LlamaModel.from_pretrained(model_id, **load_kwargs)

embedding_output_file = os.path.join("results/llama_character_embeddings_{}.jsonl".format(time.strftime('%Y%m%d', time.gmtime())))
logprint(f"Storing embeddings in {embedding_output_file}")


with open(embedding_output_file, "w", encoding="utf-8") as emb_fout:
    for (category_name, char_info) in tqdm(all_character_entries, desc="All Characters"):
        single_start_time = time.time()
        f_map_id = char_info["id"]
        movie_title = char_info["movie"]
        char_name = char_info["char"]

        if f_map_id not in map_id_to_char_data:
            logprint(f"Character {char_name} from movie {movie_title} not found in metadata (map_id).")
            continue

        w_movie_id, f_movie_id, character_name_in_meta = map_id_to_char_data[f_map_id]

        if summary_key_version == 2:
            summary_key = w_movie_id
        else:
            summary_key = (w_movie_id, char_name.lower())

        summary = movie_summaries.get(summary_key, "")
        if not summary.strip():
            logprint(f"No summary found for key: {summary_key}")
            continue

        # prompt_text = f"""[INST] Analyze the character {char_name} from {movie_title}.
        #                 Movie summary:
        #                 {summary}
        #                 Generate a compact semantic representation of this character's persona. [/INST]"""

        prompt_text = f"""The movie {movie_title} is about the character {char_name}.
                        Movie summary:
                        {summary}
                        In one word, describe {char_name}'s role:
                        """

        # Tokenize
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(embed_model.device)
        if inputs.input_ids.shape[1] == tokenizer.model_max_length:
            logprint(f"warning: {char_name}'s input is truncated.")

        with torch.no_grad():
            outputs = embed_model(**inputs, output_hidden_states=True)
        # mean pooling

        def mean_pooling(hidden_states, attention_mask):
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        pooled = mean_pooling(outputs.last_hidden_state, inputs.attention_mask)
        embedding = F.normalize(pooled, p=2, dim=1).squeeze().cpu().numpy().tolist()

        gen_output = gen_model.generate(
            inputs.input_ids.to(gen_model.device),
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False
        )
        gen_text = tokenizer.decode(gen_output[0], skip_special_tokens=True)

        record = {
            "category": category_name,
            "character_id": f_map_id,
            "movie_title": movie_title,
            "character_name": char_name,
            "embedding": embedding,
            "generated_text_sample": gen_text[:200]
        }
        emb_fout.write(json.dumps(record) + "\n")
        emb_fout.flush()

        logprint(f"Processed character {char_name} from {movie_title} - embedding size {len(embedding)}")

logprint("Finished embeddings collection.")
logprint(f"time consumption: {time.time()-overall_start_time:.2f}s")
