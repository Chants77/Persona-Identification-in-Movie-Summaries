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
    log_file = os.path.join("logs/llama_{}layer_{}words_{}.logs".format(LAYER, WORD_NUM, time.strftime('%Y%m%d', time.gmtime())))
    with open(log_file, "a", encoding="utf-8") as fout:
        fout.write(log + "\n")
    print(log)


DEBUG_MODE = True  # debug flag
SEED = 42
QUANTIZATION = False
LAYER = -2  # penultimate layer
WORD_NUM = 10

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

tvtropes_file = "../data/tvtropes.clusters.txt"
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

char_metadata_file = "../data/character.metadata.tsv"
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

plot_summaries_file = "../data/plot_summaries.txt"
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
# embed_model = LlamaModel.from_pretrained(model_id, **load_kwargs)

embedding_output_file = os.path.join("results/llama_{}layer_{}words_{}.jsonl".format(LAYER, WORD_NUM, time.strftime('%Y%m%d', time.gmtime())))
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
                        In {WORD_NUM} words, describe {char_name}'s role:
                        """

        # Tokenize
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(gen_model.device)
        if inputs.input_ids.shape[1] == tokenizer.model_max_length:
            logprint(f"warning: {char_name}'s input is truncated.")

        # with torch.no_grad():
        #     outputs = embed_model(**inputs, output_hidden_states=True)
        # mean pooling

        def mean_pooling(hidden_states, attention_mask):
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        # pooled = mean_pooling(outputs.last_hidden_state, inputs.attention_mask)
        # embedding = F.normalize(pooled, p=2, dim=1).squeeze().cpu().numpy().tolist()

        gen_output = gen_model.generate(
            inputs.input_ids.to(gen_model.device),
            max_new_tokens=WORD_NUM,
            temperature=0.0,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        # gen_text = tokenizer.decode(gen_output[0], skip_special_tokens=True)

        steps = len(gen_output.hidden_states)  # = number of new tokens
        vecs = []
        for s in range(steps):
            h_step_layer = gen_output.hidden_states[s][LAYER]  # tensor (1,1,dim)
            vecs.append(h_step_layer[0, -1, :].squeeze())  # (dim,)
            debug_print(f"{h_step_layer[0, -1, :].shape}")
        vec = torch.stack(vecs).mean(0)  # (dim,)
        embedding = F.normalize(vec.float(), p=2, dim=0).squeeze().cpu().numpy().tolist()

        # hiddens = gen_output.hidden_states[-2]  # penultimate layer
        # indices of generated tokens: last n positions
        new_tok_start = inputs.input_ids.shape[1]
        # debug_print(f"new_tok_start: {new_tok_start}, hiddens.shape: {hiddens.shape}, inputs.input_ids.shape: {inputs.input_ids.shape}")
        # new_tok_end = hiddens.shape[1]  # inclusive
        # debug_print(f"new_tok_end: {new_tok_end}")
        # new_vecs = hiddens[0, new_tok_start:new_tok_end, :]  # (n, dim)
        new_ids = gen_output.sequences[0][new_tok_start:]  # numeric
        new_tokens = tokenizer.convert_ids_to_tokens(new_ids)  # string form

        # logprint("idx | token_id | token_str            | is_special")
        # logprint("----+----------+----------------------+-----------")
        # spec_set = {tokenizer.bos_token_id,
        #             tokenizer.eos_token_id,
        #             tokenizer.convert_tokens_to_ids("<|eot_id|>")}
        #
        # for i, (tid, tok) in enumerate(zip(new_ids.tolist(), new_tokens), start=1):
        #     flag = {tokenizer.bos_token_id: "BOS",
        #             tokenizer.eos_token_id: "EOS",
        #             tokenizer.convert_tokens_to_ids("<|eot_id|>"): "EOT"}.get(tid, "")
        #     logprint(f"{i:3} | {tid:8} | {tok:<20} | {flag}")

        # 3) keep only first 3 real words (strip punctuation)
        gen_text = tokenizer.decode(gen_output.sequences[0][new_tok_start:])
        debug_print(f"Generated text: {gen_text}")
        # words = [m.group() for m in RE_WORD.finditer(gen_text)]
        # keep_n = min(3, len(words))
        # kept_vecs = new_vecs[:keep_n]  # (≤3, dim)

        # vec = new_vecs.mean(0)
        # embedding = torch.nn.functional.normalize(vec.float(), p=2, dim=0)
        # debug_print(f"Embedding shape: {embedding.shape}")

        record = {
            "category": category_name,
            "character_id": f_map_id,
            "movie_title": movie_title,
            "character_name": char_name,
            "embedding": embedding,
            "generated_text_sample": gen_text[:]
        }
        emb_fout.write(json.dumps(record) + "\n")
        emb_fout.flush()

        logprint(f"Processed character {char_name} from {movie_title} - embedding size {len(embedding)}")

logprint("Finished embeddings collection.")
logprint(f"time consumption: {time.time()-overall_start_time:.2f}s")
