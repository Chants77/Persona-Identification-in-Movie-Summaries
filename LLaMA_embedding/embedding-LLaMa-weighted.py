import torch
from transformers import AutoTokenizer, LlamaModel, LlamaForCausalLM, AutoModelForCausalLM, pipeline
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
import spacy

nlp = spacy.load("en_core_web_sm")
DEBUG_MODE = True  # debug flag
SEED = 42
weight_mode = 2  # 0: no weight, 1: last 3 layers, 2: all layers

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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

def calculate_dependency_depth(text):
    doc = nlp(text)
    depths = []
    for sent in doc.sents:
        root = [token for token in sent if token.head == token][0]
        depths.append(max([token.head.i - token.i for token in sent]))
    return max(depths) if depths else 0


def dynamic_layer_fusion(all_hidden, summary, text_length, idx, device):
    layer_weights = np.zeros(len(all_hidden))

    layer_weights[1:6] = 0.1
    layer_weights[7:20] = 0.6
    layer_weights[21:] = 0.3

    if text_length > 500:
        layer_weights[21:] *= 0.7
    if calculate_dependency_depth(summary) > 5:
        layer_weights[1:6] *= 3

    layer_weights /= layer_weights.sum()
    all_hidden = [h.to(device) for h in all_hidden]

    fused = sum(w * h[0][idx] for w, h in zip(layer_weights, all_hidden))
    return fused


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
pseudo_cls_token = "<CHAR_CLS>"

debug_print("Loading raw tokenizer...")
hf_tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False, use_fast=True)
debug_print(f"Tokenizer class: {type(hf_tokenizer).__name__}")
debug_print(f"Original vocab size: {hf_tokenizer.vocab_size}")
debug_print(f"Original special tokens: {len(hf_tokenizer.special_tokens_map)}")

debug_print(f"Adding special token: {pseudo_cls_token}")
hf_tokenizer.add_tokens([pseudo_cls_token], special_tokens=True)
debug_print(f"Updated vocab size: {hf_tokenizer.vocab_size}")
debug_print(f"New special tokens: {hf_tokenizer.special_tokens_map}")

pseudo_cls_id = hf_tokenizer.convert_tokens_to_ids(pseudo_cls_token)
debug_print(f"<CHAR_CLS> Token ID: {pseudo_cls_id} (0x{pseudo_cls_id:x})")
# assert pseudo_cls_id < len(hf_tokenizer), "Token ID out of range!"

debug_print("Loading generation model...")
base_model = LlamaForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
base_model.resize_token_embeddings(len(hf_tokenizer))
base_model.config.vocab_size = len(hf_tokenizer)
debug_print(f"Generation model vocab size: {base_model.config.vocab_size}")
debug_print(f"Generation model device: {base_model.device}")


llama_pipeline = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=hf_tokenizer,
    device_map="auto",
    temperature=0.0,
    do_sample=False,
    top_p=1.0
)

logprint("Pipeline loaded.")

hf_config = base_model.config  
hf_config.output_hidden_states = True

#model_for_hidden = LlamaModel.from_pretrained(
#    model_id,
#    config=base_model.config,
#    torch_dtype=torch.bfloat16,
#    device_map="auto"
#)

#model_for_hidden.set_input_embeddings(base_model.get_input_embeddings())

#model_for_hidden.resize_token_embeddings(len(hf_tokenizer))
#model_for_hidden.config.vocab_size = len(hf_tokenizer)


model_for_hidden = base_model.model
model_for_hidden.eval()

debug_print(f"Hidden model embeddings: {model_for_hidden.embed_tokens.weight.shape}")  # [128257, 4096]
debug_print(f"Generation model embeddings: {base_model.model.embed_tokens.weight.shape}")  # [128257, 4096]
assert model_for_hidden.embed_tokens.weight.shape == base_model.model.embed_tokens.weight.shape


pseudo_cls_id = hf_tokenizer.convert_tokens_to_ids(pseudo_cls_token)
#assert pseudo_cls_id < hf_tokenizer.vocab_size, \
#    f"Token ID {pseudo_cls_id} exceeds {hf_tokenizer.vocab_size}"

test_prompt = f"{pseudo_cls_token} Test prompt"
test_inputs = hf_tokenizer(test_prompt, return_tensors="pt").to("cuda")
try:
    test_output = model_for_hidden(**test_inputs)
    logprint("vocabulary is valid")
    debug_print(f"special token hidden states: {test_output.last_hidden_state[0, -1, :10]}")
except RuntimeError as e:
    logprint(f"{str(e)}")
    raise

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

        prompt_text = (
            f"{pseudo_cls_token} Please focus on the character {char_name} from the movie {movie_title}.\n\n"
            "Below is the movie summary:\n"
            f"{summary}\n"
        )

        # Tokenize
        inputs = hf_tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model_for_hidden.device)
        attention_mask = inputs["attention_mask"].to(model_for_hidden.device)

        device = model_for_hidden.device  # or torch.device("cuda") if you want
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model_for_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True  # or rely on config
            )

        all_hidden = outputs.hidden_states  # (layer0, layer1, ..., layerN)
        final_layer = all_hidden[-1]  # [batch_size, seq_len, hidden_dim]

        final_layer = final_layer[0]  # [seq_len, hidden_dim]

        pseudo_cls_id = hf_tokenizer.convert_tokens_to_ids(pseudo_cls_token)
        cls_positions = (input_ids[0] == pseudo_cls_id).nonzero(as_tuple=True)[0]
        # if len(cls_positions) == 0:
        #     # character_embedding = final_layer[0]
        #     character_embedding = final_layer[: 3].mean(dim=0)
        # else:
        assert len(cls_positions) > 0, "CLS token not found in input IDs"

        idx = cls_positions[0].item()
        if weight_mode == 0:
            character_embedding = final_layer[idx]
        elif weight_mode == 1:
            layer_weights = [0.3, 0.5, 0.2]
            selected_layers = [all_hidden[-i] for i in range(3, 0, -1)]
            weighted_embedding = sum(w * layer[0][idx] for w, layer in zip(layer_weights, selected_layers))
            character_embedding = weighted_embedding
        elif weight_mode == 2:
            character_embedding = dynamic_layer_fusion(all_hidden, summary, len(prompt_text), idx, device)

        character_embedding = character_embedding.cpu()
        embedding_np = character_embedding.float().numpy().tolist()

        gen_output = llama_pipeline(prompt_text, max_new_tokens=100)
        gen_text = gen_output[0]["generated_text"]

        record = {
            "category": category_name,
            "character_id": f_map_id,
            "movie_title": movie_title,
            "character_name": char_name,
            "embedding": embedding_np,
            "generated_text_sample": gen_text[:200]
        }
        emb_fout.write(json.dumps(record) + "\n")
        emb_fout.flush()

        logprint(f"Processed character {char_name} from {movie_title} - embedding size {len(embedding_np)}")

logprint("Finished embeddings collection.")
