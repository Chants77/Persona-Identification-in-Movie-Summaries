import json
import sys
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

tvtropes_file = "../data/tvtropes.clusters.cleaned.txt"

characters = []
with open(tvtropes_file, "r", encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        trope, meta_json = line.rstrip("\n").split("\t", 1)
        meta = json.loads(meta_json)
        meta["trope"] = trope
        characters.append(meta)


def build_sentences(characters, bag):
    char = characters["char"]
    sentences = []

    if bag:
        for v in bag.get("agent", []):
            sentences.append(f"{char} {v} someone.")
        for v in bag.get("patient", []):
            sentences.append(f"Someone {v} {char}.")
        for adj in bag.get("attribute", []):
            sentences.append(f"{char} is {adj}.")

    if not sentences:
        movie = characters.get("movie", "an unknown movie")
        sentences = [f"{char} is a character from {movie}."]
    return sentences


word_bag_path = "old_corenlp_filtered_word_bag.json"
with open(word_bag_path, "r", encoding="utf-8") as fh:
    bags = json.load(fh)

all_sentences = []
char_indices = []

for idx, row in enumerate(characters):
    cid = row["id"]
    bag = bags.get(cid, {})
    sents = build_sentences(row, bag)
    all_sentences.extend(sents)
    char_indices.extend([idx] * len(sents))
    print(f"Built {len(sents):,} sentences for {cid} ({row['char']})", file=sys.stderr)

print(f"Built {len(all_sentences):,} sentences "
      f"for {len(characters):,} characters.", file=sys.stderr)

model = SentenceTransformer("all-MiniLM-L6-v2")
sent_embeds = model.encode(
    all_sentences,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

dim = sent_embeds.shape[1]
n_char = len(characters)
sums = np.zeros((n_char, dim), dtype=np.float32)
counts = np.zeros(n_char, dtype=np.int32)

for sent_vec, char_idx in zip(sent_embeds, char_indices):
    sums[char_idx] += sent_vec
    counts[char_idx] += 1

counts[counts == 0] = 1
char_embeds = sums / counts[:, None]

output_file = os.path.join("../results/ling_sent_{}.jsonl".format(time.strftime('%Y%m%d', time.gmtime())))
with open(output_file, "w", encoding="utf-8") as out_f:
    for row, emb in zip(characters, char_embeds):
        rec = {
            "category": row["trope"],
            "character_id": row["id"],
            "movie_title": row.get("movie", ""),
            "character_name": row.get("char", ""),
            "embedding": emb.tolist(),
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote {len(characters):,} records → {output_file}", file=sys.stderr)

