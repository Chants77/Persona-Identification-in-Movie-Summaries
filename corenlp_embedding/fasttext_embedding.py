import json
import sys
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')
print("FastText model loaded.")

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


def generate_char_embeddings(bag):
    word_embeddings = []

    if bag:
        for v in bag.get("agent", []):
            emb = ft.get_word_vector(v)
            word_embeddings.append(emb)
        for v in bag.get("patient", []):
            emb = ft.get_word_vector(v)
            word_embeddings.append(emb)
        for adj in bag.get("attribute", []):
            emb = ft.get_word_vector(adj)
            word_embeddings.append(emb)
    n_word = len(word_embeddings)
    char_embedding = sum(word_embeddings) / n_word if n_word > 0 else np.zeros(ft.get_dimension())
    return char_embedding


word_bag_path = "old_corenlp_filtered_word_bag.json"
with open(word_bag_path, "r", encoding="utf-8") as fh:
    bags = json.load(fh)

output_file = os.path.join("../results/ling_word_{}.jsonl".format(time.strftime('%Y%m%d', time.gmtime())))
with open(output_file, "w", encoding="utf-8") as out_f:
    for idx, row in enumerate(characters):
        cid = row["id"]
        bag = bags.get(cid, {})
        char_embedding = generate_char_embeddings(bag)
        rec = {
            "category": row["trope"],
            "character_id": row["id"],
            "movie_title": row.get("movie", ""),
            "character_name": row.get("char", ""),
            "embedding": char_embedding.tolist(),
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote {len(characters):,} records → {output_file}", file=sys.stderr)

