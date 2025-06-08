import json
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import LongformerModel, LongformerTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.preprocessing import normalize
from typing import List, Dict
import os
import time
import csv
import random


SEED = 42
model = SentenceTransformer("all-MiniLM-L6-v2")


def logprint(log):
    log_file = os.path.join("../logs/sim-{}-{}.logs".format(results_file, time.strftime('%Y%m%d', time.gmtime())))
    with open(log_file, "a", encoding="utf-8") as fout:
        fout.write(log + "\n")
    print(log)

def preprocess_embeddings(embeddings):
    embeddings = normalize(embeddings, norm='l2')
    pca = PCA(
        n_components=0.95,
        whiten=False,
        random_state=SEED
    )
    embeddings = pca.fit_transform(embeddings)
    embeddings = normalize(embeddings, norm='l2')
    return embeddings


def get_category_embeddings(categories: List[str]) -> Dict[str, torch.Tensor]:
    category_embeddings = {}

    with torch.inference_mode():
        for category in tqdm(categories, desc="Categories"):
            clean_category = category.replace('_', ' ')

            embedding = model.encode(
                clean_category,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            category_embeddings[category] = embedding.cpu()

    return category_embeddings


characters = []
results_file = "ling_sent_20250526"
file_name = "../results/ling/" + results_file + ".jsonl"
with open(file_name, 'r') as f:
    for line in f:
        characters.append(json.loads(line))

categories = list({c['category'] for c in characters})

category_embeds = get_category_embeddings(categories)

char_embeddings = np.array([c['embedding'] for c in characters])

cat_embeddings = np.array([e.float().numpy() for e in category_embeds.values()])

combined_embeddings = np.concatenate([char_embeddings, cat_embeddings], axis=0)
processed_embeddings = preprocess_embeddings(combined_embeddings)

processed_char = processed_embeddings[:len(characters)]
processed_cat = processed_embeddings[len(characters):]

for i, char in enumerate(characters):
    char['embedding'] = processed_char[i].tolist()

for i, cat in enumerate(categories):
    category_embeds[cat] = torch.tensor(processed_cat[i])

results = []
for char in tqdm(characters, desc="Classifying"):
    char_embed = torch.tensor(char['embedding'])

    similarities = {
        cat: torch.cosine_similarity(char_embed, cat_embed, dim=0).item()
        for cat, cat_embed in category_embeds.items()
    }

    predicted = max(similarities, key=similarities.get)

    results.append({
        "Character": char['character_name'],
        "Movie": char['movie_title'],
        "TrueCategory": char['category'],
        "PredictedCategory": predicted,
        "SimilarityScore": similarities[predicted]
    })

y_true = [r['TrueCategory'] for r in results]
y_pred = [r['PredictedCategory'] for r in results]

logprint(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
logprint(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
logprint(f"Recall: {recall_score(y_true, y_pred, average='macro'):.4f}")
logprint(f"F1-score: {f1_score(y_true, y_pred, average='macro'):.4f}")

pd.DataFrame(results).to_csv('../results/{}_similarity_classification.csv'.format(results_file), index=False)
print("Results saved successfully.")
