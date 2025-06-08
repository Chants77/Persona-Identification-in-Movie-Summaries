import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
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
embedding_mode = 1
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()



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


def get_category_embeddings(categories: List[str], mode: int) -> Dict[str, torch.Tensor]:
    category_embeddings = {}

    with torch.inference_mode():
        for category in tqdm(categories, desc="Categories"):
            clean_category = category.replace('_', ' ')

            if mode == 1:
                inputs = tokenizer(
                    clean_category,
                    return_tensors="pt",
                    add_special_tokens=True
                ).to(model.device)

                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                embedding = hidden_states.mean(dim=1).squeeze()

            elif mode == 2:
                prompt = f"Describe the key characteristics of {clean_category} in 20 words:"
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(model.device)

                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=20,
                    temperature=0.0,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True
                )
                vecs = []
                for s in range(len(outputs.hidden_states)):
                    h_step_layer = outputs.hidden_states[s][-2]
                    vecs.append(h_step_layer[0, -1, :].squeeze())
                embedding = torch.stack(vecs).mean(0)

            category_embeddings[category] = embedding.cpu()

    return category_embeddings


characters = []
results_file = "llama_con_layer-2_20250527"
file_name = "../results/llama-con/" + results_file + ".jsonl"
with open(file_name, 'r') as f:
    for line in f:
        characters.append(json.loads(line))
categories = list({c['category'] for c in characters})

category_embeds = get_category_embeddings(categories, embedding_mode)

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
