import json
import torch
import pandas as pd
import numpy as np
from transformers import BigBirdModel, BigBirdTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.preprocessing import normalize
from typing import List, Dict
import os
import time
import csv
import random
import torch.nn.functional as F


SEED = 42
MODEL_NAME = 'google/bigbird-roberta-base'
USE_CLS = True
MAX_LENGTH = 4096
ATTENTION_TYPE = "block_sparse"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BigBirdTokenizer.from_pretrained(MODEL_NAME)
MASK = tokenizer.mask_token
model = BigBirdModel.from_pretrained(
    MODEL_NAME,
    attention_type=ATTENTION_TYPE,
    output_hidden_states=True
)
model = model.to(DEVICE)
model.eval()


def logprint(log):
    log_file = os.path.join("../logs/sim-{}-{}.logs".format(results_file, time.strftime('%Y%m%d', time.gmtime())))
    with open(log_file, "a", encoding="utf-8") as fout:
        fout.write(log + "\n")
    print(log)

def preprocess_embeddings(embeddings):
    """统一的预处理流程"""
    # L2归一化
    embeddings = normalize(embeddings, norm='l2')

    # PCA降维
    pca = PCA(
        n_components=0.95,
        whiten=False,
        random_state=SEED
    )
    embeddings = pca.fit_transform(embeddings)

    # 再次L2归一化
    embeddings = normalize(embeddings, norm='l2')
    return embeddings


def get_category_embeddings(categories: List[str]) -> Dict[str, torch.Tensor]:
    """获取分类别embedding并保持原始顺序"""
    category_embeddings = {}

    with torch.inference_mode():
        for category in tqdm(categories, desc="Categories"):
            clean_category = category.replace('_', ' ')

            prompt = f"In 2 words, describe the key characteristics of {clean_category}: {MASK * 2}."

            enc = tokenizer(prompt, return_tensors="pt",
                            max_length=4096, truncation=True).to(model.device)
            mask_pos = (enc.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            out = model(**enc)
            h = out.hidden_states[-1][0, mask_pos].mean(0)
            embedding = F.normalize(h.float(), p=2, dim=0).detach().cpu()

            category_embeddings[category] = embedding

    return category_embeddings


# 加载角色数据
characters = []
results_file = "bigbird_mask_2_20250528"
file_name = "../results/bigbird-mask/" + results_file + ".jsonl"
with open(file_name, 'r') as f:
    for line in f:
        characters.append(json.loads(line))

# 提取所有类别
categories = list({c['category'] for c in characters})

# 生成原始category embeddings
category_embeds = get_category_embeddings(categories)

# 准备预处理数据 ---------------------------------------------------------
# 提取所有character embeddings [n_characters, dim]
char_embeddings = np.array([c['embedding'] for c in characters])

# 提取category embeddings [n_categories, dim]
cat_embeddings = np.array([e.float().numpy() for e in category_embeds.values()])

# 合并数据并进行统一预处理
combined_embeddings = np.concatenate([char_embeddings, cat_embeddings], axis=0)
processed_embeddings = preprocess_embeddings(combined_embeddings)

# 分割处理后的embedding
processed_char = processed_embeddings[:len(characters)]  # 前N个是角色
processed_cat = processed_embeddings[len(characters):]  # 后M个是类别

# 更新角色embedding
for i, char in enumerate(characters):
    char['embedding'] = processed_char[i].tolist()

# 更新类别embedding
for i, cat in enumerate(categories):
    category_embeds[cat] = torch.tensor(processed_cat[i])

# 相似度计算和分类 -----------------------------------------------------
results = []
for char in tqdm(characters, desc="Classifying"):
    char_embed = torch.tensor(char['embedding'])

    # 计算与所有类别的相似度
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

# 评估结果
y_true = [r['TrueCategory'] for r in results]
y_pred = [r['PredictedCategory'] for r in results]

logprint(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
logprint(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
logprint(f"Recall: {recall_score(y_true, y_pred, average='macro'):.4f}")
logprint(f"F1-score: {f1_score(y_true, y_pred, average='macro'):.4f}")

# 保存结果
pd.DataFrame(results).to_csv('../results/{}_similarity_classification.csv'.format(results_file), index=False)
print("Results saved successfully.")
