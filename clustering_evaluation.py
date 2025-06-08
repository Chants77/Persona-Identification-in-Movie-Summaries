import json
import numpy as np
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, Birch
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import itertools
from joblib import Parallel, delayed
import os
import time
import random
import torch
import csv
import umap as umap

DEBUG_MODE = True
SEED = 42
Embedding_mode = 2

PREPROCESS_CONFIG = {
    'normalization': 'l2',          # None, 'zscore', 'minmax', 'l2'
    'pca': {                        # None or dict with params
        'n_components': 0.95,
        'whiten': False
    },
    'umap': None,                   # None or dict with n_components
    'l2_normalize_after': True
}


CLUSTERING_CONFIG = {
    'method': 'kmeans',
    'params': {
        'n_init': 50,
        'algorithm': 'elkan',
    }

    # 'method': 'hierarchical',
    # 'params': {},

    # 'method': 'gmm',
    # 'params': {'covariance_type': 'full', 'tol': 1e-3},

    # 'method': 'spectral',
    # 'params': {'n_init': 20, 'assign_labels': 'kmeans'},

    # 'method': 'birch',
    # 'params': {'threshold': 0.5, 'branching_factor': 50},
}


def debug_print(msg):
    if DEBUG_MODE:
        log_file = os.path.join("logs/clustering-debug-{}.logs".format(time.strftime('%Y%m%d', time.gmtime())))
        with open(log_file, "a", encoding="utf-8") as fout:
            fout.write(msg + "\n")
        print(f"[DEBUG] {msg}")

def logprint(log):
    log_file = os.path.join("logs/clustering-{}-{}.logs".format(results_file, time.strftime('%Y%m%d', time.gmtime())))
    with open(log_file, "a", encoding="utf-8") as fout:
        fout.write(log + "\n")
    print(log)

def load_embeddings(embedding_file):
    embeddings = []
    labels = []
    char2label = {}
    with open(embedding_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            embeddings.append(data['embedding'])
            labels.append(data['category'])
            char2label[data['character_id']] = data['category']
    return np.array(embeddings), np.array(labels), char2label


def load_and_fuse_embeddings(files_and_weights):
    all_embeddings = []
    labels = []
    char2label = {}

    embeddings_dict = {}
    for file_path, weight in files_and_weights:
        embeddings_dict[file_path] = {'embeddings': [], 'weight': weight}

    char_data = {}
    for file_path, _ in files_and_weights:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                char_id = data['character_id']
                if char_id not in char_data:
                    char_data[char_id] = {'embeddings': {}, 'category': data['category']}
                char_data[char_id]['embeddings'][file_path] = data['embedding']

    valid_chars = []
    for char_id, data in char_data.items():
        if len(data['embeddings']) == len(files_and_weights):
            valid_chars.append(char_id)

    debug_print(
        f"Total characters: {len(char_data)}, Valid characters with embeddings in all files: {len(valid_chars)}")

    valid_chars.sort()

    fused_embeddings = []
    labels = []
    char2label = {}

    for char_id in valid_chars:
        data = char_data[char_id]
        char_embeddings = []

        for file_path, weight in files_and_weights:
            char_embeddings.append(np.array(data['embeddings'][file_path]))

        fused_embedding = np.zeros_like(char_embeddings[0])
        for i, (file_path, weight) in enumerate(files_and_weights):
            fused_embedding += char_embeddings[i] * weight

        fused_embeddings.append(fused_embedding)
        labels.append(data['category'])
        char2label[char_id] = data['category']

        debug_print(f"Created {len(fused_embeddings)} fused embeddings")
    return np.array(fused_embeddings), np.array(labels), char2label
def variation_of_info(gold_clusters, pred_clusters):
    n = len(gold_clusters)
    contingency = np.zeros((len(set(gold_clusters)), len(set(pred_clusters))))
    unique_gold, gold_ids = np.unique(gold_clusters, return_inverse=True)
    unique_pred, pred_ids = np.unique(pred_clusters, return_inverse=True)

    for i in range(n):
        contingency[gold_ids[i], pred_ids[i]] += 1
    contingency = contingency / n

    h_gold = -np.sum(contingency.sum(1) * np.log2(contingency.sum(1) + 1e-12))
    h_pred = -np.sum(contingency.sum(0) * np.log2(contingency.sum(0) + 1e-12))
    mi = mutual_info_score(gold_clusters, pred_clusters)

    return h_gold + h_pred - 2 * mi


def permutation_test(gold, pred, metric, n_perm=1000, higher_is_better=True):
    true = metric(gold, pred)
    if higher_is_better:
        comp = lambda x: x >= true
    else:
        comp = lambda x: x <= true
    cnt = sum(comp(metric(gold, np.random.permutation(pred))) for _ in range(n_perm))
    return true, (cnt + 1) / (n_perm + 1)


def cluster_purity(gold_clusters, pred_clusters):
    from sklearn.metrics.cluster import contingency_matrix
    contingency = contingency_matrix(gold_clusters, pred_clusters)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


def analyze_embeddings(embeddings):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=2).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    avg_min_dist = np.mean(distances[:, 1])
    logprint(f"Average nearest neighbor distance: {avg_min_dist:.4f}")

    var_per_feature = np.var(embeddings, axis=0)
    logprint(f"Feature variance summary: min={np.min(var_per_feature):.4f}, "
             f"median={np.median(var_per_feature):.4f}, max={np.max(var_per_feature):.4f}")


def baseline_purity(gold, pred, n=1000):
    counts = np.bincount(pred)
    base = []
    for _ in range(n):
        s = np.concatenate([np.repeat(i,c) for i,c in enumerate(counts)])
        np.random.shuffle(s)
        base.append(cluster_purity(gold, s))
    return np.mean(base)

def preprocess_embeddings(embeddings, config):
    if config['normalization'] == 'zscore':
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    elif config['normalization'] == 'minmax':
        scaler = MinMaxScaler()
        embeddings = scaler.fit_transform(embeddings)
    elif config['normalization'] == 'l2':
        embeddings = normalize(embeddings, norm='l2')

    if config['pca'] is not None:
        pca = PCA(
            n_components=config['pca']['n_components'],
            whiten=config['pca']['whiten'],
            random_state=SEED
        )
        embeddings = pca.fit_transform(embeddings)

    if config['umap'] is not None:
        reducer = umap.UMAP(
            n_components=config['umap']['n_components'],
            random_state=SEED
        )
        embeddings = reducer.fit_transform(embeddings)

    if config['l2_normalize_after']:
        embeddings = normalize(embeddings, norm='l2')

    return embeddings

def make_clusterer(method, P, params):
    if method == 'kmeans':
        return KMeans(n_clusters=P, random_state=SEED, **params)
    if method == 'hierarchical':
        return AgglomerativeClustering(n_clusters=P, linkage='complete')
    if method == 'gmm':
        return GaussianMixture(n_components=P, random_state=SEED)
    if method == 'spectral':
        return SpectralClustering(n_clusters=P, random_state=SEED, n_init=10, assign_labels='kmeans')
    if method == 'birch':
        # BIRCH will internally pick sub‑clusters, then we label them with P later
        return Birch(n_clusters=P)
    raise ValueError(f'Unsupported clustering method: {method}')

def evaluate_clustering(embeddings, gold_labels, config):
    results = {}

    analyze_embeddings(embeddings)

    method = config['method']
    base_params = config['params']

    P_values = [25, 50, 100]

    for P in P_values:
        logprint(f'Evaluating {method}   P={P if P > 0 else "N/A"}')
        model = make_clusterer(method, P if P > 0 else 2, base_params)
        pred = model.fit_predict(embeddings)

        valid = pred >= 0
        vi = variation_of_info(gold_labels[valid], pred[valid]) if valid.sum() else np.nan
        purity = cluster_purity(gold_labels[valid], pred[valid]) if valid.sum() else np.nan

        if valid.sum():  # 确保有有效样本
            # 分配真实标签的核心逻辑
            cluster_to_true_label = {}
            for cluster_id in np.unique(pred[valid]):
                # 1. 获取当前聚类中的所有样本
                mask = (pred == cluster_id)
                cluster_samples = gold_labels[mask]

                # 2. 找出聚类中最常见的真实标签
                unique_labels, counts = np.unique(cluster_samples, return_counts=True)
                true_label = unique_labels[np.argmax(counts)]

                # 3. 为该聚类分配这个真实标签
                cluster_to_true_label[cluster_id] = true_label

            # 4. 创建预测标签数组
            assigned_labels = np.array([cluster_to_true_label[c] for c in pred])

            # 5. 计算分类指标
            accuracy = accuracy_score(gold_labels[valid], assigned_labels[valid])
            precision = precision_score(gold_labels[valid], assigned_labels[valid], average='macro', zero_division=0)
            recall = recall_score(gold_labels[valid], assigned_labels[valid], average='macro', zero_division=0)
            f1 = f1_score(gold_labels[valid], assigned_labels[valid], average='macro', zero_division=0)
        else:
            accuracy = precision = recall = f1 = np.nan

        results[P] = {
            'vi_bits': vi,
            'purity': purity,
            'n_clusters': int(np.unique(pred[valid]).size),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        logprint(f'  VI={vi:.3f} bits | Purity={purity * 100:.1f}% | clusters={results[P]["n_clusters"]}')
        logprint(f'  Accuracy={accuracy * 100:.1f}% | Precision={precision * 100:.1f}% | '
                 f'Recall={recall * 100:.1f}% | F1={f1 * 100:.1f}%')

    return results

    # for P in [25, 50, 100]:
    #     logprint(f"Evaluating with P={P} personas...")
    #     kmeans = KMeans(n_clusters=P, n_init=50, random_state=42,
    #                    algorithm='elkan', max_iter=500)
    #     pred_labels = kmeans.fit_predict(embeddings)
    #
    #     vi_score = variation_of_info(gold_labels, pred_labels)
    #
    #     purity = cluster_purity(gold_labels, pred_labels)
    #
    #     # base = baseline_purity(gold_labels, pred_labels)
    #     # logprint(f"Purity={purity * 100:4.1f}%  (↑{(purity - base) * 100:4.1f} pp)")
    #
    #     _, vi_p_value = permutation_test(gold_labels, pred_labels, variation_of_info, higher_is_better=False)
    #     _, purity_p_value = permutation_test(gold_labels, pred_labels, cluster_purity, higher_is_better=True)
    #
    #     results[P] = {
    #         'VI': (vi_score, vi_p_value),
    #         'Purity': (purity, purity_p_value)
    #     }

    # return results


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

overall_start_time = time.time()
if Embedding_mode == 1:
    results_file = "llama_-2layer_1words_20250520"
    file_name = "results/" + results_file + ".jsonl"
    embeddings, gold_labels, _ = load_embeddings(file_name)
    debug_print(f"Loaded {len(embeddings)} embeddings and {len(gold_labels)} labels from {file_name}.")
elif Embedding_mode == 2:
    FILES_AND_WEIGHTS = [
        ('results/llama-gen/pt5_llama_-2layer_1words_20250526.jsonl', 0.3),
        ('results/llama-gen/pt5_llama_-3layer_1words_20250526.jsonl', 0.3),
        ('results/llama-gen/pt5_llama_-4layer_1words_20250526.jsonl', 0.4)
    ]
    results_file = "fused_embeddings"
    embeddings, gold_labels, _ = load_and_fuse_embeddings(FILES_AND_WEIGHTS)

unique_labels = np.unique(gold_labels)
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
gold_indices = np.array([label_to_idx[label] for label in gold_labels])
logprint(f"Number of unique labels: {len(unique_labels)}")

logprint("Evaluating clustering...")
logprint(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time))}")
start_time = time.time()
# emb = embeddings
emb = preprocess_embeddings(embeddings, PREPROCESS_CONFIG)
results = evaluate_clustering(emb, gold_indices, CLUSTERING_CONFIG)
logprint(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")

# logprint("\nVariation of Information Results:")
# for P in [25, 50, 100]:
#     vi_score, p_value = results[P]['VI']
#     logprint(f"P={P}: VI={vi_score:.2f} bits (p<{p_value:.3f})")
#
# logprint("\nPurity Results:")
# for P in [25, 50, 100]:
#     purity, p_value = results[P]['Purity']
#     logprint(f"P={P}: Purity={purity * 100:.1f}% (p<{p_value:.3f})")




