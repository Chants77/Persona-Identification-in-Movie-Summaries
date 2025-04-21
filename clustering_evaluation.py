import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import itertools
from joblib import Parallel, delayed
import os
import time
import random
import torch
import csv
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, normalize
from sklearn.decomposition import PCA

def debug_print(msg):
    if DEBUG_MODE:
        log_file = os.path.join("logs/clustering-debug-{}.logs".format(time.strftime('%Y%m%d', time.gmtime())))
        with open(log_file, "a", encoding="utf-8") as fout:
            fout.write(msg + "\n")
        print(f"[DEBUG] {msg}")

def logprint(log):
    log_file = os.path.join("logs/clustering-{}.logs".format(time.strftime('%Y%m%d', time.gmtime())))
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

def preprocess_embeddings(embeddings):
    # pre
    # scaler = RobustScaler(unit_variance=True)
    # scaled = scaler.fit_transform(embeddings)
    #
    # pca = PCA(n_components=0.95, random_state=42)
    # return pca.fit_transform(scaled)

    # pre2
    # vecs = normalize(embeddings, norm="l2")
    # vecs = PCA(n_components=0.95, whiten=True,
    #            random_state=42).fit_transform(vecs)

    # pre3
    # vecs = normalize(embeddings, norm="l2")
    # vecs = PCA(n_components=0.95, whiten=True,
    #            random_state=42).fit_transform(vecs)
    # vecs = normalize(vecs, norm="l2", axis=1)

    # pre4
    vecs = normalize(embeddings, norm="l2")
    vecs = PCA(n_components=0.95, whiten=False,
               random_state=42).fit_transform(vecs)
    vecs = normalize(vecs, norm="l2", axis=1)
    return vecs


def evaluate_clustering(embeddings, gold_labels, n_jobs=-1):
    results = {}

    # scaler = StandardScaler()
    # norm_embeddings = scaler.fit_transform(embeddings)
    analyze_embeddings(embeddings)

    for P in [25, 50, 100]:
        logprint(f"Evaluating with P={P} personas...")
        kmeans = KMeans(n_clusters=P, n_init=50, random_state=42,
                       algorithm='elkan', max_iter=500)
        pred_labels = kmeans.fit_predict(embeddings)
        # kmeans = SphericalKMeans(
        #     n_clusters=P,
        #     n_init=50,
        #     max_iter=500,
        #     random_state=42)
        # pred_labels = kmeans.fit_predict(embeddings)

        vi_score = variation_of_info(gold_labels, pred_labels)

        purity = cluster_purity(gold_labels, pred_labels)

        # base = baseline_purity(gold_labels, pred_labels)
        # logprint(f"Purity={purity * 100:4.1f}%  (↑{(purity - base) * 100:4.1f} pp)")

        _, vi_p_value = permutation_test(gold_labels, pred_labels, variation_of_info, higher_is_better=False)
        _, purity_p_value = permutation_test(gold_labels, pred_labels, cluster_purity, higher_is_better=True)

        results[P] = {
            'VI': (vi_score, vi_p_value),
            'Purity': (purity, purity_p_value)
        }

    return results


DEBUG_MODE = True  # debug flag
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

overall_start_time = time.time()
file_name = "results/llama_embeddings_pooling_20250419.jsonl"
embeddings, gold_labels, _ = load_embeddings(file_name)
debug_print(f"Loaded {len(embeddings)} embeddings and {len(gold_labels)} labels from {file_name}.")

unique_labels = np.unique(gold_labels)
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
gold_indices = np.array([label_to_idx[label] for label in gold_labels])
logprint(f"Number of unique labels: {len(unique_labels)}")

logprint("Evaluating clustering...")
logprint(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time))}")
start_time = time.time()
emb = embeddings
emb = preprocess_embeddings(embeddings)
results = evaluate_clustering(emb, gold_indices, n_jobs=4)
logprint(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")

logprint("\nVariation of Information Results:")
for P in [25, 50, 100]:
    vi_score, p_value = results[P]['VI']
    logprint(f"P={P}: VI={vi_score:.2f} bits (p<{p_value:.3f})")

logprint("\nPurity Results:")
for P in [25, 50, 100]:
    purity, p_value = results[P]['Purity']
    logprint(f"P={P}: Purity={purity * 100:.1f}% (p<{p_value:.3f})")




