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

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

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


def preprocess_embeddings(embeddings):
    scaler = RobustScaler(unit_variance=True)
    scaled = scaler.fit_transform(embeddings)

    pca = PCA(n_components=0.95, random_state=42)
    return pca.fit_transform(scaled)


def analyze_embeddings(embeddings):
    mean = np.mean(embeddings, axis=0)
    variance = np.var(embeddings, axis=0)
    print(f"mean: [{np.min(mean):.4f}, {np.max(mean):.4f}]")
    print(f"variance: [{np.min(variance):.4f}, {np.max(variance):.4f}]")

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"L2 normalization: mean={np.mean(norms):.4f}, std norm={np.std(norms):.4f}")

    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=2).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    avg_min_dist = np.mean(distances[:, 1])
    print(f"average mean distance: {avg_min_dist:.4f}")


def analyze_class_separation(embeddings, labels):
    from sklearn.metrics.pairwise import cosine_distances
    intra_dist, inter_dist = [], []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = (labels == label)
        same_class = embeddings[mask]
        if len(same_class) > 1:
            intra_dist.extend(cosine_distances(same_class).flatten())

        diff_class = embeddings[~mask]
        if len(diff_class) > 0:
            inter_dist.extend(cosine_distances(same_class[:10], diff_class[:10]).flatten())

    print(f"average distance in the same category: {np.mean(intra_dist):.4f}")
    print(f"average distance between categories: {np.mean(inter_dist):.4f}")

def visualize_embeddings(embeddings, labels):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Character Embeddings")
    plt.savefig("embedding_visualization.png")
    plt.close()


embeddings, gold_labels, _ = load_embeddings("results/llama_embeddings_weight1_20250419.jsonl")
# emb = preprocess_embeddings(embeddings)
emb = embeddings
analyze_embeddings(emb)
analyze_class_separation(emb, gold_labels)
# visualize_embeddings(emb, gold_labels)

