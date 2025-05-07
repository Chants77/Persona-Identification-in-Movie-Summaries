import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


def variation_of_info(gold_clusters, pred_clusters):
    n = len(gold_clusters)
    contingency = contingency_matrix(gold_clusters, pred_clusters)
    contingency = contingency / n

    h_gold = -np.sum(contingency.sum(1) * np.log2(contingency.sum(1) + 1e-12))
    h_pred = -np.sum(contingency.sum(0) * np.log2(contingency.sum(0) + 1e-12))
    mi = mutual_info_score(gold_clusters, pred_clusters)

    return h_gold + h_pred - 2 * mi


def cluster_purity(gold_clusters, pred_clusters):
    contingency = contingency_matrix(gold_clusters, pred_clusters)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


np.random.seed(42)
n_samples = 501
n_true_clusters = 72

true_labels = np.repeat(np.arange(n_true_clusters), n_samples // n_true_clusters)
true_labels = np.concatenate([true_labels, [n_true_clusters - 1] * (n_samples - len(true_labels))])
np.random.shuffle(true_labels)


def generate_embeddings(labels, quality):
    np.random.seed(42)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    embeddings = np.zeros((len(labels), 2))

    spacing = 5 + 20 * quality ** 1.5
    noise_std = 2.5 * (1 - quality) ** 2.5
    grid_size = int(np.sqrt(n_clusters) * (0.5 + 2 * quality))

    centers = {}
    for idx, cluster in enumerate(unique_labels):
        if quality < 0.1:
            centers[cluster] = np.random.uniform(-50, 50, 2)
        else:
            row = idx // grid_size
            col = idx % grid_size
            base = np.array([row * spacing, col * spacing])
            offset = np.random.normal(0, spacing * (1 - quality) / 2, 2)
            centers[cluster] = base + offset

    for i, cluster in enumerate(labels):
        noise = np.random.normal(0, noise_std, 2)
        embeddings[i] = centers[cluster] + noise

    return embeddings


qualities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
embeddings = {q: generate_embeddings(true_labels, q) for q in qualities}

plt.figure(figsize=(15, 8))
for i, (q, emb) in enumerate(embeddings.items(), 1):
    plt.subplot(2, 3, i)
    plt.scatter(emb[:, 0], emb[:, 1], c=true_labels, s=8, alpha=0.5, cmap='tab20')
    plt.title(f"Quality={q}", fontsize=9)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

# n_clusters_list = [20, 40, 60, 80, 100]
n_clusters_list = [25, 50, 75, 100]
metrics = {q: {'VI': [], 'Purity': []} for q in qualities}

for q in qualities:
    X = embeddings[q]
    for n_clusters in n_clusters_list:
        pred = KMeans(n_clusters, n_init=10, random_state=42).fit_predict(X)
        metrics[q]['VI'].append(variation_of_info(true_labels, pred))
        metrics[q]['Purity'].append(cluster_purity(true_labels, pred))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for q in qualities:
    ax1.plot(n_clusters_list, metrics[q]['VI'], marker='o', label=f'q={q}')
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("VI")
ax1.set_title("Variation of Information")
ax1.grid(True, alpha=0.3)

for q in qualities:
    ax2.plot(n_clusters_list, metrics[q]['Purity'], marker='o', label=f'q={q}')
ax2.set_xlabel("Number of Clusters")
ax2.set_ylabel("Purity")
ax2.set_title("Cluster Purity")
ax2.grid(True, alpha=0.3)

plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
plt.tight_layout()
plt.show()
