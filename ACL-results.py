# save as filter_personas.py
import csv, sys, json
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


posterior_file = "ACL-results/reg50.100.lda.log.txt"        # path to the big file

target_ids = set()  # set to hold target character IDs

tvtropes_file = "data/tvtropes.clusters.cleaned.txt"

gold = {}

with open(tvtropes_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t', 1)
        if len(parts) != 2:
            continue
        category_name, char_info_str = parts
        char_info = json.loads(char_info_str)
        # print(f"Category: {category_name}, Character: {char_info['char']}, Movie: {char_info['movie']}")

        target_ids.add(char_info['id'])  # add character ID to the set
        gold[char_info['id']] = category_name
        print(f"Added character ID: {char_info['char']} from category: {category_name}")

pred = {}

with open(posterior_file, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        if not row:            # skip blank lines
            print("Skipping empty line")
            continue
        char_id = row[0]
        # print(f"Processing character ID: {char_id}")
        if char_id in target_ids:
            persona_mode = row[6]          # column 5 (0-based) = cluster id
            # You can also grab the full 50-dimensional posterior:
            posterior = row[7:57]
            pred[char_id] = persona_mode
            print(f"{char_id}\t{persona_mode}\t{row[2]}")  # id, persona, movie

def variation_of_info(gold_clusters, pred_clusters):
    n = len(gold_clusters)
    contingency = np.zeros((len(set(gold_clusters)), len(set(pred_clusters))))
    unique_gold, gold_ids = np.unique(gold_clusters, return_inverse=True)
    unique_pred, pred_ids = np.unique(pred_clusters, return_inverse=True)

    for i in range(n):
        print(f"Processing pair: {gold_ids[i]} -> {pred_ids[i]}")
        contingency[gold_ids[i], pred_ids[i]] += 1
    contingency = contingency / n

    h_gold = -np.sum(contingency.sum(1) * np.log2(contingency.sum(1) + 1e-12))
    h_pred = -np.sum(contingency.sum(0) * np.log2(contingency.sum(0) + 1e-12))
    mi = mutual_info_score(gold_clusters, pred_clusters)

    return h_gold + h_pred - 2 * mi


def cluster_purity(gold_clusters, pred_clusters):
    from sklearn.metrics.cluster import contingency_matrix
    contingency = contingency_matrix(gold_clusters, pred_clusters)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


gold_info = []
pred_info = []

for char_id in pred.keys():
    if char_id not in gold:
        print(f"Warning: Character ID {char_id} in predictions but not in gold data.")
    gold_info.append(gold[char_id])
    pred_info.append(pred[char_id])

gold_clusters = np.array(gold_info)
pred_clusters = np.array(pred_info)

cluster_to_true_label = {}
for cluster_id in np.unique(pred_clusters):
    # 1. 获取当前聚类中的所有样本
    mask = (pred_clusters == cluster_id)
    cluster_samples = gold_clusters[mask]

    # 2. 找出聚类中最常见的真实标签
    unique_labels, counts = np.unique(cluster_samples, return_counts=True)
    true_label = unique_labels[np.argmax(counts)]

    # 3. 为该聚类分配这个真实标签
    cluster_to_true_label[cluster_id] = true_label

# 4. 创建预测标签数组
assigned_labels = np.array([cluster_to_true_label[c] for c in pred_clusters])

# 5. 计算分类指标
accuracy = accuracy_score(gold_clusters, assigned_labels)
precision = precision_score(gold_clusters, assigned_labels, average='macro', zero_division=0)
recall = recall_score(gold_clusters, assigned_labels, average='macro', zero_division=0)
f1 = f1_score(gold_clusters, assigned_labels, average='macro', zero_division=0)

print(f"VI (bits): {variation_of_info(gold_clusters, pred_clusters):.4f}")
print(f"Purity   : {cluster_purity(gold_clusters, pred_clusters):.4f}")
print(f'  Accuracy={accuracy * 100:.1f}% | Precision={precision * 100:.1f}% | '
         f'Recall={recall * 100:.1f}% | F1={f1 * 100:.1f}%')