import csv
import json
from transformers import pipeline
import torch
import random
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def test_set_statistics():
    tvtropes_file = "data/tvtropes.clusters.txt"
    category_to_characters = {}
    all_categories = set()

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
            print(f"Category: {category_name}, Character: {char_info['char']}, Movie: {char_info['movie']}")

            if category_name not in category_to_characters:
                category_to_characters[category_name] = []
            category_to_characters[category_name].append(char_info)
            all_categories.add(category_name)

    all_categories = sorted(all_categories)

    char_metadata_file = "data/character.metadata.tsv"
    id_to_char_data = {}
    map_id_to_char_data = {}

    with open(char_metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            w_movie_id = row[0]
            f_movie_id = row[1]
            map_id = row[10]
            freebase_char_id = row[11]
            character_name = row[3]
            id_to_char_data[freebase_char_id] = (w_movie_id, f_movie_id, character_name)
            map_id_to_char_data[map_id] = (w_movie_id, f_movie_id, character_name)

    plot_summaries_file = "data/plot_summaries.txt"
    movie_summaries = {}

    with open(plot_summaries_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            w_movie_id, summary = row
            movie_summaries[w_movie_id] = summary

    all_character_entries = []
    for category_name, char_list in category_to_characters.items():
        for char_info in char_list:
            all_character_entries.append((category_name, char_info))

    results = []
    max_roberta = (None, 0)
    min_roberta = (None, float('inf'))
    max_llama = (None, 0)
    min_llama = (None, float('inf'))
    print("Max length for Roberta:", max_roberta[1])
    print("\nInitializing pipelines...")

    roberta_qa = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2"
    )
    roberta_tokenizer = roberta_qa.tokenizer

    llama_gen = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        temperature=0.0,
        do_sample=False,
        top_p=1.0
    )
    llama_tokenizer = llama_gen.tokenizer

    print("\nProcessing summaries:")
    for (category_name, char_info) in tqdm(all_character_entries, desc="All Characters"):
        single_start_time = time.time()
        f_map_id = char_info["id"]
        movie_title = char_info["movie"]
        char_name = char_info["char"]

        if f_map_id not in map_id_to_char_data:
            print(f"Character {char_name} from movie {movie_title} not found in metadata (map_id).")
            continue

        w_movie_id, f_movie_id, character_name_in_meta = map_id_to_char_data[f_map_id]

        summary_key = w_movie_id

        summary = movie_summaries.get(summary_key, "")

        roberta_tokens = roberta_tokenizer(
            summary,
            add_special_tokens=False,
            return_attention_mask=False,
            return_length=True
        )["input_ids"].shape[1]

        llama_tokens = llama_tokenizer(
            summary,
            add_special_tokens=False,
            return_attention_mask=False,
            return_length=True
        ).input_ids.shape[1]

        results.append({
            "summary key": summary_key,
            "roberta": roberta_tokens,
            "llama": llama_tokens
        })

        if roberta_tokens > max_roberta[1]:
            max_roberta = (summary_key, roberta_tokens)
        if roberta_tokens < min_roberta[1]:
            min_roberta = (summary_key, roberta_tokens)
        if llama_tokens > max_llama[1]:
            max_llama = (summary_key, llama_tokens)
        if llama_tokens < min_llama[1]:
            min_llama = (summary_key, llama_tokens)

    df = pd.DataFrame(results)
    df.to_csv("results/token_counts.csv", index=True)
    print("\nSaved token counts to token_counts.csv")

    print(f"\nRoberta - Longest: {max_roberta[1]} tokens (doc: {max_roberta[0]})")
    print(f"Roberta - Shortest: {min_roberta[1]} tokens (doc: {min_roberta[0]})")
    print(f"Llama - Longest: {max_llama[1]} tokens (doc: {max_llama[0]})")
    print(f"Llama - Shortest: {min_llama[1]} tokens (doc: {min_llama[0]})")

    return df


def plot_results(df):
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df['roberta'], color="blue", label='Roberta', kde=True, bins=30)
    sns.histplot(df['llama'], color="red", label='Llama', kde=True, bins=30, alpha=0.5)
    plt.title("Token Length Distribution")
    plt.xlabel("Token Count")
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df[['roberta', 'llama']], palette="Set2")
    plt.xticks([0, 1], ['Roberta', 'Llama'])
    plt.title("Token Length Comparison")
    plt.ylabel("Token Count")

    plt.tight_layout()
    plt.savefig("token_comparison.png")
    plt.show()


if __name__ == "__main__":
    # result_df = test_set_statistics()
    # plot_results(result_df)

    sns.set(style="whitegrid", palette="pastel")
    # plt.rcParams['font.family'] = 'DejaVu Sans'

    df = pd.read_csv("results/token_counts.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    bins = list(range(0, 2700, 100))
    colors = {'roberta': '#4C72B0', 'llama': '#DD8452'}

    sns.histplot(df['roberta'],
                 bins=bins,
                 color=colors['roberta'],
                 alpha=0.8,
                 label='RoBERTa',
                 edgecolor='white',
                 linewidth=0.5,
                 ax=ax1)

    sns.histplot(df['llama'],
                 bins=bins,
                 color=colors['llama'],
                 alpha=0.8,
                 label='Llama',
                 edgecolor='white',
                 linewidth=0.5,
                 ax=ax1)

    stats_text = f'''
    RoBERTa:
    mean = {df.roberta.mean():.1f}
    median = {df.roberta.median():.1f}

    Llama:
    mean = {df.llama.mean():.1f}
    median = {df.llama.median():.1f}
    '''

    ax1.text(0.95, 0.95, stats_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))

    ax1.set_title("Token distribution", fontsize=14, pad=20)
    ax1.set_xlabel("Token amount", labelpad=10)
    ax1.set_ylabel("doc amount", labelpad=10)
    ax1.legend()

    scatter = sns.scatterplot(
        x='roberta',
        y='llama',
        data=df,
        alpha=0.6,
        color='#55A868',
        edgecolor='white',
        linewidth=0.3,
        ax=ax2
    )

    max_val = max(df[['roberta', 'llama']].max())
    sns.regplot(x='roberta', y='llama', data=df,
                scatter=False,
                color='#C44E52',
                line_kws={'linestyle': '--', 'alpha': 0.7},
                ax=ax2)
    ax2.plot([0, max_val], [0, max_val],
             color='#4C72B0',
             linestyle=':',
             linewidth=1.5,
             label='y = x')

    corr = df[['roberta', 'llama']].corr().iloc[0, 1]
    ax2.text(0.05, 0.95, f'Pearson r = {corr:.2f}',
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    ax2.set_title("Token", fontsize=14, pad=20)
    ax2.set_xlabel("RoBERTa Token Amount", labelpad=10)
    ax2.set_ylabel("Llama Token Amount", labelpad=10)
    ax2.legend()

    plt.savefig('token_analysis.png', dpi=300, bbox_inches='tight')

