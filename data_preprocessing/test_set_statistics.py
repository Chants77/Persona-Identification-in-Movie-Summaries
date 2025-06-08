import csv
import json
from transformers import pipeline
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def test_set_statistics():
    tvtropes_file = "../data/tvtropes.clusters.cleaned.txt"
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

    char_metadata_file = "../data/character.metadata.tsv"
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

    plot_summaries_file = "../data/plot_summaries.txt"
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
    df.to_csv("../results/token_counts.csv", index=True)
    print("\nSaved token counts to token_counts.csv")

    print(f"\nRoberta - Longest: {max_roberta[1]} tokens (doc: {max_roberta[0]})")
    print(f"Roberta - Shortest: {min_roberta[1]} tokens (doc: {min_roberta[0]})")
    print(f"Llama - Longest: {max_llama[1]} tokens (doc: {max_llama[0]})")
    print(f"Llama - Shortest: {min_llama[1]} tokens (doc: {min_llama[0]})")

    return df

result_df = test_set_statistics()