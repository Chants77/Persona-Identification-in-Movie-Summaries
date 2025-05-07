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


def logprint(log):
    log_file = os.path.join("logs/llama-{}.logs".format(time.strftime('%Y%m%d', time.gmtime())))
    with open(log_file, "a", encoding="utf-8") as fout:
        fout.write(log + "\n")
    print(log)

def parse_response(response_text):
    try:
        data = json.loads(response_text)
        return data
    except json.JSONDecodeError:
        print("Failed to parse JSON response.")
        return None


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

overall_start_time = time.time()
logprint("Start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time)))

tvtropes_file = "../data/tvtropes.clusters.txt"
category_to_characters = {}
all_categories = set()

if torch.cuda.is_available():
    logprint("CUDA is available. GPU will be used if there's enough memory.")
else:
    logprint("CUDA not available. The code will fall back to CPU or partial CPU/GPU usage.")


with open(tvtropes_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t', 1)
        if len(parts) != 2:
            logprint(f"Invalid line: {line}")
            continue
        category_name, char_info_str = parts
        char_info = json.loads(char_info_str)
        logprint(f"Category: {category_name}, Character: {char_info['char']}, Movie: {char_info['movie']}")

        if category_name not in category_to_characters:
            category_to_characters[category_name] = []
        category_to_characters[category_name].append(char_info)
        all_categories.add(category_name)

all_categories = sorted(all_categories)
logprint(f"Loaded {len(all_categories)} categories")

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
summary_key_version = 0

with open(plot_summaries_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) == 2:
            w_movie_id, summary = row
            movie_summaries[w_movie_id] = summary
            summary_key_version = 2
        elif len(row) == 3:
            w_movie_id, c_name, short_summary = row
            movie_summaries[(w_movie_id, c_name.lower())] = short_summary
            summary_key_version = 3


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

llama_pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    temperature=0.0,
    do_sample=False,
    top_p=1.0
)
logprint("Pipeline loaded.")

logprint(f"Total categories: {len(all_categories)}")
categories_context_str = "The possible categories are: " + ", ".join(all_categories) + ". "

all_character_entries = []
for category_name, char_list in category_to_characters.items():
    for char_info in char_list:
        all_character_entries.append((category_name, char_info))

# max_context = 0
# min_context = 2000

y_true = []
y_pred = []
results_list = []
invalid_count = 0

# correct = 0
for (category_name, char_info) in tqdm(all_character_entries, desc="All Characters"):
        single_start_time = time.time()
        f_map_id = char_info["id"]
        movie_title = char_info["movie"]
        char_name = char_info["char"]

        if f_map_id not in map_id_to_char_data:
            logprint(f"Character {char_name} from movie {movie_title} not found in metadata (map_id).")
            continue

        w_movie_id, f_movie_id, character_name_in_meta = map_id_to_char_data[f_map_id]

        if summary_key_version == 2:
            summary_key = w_movie_id
        else:
            summary_key = (w_movie_id, char_name.lower())

        summary = movie_summaries.get(summary_key, "")
        if not summary.strip():
            logprint(f"No summary found for key: {summary_key}")
            continue

        context = categories_context_str + summary

        question = f"Which category best describes the character {char_name} from the movie {movie_title}? Give me only the category."

        messages = [
            {"role": "system", "content": "You are a bot that responds to film character classification queries."},
            {"role": "user", "content": question + "\n\n" + context},
        ]

        max_attempts = 3
        attempt = 0
        predicted_cat = "No valid category"

        while predicted_cat == "No valid category" and attempt < max_attempts:
            attempt += 1
            outputs = llama_pipeline(messages, max_new_tokens=256)
            generated_text = outputs[0]["generated_text"][-1]

            generated_cat = generated_text['content']

            predicted_cat = "No valid category"
            for cat in all_categories:
                if cat in generated_cat:
                    predicted_cat = cat
                    break

            if predicted_cat == "No valid category":
                logprint(f"(Attempt {attempt}) generated_cat: {generated_cat}")
                messages.append({"role": "user", "content": generated_cat + " is not valid."})
                logprint(json.dumps(messages, indent=2))

        if predicted_cat == "No valid category":
            invalid_count += 1

        y_true.append(category_name)
        y_pred.append(predicted_cat)
        # if category_name == predicted_cat:
        #     correct += 1
        single_end_time = time.time()

        # Logging
        logprint(f"Time taken for character {char_name} from movie {movie_title}: {single_end_time - single_start_time:.2f} seconds")
        # logprint(f"Character: {char_name} (Movie: {movie_title})")
        # logprint(f"Q: {question}")
        # logprint(f"response: {generated_text}")
        # logprint(f"Model Output: {generated_cat}")
        logprint(f"Predicted Category: {predicted_cat}")
        logprint(f"True Category: {category_name}")
        # logprint('Current correct: ', correct)
        logprint("-------------------------------------------------------")
        results_list.append([f_map_id, char_name, movie_title, category_name, predicted_cat])

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)


logprint("Invalid count: " + str(invalid_count))
logprint("Accuracy: " + str(accuracy))
logprint("Precision: " + str(precision))
logprint("Recall: " + str(recall))
logprint("F1-score: " + str(f1))

category_counts = defaultdict(lambda: {"correct": 0, "total": 0})

for true_cat, pred_cat in zip(y_true, y_pred):
    category_counts[true_cat]["total"] += 1
    if pred_cat == true_cat:
        category_counts[true_cat]["correct"] += 1

logprint("Per-Category Accuracy:")
for cat in all_categories:
    total = category_counts[cat]["total"]
    correct = category_counts[cat]["correct"]
    if total > 0:
        cat_acc = correct / total
        logprint(f"  {cat} -> {cat_acc:.3f} ({correct}/{total})")
    else:
        logprint(f"  {cat} -> No samples in ground truth")

csv_filename = os.path.join("results/results-{}.csv".format(time.strftime('%Y%m%d', time.gmtime())))
with open(csv_filename, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Character", "Movie", "TrueCategory", "PredictedCategory", "Reason"])
    for row in results_list:
        writer.writerow(row)

logprint(f"Results saved to {csv_filename}")

overall_end_time = time.time()
logprint("End time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end_time)))
logprint(f"Total runtime: {overall_end_time - overall_start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files")
    args, leftover = parser.parse_known_args()

    if args.outputs:
        for fname in args.outputs:
            with open(fname, "wt", encoding="utf-8") as ofile:
                ofile.write("LLaMa pipeline finished.\n")


