import csv
import json
from transformers import pipeline
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


tvtropes_file = "../data/tvtropes.clusters.txt"
category_to_characters = {}
all_categories = set()

def logprint(log):
    log_file = os.path.join("./logs/roberta-{}.logs".format(time.strftime('%Y%m%d', time.gmtime())))
    with open(log_file, "a") as fout:
        fout.write(log + "\n")
    print(log)


with open(tvtropes_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t', 1)
        if len(parts) != 2:
            print(f"Invalid line: {line}")
            continue
        category_name, char_info_str = parts
        char_info = json.loads(char_info_str)
        print(f"Category: {category_name}, Character: {char_info['char']}, Movie: {char_info['movie']}")

        if category_name not in category_to_characters:
            category_to_characters[category_name] = []
        category_to_characters[category_name].append(char_info)
        all_categories.add(category_name)

all_categories = sorted(all_categories)  # sort for consistency
print(f"Loaded {len(all_categories)} categories")

char_metadata_file = "../data/character.metadata.tsv"
id_to_char_data = {}
map_id_to_char_data = {}

with open(char_metadata_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        w_movie_id = row[0]
        f_movie_id = row[1]
        freebase_char_id = row[11]
        character_name = row[3]
        map_id = row[10]
        id_to_char_data[freebase_char_id] = (w_movie_id, f_movie_id, character_name)
        map_id_to_char_data[map_id] = (w_movie_id, f_movie_id, character_name)

# plot_summaries_file = "data/plot_summaries.txt"
plot_summaries_file = "../data/shortened_summaries.txt"
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

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)
qa_pipeline_tokenizer = qa_pipeline.tokenizer

print("Pipeline tokenizer max length:", qa_pipeline_tokenizer.model_max_length)

categories_context_str = "The possible categories are: " + ", ".join(all_categories) + ". "

y_true = []
y_pred = []

max_context = 0
min_context = 2000

for category_name, char_list in category_to_characters.items():
    for char_info in char_list:
        f_map_id = char_info["id"]
        movie_title = char_info["movie"]
        char_name = char_info["char"]

        if f_map_id not in map_id_to_char_data:
            print(f"Character {char_name} from movie {movie_title} not found in metadata.")
            continue

        w_movie_id, f_movie_id, character_name_in_meta = map_id_to_char_data[f_map_id]

        if summary_key_version == 2:
            summary_key = w_movie_id
        elif summary_key_version == 3:
            summary_key = (w_movie_id, char_name.lower())
        summary = movie_summaries.get(summary_key, "")
        if not summary.strip():
            continue

        question = f"Which category best describes the character {char_name} from the movie {movie_title}?"
        context = categories_context_str + summary
        # print(f"context: {context}")

        try:
            encoded_context = qa_pipeline_tokenizer(
                context,
                truncation=False,
                max_length=qa_pipeline_tokenizer.model_max_length,
                return_tensors="pt"
            )
            print("Number of tokens fed into pipeline:", encoded_context["input_ids"].shape[1])

            if encoded_context["input_ids"].shape[1] > max_context:
                max_context = encoded_context["input_ids"].shape[1]
            if encoded_context["input_ids"].shape[1] < min_context:
                min_context = encoded_context["input_ids"].shape[1]

            encoded_categories_context = qa_pipeline_tokenizer(
                categories_context_str,
                truncation=False,
                max_length=qa_pipeline_tokenizer.model_max_length,
                return_tensors="pt"
            )
            print("Number of category tokens:", encoded_categories_context["input_ids"].shape[1])

            encoded_question = qa_pipeline_tokenizer(
                question,
                truncation=False,
                max_length=qa_pipeline_tokenizer.model_max_length,
                return_tensors="pt"
            )
            print("Number of question tokens:", encoded_question["input_ids"].shape[1])

            result = qa_pipeline(question=question, context=context)
            answer = result.get('answer', 'No answer found')

            print(f"Character: {char_name} (from {movie_title})")
            print(f"Q: {question}")
            print(f"A: {answer}")
            print(f"True Category: {category_name}")
            print("-------------------------------------------------------")
            y_true.append(category_name)
            y_pred.append(answer)

        except Exception as e:
            print(f"Error processing character {char_name} from movie {movie_title}: {e}")
            continue


print(f"Max context length: {max_context}")
print(f"Min context length: {min_context}")

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
