import os
import subprocess
import tempfile
import json
import csv
import time
from tqdm import tqdm

CORENLP_DIR = "../stanford-corenlp-4.5.9"
OUTPUT_DIR = "./xml_output"
JAVA_MEMORY = "4g"
MAX_WAIT_SECONDS = 30

def wait_for_file(path, max_wait):
    start_time = time.time()
    while not os.path.exists(path):
        if time.time() - start_time > max_wait:
            return False
        time.sleep(0.5)
    return True

os.makedirs(OUTPUT_DIR, exist_ok=True)

base_command = (
    f"java -Xmx{JAVA_MEMORY} -cp '{CORENLP_DIR}/*' " 
    "edu.stanford.nlp.pipeline.StanfordCoreNLP "
    "-annotators tokenize,ssplit,pos,lemma,ner,parse,coref "
    "-coref.algorithm neural "
    "-outputFormat xml "
    f"-outputDirectory {OUTPUT_DIR} "
)


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
            print(f"Invalid line: {line}")
            continue
        category_name, char_info_str = parts
        char_info = json.loads(char_info_str)

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

print(f"Total categories: {len(all_categories)}")

all_character_entries = []

for category_name, char_list in category_to_characters.items():
    for char_info in char_list:
        all_character_entries.append((category_name, char_info))

for (category_name, char_info) in tqdm(all_character_entries, desc="All Characters"):
    single_start_time = time.time()
    f_map_id = char_info["id"]
    movie_title = char_info["movie"]
    char_name = char_info["char"]

    if f_map_id not in map_id_to_char_data:
        print(f"Character {char_name} from movie {movie_title} not found in metadata (map_id).")
        continue

    w_movie_id, f_movie_id, character_name_in_meta = map_id_to_char_data[f_map_id]

    if summary_key_version == 2:
        summary_key = w_movie_id
    else:
        summary_key = (w_movie_id, char_name.lower())

    summary = movie_summaries.get(summary_key, "")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        tmp.write(summary)
        tmp_path = os.path.abspath(tmp.name)
        tmp_basename = os.path.basename(tmp_path)

    orig_xml_name = tmp_basename+ ".xml"
    orig_path = os.path.join(OUTPUT_DIR, orig_xml_name)
    new_path = os.path.join(OUTPUT_DIR, f"{w_movie_id}.xml")

    for path in [orig_path, new_path]:
        if os.path.exists(path):
            os.remove(path)

    command = f"{base_command} -file '{tmp_path}' -outputExtension .xml"

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"CoreNLP output：\n{result.stdout}")
        if result.stderr:
            print(f"CoreNLP error：\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"failed：{e}\n error output：{e.stderr}")
        os.remove(tmp_path)
        continue

    if wait_for_file(orig_path, MAX_WAIT_SECONDS):
        os.rename(orig_path, new_path)
        print(f"generated：{new_path}")
    else:
        print(f"no output file {orig_path}")
        possible_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".xml")]
        print(f"current XMLs：{possible_files}")

    os.remove(tmp_path)
    single_end_time = time.time()
    print(f"{w_movie_id} cost：{single_end_time - single_start_time:.2f}s")

print("Finished.")

