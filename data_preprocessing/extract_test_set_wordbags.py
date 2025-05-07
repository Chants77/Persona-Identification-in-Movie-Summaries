import json
from collections import Counter

word_bag_path = "../word_bag.json"
tvtropes_file = "../data/tvtropes.clusters.cleaned.txt"
output_path = "../filtered_word_bag.json"
missing_ids_output_path = "missing_ids.txt"

with open(word_bag_path, 'r', encoding='utf-8') as f:
    all_entities = json.load(f)

target_ids = []
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
        # print(f"Category: {category_name}, Character: {char_info['char']}, Movie: {char_info['movie']}, ID: {char_info['id']}")
        target_ids.append(char_info['id'].lower())

element_counts = Counter(target_ids)
duplicates = [item for item, count in element_counts.items() if count > 1]
print(f'duplicate ids: {len(duplicates)}')
print(duplicates)

filtered_entities = {}
missing_ids = []

for entity_id in target_ids:
    if entity_id in all_entities:
        filtered_entities[entity_id] = all_entities[entity_id]
    else:
        missing_ids.append(entity_id)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(filtered_entities, f, ensure_ascii=False, indent=2)

print(f"saved {len(filtered_entities)}/{len(target_ids)} characters in {output_path}")
print(f"didn't find：{len(missing_ids)}")

if missing_ids:
    print("\ndidn't find：")
    for idx, missing_id in enumerate(missing_ids, 1):
        print(f"{idx}. {missing_id}")

    if missing_ids_output_path:
        with open(missing_ids_output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(missing_ids))
        print(f"\nsave missing ids in {missing_ids_output_path}")

