import json
from collections import defaultdict


def read_character_metadata(character_metadata_path):
    metadata = defaultdict(dict)
    with open(character_metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 13:
                continue

            char_id = parts[10].lower()
            gender = parts[5]
            metadata[char_id]['gender'] = gender

    return metadata


def process_data(data_path, character_metadata):
    entities = defaultdict(lambda: {'agent': [], 'patient': [], 'attribute': []})

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue

            agent_line = parts[1]
            patient_line = parts[2]
            modifier_line = parts[3]

            for word in agent_line.split():
                elements = word.split(':')
                if len(elements) < 6:
                    continue
                entity_id = elements[0].lower()
                verb = elements[3]
                entities[entity_id]['agent'].append(verb)

            for word in patient_line.split():
                elements = word.split(':')
                if len(elements) < 6:
                    continue
                entity_id = elements[0].lower()
                verb = elements[3]
                entities[entity_id]['patient'].append(verb)

            for word in modifier_line.split():
                elements = word.split(':')
                if len(elements) < 5:
                    continue
                entity_id = elements[0].lower()
                attribute = elements[3]
                entities[entity_id]['attribute'].append(attribute)

    for entity_id in entities:
        if entity_id in character_metadata:
            meta = character_metadata[entity_id]
            if meta['gender'] in ('M', 'F'):
                entities[entity_id]['attribute'].append(f"gender:{meta['gender']}")
            # if meta['age_bucket'] != -1:
            #     entities[entity_id]['attribute'].append(f"age:{meta['age_bucket']}")

    for entity_id in entities:
        for key in ['agent', 'patient', 'attribute']:
            entities[entity_id][key] = list(set(entities[entity_id][key]))

    return dict(entities)


def main(character_meta_path, data_path, output_path):
    character_metadata = read_character_metadata(character_meta_path)
    result = process_data(data_path, character_metadata)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    character_metadata = "data/character.metadata.tsv"
    data_file = "../../persona/data/all.data"
    output_file = "word_bag.json"

    main(character_metadata, data_file, output_file)
