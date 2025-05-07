import sys
import csv
import json
import re
import spacy
import coreferee


def main():
    tvtropes_file = '../data/tvtropes.clusters.txt'
    char_metadata_file = '../data/character.metadata.tsv'
    plot_summaries_file = '../data/plot_summaries.txt'
    output_file = '../data/shortened_summaries.txt'

    tvtropes_data = []
    with open(tvtropes_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            category, char_info_str = parts
            try:
                char_info = json.loads(char_info_str)
                tvtropes_data.append({
                    "category": category,
                    "char_name": char_info.get("char", ""),
                    "movie_name": char_info.get("movie", ""),
                    "actor_name": char_info.get("actor", ""),
                    "freebase_map_id": char_info.get("id", ""),
                })
            except Exception as e:
                print(f"[Warning] Could not parse JSON on line: {line}\n  Error: {e}")
                continue
    print(f"Loaded {len(tvtropes_data)} entries from {tvtropes_file}.")

    id_to_char_data = {}
    map_id_to_char_data = {}
    with open(char_metadata_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            w_movie_id = row[0]
            f_movie_id = row[1]
            freebase_char_id = row[11]
            character_name = row[3]
            map_id = row[10]
            id_to_char_data[freebase_char_id] = (w_movie_id, f_movie_id, character_name)
            map_id_to_char_data[map_id] = (w_movie_id, f_movie_id, character_name)

    movie_summaries = {}
    with open(plot_summaries_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) == 2:
                w_movie_id, summary = row
                movie_summaries[w_movie_id] = summary

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("coreferee")

    count_found = 0
    total = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in tvtropes_data:
            total += 1
            freebase_map_id = entry["freebase_map_id"]
            if freebase_map_id not in map_id_to_char_data:
                continue

            w_movie_id, f_movie_id, meta_char_name = map_id_to_char_data[freebase_map_id]
            if w_movie_id not in movie_summaries:
                continue

            summary = movie_summaries[w_movie_id].strip()
            if not summary:
                continue

            character_name = meta_char_name if meta_char_name else entry["char_name"]

            shortened_summary = shorten_context_with_coref(nlp, summary, character_name)
            if not shortened_summary:
                shortened_summary = summary

            fout.write(f"{w_movie_id}\t{character_name}\t{shortened_summary}\n")
            count_found += 1

    print(f"Processed {total} entries from {tvtropes_file}.")
    print(f"Wrote {count_found} lines to {output_file}.")


def shorten_context_with_coref(nlp, summary_text, character_name):
    doc = nlp(summary_text)
    mentions_set = get_character_mentions(doc, summary_text, character_name)
    if not mentions_set:
        return []

    relevant_sents = []
    for sent in doc.sents:
        sent_token_range = set(range(sent.start, sent.end))

        overlap = mentions_set.intersection(sent_token_range)
        if overlap:
            relevant_sents.append(sent.text)

    shortened = " ".join(relevant_sents).strip()
    print(f"Shortened summary for {character_name}: {shortened}")
    return shortened


def get_character_mentions(doc, text, character_name):
    target_name_lower = character_name.lower()
    token_index_set = set()

    for chain in doc._.coref_chains.chains:
        if any(target_name_lower in str(mention).lower() for mention in chain.mentions):
            for mention in chain.mentions:
                token_index_set.update(mention.token_indexes)

    return token_index_set


if __name__ == "__main__":
    main()
