import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict

Record = Dict[str, str]


def read_records(input_file) -> List[Record]:
    records: List[Record] = []
    category_counts: Counter[str] = Counter()

    with open(input_file, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            try:
                category, obj = line.split("\t", 1)
            except ValueError:
                category, obj = line.split(maxsplit=1)

            data = json.loads(obj)
            records.append({"line": line, "category": category, "id": data["id"]})
            category_counts[category] += 1

    return records, category_counts


def choose_categories(records: List[Record], category_counts: Counter) -> Dict[str, str]:
    id_to_category: Dict[str, str] = {}

    for rec in records:
        cid, cat = rec["id"], rec["category"]

        if cid not in id_to_category:
            id_to_category[cid] = cat
            continue

        current = id_to_category[cid]
        if category_counts[cat] > category_counts[current]:
            id_to_category[cid] = cat

    return id_to_category


def write_filtered(records: List[Record], keep_map: Dict[str, str], output_file) -> int:
    kept = [rec["line"] for rec in records if rec["category"] == keep_map[rec["id"]]]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(kept))
        f.write("\n")

    return len(kept)


def main() -> None:
    input_file = '../data/tvtropes.clusters.txt'
    output_file = '../data/tvtropes.clusters.cleaned.txt'

    records, category_counts = read_records(input_file)
    id_preference = choose_categories(records, category_counts)
    kept = write_filtered(records, id_preference, output_file)

    print(f"Kept {kept:,} of {len(records):,} lines → {output_file}")


if __name__ == "__main__":
    main()
