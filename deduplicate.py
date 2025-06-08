import json
import sys
in_path = 'results/bin/sota_llama_10words_20250426.jsonl'
out_path = 'results/bin/sota_llama_10words_20250426_dedup.jsonl'

seen_ids = set()
kept, skipped = 0, 0

with open(in_path, "r", encoding="utf-8") as fin, \
     open(out_path, "w", encoding="utf-8") as fout:

    for line in fin:
        record = json.loads(line)

        cid = record.get("character_id")
        if cid is None:
            print("record without 'character_id' key, skipping",
                  file=sys.stderr)
            skipped += 1
            continue

        if cid in seen_ids:
            skipped += 1
            continue

        seen_ids.add(cid)
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        kept += 1

print(f"Kept {kept:,} records, skipped {skipped:,} duplicates.",
      file=sys.stderr)
