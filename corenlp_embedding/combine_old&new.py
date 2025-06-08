import json

with open('new_neural_filtered_word_bag.json', 'r') as f:
    a_data = json.load(f)

with open('old_corenlp_filtered_word_bag.json', 'r') as f:
    b_data = json.load(f)

merged_data = a_data.copy()
for key, value in b_data.items():
    if key not in merged_data:
        merged_data[key] = value

with open('combined_old&new.json', 'w') as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print("saved to combined_old&new.json")
