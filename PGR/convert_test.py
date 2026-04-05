import pickle
import json
import os

input_path = '../data/factkg_test.pickle'
output_path = '../data/factkg_test_pgr.jsonl'

print(f"Reading {input_path}...")
with open(input_path, 'rb') as f:
    data = pickle.load(f)

print(f"Converting to {output_path}...")
with open(output_path, 'w', encoding='utf-8') as f:
    for i, (claim, meta) in enumerate(data.items()):
        # PGR format requires json lines with question_id, question, entity_set, Label
        obj = {
            "question_id": i + 1,
            "question": claim,
            "entity_set": meta.get('Entity_set', []),
            "Label": meta.get('Label', [])
        }
        f.write(json.dumps(obj) + "\n")

print("Conversion complete!")
