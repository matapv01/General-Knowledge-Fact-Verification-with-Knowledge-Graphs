import json

input_file = '/home/vnson/vnson/FactKG/SimGRAG/results/FactKG_query_qwen_122b.txt'
output_file = '/home/vnson/vnson/FactKG/SimGRAG/results/FactKG_query_qwen_122b_cleaned.txt'

lines_to_keep = []
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            try:
                d = json.loads(line)
                if 'error_message' not in d:
                    lines_to_keep.append(line)
            except:
                pass

    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(lines_to_keep)

    print(f"\n=> CLEANUP SUCCESS: Kept {len(lines_to_keep)} correct lines, deleted all errors!")
except Exception as e:
    print(f"\n=> ERROR: {e}")
