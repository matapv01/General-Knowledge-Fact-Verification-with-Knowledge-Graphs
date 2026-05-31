import pickle
import os
import ast
from kg import KG
import numpy as np
import pandas as pd
import glob
from argparse import ArgumentParser
from thefuzz import process

from multiprocessing import Pool
from functools import partial



import re
import json
import ast
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from thefuzz import process
stop_words = set(stopwords.words('english'))


# Ensure you have the necessary NLTK data files
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# tmp_dict={}
tmp_dict = pickle.load(open('tmp_dict.pickle', 'rb'))

parser = ArgumentParser()
parser.add_argument("--data_path", default="/global/D1/projects/HOST/Datasets/factKG_ifi/full/")
parser.add_argument("--dbpedia_path",default="/global/D1/projects/HOST/Datasets/factKG_ifi/dbpedia/dbpedia_2015_undirected.pickle")
parser.add_argument("--set", choices=["test", "train", "val"], default="train")
parser.add_argument("--outputPath", default="./llm_all_one_hop")
parser.add_argument("--claim_match", action="store_true", help="If set, fuzzy match relations with claims rather than using all walkable nodes.")


parser.add_argument("--num_proc", type=int, default=10)

args = parser.parse_args()
print(args)


kg = KG(pickle.load(open(args.dbpedia_path, 'rb')))
df = pd.read_csv(args.data_path + f'{args.set}.csv')

output_dir = f"{args.outputPath}_{args.set}"
os.makedirs(output_dir, exist_ok=True)
dfx = df[~df.index.isin([int(f.split('/')[-1].split('.')[0]) for f in glob.glob(f'{output_dir}/**.json', recursive=True)])]
# dfx= df.sample(1000, random_state=42)

print("Total rows to process", len(dfx))

# matched_len=[]

def process_row(index, row):
    entities = ast.literal_eval(row["Entity_set"])
    save_json_as = f"{output_dir}/{index}.json"
    if os.path.exists(save_json_as):
        return
    # breakpoint()
    # tmp_dict[index] = {e: list(kg.kg.get(e,{}).keys()) for e in entities}
    # val= tmp_dict.get(index)
    val=  {e: list(kg.kg.get(e,{}).keys()) for e in entities}
    resolved_path_ = kg.search(list(val.keys()), {key: [[item] for item in value] for key, value in val.items()})
    if not args.claim_match:
        resolved_json = resolved_path_['connected']+ resolved_path_['walkable']
        pass
    else: ## fuzzy matching with claims
        all_relations = [value for sublist in val.values() for value in sublist]
        tokens = [word for word in word_tokenize(row.Sentence) if len(word) > 1 and word.lower() not in stop_words]
        thrs=70
        matched_relations = list(set([match[0] for token in tokens for match in process.extract(token, all_relations, limit=3) if match[1] > thrs]))
        resolved_path_= kg.search(list(val.keys()), {key: [[e] for e in matched_relations] for key, value in val.items()})
        resolved_json = resolved_path_['connected']+ resolved_path_['walkable']

    print(index, resolved_json)
    with open(save_json_as, 'w') as f:
        json.dump(resolved_json, f, ensure_ascii=False)



partial_process_row = partial(process_row)
with Pool(processes=args.num_proc) as pool:
    pool.starmap(partial_process_row, dfx.iterrows())

# for index, row in dfx.iterrows():
#     process_row(index, row)

# breakpoint()
# with open('tmp_dict.pickle', 'wb') as handle:
#     pickle.dump(tmp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



paths_to_str2 = lambda paths: [evidence[0]+" >- "+evidence[1]  +  " -> "+evidence[2] if "~" not in evidence[1] else evidence[2]+" >- "+evidence[1][1:]+" -> "+evidence[0] for evidence in paths]

import tqdm
jsons = glob.glob(f'{args.outputPath}_{args.set}/**/*.json', recursive=True)
print("Total rows to process", len(df))

output_file_name = args.outputPath + f"/{args.set}.csv"
print("Processsed files will be saved in ", output_file_name)
os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

sentence_label=[]
for file in tqdm.tqdm(jsons):
    file_id = int(file.split('/')[-1].split('.')[0])
    try:
        data = json.load(open(file))
        row = df.iloc[file_id]
        # breakpoint()
        path_string= " | ".join(paths_to_str2(data))
        new_input= f"Claim: {row.Sentence} Evidence: {path_string}"
        sentence_label.append((file_id, new_input, row.Label))
    except Exception as e:
        print(e)
        breakpoint()

# save sentence_label as Sentence,Label csv file 
df = pd.DataFrame(sentence_label, columns=["rowID", "Sentence", "Label"]).sort_values(by=["rowID"]).drop(columns=["rowID"])
df.to_csv(output_file_name, index=False)



#python  llm_filter_relation_all_one_hop.py  --outputPath ./llm_all_one_hop
#python  llm_filter_relation_all_one_hop.py  --outputPath ./llm_all_one_hop_match_claim --claim_match
