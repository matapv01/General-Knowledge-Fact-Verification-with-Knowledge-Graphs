import sys
sys.path.append('..')

import time
import json
from tqdm import tqdm

from src.llm import LLM
import prompts.answer_FactKG
import prompts.rewrite_FactKG
from src.dataset import FactKG
from src.retriever import Retriever
from src.utils import check_answer
from src.utils import extract_graph

# load configs
configs = json.load(open('../configs/FactKG.json'))

# load dataset
dataset = FactKG(configs)
KG = dataset.get_KG()
type_to_nodes = dataset.get_type_to_nodes()
all_queries = dataset.get_queries()
all_groundtruths = dataset.get_groundtruths()

# load LLM
llm = LLM(configs)

# load retriever
retriever = Retriever(configs, KG, type_to_nodes)

# run for each query
def run(query, groundtruths):
	global retriever
	res = {
		'query': query,
		'groundtruths': groundtruths,
		'retriever_configs': configs['retriever'],
		'llm_configs': configs['llm'],
		'rewrite_shot': configs['rewrite_shot'],
		'answer_shot': configs['answer_shot'],
	}
	
	try:
		# rewrite
		start = time.time()
		res['rewrite_prompt'] = prompts.rewrite_FactKG.get(query, shot=res['rewrite_shot'])
		res['rewrite_llm_output'] = llm.chat(res['rewrite_prompt'])
		res['rewrite_time'] = time.time() - start
  
		# extract graph
		res['query_graph'] = extract_graph(res['rewrite_llm_output'])
		
		# subgraph matching
		start = time.time()
		res['retrieval_details'] = retriever.retrieve(res['query_graph'], mode='greedy')
		res['evidences'] = [each[1] for each in res['retrieval_details']['results']]
		res['retrieval_time'] = time.time() - start

		# answer
		start = time.time()
		res['answer_prompt'] = prompts.answer_FactKG.get(res['query'], res['evidences'], shot=res['answer_shot'])
		res['answer_llm_output'] = llm.chat(res['answer_prompt'])
		res['answer_time'] = time.time() - start
  
		# check answer
		res['correct'] = check_answer(res['answer_llm_output'], groundtruths)
  
	except Exception as e:
		res['error_message'] = str(e)
		if "closed channel" in str(e):
			time.sleep(5)
			print("\\nRestarting connection to Milvus due to closed channel...")
			from src.retriever import Retriever
			retriever = Retriever(configs, KG, type_to_nodes)
  
	return res

# run for all queries
result_file = configs["output_filename"]
import os
processed_queries = set()
if os.path.exists(result_file):
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'error_message' not in data:
                        processed_queries.add(data['query'])
                except:
                    pass

for query, groundtruths in tqdm(zip(all_queries, all_groundtruths), total=len(all_queries)):
    if query in processed_queries:
        continue
    res = run(query, groundtruths)
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(res, ensure_ascii=False) + '\n')