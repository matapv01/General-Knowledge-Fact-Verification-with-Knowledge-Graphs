import sys
sys.path.append('..')

import json
from src.dataset import MetaQA
from src.indexer import Indexer

# load configs
configs = json.load(open('../configs/metaQA_3hop.json'))

# load dataset
dataset = MetaQA(configs)
KG = dataset.get_KG()

# build index
indexer = Indexer(configs)
indexer.build_index(KG)