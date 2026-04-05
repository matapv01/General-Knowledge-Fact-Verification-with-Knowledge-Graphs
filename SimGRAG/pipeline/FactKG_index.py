import sys
sys.path.append('..')

import json
from src.dataset import FactKG
from src.indexer import Indexer

# load configs
configs = json.load(open('../configs/FactKG.json'))

# load dataset
dataset = FactKG(configs)
KG = dataset.get_KG()
type_to_nodes = dataset.get_type_to_nodes()

# build index
indexer = Indexer(configs)
indexer.build_index(KG, type_to_nodes)