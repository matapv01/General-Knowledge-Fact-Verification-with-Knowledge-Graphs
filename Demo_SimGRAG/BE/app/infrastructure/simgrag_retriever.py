import sys
import os
import copy

baseline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Baseline", "with_evidence", "SimGRAG"))
if baseline_path not in sys.path:
    sys.path.append(baseline_path)

from src.retriever import Retriever

class KGProxy:
    def __init__(self, neo4j_adapter):
        self.neo4j = neo4j_adapter
        self.cache = {}
        
    def __len__(self):
        return 5000000 # Dummy limit để lừa bypass check len của src_retriever
        
    def __contains__(self, item):
        return True # Mọi Node ID từ Milvus coi như tồn tại
        
    def __getitem__(self, node_id):
        if node_id in self.cache:
            return self.cache[node_id]
        neighbors = self.neo4j.get_1hop_neighbors(node_id, limit=30000)
        res = {}
        for nb in neighbors:
            rel = nb['relation']
            tail = nb['tail']
            if rel not in res:
                res[rel] = []
            # Baseline lưu undirected => neighbor có thể theo cả 2 chiều, hàm neo4j_adapter trả tail.
            res[rel].append(tail)
        self.cache[node_id] = res
        return res

class MockModel:
    def encode(self, texts):
        # Trả về nguyên gốc list(Text) thay vì Vector vì chúng ta gắn Milvus nhúng sau
        return texts

class MockNodeStore:
    def __init__(self, milvus):
        self.milvus = milvus
    def search(self, vectors, topk):
        # vectors lúc này là list Text string do MockModel trả ra
        res = []
        for v in vectors:
            cands = self.milvus.search_similar_nodes(v, top_k=topk)
            # Phải bọc abs() vì milvus đôi lúc trả distance -0.0001 do float precision
            row = [{'entity': {'name': c['entity_id']}, 'distance': abs(c['distance'])} for c in cands]
            res.append(row)
        return res

class MockRelStore:
    def __init__(self, milvus):
        self.milvus = milvus
    def search(self, vectors, topk):
        res = []
        for v in vectors:
            cands = self.milvus.search_similar_relations(v, top_k=topk)
            row = [{'entity': {'name': c['relation_id']}, 'distance': abs(c['distance'])} for c in cands]
            res.append(row)
        return res

import json

class SimgragBaselineRetriever(Retriever):
    def __init__(self, milvus_adapter, neo4j_adapter, relations_map):
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "simgrag_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.configs = json.load(f)
            
        self.relations_map = relations_map
        self.milvus = milvus_adapter
        
        # --- KẾ THỪA VÀ MOCK LẠI CÁC THUỘC TÍNH CỦA DEPENDENCY GỐC ---
        self.KG = KGProxy(neo4j_adapter)
        self.model = MockModel()
        self.timeout = self.configs['retriever']['timeout']
        self.final_topk = self.configs['retriever']['final_topk']
        
        self.node_vector_store = MockNodeStore(milvus_adapter)
        self.node_sim_topk = self.configs['retriever']['node_sim_topk']
        
        self.relation_vector_store = MockRelStore(milvus_adapter)
        self.relation_sim_topk = self.configs['retriever']['relation_sim_topk']
        
        self.use_type_candidates = False
        self.type_vector_store = None
        self.type_to_nodes = None
        
    def retrieve_subgraph(self, query_graph):
        """
        Hàm Wrapper để format đầu ra và dịch ID -> Text. 
        Mọi tính toán Isomorphism DFS đều dùng super().retrieve từ Baseline gốc.
        """
        if not query_graph: return []
        
        # Chạy thuật toán truy vấn Đồ thị nguyên bản của SimGRAG!
        output = super().retrieve(query_graph, mode='greedy')
        results = output.get("results", [])
        
        # Định dạng và Dịch Ids
        all_evidences = []
        all_ids = set()
        
        for result in results:
            # Baseline trả về tuple (score, graph, reuse_nodes)
            score, graph, reuse_nodes = result
            for h, r, t in graph:
                all_ids.add(h)
                all_ids.add(t)
                
        id_to_name = self.milvus.get_entity_names_by_ids(list(all_ids))
        
        for result in results:
            score, graph, reuse_nodes = result
            current_evidence = []
            for h, r, t in graph:
                h_name = id_to_name.get(h, h)
                t_name = id_to_name.get(t, t)
                r_name = self.relations_map.get(r, r)
                current_evidence.append((h_name, r_name, t_name))
            all_evidences.append(current_evidence)
            
        return all_evidences
