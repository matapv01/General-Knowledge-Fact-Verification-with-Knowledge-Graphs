from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
import threading

class MilvusAdapter:
    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port
        self.node_collection = None
        self.rel_collection = None
        # Load embedding model immediately when adapter is created
        self.embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True, device='cpu')
        self.embedding_lock = threading.Lock()

    def connect(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
            self.node_collection = Collection("WikidataNodes")
            self.node_collection.load()
            
            try:
                self.rel_collection = Collection("WikidataRelations")
                self.rel_collection.load()
            except Exception as e:
                print(f"Lỗi load WikidataRelations: {e}")
        except Exception as e:
            print(f"Lỗi kết nối Milvus: {e}")

    def _get_embedding(self, text: str):
        try:
            with self.embedding_lock:
                emb = self.embedding_model.encode(text, normalize_embeddings=True)
            return emb.tolist()
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0]*768

    def search_similar_nodes(self, query_text: str, top_k=3):
        if not self.node_collection:
            self.connect()
            
        vector = self._get_embedding(query_text)
        search_params = {"metric_type": "L2", "params": {"ef": max(64, top_k + 10)}}
        
        try:
            results = self.node_collection.search(
                data=[vector], 
                anns_field="embedding", 
                param=search_params, 
                limit=top_k, 
                output_fields=["entity_id", "entity_name"]
            )
            retrieved = []
            for hit in results[0]:
                retrieved.append({
                    "entity_id": hit.entity_id,
                    "entity_name": hit.entity_name,
                    "distance": hit.distance
                })
            return retrieved
        except Exception as e:
            print(f"Lỗi tìm kiếm Node Milvus: {e}")
            return []

    def search_similar_relations(self, query_text: str, top_k=3):
        if not self.rel_collection:
            self.connect()
            if not self.rel_collection: return []
            
        vector = self._get_embedding(query_text)
        search_params = {"metric_type": "L2", "params": {"ef": max(64, top_k + 10)}}
        
        try:
            results = self.rel_collection.search(
                data=[vector], 
                anns_field="embedding", 
                param=search_params, 
                limit=top_k, 
                output_fields=["relation_id", "relation_name"]
            )
            retrieved = []
            for hit in results[0]:
                retrieved.append({
                    "relation_id": hit.relation_id,
                    "relation_name": hit.relation_name,
                    "distance": hit.distance
                })
            return retrieved
        except Exception as e:
            print(f"Lỗi tìm kiếm Relation Milvus: {e}")
            return []

    def get_entity_names_by_ids(self, entity_ids: list) -> dict:
        if not self.node_collection:
            self.connect()
        if not entity_ids: return {}
        
        try:
            import json
            # Bỏ filter db_id, dùng ID format string vì field entity_id của node là varchar
            # Và giới hạn batch query để khỏi lỗi syntax expr Milvus (tối đa 1000 phần tử)
            ids_str = json.dumps(list(entity_ids))
            expr = f"entity_id in {ids_str}"
            res = self.node_collection.query(expr=expr, output_fields=["entity_id", "entity_name"])
            return {item["entity_id"]: item["entity_name"] for item in res}
        except Exception as e:
            print(f"Lỗi Bulk tra tên Milvus: {e}")
            return {}
