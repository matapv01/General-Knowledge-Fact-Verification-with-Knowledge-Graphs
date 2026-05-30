import requests
from pymilvus import connections, Collection


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


class MilvusAdapter:
    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port
        self.node_collection = None
        self.rel_collection = None

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

    def _get_embedding(self, text: str) -> list:
        """
        Sử dụng HuggingFace SentenceTransformer để mã hóa truy vấn đồng bộ với lớp Indexer offline
        """
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer('all-mpnet-base-v2', device=device)
            # Mã hoá Query bằng GPU/CPU
            query_vector = model.encode([text], convert_to_numpy=True)[0]
            
            # TUYỆT ĐỐI KHÔNG NORMALIZE L2
            # Vì Data lúc Index cũng KHÔNG ĐƯỢC NORMALIZE.
            # (Xem luồng UDF của file 2_vectorize_embeddings.py).
            # Nếu Query Normalize (Độ lớn=1) mà DB không Normalize (Độ lớn > 10) thì Milvus L2 Search sẽ ra các kết quả sai.
                
            return query_vector.tolist()
        except Exception as e:
            raise RuntimeError(f"Lỗi mã hoá truy vấn HuggingFace: {e}")

    def search_similar_nodes(self, query_text: str, top_k=3):
        if not self.node_collection:
            self.connect()

        vector = self._get_embedding(query_text)
        # ef phải >= top_k để HNSW tìm đủ candidate candidates
        # Với top_k lớn (16384 từ config), ef cần lớn tương đương
        ef_value = max(top_k * 2, 256)
        search_params = {"metric_type": "L2", "params": {"ef": ef_value}}

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
                    "entity_id": hit.entity.get("entity_id", ""),
                    "entity_name": hit.entity.get("entity_name", ""),
                    "distance": abs(hit.distance)
                })
            return retrieved
        except Exception as e:
            print(f"Lỗi tìm kiếm Node Milvus: {e}")
            return []

    def search_similar_relations(self, query_text: str, top_k=3):
        if not self.rel_collection:
            self.connect()
            if not self.rel_collection:
                return []

        vector = self._get_embedding(query_text)
        ef_value = max(top_k * 2, 256)
        search_params = {"metric_type": "L2", "params": {"ef": ef_value}}

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
                    "relation_id": hit.entity.get("relation_id", ""),
                    "relation_name": hit.entity.get("relation_name", ""),
                    "distance": abs(hit.distance)
                })
            return retrieved
        except Exception as e:
            print(f"Lỗi tìm kiếm Relation Milvus: {e}")
            return []

    def get_entity_names_by_ids(self, entity_ids: list) -> dict:
        if not self.node_collection:
            self.connect()
        if not entity_ids:
            return {}

        try:
            import json
            # Chia thành batch tối đa 1000 phần tử để tránh lỗi syntax expr Milvus
            result_map = {}
            batch_size = 1000
            for i in range(0, len(entity_ids), batch_size):
                batch = entity_ids[i:i + batch_size]
                ids_str = json.dumps(list(batch))
                expr = f"entity_id in {ids_str}"
                res = self.node_collection.query(
                    expr=expr,
                    output_fields=["entity_id", "entity_name"]
                )
                for item in res:
                    result_map[item["entity_id"]] = item["entity_name"]
            return result_map
        except Exception as e:
            print(f"Lỗi Bulk tra tên Milvus: {e}")
            return {}
