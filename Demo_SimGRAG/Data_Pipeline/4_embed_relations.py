import pandas as pd
import requests
import numpy as np
from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os

"""
HƯỚNG DẪN HOẠT ĐỘNG:
Cũng giống như Nodes, thuật toán SimGRAG cần phải so khớp phương hướng chui rút DFS của Đồ Thị.
Do đó thuật toán không đi chệch sang hướng sai, mà nó phải đi vào đúng Hướng Liên Kết Ngữ Nghĩa nhất (Relation).
File này đọc bảng Danh sách Quan Hệ (VD: 'was born in', 'is located in') -> Vector hoá -> Lưu vào Milvus Collection 2 (WikidataRelations).

Vector lưu ở đây đã được Normalize (Chuẩn hoá thành đơn vị L2) để sau tính Cosine hiệu năng tốt hơn.
"""

path = "/home/llm/MinhPV/General-Knowledge-Fact-Verification-with-Knowledge-Graphs/Demo_SimGRAG/output/relations.parquet"
df = pd.read_parquet(path)
print(f"Tổng số Mối quan hệ (Relations) cần cấu trúc: {len(df)}")

connections.connect("default", host="localhost", port="19530")
col_name = "WikidataRelations"
if utility.has_collection(col_name):
    print(f"Xoá collection cũ: {col_name}")
    utility.drop_collection(col_name)

fields = [
    FieldSchema(name="db_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="relation_id", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="relation_name", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="Wikidata Relations for SimGRAG Baseline")
col = Collection(col_name, schema)

# Tạo Index
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64}
}
print("Đang tạo Index...")
col.create_index(field_name="embedding", index_params=index_params)

def get_embedding(text):
    try:
        url = "http://localhost:11434/api/embeddings"
        payload = {"model": "nomic-embed-text", "prompt": text}
        response = requests.post(url, json=payload, timeout=10)
        return response.json().get("embedding", [0.0]*768)
    except:
        return [0.0]*768

batch_size = 128
total = len(df)

rel_ids = []
rel_names = []
embeddings = []

print("Bắt đầu nhúng Vector và đẩy vào Milvus...")
for i in tqdm(range(total)):
    row = df.iloc[i]
    r_id = str(row['relation_id'])
    r_name = str(row['relation_name'])
    
    emb = get_embedding(r_name)
    
    rel_ids.append(r_id)
    rel_names.append(r_name)
    embeddings.append(emb)
    
    if len(rel_ids) >= batch_size or i == total - 1:
        # Chuẩn hóa L2 norm cho các vector (để L2 distance hoạt động tốt)
        embs_np = np.array(embeddings)
        norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
        norms[norms == 0] = 1 # Tránh chia cho 0
        embs_np = embs_np / norms
        
        insert_data = [
            rel_ids,
            rel_names,
            embs_np.tolist()
        ]
        col.insert(insert_data)
        rel_ids, rel_names, embeddings = [], [], []

col.flush()
print(f"Đã nạp xong {col.num_entities} Quan hệ vào Milvus!")
