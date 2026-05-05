import argparse
import pandas as pd
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from neo4j import GraphDatabase
import os

def load_to_milvus(parquet_path):
    print("-----------------------------------------")
    print("🔥 ĐANG KẾT NỐI TỚI MILVUS VECTOR DB...")
    connections.connect("default", host="localhost", port="19530")
    
    collection_name = "WikidataNodes"
    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} đã tồn tại. Đang xóa dữ liệu cũ...")
        utility.drop_collection(collection_name)
        
    print("Tiến hành tạo Schema cho Vector Nodes...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="entity_id", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields, "Wikidata5m Nodes Embedded")
    collection = Collection(collection_name, schema)
    
    print(f"Đang đọc dữ liệu Parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    # Loại bỏ dữ liệu None nếu có
    df['entity_name'] = df['entity_name'].fillna("")
    
    print("Bắt đầu Insert (Batch processing)...")
    batch_size = 10000
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        entities = [
            batch['entity_id'].tolist(),
            batch['entity_name'].tolist(),
            batch['embedding'].tolist()
        ]
        collection.insert(entities)
        print(f" -> Đã chèn thành công {min(i+batch_size, len(df))} / {len(df)} đỉnh vào Milvus.")
        
    print("Đang tạo Index bề mặt (HNSW) cho Vector DB. Việc này giúp Semantic Search nhanh như chớp...")
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("✅ Hoàn tất Import Nodes vào VectorDB!")

def load_to_neo4j(edges_parquet_path):
    print("-----------------------------------------")
    print("🕸️ ĐANG KẾT NỐI TỚI NEO4J GRAPH DB...")
    URI = "neo4j://localhost:7689"
    AUTH = ("neo4j", "password") # Tài khoản pass mặc định như trong docker-compose
    
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    print(f"Đang đọc dữ liệu cạnh Parquet: {edges_parquet_path}")
    df = pd.read_parquet(edges_parquet_path)
    
    with driver.session() as session:
        print("Tạo chỉ mục (Index) trên Graph để tăng tốc khởi tạo...")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        
        print("Bắt đầu Import các Quan Hệ (Relations)...")
        # Sử dụng UNWIND của Cypher giúp batch import rất khỏe
        query = """
        UNWIND $batch AS row
        MERGE (h:Entity {id: row.head_id})
        MERGE (t:Entity {id: row.tail_id})
        MERGE (h)-[r:RELATION {type: row.relation_id}]->(t)
        """
        
        batch_size = 10000
        # Tránh OOM do biến toàn bộ df thành dict cùng lúc
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch = batch_df.to_dict('records')
            session.run(query, batch=batch)
            print(f" -> Đã nối thành công {min(i+batch_size, len(df))} / {len(df)} cạnh trên Map.")
            
    driver.close()
    print("✅ Hoàn tất Import Edges vào GraphDB!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", default="./output/nodes_vectorized.parquet")
    parser.add_argument("--edges", default="./output/edges.parquet")
    parser.add_argument("--milvus", action="store_true", help="Bật cờ này để Load Nodes vào thư viện Milvus VectorDB")
    parser.add_argument("--neo4j", action="store_true", help="Bật cờ này để Load Edges vào Neo4j GraphDB")
    args = parser.parse_args()
    
    if not args.milvus and not args.neo4j:
        print("⚠️ Chú ý: Bạn hãy truyền flag --milvus và/hoặc --neo4j để tôi biết bạn muốn nạp cái gì nhé!")
        print("Ví dụ: uv run Data_Pipeline/3_load_to_db.py --milvus --neo4j")
    
    if args.milvus:
        if os.path.exists(args.nodes):
            load_to_milvus(args.nodes)
        else:
            print(f"Không tìm thấy file {args.nodes}")
            
    if args.neo4j:
        if os.path.exists(args.edges):
            load_to_neo4j(args.edges)
        else:
            print(f"Không tìm thấy file {args.edges}")