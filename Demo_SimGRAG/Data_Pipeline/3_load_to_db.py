import argparse
from pyspark.sql import SparkSession

"""
HƯỚNG DẪN HOẠT ĐỘNG (DUAL-DATABASE LOAD với Spark):
Bước này chịu trách nhiệm Load bộ Dataset Parquet đã gia công nhúng vector vào 2 nền tảng vật lý tách biệt.

Spark đọc Parquet phân tán theo từng partition — tránh OOM khi load toàn bộ file vào RAM 1 process.
Mỗi partition được xử lý bởi 1 Worker độc lập qua foreachPartition:
- Mỗi Worker tự tạo connection riêng tới DB (pattern chuẩn cho Spark → external DB write)
- Nhiều Worker ghi song song (concurrent writes) vào Milvus / Neo4j như nhiều client độc lập

1. load_to_milvus: Đọc cột Vector do AI nặn ra bơm vào CSDL Vector Milvus.
   Mục đích là tạo Index HNSW để sau truy vấn "Barack Obama" sẽ bắn vào Milvus lấy được ID góc (Q76).

2. load_to_neo4j: Đọc các Cạnh cấu trúc (Edges), dùng Cypher UNWIND batch import.
   Mục đích là nối lại rễ đồ thị, phục vụ thuật toán DFS Isomorphism ở chặng Retrieval SimGRAG.
"""


def load_to_milvus(nodes_parquet_path, milvus_host="localhost", milvus_port="19530"):
    print("-----------------------------------------")
    print("🔥 BẮT ĐẦU LOAD NODES VÀO MILVUS VECTOR DB...")

    spark = SparkSession.builder \
        .appName("Load_Milvus") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "16g") \
        .getOrCreate()

    print(f"Đang đọc Parquet phân tán: {nodes_parquet_path}")
    nodes_df = spark.read.parquet(nodes_parquet_path)
    total = nodes_df.count()
    print(f"Tổng số Nodes cần nạp: {total}")

    # Tạo Schema và Index HNSW một lần trên Driver trước khi Worker ghi
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
    connections.connect("default", host=milvus_host, port=milvus_port)

    collection_name = "WikidataNodes"
    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} đã tồn tại. Đang xóa dữ liệu cũ...")
        utility.drop_collection(collection_name)

    print("Tạo Schema cho Vector Nodes...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="entity_id", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields, "Wikidata5m Nodes Embedded")
    collection = Collection(collection_name, schema)

    print("Tạo Index HNSW (Hierarchical Navigable Small World)...")
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index HNSW đã sẵn sàng.")

    # Dồn toàn bộ dữ liệu về luồng đơn (Single Thread/Driver) để kiểm soát tốc độ Insert an toàn.
    # Ngăn chặn Milvus nghẽn cổ chai (Connection refuse) và vượt Heap OOM của Spark
    batch_size = 500 # Số lượng bản ghi mỗi Worker gửi vào Milvus một lần, điều chỉnh để cân bằng giữa tốc độ và ổn định
    nodes_df = nodes_df.coalesce(1) # Dồn về 1 partition để chỉ có 1 Worker duy nhất ghi vào Milvus, tránh tình trạng quá tải khi nhiều Worker cùng ghi đồng thời

    def write_partition_to_milvus(partition_rows):
        import time
        """
        Hàm này chạy trên từng Spark Worker độc lập.
        Mỗi Worker tự mở connection riêng tới Milvus — tránh share connection giữa các thread.
        """
        from pymilvus import connections, Collection
        connections.connect("default", host=milvus_host, port=milvus_port)
        col = Collection("WikidataNodes")

        batch_ids, batch_names, batch_vecs = [], [], []
        for row in partition_rows:
            entity_id = str(row["entity_id"]) if row["entity_id"] else ""
            entity_name = str(row["entity_name"]) if row["entity_name"] else ""
            embedding = list(row["embedding"]) if row["embedding"] else [0.0] * 768

            batch_ids.append(entity_id)
            batch_names.append(entity_name)
            batch_vecs.append(embedding)

            if len(batch_ids) >= batch_size:
                try:
                    col.insert([batch_ids, batch_names, batch_vecs])
                    time.sleep(0.1) # Nghỉ nhỏ giữa các batch để tránh spam sập Milvus
                except Exception as e:
                    print(f"Bỏ qua lỗi Insert Milvus batch này: {e}")
                batch_ids, batch_names, batch_vecs = [], [], []

        if batch_ids:
            try:
                col.insert([batch_ids, batch_names, batch_vecs])
            except:
                pass

    print("Bắt đầu ghi song song vào Milvus (foreachPartition)...")
    nodes_df.foreachPartition(write_partition_to_milvus)

    # Flush để đảm bảo tất cả segment được persist
    collection.flush()
    print(f"✅ Hoàn tất Import {collection.num_entities} Nodes vào VectorDB!")
    spark.stop()


def load_to_neo4j(edges_parquet_path, neo4j_uri="neo4j://localhost:7689",
                  neo4j_user="neo4j", neo4j_password="password"):
    print("-----------------------------------------")
    print("🕸️ BẮT ĐẦU LOAD EDGES VÀO NEO4J GRAPH DB...")

    spark = SparkSession.builder \
        .appName("Load_Neo4j") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "16g") \
        .getOrCreate()

    print(f"Đang đọc Parquet phân tán: {edges_parquet_path}")
    edges_df = spark.read.parquet(edges_parquet_path)
    total = edges_df.count()
    print(f"Tổng số Edges cần nạp: {total}")

    # Tạo Constraint Index trên Driver một lần trước khi Worker ghi
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        print("Tạo Constraint Index trên Graph để tăng tốc khởi tạo...")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
    driver.close()

    # Dồn dữ liệu về Single Thread/Driver để giảm áp lực lên driver database Neo4j
    batch_size = 500
    edges_df = edges_df.coalesce(1)

    query = """
    UNWIND $batch AS row
    MERGE (h:Entity {id: row.head_id})
    MERGE (t:Entity {id: row.tail_id})
    MERGE (h)-[r:RELATION {type: row.relation_id}]->(t)
    """

    def write_partition_to_neo4j(partition_rows):
        """
        Hàm này chạy trên từng Spark Worker độc lập.
        Mỗi Worker tự mở connection riêng tới Neo4j — tránh share driver giữa các thread.
        """
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        batch = []
        with driver.session() as session:
            for row in partition_rows:
                batch.append({
                    "head_id": str(row["head_id"]),
                    "relation_id": str(row["relation_id"]),
                    "tail_id": str(row["tail_id"])
                })
                if len(batch) >= batch_size:
                    try:
                        session.run(query, batch=batch)
                    except Exception as e:
                        print(f"Bỏ qua lỗi Insert Neo4j batch này: {e}")
                    batch = []
            if batch:
                try:
                    session.run(query, batch=batch)
                except:
                    pass

        driver.close()

    print("Bắt đầu ghi song song vào Neo4j (foreachPartition)...")
    edges_df.foreachPartition(write_partition_to_neo4j)

    print("✅ Hoàn tất Import Edges vào GraphDB!")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", default="./output/nodes_vectorized.parquet")
    parser.add_argument("--edges", default="./output/edges.parquet")
    parser.add_argument("--milvus", action="store_true", help="Bật cờ này để Load Nodes vào Milvus VectorDB")
    parser.add_argument("--neo4j", action="store_true", help="Bật cờ này để Load Edges vào Neo4j GraphDB")
    parser.add_argument("--milvus-host", default="localhost")
    parser.add_argument("--milvus-port", default="19530")
    parser.add_argument("--neo4j-uri", default="neo4j://localhost:7689")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    args = parser.parse_args()

    if not args.milvus and not args.neo4j:
        print("⚠️ Chú ý: Hãy truyền flag --milvus và/hoặc --neo4j để chỉ định cần nạp cái gì.")
        print("Ví dụ: uv run Data_Pipeline/3_load_to_db.py --milvus --neo4j")

    if args.milvus:
        import os
        if os.path.exists(args.nodes):
            load_to_milvus(args.nodes, milvus_host=args.milvus_host, milvus_port=args.milvus_port)
        else:
            print(f"Không tìm thấy file {args.nodes}")

    if args.neo4j:
        import os
        if os.path.exists(args.edges):
            load_to_neo4j(args.edges, neo4j_uri=args.neo4j_uri,
                          neo4j_user=args.neo4j_user, neo4j_password=args.neo4j_password)
        else:
            print(f"Không tìm thấy file {args.edges}")
