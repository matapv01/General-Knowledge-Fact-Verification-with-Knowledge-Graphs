import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

def create_spark_session(app_name="Wikidata_Extractor"):
    # Khởi tạo Spark Session với cấu hình phù hợp với RAM
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "16g") \
        .getOrCreate()

def process_wikidata5m(entity_path, relation_path, triplet_path, output_dir):
    spark = create_spark_session()
    sc = spark.sparkContext
    
    # Hàm Map phân rã file Entity (tách bằng Tab)
    def parse_entity(line):
        parts = line.split('\t')
        ent_id = parts[0]
        # Chọn tên đầu tiên làm tên chính (bỏ qua alias để tối ưu bộ nhớ nếu không cần)
        ent_name = parts[1] if len(parts) > 1 else ""
        return (ent_id, ent_name)

    print("1. Xử lý Entities (Nodes)...")
    entity_rdd = sc.textFile(entity_path).map(parse_entity)
    entity_schema = StructType([
        StructField("entity_id", StringType(), False),
        StructField("entity_name", StringType(), True)
    ])
    nodes_df = spark.createDataFrame(entity_rdd, entity_schema).dropDuplicates(["entity_id"])
    print(" -> Lưu Nodes dạng Parquet...")
    nodes_df.write.mode("overwrite").parquet(f"{output_dir}/nodes.parquet")

    print("\n2. Xử lý Relations...")
    relation_rdd = sc.textFile(relation_path).map(parse_entity)
    relation_schema = StructType([
        StructField("relation_id", StringType(), False),
        StructField("relation_name", StringType(), True)
    ])
    relations_df = spark.createDataFrame(relation_rdd, relation_schema).dropDuplicates(["relation_id"])
    print(" -> Lưu Relations dạng Parquet...")
    relations_df.write.mode("overwrite").parquet(f"{output_dir}/relations.parquet")

    print("\n3. Xử lý Triplets (Edges)...")
    def parse_triplet(line):
        parts = line.split('\t')
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
        return None
        
    triplet_rdd = sc.textFile(triplet_path).map(parse_triplet).filter(lambda x: x is not None)
    triplet_schema = StructType([
        StructField("head_id", StringType(), False),
        StructField("relation_id", StringType(), False),
        StructField("tail_id", StringType(), False)
    ])
    edges_df = spark.createDataFrame(triplet_rdd, triplet_schema).dropDuplicates()
    
    print(" -> Phân vùng Edges và lưu dạng Parquet...")
    # Tối ưu: Phân mảnh theo head_id (Hash Partitioning) giúp GraphDB nạp nhanh hơn sau này
    edges_df.repartition(64, "head_id").write.mode("overwrite").parquet(f"{output_dir}/edges.parquet")

    print("\nHoàn tất quy trình MapReduce Extract cho Wikidata5m!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Đường dẫn chuẩn tới folder tải về
    parser.add_argument("--entity", default="data/raw/wikidata/wikidata5m_alias/wikidata5m_entity.txt")
    parser.add_argument("--relation", default="data/raw/wikidata/wikidata5m_alias/wikidata5m_relation.txt")
    parser.add_argument("--triplet", default="data/raw/wikidata/wikidata5m_all_triplet.txt/wikidata5m_all_triplet.txt")
    parser.add_argument("--output", default="./output")
    args = parser.parse_args()
    
    process_wikidata5m(args.entity, args.relation, args.triplet, args.output)
