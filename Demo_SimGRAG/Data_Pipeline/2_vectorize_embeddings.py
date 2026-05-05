from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd
import argparse

import time

# Khởi tạo mô hình Embedding (Ollama hoặc Nomic)
# Lưu ý: Mô hình sẽ được khởi tạo trên từng Worker thay vì trên Driver.
def embed_text_batch(texts: pd.Series) -> pd.Series:
    """
    Pandas UDF (User Defined Function): 
    Spark tự gom các text thành các lô (batch) và gọi hàm này song song.
    """
    import requests
    import json
    
    api_url = "http://localhost:11434/api/embed"
    model_name = "nomic-embed-text"
    
    valid_texts = texts.fillna("").astype(str).tolist()
    
    try:
        results = []
        chunk_size = 32  # Giảm chunk_size cực mạnh để tránh ngộp VRAM / Ollama lỗi trả về rác cache
        for i in range(0, len(valid_texts), chunk_size):
            chunk = valid_texts[i:i+chunk_size]
            payload = {"model": model_name, "input": chunk, "keep_alive": "5m"}
            
            # Retry logic để đảm bảo không bị timeout
            max_retries = 3
            res_json = {}
            for attempt in range(max_retries):
                try:
                    res = requests.post(api_url, json=payload, timeout=60)
                    if res.status_code == 200:
                        res_json = res.json()
                        break
                    else:
                        time.sleep(1)
                except Exception:
                    time.sleep(2)
            
            if 'embeddings' in res_json and len(res_json['embeddings']) == len(chunk):
                results.extend(res_json['embeddings'])
            else:
                # Nếu API thực sự nổ, đành gán vector random (hoặc 0) để không gián đoạn
                results.extend([[0.0001] * 768] * len(chunk))
        
        if len(results) == len(texts):
            return pd.Series(results)
    except Exception as e:
        print(f"Ollama error: {e}")
        
    return pd.Series([[0.0001] * 768] * len(texts))

def vectorize_nodes(input_dir, output_dir, limit=None):
    spark = SparkSession.builder \
        .appName("Wikidata_Vectorize") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "2g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "500") \
        .getOrCreate()

    embed_udf = pandas_udf(embed_text_batch, returnType=ArrayType(FloatType()))

    print("Đọc Nodes...")
    nodes_df = spark.read.parquet(f"{input_dir}/nodes.parquet")
    
    if limit:
        print(f"⚠️ Chế độ DEMO: Chỉ lấy {limit} dòng để test cho nhanh...")
        nodes_df = nodes_df.limit(limit)

    # Bóp lại số lượng phân vùng song song xuống đúng bằng số NUM_PARALLEL của Ollama
    # 16 luồng Spark sẽ call map 1-1 với 16 luồng của Ollama -> Không bị nghẽn mạng!
    nodes_df = nodes_df.repartition(16)

    print("Thực hiện Vectorize song song...")
    # Pha Map: Tính vector cho trường entity_name
    vectorized_df = nodes_df.withColumn("embedding", embed_udf(col("entity_name")))

    print("Ghi kết quả ra Vector DB Format (Parquet)...")
    vectorized_df.write.mode("overwrite").parquet(f"{output_dir}/nodes_vectorized.parquet")
    print("Hoàn thành!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./output", help="Thư mục chứa nodes đã extract")
    parser.add_argument("--output", default="./output", help="Thư mục ghi kết quả vectorized")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số dòng để test cho lẹ (bỏ trống nếu muốn chạy FULL)")
    args = parser.parse_args()
    
    vectorize_nodes(args.input, args.output, limit=args.limit)
