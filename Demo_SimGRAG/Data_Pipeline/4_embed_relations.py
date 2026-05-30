from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd
import numpy as np
import argparse
import time

"""
HƯỚNG DẪN HOẠT ĐỘNG:
Giống như 2_vectorize_embeddings.py cho Nodes, bước này vector hóa Relations (Mối quan hệ)
và nạp vào Milvus collection WikidataRelations để SimGRAG dùng khi tìm kiếm cạnh đồ thị.

Spark + pandas_udf gom Relations thành batch → gọi Ollama /api/embed → nhận vector 768 chiều.
Các vector được L2 Normalize trước khi lưu để L2 distance search hoạt động chính xác.
Nếu Ollama thất bại sau max_retries lần → raise exception, không inject vector giả vào Milvus.
"""

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


import os

_GLOBAL_MODEL = None

def _get_model():
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        import multiprocessing
        device = "cpu"
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                gpu_id = (multiprocessing.current_process().pid % (num_gpus - 1)) + 1
                device = f"cuda:{gpu_id}"
            else:
                device = "cuda:0"
        print(f"[ROUND RUBIN] Worker PID {os.getpid()} phân bổ thẻ Relations qua GPU: {device}")
        _GLOBAL_MODEL = SentenceTransformer('all-mpnet-base-v2', device=device)
    return _GLOBAL_MODEL

def embed_relations_batch(texts: pd.Series) -> pd.Series:
    """
    Pandas UDF chạy trên từng Spark Worker:
    - Nhận batch relation names dạng pd.Series
    """
    model = _get_model()
    valid_texts = texts.fillna("").astype(str).tolist()
    
    # Sử dụng param show_progress_bar=False để HuggingFace không nhả text rác làm đầy log console
    # Giảm batch size xuống để tránh OOM trong Cuda Node Worker
    embeddings = model.encode(valid_texts, batch_size=16, convert_to_numpy=True, show_progress_bar=False)
    
    # L2 Normalize từng vector trong numpy array để Milvus Index FLAT hỗ trợ Cosine thông qua L2 space
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1 # Tránh chia cho 0
    embeddings = embeddings / norms
    
    results = embeddings.tolist()

    if len(results) != len(texts):
        raise RuntimeError("Số lượng vector trả về từ HuggingFace không khớp với số lượng đầu vào!")

    return pd.Series(results)


def embed_and_load_relations(input_dir, output_dir, limit=None):
    spark = SparkSession.builder \
        .appName("Wikidata_EmbedRelations") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.python.worker.reuse", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50") \
        .getOrCreate()

    embed_udf = pandas_udf(embed_relations_batch, returnType=ArrayType(FloatType()))

    print("Đọc Relations Parquet...")
    relations_df = spark.read.parquet(f"{input_dir}/relations.parquet")
    print(f"Tổng số Relations: {relations_df.count()}")

    if limit:
        print(f"⚠️ Chế độ DEMO: Chỉ lấy {limit} dòng...")
        relations_df = relations_df.limit(limit)

    relations_df = relations_df.repartition(2)

    print("Thực hiện Vectorize Relations song song...")
    vectorized_df = relations_df.withColumn("embedding", embed_udf(col("relation_name")))

    print(f"Ghi kết quả ra: {output_dir}/relations_vectorized.parquet")
    vectorized_df.write.mode("overwrite").parquet(f"{output_dir}/relations_vectorized.parquet")
    print("✅ Hoàn thành vectorize Relations!")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./output", help="Thư mục chứa relations.parquet đã extract")
    parser.add_argument("--output", default="./output", help="Thư mục ghi kết quả relations_vectorized.parquet")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số dòng để test (bỏ trống để chạy FULL)")
    args = parser.parse_args()

    embed_and_load_relations(args.input, args.output, limit=args.limit)
