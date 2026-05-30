from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd
import argparse
import time

import os

# Biến Toàn Cục Cache Mô Hình Tại Cấp Độ Worker Process
_GLOBAL_MODEL = None
_GLOBAL_DEVICE = None

def _get_model():
    """
    Hàm Singleton Factory: Chỉ load SentenceTransformer 1 lần duy nhất cho mỗi quá trình Worker của Spark.
    Tránh tình trạng file Parquet nhiều chunk khiến Spark nạp lại Pytorch Model mỗi nhịp gây overhead và in rác (như lỗi vòng lặp Loader).
    """
    global _GLOBAL_MODEL, _GLOBAL_DEVICE
    if _GLOBAL_MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        import multiprocessing
        
        # Thay vì set cứng TARGET_GPU_ID = 2 dễ bị "đụng độ" VRAM giữa các Worker khiến cho 1 con GPU quá tải
        # Chúng ta dùng LOAD BALANCING phân bổ TẤT CẢ Worker chia đều ra cho toàn bộ Card Màn Hình có sẵn
        # bằng cách dùng Modulo theo Process PID (Chưa tính đến tiến trình khác, nhưng sẽ giúp tránh thắt cổ chai ở 1 GPU đơn lẻ)
        device = "cpu"
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                # Tránh nhồi vô GPU 0 (thường là GPU hệ thống đang bị chiếm tài nguyên)
                # Phân đều các Worker của Spark qua các GPU: 1 và 2 (hoặc cao hơn nếu máy có)
                gpu_id = (multiprocessing.current_process().pid % (num_gpus - 1)) + 1
                device = f"cuda:{gpu_id}"
            else:
                device = "cuda:0"
                
        print(f"[ROUND RUBIN] Worker PID {os.getpid()} được Load Balancer phân bổ vào tải Model tại GPU: {device}")
        _GLOBAL_MODEL = SentenceTransformer('all-mpnet-base-v2', device=device)
        _GLOBAL_DEVICE = device
        
    return _GLOBAL_MODEL

# Pandas UDF phân tán sẽ được map đến Worker thông qua Arrow
def embed_text_batch(texts: pd.Series) -> pd.Series:
    """
    HƯỚNG DẪN HOẠT ĐỘNG (PANDAS UDF - User Defined Function)
    """
    model = _get_model()
    
    valid_texts = texts.fillna("").astype(str).tolist()
    
    # Model sẽ tính toán sử dụng Pipeline Encode, tránh load lại Weights HuggingFace liên tục
    embeddings = model.encode(valid_texts, batch_size=16, convert_to_numpy=True, show_progress_bar=False)

    results = embeddings.tolist()
    
    if len(results) != len(texts):
         raise RuntimeError("Số lượng vector sinh ra khác với số text đầu vào từ HuggingFace.")
         
    return pd.Series(results)


def vectorize_nodes(input_dir, output_dir, limit=None):
    spark = SparkSession.builder \
        .appName("Wikidata_Vectorize") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.python.worker.reuse", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50") \
        .getOrCreate()

    embed_udf = pandas_udf(embed_text_batch, returnType=ArrayType(FloatType()))

    print("Đọc Nodes...")
    nodes_df = spark.read.parquet(f"{input_dir}/nodes.parquet")

    if limit:
        print(f"⚠️ Chế độ DEMO: Chỉ lấy {limit} dòng để test cho nhanh...")
        nodes_df = nodes_df.limit(limit)

    # Cắt nhỏ số process Python xuống 2 để dập tránh Out Of Memory CUDA khi có cục text to
    # tất nhiên số worker bây giờ sẽ chỉ còn 2, nó sẽ phải xử lý tuần tự 2 chunk của Parquet, nhưng sẽ ổn định hơn khi chạy trên GPU có VRAM hạn chế
    # các worker còn lại vẫn ở đó, chỉ là không được cấp GPU để xử lý thôi
    nodes_df = nodes_df.repartition(2)

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
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số dòng để test (bỏ trống nếu muốn chạy FULL)")
    args = parser.parse_args()

    vectorize_nodes(args.input, args.output, limit=args.limit)
