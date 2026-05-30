# Data Pipeline - MapReduce cho Knowledge Graph khổng lồ

Mục đích của thư mục này là giải quyết vấn đề Tràn RAM (OOM) khi phải khởi tạo một kho Wikidata khổng lồ lên bộ nhớ thuần túy.

Chúng ta sử dụng **Apache Spark (PySpark)**, một công cụ tính toán song song dựa trên tư tưởng của MapReduce, thay cho vòng `for` chạy tuần tự bằng Pandas.

## Cài đặt thư viện:
Cài đặt Spark, arrow, sentence-transformers, torch (để mã hoá Vector bằng Local GPU tốc độ cao):
```bash
uv add pyspark pyarrow pandas pymilvus neo4j sentence-transformers torch
```

## Giải pháp Tối ưu Tốc độ: HuggingFace (`all-mpnet-base-v2`)
Để tránh lỗi Timeout/OOM mạng như khi dùng `Ollama/nomic`, pipeline hiện tại đã được nâng cấp để tận dụng Card Đồ hoạ (GPU) thông qua model `SentenceTransformer`. Model này sẽ được tự động tải và cache vào hệ thống trong lần chạy đầu tiên.

> **Quan trọng:** Toàn bộ dữ liệu **Nodes** và **Relations** đều được vector hóa bằng model `all-mpnet-base-v2`. Trong mã nguồn gốc, file query BE `milvus_adapter` cũng đã được đổi lại nhúng bằng `HuggingFace`. Điều này đảm bảo **tính nhất quán** của Vector Space, làm cho ANN Search chính xác 100%.

## Các bước chạy Pipeline thực tế:

### Bước 0: Tải dữ liệu Raw
Gồm các file thực thể và quan hệ từ tập dữ liệu Wikidata5m.
```bash
uv run Data_Pipeline/0_download_data.py
```

### Bước 1: Extract & Deduplicate (Map & Reduce)
Spark đọc song song 3 file text raw (entities, relations, triplets), tách bằng `\t`, loại bỏ trùng lặp và ghi xuống định dạng Parquet phân vùng theo `head_id`.
```bash
uv run Data_Pipeline/1_extract_triplets.py
```
*Kết quả:* Các file `nodes.parquet`, `relations.parquet`, `edges.parquet` sẽ được sinh ra ở thư mục `output/`

### Bước 2: Parallel Vectorization + Load Balancing trên GPUs
Spark dùng Pandas UDF + Arrow Flight để thu gom mảng text đưa vào mô hình HuggingFace. Hệ thống **tự động chia tải** (Round Robin) các worker tới những Card Màn Hình rỗng (`cuda:1`, `cuda:2`) để tránh lỗi CUDA OOM, và khởi tạo Singleton Model bằng Pytorch đúc sẵn trong RAM.
```bash
uv run Data_Pipeline/2_vectorize_embeddings.py
uv run Data_Pipeline/4_embed_relations.py
```
*(Ghi chú: PySpark Arrow Memory đã được giới hạn `maxRecordsPerBatch=50` và giảm Process Partition để không dồn ứ I/O dữ liệu khiến mô hình Python Worker kiệt sức).* Kết quả Relations vector còn được **L2 Normalize** để Index HNSW search trên Milvus chuẩn không gian.

### Bước 3: Đưa dữ liệu đã Vector hoá vào Database (Neo4j & Milvus)
Do Docker Neo4j/Milvus dễ bị quá tải Network và sập Heap (`Connection refused/Java Heap Space`) khi nhận Batch Insert từ nhiều kết nối mạng song song. Script tiến hành ép dòng Parquet hợp lại thành Single Pipeline Sequential để đẩy dữ liệu ổn định và an toàn lặp lại:

Đầu tiên, hãy khởi động cụm Database (Milvus & Neo4j) qua Docker:
```bash
docker compose up -d
```
*(Ghi chú: Neo4j đã được cấu hình chạy ở port 7476/7689 để tránh đụng độ với container có sẵn của bạn)*

Sau khi DB đã ở trạng thái Running, chạy script sau để đẩy dữ liệu vào kho:
```bash
uv run python Data_Pipeline/3_load_to_db.py --milvus --neo4j
```
*(Tùy chọn nâng cao: `--milvus-host`, `--milvus-port`, `--neo4j-uri`, `--neo4j-user`, `--neo4j-password`)*

### Bước 5: Kiểm tra trạng thái Database
Liệt kê các Collection hiện có trong Milvus để đảm bảo dữ liệu đã đổ thành công:
```bash
uv run python Data_Pipeline/5_list_collections.py
```
