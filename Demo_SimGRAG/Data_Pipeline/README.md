# Data Pipeline - MapReduce cho Knowledge Graph khổng lồ

Mục đích của thư mục này là giải quyết vấn đề Tràn RAM (OOM) khi phải khởi tạo một kho Wikidata khổng lồ lên bộ nhớ thuần túy của Baseline cũ.

Chúng ta sử dụng **Apache Spark (PySpark)**, một công cụ tính toán song song dựa trên tư tưởng của MapReduce, thay cho vòng `for` chạy tuần tự bằng Pandas.

## Cài đặt thư viện:
Cài đặt Spark và arrow (để giao tiếp tốt hơn giữa C++ memory framework và Python) bằng lệnh:
```bash
uv add pyspark pyarrow requests pandas
```

## Setup Backend AI nội bộ (Ollama)
Để chạy được bước tạo Vector, máy tính hiện tại cần khởi động Ollama và chạy model text-embedding.
1. Khởi động Ollama ở chế độ nền (port mặc định 11434):
   ```bash
   ollama serve
   ```
2. Tải model nhúng văn bản về máy cục bộ:
   ```bash
   ollama pull nomic-embed-text
   ```

## Các bước chạy Pipeline thực tế:

### Bước 0: Tải dữ liệu Raw
Gồm các file thực thể và quan hệ từ tập dữ liệu Wikidata5m.
```bash
uv run Data_Pipeline/0_download_data.py
```

### Bước 1: Extract & Deduplicate (Map & Reduce)
Đọc file Text, phân vùng, loại bỏ trùng lặp và ghi xuống định dạng Parquet.
```bash
uv run Data_Pipeline/1_extract_triplets.py
```
*Kết quả:* Các file `nodes.parquet`, `relations.parquet`, `edges.parquet` sẽ được sinh ra ở thư mục `output/`

### Bước 2: Parallel Vectorization (Bắn nhúng Vector song song)
Tận dụng Pandas UDF của Spark đóng gói lệnh gửi đi dưới dạng mô hình Batch, kết nối tới con Ollama local.
```bash
uv run Data_Pipeline/2_vectorize_embeddings.py
```
*(Ghi chú: PySpark Arrow Memory đã được cấp tối đa `8g` Heap và `2g` Off-Heap để phòng thủ lỗi Memory Leak khi chuyển buffer sang Batch API)*

### Bước 3: Load vào hệ quản trị đồ thị (Neo4j) & VectorDB (Milvus)
File Parquet thành quả sẽ được nạp thẳng vào Database dùng làm RAG (Retrieval-Augmented Generation).

Đầu tiên, hãy khởi động cụm Database (Milvus & Neo4j) qua Docker:
```bash
docker compose up -d
```
*(Ghi chú: Neo4j đã được cấu hình chạy ở port 7476/7689 để tránh đụng độ với container có sẵn của bạn)*

Sau khi DB đã ở trạng thái Running, chạy script sau để đẩy dữ liệu vào kho:
```bash
uv run python Data_Pipeline/3_load_to_db.py --milvus --neo4j
```
*(Hệ thống sử dụng Batch Insert và sinh Index HNSW để tăng tốc độ Import)*

### Bước 4: Embedding các Relation (Mối quan hệ)
Mã hóa thông tin relation trên đồ thị thành Vector và đẩy lên Milvus collection `WikidataRelations`:
```bash
uv run python Data_Pipeline/4_embed_relations.py
```

### Bước 5: Kiểm tra trạng thái Database
Liệt kê các Collection hiện có trong Milvus để đảm bảo dữ liệu đã đổ thành công:
```bash
uv run python Data_Pipeline/5_list_collections.py
```
