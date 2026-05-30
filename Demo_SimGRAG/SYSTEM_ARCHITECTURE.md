# SYSTEM ARCHITECTURE - KNOWLEDGE GRAPH PIPELINE

Hệ thống được thiết kế theo tư tưởng **SimGRAG** nhằm kết hợp Graph Retrieval và Text Retrieval, giải quyết bài toán RAG trên các kho tri thức lớn mà không bị Memory Leak.

## Sơ đồ Kiến trúc

```mermaid
graph TD
    subgraph Data_Pipeline [Data Pipeline (Offline)]
        raw(Dữ liệu thô Wikidata) --> E[1_extract_triplets: Spark MapReduce]
        E --> P1(nodes.parquet)
        E --> P2(relations.parquet)
        E --> P3(edges.parquet)
        
        P1 --> V1[2_vectorize: Spark + HuggingFace GPU]
        P2 --> V2[4_embed: Spark + HuggingFace GPU]
        
        V1 --> VP1(nodes_embedded.parquet)
        V2 --> VP2(relations_embedded.parquet)
        
        VP1 & VP2 & P3 --> L[3_load_to_db: Spark foreachPartition]
    end
    
    subgraph Storage [Database Layer]
        L --> M[(Milvus: VectorDB)]
        L --> N[(Neo4j: GraphDB)]
    end
    
    subgraph Backend_Online [Backend (Online)]
        Req(User Ask) --> BE[FastAPI App]
        BE --> LM[LLM API via .env]
        BE --> S[SimGRAG Retriever + HuggingFace]
        S --> M
        S --> N
        S -.-> Res(Answer)
    end
    
    Backend_Online -- Chạy trên --> Docker(Docker Compose)
    Storage -- Chạy trên --> Docker
```

## Giải thích Luồng Dữ liệu (Offline / Data Pipeline)

Nhờ áp dụng hệ sinh thái Apache Spark và Parquet, hệ thống xử lý Graph bằng MapReduce theo 3 pha:

1. **Extraction (Trích xuất & Làm sạch):**
   - Đọc Text Raw song song.
   - Distinct qua Shuffle của Spark tránh trùng ID.
   - Ghi định dạng cột (Parquet) giúp tiết kiệm dung lượng đĩa và tối ưu IO cho bước sau.

2. **Vectorization (Nhúng song song):**
   - Dùng Pandas UDF (User Defined Function) của Spark kết hợp Apache Arrow.
   - Spark sẽ cắt file Parquet lớn thành các Arrow Batch (chuẩn in-memory cột).
   - Sử dụng thư viện **HuggingFace (`sentence-transformers/all-mpnet-base-v2`)** cục bộ với sự hỗ trợ của Torch GPU (`cuda`), giúp Vector hoá hàng chục ngàn node nhanh gấp nhiều lần so với gửi HTTP API và tránh lỗi Memory Leak hiệu quả!
   - Không sinh Vector giả rác (`[0.0]*768`) như phiên bản code cũ.
   - **Đặc biệt**: Các Vector Push thẳng vào DB mà *KHÔNG ÁP DỤNG L2 NORMALIZE*.

3. **Loading (Chèn an toàn CSDL qua Streaming):**
   - Thay vì 1 script đọc toàn bộ 10GB Parquet lên vùng nhớ Python (gây OOM máy tính) hay mở nhiều `foreachPartition` làm sập Docker Milvus và Neo4j vì Connection Refused.
   - Spark tiến hành `coalesce(1)` dồn dữ liệu về Single Thread pipeline streaming. Dữ liệu cực lớn sẽ được kéo từ ổ cứng lên theo từng Batch 500 records đẩy vào Localhost và xoá khỏi RAM liên tục để duy trì độ ổn định vững chãi.
   - Đảm bảo CSDL trên cục bộ không bao giờ bị nghẽn mạng!

## Dịch Vụ Cốt Lõi (Online / Streaming)

- **Vector Database (Milvus):** Chạy trên `:19530`, đảm nhận ANN Search L2 Distance cho Index `Entity` và Index `Relation`. Cơ chế HNSW được dùng với hệ số `ef` linh hoạt theo `top_k`. **Lưu ý Cực Kỳ Quan Trọng**: Retriever phải mã hóa Query bằng chính `all-mpnet-base-v2` và *không được L2 Normalize*, để đồng bộ cùng cấp độ Magnitude với Data trong Milvus!
- **Graph Database (Neo4j):** Chạy trên `:7476` (bolt `:7689`). Chứa Topology Graph. Nhờ VectorDB trả về Anchor Node, hệ thống neo thẳng vào Neo4j và traverse ra các Head/Tail kề.
- **Backend (FastAPI):** Kết nối Milvus + Neo4j + LLM. Logic Isomorphism Subgraph Search được bọc trong mô hình Use Case của Clean Architecture.
- **LLM Engine:** Cho phép tuỳ biến hoàn toàn qua biến môi trường (File `.env`). Hỗ trợ Ollama Local, OpenAI Platform, hoặc NVIDIA NIM (VD: `qwen/qwen3.5-122b-a10b` kết hợp `OPENAI_BASE_URL`).
