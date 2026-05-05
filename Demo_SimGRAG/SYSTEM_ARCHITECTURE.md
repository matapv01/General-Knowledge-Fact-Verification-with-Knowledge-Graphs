# Kiến Trúc Hệ Thống & Luồng Dữ Liệu (System Architecture & Data Flow)
Dự án: **General Knowledge Fact Verification with Knowledge Graphs (SimGRAG)**

Tài liệu này mô tả chi tiết toàn bộ vòng đời của hệ thống: từ lúc nhập dữ liệu thô khổng lồ (Wikidata), xử lý qua MapReduce, lưu trữ kép trên cơ sở dữ liệu Vector & Đồ thị, cho đến cách thuật toán SimGRAG truy vấn qua lại để xác thực một nhận định (Fact Verification).

---

## PHẦN 1: TỔNG QUAN HẠ TẦNG (INFRASTRUCTURE)

Hệ thống được thiết kế theo tư tưởng **Clean Architecture** kết hợp với **RAG đa phương thức (Hybrid RAG)**. Dữ liệu được chia làm 3 mảng lưu trữ tách biệt để đảm nhận các vai trò khác nhau:

1. **Graph Database (Neo4j - Cổng 7689/7476):** Lưu trữ *cấu trúc* và *liên kết* (Topology) của các Node/Edge theo chiều sâu. Hỗ trợ truy xuất đồ thị nhiều chặng (Multi-hop).
2. **Vector Database (Milvus - Cổng 19530):** Lưu trữ *ngữ nghĩa* của các Vector Embeddings. Đảm nhiệm chức năng đối soát chuỗi tự nhiên sang ID của đồ thị (Semantic Search).
3. **LLM Engine (Ollama/NVIDIA - Cổng 11434):** Nền tảng nhúng (Nomic-embed-text) và 추 luận ngôn ngữ (Llama 3/Qwen...) dùng để bóc tách câu hỏi và rút ra kết luận cuối.

---

## PHẦN 2: CHUẨN BỊ DỮ LIỆU (ETL PIPELINE & MAPREDUCE)

Kiến thức nền tảng KG (như Wikidata5m) chứa hàng chục triệu Nodes (đỉnh) và Edges (cạnh) với dung lượng lên đến vài chục GB. Nếu tải tất cả vào bộ nhớ RAM bằng Pandas thông thường để Join (kết bảng) thì hệ thống sẽ sập (Out of Memory) lập tức. Do đó, việc xử lý được chia nhỏ qua 4 bước nghiêm ngặt nằm trong folder `Data_Pipeline/`:

### Bước 1: Trích xuất Triplets với MapReduce (Apache Spark)
- **Tập tin:** `1_extract_triplets.py`
- **Vấn đề:** Dữ liệu Entity thô thường ở dạng liên kết ID (VD: `Q76 \t P31 \t Q30`). Ở một file khác (aliases/labels) mới chứa từ điển `Q76 -> "Barack Obama"`. Để biến chuỗi liên kết vô nghĩa này thành bộ 3 từ vựng `(Head Label, Relation Label, Tail Label)`, ta phải Join dữ liệu.
- **Tiếp cận MapReduce:** Hệ thống đưa dữ liệu đi qua cụm **Apache Spark**.
  - **Quá trình Map (Ánh xạ):** Spark đọc song song các file khổng lồ, chia nhỏ ra cho nhiều Worker. Mỗi Worker biến một dòng Triplets thô thành cặp Key-Value định tuyến. Ví dụ: Phát xạ `Key=Q76`, `Value={Relation: P31, Tail: Q30}`. Tại nhánh khác, phát xạ `Key=Q76`, `Value={Label: "Barack Obama"}`.
  - **Quá trình Shuffle & Reduce (Gom nhóm & Rút gọn):** Các Worker trộn dữ liệu lại (Shuffle). Những Dữ liệu có chung Key `Q76` được gom về chung một mối. Tại đây, hàm Reduce lập mảnh ghép "Barack Obama" gán ngược vào mạng lưới Triplets.
- **Biến đổi tương tự với Relation (P-Code) và Tail Node.**
- **Kết quả:** Ta thu được một file Parquet sạch chứa các tập `(Q-Head_ID, "Head String", P-Relation_ID, "Relation String", Q-Tail_ID, "Tail String")` hoàn chỉnh ngữ nghĩa.

### Bước 2 & Bước 3: Sinh Vector Embeddings (Heavy GPU Task)
- **Tập tin:** `2_vectorize_embeddings.py` (Cho Nodes) & `4_embed_relations.py` (Cho Quan hệ).
- **Hoạt động:** Lấy các cột chuỗi (Strings) vừa Join được ở MapReduce, bơm vào Local LLM Embedding (`nomic-embed-text`).
- **Bản chất quá trình:** Biểu diễn từ ngữ (VD: "Barack Obama") thành một ma trận Vector toán học (Ví dụ gồm 768 chiều nhị phân).
- **Mục đích sinh Vector:** Sau này khi User gõ "Ông Ô-ba-ma" (bị sai chính tả hoặc khác đại từ), máy tính không so khớp chuỗi chữ cái (Exact Match) được, nhưng Vector toán học của chúng khi chiếu lên không gian cosine (Cosine Similarity) lại nằm sát rạt nhau. Điều này tạo nên Semantic Search.

### Bước 4: Lưu Trữ Phân Tách Kép (Dual-Storage Upload)
- **Tập tin:** `3_load_to_db.py`
- Dữ liệu ở quá trình trên được tách làm hai nửa lưu trữ song song:
  - **Milvus (VectorDB):** Lưu trữ các Vector nhúng. Database này tối ưu hóa thuật toán HNSW / IVF_FLAT để "Rà quét ma trận" cực nhanh.
  - **Neo4j (GraphDB):** Insert trực tiếp các Nodes Q-Id và chằng chịt các quan hệ P-Id thành đồ thị. Database này không lưu Vector mà tập trung lập chỉ mục cấu trúc nhánh (Topology) để "Chạy xe dọc theo các đường nối".

---

## PHẦN 3: HỆ THỐNG TRUY VẤN SimGRAG (RETRIEVE PIPELINE)

Khi hệ thống hoạt động (`test_rag.py`) và nhận một Nhận định (Claim) từ người dùng (Vd: *"Barack Obama was the 44th president of the United States."*), luồng Retrieval (truy xuất chắt lọc thông tin) đi qua 4 giai đoạn cực ký tinh vi.

### Giai đoạn 3.1: Đóng Graph Nháp (LLM Rewriter)
- **Cơ chế:** Dùng Zero/Few-shot LLM cắt câu Claim thành mảnh Graph.
- Thuật toán LLM tự động xuất ra bộ 3 nháp (Draft Triplets): `('Barack Obama', 'was the 44th president of', 'United States')`. Lưu ý: Đây là những từ vựng mà LLM "tưởng tượng", hoàn toàn chưa biết chúng có tồn tại hay ID là gì trong Database.

### Giai đoạn 3.2: Truy Hồi Mỏ Neo (Semantic Search - Milvus)
- Không phải mang cả cái Graph Nháp ném lên Neo4j tìm. Máy lấy ra khối thực thể chính (Head/Tail - VD: "Barack Obama").
- Nén Text này thành Vector (giống hệt Bước 2 ở ETL).
- **Query Milvus:** Bắn Vector này vào Milvus Database. Milvus áp dụng phép tính Cosine / L2 distance quét qua 5 triệu Vector và trả về Top K các Node có cự ly gần nhất.
- **Kết quả:** Ta "bắt" được cái neo ID trỏ vào Graph: `Q76` (Đích thực là Barack Obama trên đồ thị gốc). Gọi đây là **Anchor Nodes**.

### Giai đoạn 3.3: Thuật Toán Truy Vết Hình Thái học (DFS Isomorphism Retrieve - Neo4j)
- Trọng tâm lõi của SimGRAG nằm ở đây (`retriever.py`). Khoan thẳng từ Anchor Node `Q76` xuống đáy Database Neo4j bằng thuật toán **DFS (Depth-First Search - Tìm kiếm theo chiều sâu)**.
- **Cơ chế Semantic Edge Matching (Khớp cạnh ngữ nghĩa):** Đứng ở đỉnh `Q76`, có thể có hàng ngàn con đường rẽ nhánh (Relations). Hệ thống không đi mù, mà nó tính toán Dot Product (Nhân vô hướng) giữa Vector góc của "Quan hệ nháp" (Từ khóa `'was the 44th president of'`) so với các Vector của "Quan hệ thực" (VD: `P31: position held`).
- Nếu điểm số > Ngưỡng (Threshold), DFS bám theo cạnh đó tiến xuống đuôi.
- Khi chạm đuôi, DFS kiểm tra tính Isomorphism (Đồng Cấu): Nếu Đỉnh đuôi thu được từ Neo4j trùng khớp ngữ nghĩa (Semantic match) với Đỉnh đuôi Nháp của LLM (`'United States'`), thì nhánh đồ thị đó hợp lệ.
- Hành động này nhổ rễ cắn đứt các nhánh Nhiễu và Ảo giác đi.

### Giai đoạn 3.4: Rút Trích Context và LLM Quyết Định (Fact Generator)
- Mảng đồ thị thu nhỏ tìm được từ Neo4j được bứt ra, dịch về Text thuần (Ví dụ: `Barack Obama --> holds position --> President of the United States`).
- Cuối cùng, LLM nhận System Prompt: `Dựa vào bằng chứng: [Subgraph Text] ... Hãy xác minh câu nói [Claim gốc]`.
- LLM Output ra: **`SUPPORTED`** (Đúng thực tế), **`REFUTED`** (Sai / Bị bác bỏ) hoặc **`NOT ENOUGH EVIDENCE`** (Không tra ra đường đi DFS thỏa mãn).

---

## TỔNG KẾT
Nhờ việc bẻ đôi kiến trúc ra làm RAG Vector (chuyên xử lý Ngữ nghĩa tự nhiên - Milvus) và Graph Traversal (chuyên kiểm tra Sự logic về kết cấu tri thức - Neo4j), mô hình SimGRAG chống lại được hội chứng hoang tưởng kiến thức (Hallucination) từ LLM một cách hiệu quả, và không bị quá tải so với việc nhồi nhét cả trăm ngàn chữ vào Context Window.