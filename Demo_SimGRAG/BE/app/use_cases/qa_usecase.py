from app.infrastructure.simgrag_adapter import SimGRAGAdapter
from app.infrastructure.milvus_adapter import MilvusAdapter
from app.infrastructure.neo4j_adapter import Neo4jAdapter
from app.infrastructure.ollama_adapter import OllamaEmbeddingAdapter
from app.infrastructure.simgrag_retriever import SimgragBaselineRetriever
from app.domain.schemas import QueryResponse
import pandas as pd
import os
import sys

# Chèn đường dẫn tới source code gốc của SimGRAG để tái sử dụng Prompts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Baseline", "with_evidence", "SimGRAG")))
try:
    from prompts import answer_FactKG
except ImportError:
    answer_FactKG = None

class QAUseCase:
    def __init__(self, simgrag_adapter: SimGRAGAdapter):
        self.llm_adapter = simgrag_adapter
        self.embed_adapter = OllamaEmbeddingAdapter()
        self.vector_db = MilvusAdapter()
        self.graph_db = Neo4jAdapter()
        
        # Load bảng dịch tên Mối Quan Hệ (P-IDs thành chữ) cho tự nhiên
        self.relations_map = self._load_relations()
        self.simgrag_retriever = SimgragBaselineRetriever(self.vector_db, self.graph_db, self.relations_map)

    def _load_relations(self):
        try:
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "relations.parquet"))
            df = pd.read_parquet(path)
            return dict(zip(df["relation_id"], df["relation_name"]))
        except Exception:
            return {}

    def execute(self, query: str) -> QueryResponse:
        try:
            print(f"1. RAG - Đang dùng LLM dịch Nhận định sang Graph Nháp (Rewrite)...")
            from prompts import rewrite_FactKG
            rewrite_prompt = rewrite_FactKG.get(query, shot=12)
            rewrite_llm_output = self.llm_adapter.chat_llm(rewrite_prompt)
            print(f"   [LLM Draft Graph]:\n{rewrite_llm_output}")
            
            # Parse chuỗi LLM thành List of Tuples
            import re
            import ast
            query_graph = []
            match = re.search(r'\{.*\}', rewrite_llm_output, re.DOTALL)
            if match:
                try:
                    parsed = ast.literal_eval(match.group(0))
                    query_graph = parsed.get("graph", [])
                except:
                    pass

            if not query_graph:
                 raise Exception("LLM trích xuất Graph Nháp thất bại.")

            print("2. RAG - Đang tìm kiếm Đồ thị đồng cấu (Isomorphism) trên Milvus+Neo4j (SimGRAG)...")
            simgrag_evidences = self.simgrag_retriever.retrieve_subgraph(query_graph)
            
            # Format về String để truyền vào prompt custom hoặc show frontend
            evidence_texts = []
            for ev_list in simgrag_evidences:
                for head, rel, tail in ev_list:
                    evidence_texts.append(f"[{head}] - {rel} -> [{tail}]")

            if not evidence_texts:
                context_str = "Không tìm thấy thông tin đồ thị tri thức nào phù hợp."
            else:
                context_str = "\n".join(f"- {ev}" for ev in evidence_texts)
            
            print("3. RAG - Đang đóng gói thông tin và gửi qua LLM (Nvidia/Local)...")
            
            if answer_FactKG is not None:
                print(">>> Sử dụng Prompt gốc từ SimGRAG (prompts.answer_FactKG) <<<")
                prompt = answer_FactKG.get(query, simgrag_evidences, shot=12)
            else:
                print(">>> Sử dụng Prompt Custom (Không tìm thấy thư mục SimGRAG) <<<")
                prompt = (
                    "Bạn là một chuyên gia Fact Verification (Kiểm chứng sự thật) dựa trên Đồ thị Tri thức (Knowledge Graph).\n"
                    "Nhiệm vụ của bạn là đánh giá tính đúng đắn của 'Nhận định (Claim)' được cung cấp, DỰA HOÀN TOÀN vào các chứng cứ (Evidence) trích xuất từ database dưới đây.\n"
                    "Hãy phân tích và đưa ra 1 trong 3 kết luận cuối cùng:\n"
                    "- SUPPORTED: (Nếu chứng cứ ủng hộ nhận định)\n"
                    "- REFUTED: (Nếu chứng cứ bác bỏ hoặc đi ngược lại nhận định)\n"
                    "- NOT ENOUGH EVIDENCE: (Nếu chứng cứ không chứa đủ thông tin để kiểm chứng)\n\n"
                    "Hãy giải thích quá trình suy luận của bạn một cách ngắn gọn trước khi đưa ra kết luận.\n\n"
                    f"### CHỨNG CỨ (EVIDENCE TỪ KNOWLEDGE GRAPH):\n{context_str}\n\n"
                    f"### NHẬN ĐỊNH CẦN KIỂM CHỨNG (CLAIM):\n{query}\n\n"
                    "### PHÂN TÍCH VÀ KẾT LUẬN CỦA BẠN:"
                )
            
            answer = self.llm_adapter.chat_llm(prompt)
            print("4. RAG - LLM đã trả về câu trả lời!")
            
            return QueryResponse(
                query=query,
                evidences=evidence_texts,
                answer=answer
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            return QueryResponse(
                query=query,
                answer="Lỗi xảy ra trong quá trình xử lý luồng RAG (Milvus/Neo4j).",
                error=str(e)
            )
