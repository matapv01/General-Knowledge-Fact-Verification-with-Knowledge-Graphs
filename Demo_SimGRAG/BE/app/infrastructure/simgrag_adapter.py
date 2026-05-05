import sys
import os

# Liên kết với source code Baseline/SimGRAG
simgrag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'Baseline', 'with_evidence', 'SimGRAG'))
sys.path.insert(0, simgrag_path)

from src.llm import LLM
from app.core.config import settings

class SimGRAGAdapter:
    def __init__(self):
        self.mock_configs = {
            "llm": {
                "base_url": settings.LLM_BASE_URL,
                "api_key": settings.LLM_API_KEY,
                "model": settings.LLM_MODEL,
                "temperature": settings.TEMPERATURE,
                "top_p": settings.TOP_P,
                "max_tokens": settings.MAX_TOKENS
            }
        }
        self.llm = LLM(self.mock_configs)

    def chat_llm(self, prompt: str) -> str:
        return self.llm.chat(prompt)

    # Trong tương lai sẽ có hàm query qua retriever/indexer
    # def retrieve(self, query: str):
    #     pass
