import os
from dotenv import load_dotenv
from pydantic import BaseModel

class Config(BaseModel):
    LLM_API_KEY: str
    LLM_MODEL: str
    LLM_BASE_URL: str
    TEMPERATURE: float = 0.60
    TOP_P: float = 0.95
    MAX_TOKENS: int = 16384
    RETRIEVER_TOP_K: int = 5
    REWRITE_SHOT: int = 0
    ANSWER_SHOT: int = 0

def load_config() -> Config:
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))
    load_dotenv(env_path)
    
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    default_model = "gpt-3.5-turbo" if api_key != "ollama" else "llama3"
    model = os.getenv("OPENAI_MODEL", default_model)
    
    default_base_url = "https://api.openai.com/v1" if api_key != "ollama" else "http://localhost:11434/v1"
    base_url = os.getenv("OPENAI_BASE_URL", default_base_url)

    return Config(
        LLM_API_KEY=api_key,
        LLM_MODEL=model,
        LLM_BASE_URL=base_url
    )

settings = load_config()
