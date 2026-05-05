from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    rewrite_output: Optional[str] = None
    evidences: Optional[List[str]] = None
    answer: str
    error: Optional[str] = None
