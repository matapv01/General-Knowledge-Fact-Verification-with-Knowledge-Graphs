from fastapi import APIRouter, HTTPException, Depends
from app.domain.schemas import QueryRequest, QueryResponse
from app.infrastructure.simgrag_adapter import SimGRAGAdapter
from app.use_cases.qa_usecase import QAUseCase

router = APIRouter()

# Dependency Injection
def get_qa_usecase():
    adapter = SimGRAGAdapter()
    return QAUseCase(simgrag_adapter=adapter)

@router.post("/query", response_model=QueryResponse)
def query_kg(request: QueryRequest, uc: QAUseCase = Depends(get_qa_usecase)):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return uc.execute(request.query)

@router.get("/health")
def health_check():
    return {"status": "ok", "message": "SimGRAG Backend is running"}
