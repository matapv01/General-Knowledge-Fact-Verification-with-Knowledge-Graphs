from fastapi import FastAPI
from app.api.endpoints import router

def create_app() -> FastAPI:
    app = FastAPI(title="SimGRAG Demo API", version="1.0.0")
    
    # Đăng ký các endpoints
    app.include_router(router, prefix="/api/v1")
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # uv run BE/app/main.py
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
