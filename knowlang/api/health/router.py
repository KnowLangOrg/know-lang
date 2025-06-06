from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/health")
async def ping():
    return JSONResponse(content={"status": "healthy"}, status_code=200)
