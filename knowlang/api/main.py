import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from knowlang.api.base import ApiModelRegistry
from knowlang.api.chat.router import router as chat_router

# Initialize FastAPI app
app = FastAPI(
    title="KnowLang FastAPI",
    description="A FastAPI server for the KnowLang project, providing endpoints for all KnowLang features.",
    version="1.0.0",
)

# Custom OpenAPI schema generation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="KnowLang API",
        version="1.0.0",
        description="A FastAPI server for the KnowLang project, providing endpoints for all KnowLang features.",
        routes=app.routes,
    )
    
    # Add our models to components/schemas
    openapi_schema["components"]["schemas"].update(ApiModelRegistry.get_all_schemas())
    
    app.openapi_schema = openapi_schema

    return app.openapi_schema

app.openapi = custom_openapi

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
router_prefix = "/api/v1"
app.include_router(chat_router, prefix=router_prefix, tags=["chat"])

if __name__ == "__main__":
    # Run the server if this file is executed directly
    uvicorn.run("knowlang.api.main:app", host="0.0.0.0", port=8080, reload=True)
