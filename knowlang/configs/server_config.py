from pydantic import Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    port: int = Field(default=8080, description="Port on which the server will run")
