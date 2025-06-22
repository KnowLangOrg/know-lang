from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel import SQLModel
from typing import List

from knowlang.assets.config import DatabaseConfig
from knowlang.assets.models import DomainManagerData, GenericAssetData, GenericAssetChunkData


class KnowledgeSqlDatabase:
    def __init__(self, config: DatabaseConfig):
        self.engine = create_async_engine(config.connection_url)
        self.AsyncSession = async_sessionmaker(bind=self.engine)
        SQLModel.metadata.create_all(self.engine)
    
    async def index_assets(self, assets: List[GenericAssetData]):
        """Index a new asset into the database."""
        async with self.AsyncSession() as session:
            try:
                session.add_all(assets)
                await session.commit()
            except SQLAlchemyError as e:
                await session.rollback()
                raise e

