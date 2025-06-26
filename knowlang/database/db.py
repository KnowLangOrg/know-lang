from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, String, select
from typing import Dict, List

from knowlang.assets.config import DatabaseConfig

Base = declarative_base()

DOMAIN_TABLE_NAME = 'domains'
ASSET_TABLE_NAME = 'assets'
ASSET_CHUNK_TABLE_NAME = 'asset_chunks'

class DomainManagerOrm(Base):
    __tablename__ = DOMAIN_TABLE_NAME
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    meta = Column(String, nullable=True)
    assets = relationship(
        "GenericAssetOrm",
        back_populates="domain",
        cascade="all, delete-orphan"
    )

class GenericAssetOrm(Base):
    __tablename__ = ASSET_TABLE_NAME
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    domain_id = Column(String, ForeignKey(f'{DOMAIN_TABLE_NAME}.id'), nullable=False)
    asset_hash = Column(String, nullable=True)
    meta = Column(String, nullable=True)
    asset_chunks = relationship(
        "GenericAssetChunkOrm",
        back_populates="asset",
        cascade="all, delete-orphan"
    )
    domain = relationship(
        "DomainManagerOrm",
        back_populates="assets",
        lazy= "joined",
    )

class GenericAssetChunkOrm(Base):
    __tablename__ = ASSET_CHUNK_TABLE_NAME

    id = Column(String, primary_key=True, index=True)
    asset_id = Column(String, ForeignKey(f'{ASSET_TABLE_NAME}.id'), nullable=False)
    meta = Column(String, nullable=True)
    asset = relationship(
        "GenericAssetOrm",
        back_populates="asset_chunks",
        lazy="joined",
    )


class KnowledgeSqlDatabase:
    def __init__(self, config: DatabaseConfig):
        self.engine = create_async_engine(config.connection_url)
        self.AsyncSession = async_sessionmaker(bind=self.engine)

    async def create_schema(self):
        """Create the database schema if it does not exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def upsert_assets(self, assets: List[GenericAssetChunkOrm]):
        """Index a new asset into the database."""
        async with self.AsyncSession() as session:
            try:
                for asset in assets:
                    session.merge(asset)  # Use merge to handle upsert
                await session.commit()
            except SQLAlchemyError as e:
                await session.rollback()
                raise e

    async def get_asset_hash(self, asset_ids: List[str]) -> Dict[str, str]:
        """Retrieve assets from the database."""
        async with self.AsyncSession() as session:
            result =  await session.execute(
                select(GenericAssetOrm.id, GenericAssetOrm.asset_hash).where(GenericAssetOrm.id.in_(asset_ids))
            )
        
        return {row[0]: row[1] for row in result.fetchall()}

    async def get_chunks_given_assets(self, asset_ids: List[str]) -> List[GenericAssetChunkOrm]:
        """Retrieve asset chunks for a given asset."""
        async with self.AsyncSession() as session:
            result = await session.execute(
                select(GenericAssetChunkOrm).where(GenericAssetChunkOrm.asset_id.in_(asset_ids))
            )
            return result.scalars().all()