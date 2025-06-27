from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, String, delete, select
from sqlalchemy.dialects.sqlite import insert
from typing import Dict, List

from knowlang.assets.config import DatabaseConfig
from knowlang.utils.fancy_log import FancyLogger

LOG = FancyLogger(__name__)

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
    
    async def upsert_domains(self, domains: List[DomainManagerOrm]):
        """Index a new domain into the database."""
        async with self.AsyncSession() as session:
            try:
                for domain in domains:
                    await session.merge(domain)
                await session.commit()
            except SQLAlchemyError as e:
                await session.rollback()
                raise e
    
    async def upsert_assets(self, assets: List[GenericAssetOrm]):
        """Index a new asset into the database."""
        if not assets:
            LOG.debug("No assets to upsert")
            return
            
        LOG.debug(f"Starting upsert for {len(assets)} assets")
        
        async with self.AsyncSession() as session:
            try:
                # Convert ORM objects to dictionaries
                asset_dict_list = []
                for asset in assets:
                    asset_dict = {
                        'id': asset.id,
                        'name': asset.name,
                        'domain_id': asset.domain_id,
                        'asset_hash': asset.asset_hash,
                        'meta': asset.meta
                    }
                    asset_dict_list.append(asset_dict)
                
                # Use SQLite's ON CONFLICT for upsert
                stmt = insert(GenericAssetOrm).values(asset_dict_list)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['id'],
                    set_=dict(
                        name=stmt.excluded.name,
                        domain_id=stmt.excluded.domain_id,
                        asset_hash=stmt.excluded.asset_hash,
                        meta=stmt.excluded.meta
                    )
                )

                await session.execute(stmt)
                await session.commit()
                
            except SQLAlchemyError as e:
                await session.rollback()
                LOG.error(f"Failed to upsert assets: {e}")
                raise e
    
    async def index_asset_chunks(self, asset_chunks: List[GenericAssetChunkOrm]):
        """Index new asset chunks into the database."""

        # We don't upsert assset chunks, since the cascade delete on asset will handle it.
        async with self.AsyncSession() as session:
            try:
                session.add_all(asset_chunks)
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
    
    async def get_all_asset_ids_for_domain(self, domain_id: str) -> List[str]:
        """Get all asset IDs currently stored for a domain."""
        async with self.AsyncSession() as session:
            result = await session.execute(
                select(GenericAssetOrm.id).where(GenericAssetOrm.domain_id == domain_id)
            )
            return [row[0] for row in result.fetchall()]
    
    async def delete_assets_by_ids(self, asset_ids: List[str]) -> None:
        """Delete assets by their IDs. Returns count of deleted assets."""
        if not asset_ids:
            return 0
            
        async with self.AsyncSession() as session:
            try:
                # Delete assets (chunks will be deleted via cascade)
                result = await session.execute(
                    delete(GenericAssetOrm).where(GenericAssetOrm.id.in_(asset_ids))
                )
                deleted_count = result.rowcount
                await session.commit()
                LOG.debug(f"Deleted {deleted_count} assets with IDs: {asset_ids}")
            except SQLAlchemyError as e:
                await session.rollback()
                LOG.error(f"Failed to delete assets with IDs {asset_ids}: {e}")
                raise e