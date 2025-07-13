from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, String, select, tuple_
from typing import Dict, List
from contextlib import asynccontextmanager

from knowlang.utils.fancy_log import FancyLogger
from knowlang.database.config import DatabaseConfig

LOG = FancyLogger(__name__)

Base = declarative_base()

DOMAIN_TABLE_NAME = "domains"
ASSET_TABLE_NAME = "assets"
ASSET_CHUNK_TABLE_NAME = "asset_chunks"


class DomainManagerOrm(Base):
    __tablename__ = DOMAIN_TABLE_NAME
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    meta = Column(String, nullable=True)
    assets = relationship(
        "GenericAssetOrm",
        back_populates="domain",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class GenericAssetOrm(Base):
    __tablename__ = ASSET_TABLE_NAME
    # Since ids are relative path, so we use a composite primary key to make sure it's unique across domain.
    id = Column(String, primary_key=True, index=True)
    domain_id = Column(
        String, ForeignKey(f"{DOMAIN_TABLE_NAME}.id"), nullable=False, primary_key=True
    )
    name = Column(String, index=True)
    asset_hash = Column(String, nullable=True)
    meta = Column(String, nullable=True)
    asset_chunks = relationship(
        "GenericAssetChunkOrm",
        back_populates="asset",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    domain = relationship(
        "DomainManagerOrm",
        back_populates="assets",
        lazy="selectin",
    )


class GenericAssetChunkOrm(Base):
    __tablename__ = ASSET_CHUNK_TABLE_NAME

    id = Column(String, primary_key=True, index=True)
    asset_id = Column(
        String, ForeignKey(f"{ASSET_TABLE_NAME}.id"), nullable=False, primary_key=True
    )
    meta = Column(String, nullable=True)
    asset = relationship(
        "GenericAssetOrm",
        back_populates="asset_chunks",
        lazy="selectin",
    )


class KnowledgeSqlDatabase:
    def __init__(self, config: DatabaseConfig):
        self.engine = create_async_engine(config.connection_url)
        self.AsyncSession = async_sessionmaker(bind=self.engine)

    async def create_schema(self):
        """Create the database schema if it does not exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def get_session(self, auto_commit: bool = True):
        """Context manager to get a database session."""
        async with self.AsyncSession() as session:
            try:
                yield session
                if auto_commit:
                    await session.commit()
            except SQLAlchemyError as e:
                import traceback

                await session.rollback()
                LOG.error(traceback.format_exc())
                LOG.error(f"Database operation failed: {e}")
                raise e

    async def upsert_domains(
        self, session: AsyncSession, domains: List[DomainManagerOrm]
    ):
        """Index a new domain into the database."""
        for domain in domains:
            await session.merge(domain)

    async def upsert_assets(self, session: AsyncSession, assets: List[GenericAssetOrm]):
        """Index a new asset into the database."""
        if not assets:
            LOG.debug("No assets to upsert")
            return

        # 1. Cascade delete existing assets before upsert
        stale_assets = await session.execute(
            select(GenericAssetOrm).where(
                tuple_(GenericAssetOrm.id, GenericAssetOrm.domain_id).in_(
                    [(asset.id, asset.domain_id) for asset in assets]
                )
            )
        )
        for asset in stale_assets.scalars().all():
            await session.delete(asset)
        await session.flush()

        # 2. Upsert assets
        session.add_all(assets)

    async def index_asset_chunks(
        self, session: AsyncSession, asset_chunks: List[GenericAssetChunkOrm]
    ):
        """Index new asset chunks into the database."""

        # We don't upsert assset chunks, since the cascade delete on asset will handle it.
        session.add_all(asset_chunks)

    async def get_asset_hash(
        self, session: AsyncSession, assets: List[GenericAssetOrm]
    ) -> Dict[str, str]:
        """Retrieve assets from the database."""
        target_keys = [(a.id, a.domain_id) for a in assets]

        result = await session.execute(
            select(GenericAssetOrm.id, GenericAssetOrm.asset_hash).where(
                tuple_(GenericAssetOrm.id, GenericAssetOrm.domain_id).in_(target_keys)
            )
        )

        return {row[0]: row[1] for row in result.fetchall()}

    async def get_chunks_given_assets(
        self, session: AsyncSession, asset_ids: List[str]
    ) -> List[GenericAssetChunkOrm]:
        """Retrieve asset chunks for a given asset."""
        result = await session.execute(
            select(GenericAssetChunkOrm).where(
                GenericAssetChunkOrm.asset_id.in_(asset_ids)
            )
        )
        return result.scalars().all()

    async def get_all_asset_ids_for_domain(
        self, session: AsyncSession, domain_id: str
    ) -> List[str]:
        """Get all asset IDs currently stored for a domain."""
        result = await session.execute(
            select(GenericAssetOrm.id).where(GenericAssetOrm.domain_id == domain_id)
        )
        return [row[0] for row in result.fetchall()]

    async def delete_assets_by_ids(
        self, session: AsyncSession, asset_ids: List[str]
    ) -> None:
        """Delete assets by their IDs. Returns count of deleted assets."""
        if not asset_ids:
            return 0

        target_assets = await session.execute(
            select(GenericAssetOrm).where(GenericAssetOrm.id.in_(asset_ids))
        )

        for asset in target_assets.scalars().all():
            await session.delete(asset)
        LOG.debug(
            f"Deleted {len(target_assets.scalars().all())} assets with IDs: {asset_ids}"
        )
