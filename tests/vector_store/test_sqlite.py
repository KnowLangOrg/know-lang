from __future__ import annotations

import json
import json
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy
from sqlalchemy.orm import sessionmaker

from knowlang.configs import AppConfig, DBConfig, EmbeddingConfig
from knowlang.vector_stores.base import (
    SearchResult,
    # VectorStoreError, # Not used yet with new structure
    VectorStoreInitError,
)
from knowlang.vector_stores.sqlite import SqliteVectorStore, VectorDocumentModel

# Default values for testing
TEST_DB_PATH = "sqlite:///test.db"  # Updated to be a SQLAlchemy DSN
TEST_TABLE_NAME = "test_vector_table"
TEST_EMBEDDING_DIM = 768  # More realistic dimension
TEST_CONTENT_FIELD = "document_content"
DEFAULT_SIMILARITY_METRIC = "cosine"


@pytest.fixture
def mock_app_config():
    """Fixture for a mock AppConfig configured for SQLite with new constants."""
    return AppConfig(
        db=DBConfig(
            provider="sqlite",
            path=TEST_DB_PATH,
            collection_name=TEST_TABLE_NAME,  # Updated constant
            content_field=TEST_CONTENT_FIELD,
            similarity_metric=DEFAULT_SIMILARITY_METRIC,
            # Additional SQLAlchemy specific fields if any, e.g. use_memory_db for tests
        ),
        embedding=EmbeddingConfig(
            provider="fake",
            dimension=TEST_EMBEDDING_DIM,
        ),
    )


@pytest.fixture
def mock_sqlalchemy_engine(monkeypatch):
    """Mocks sqlalchemy.create_engine and the engine, session factory, and session."""
    mock_engine_instance = MagicMock(spec=sqlalchemy.engine.Engine)
    mock_connection = MagicMock(spec=sqlalchemy.engine.Connection)
    # Make connect() return a context manager
    mock_connect_context_manager = MagicMock()
    mock_connect_context_manager.__enter__.return_value = mock_connection
    mock_connect_context_manager.__exit__.return_value = None
    mock_engine_instance.connect.return_value = mock_connect_context_manager

    mock_session_instance = MagicMock(spec=sqlalchemy.orm.Session)
    mock_session_factory = MagicMock(return_value=mock_session_instance)

    # Path to SQLAlchemy's create_engine, assuming it's imported as `import sqlalchemy`
    # or `from sqlalchemy import create_engine` in the module under test.
    # If SqliteVectorStore imports create_engine from sqlalchemy.engine, adjust path.
    # Let's assume `knowlang.vector_stores.sqlite` does `import sqlalchemy`
    mock_create_engine = MagicMock(return_value=mock_engine_instance)
    monkeypatch.setattr("sqlalchemy.create_engine", mock_create_engine)

    # Also mock the sessionmaker used in SqliteVectorStore if it's directly imported
    # Assuming it's `from sqlalchemy.orm import sessionmaker`
    monkeypatch.setattr("sqlalchemy.orm.sessionmaker", MagicMock(return_value=mock_session_factory))

    return {
        "create_engine": mock_create_engine,
        "engine": mock_engine_instance,
        "connection": mock_connection,
        "session_factory": mock_session_factory,
        "session": mock_session_instance,
    }


@pytest.fixture
def mock_sqlite_vec(monkeypatch):
    """Mocks the sqlite_vec module and its functions."""
    mock_sv = MagicMock()
    mock_sv.load = MagicMock()
    mock_sv.serialize_float32 = MagicMock(return_value=b"serialized_embedding")
    # If other sqlite_vec functions are used, mock them here too.

    # Patch where sqlite_vec is imported in the SqliteVectorStore module
    # Assuming `from knowlang.vector_stores.sqlite import sqlite_vec` or similar
    monkeypatch.setattr("knowlang.vector_stores.sqlite.sqlite_vec", mock_sv)
    return mock_sv


@pytest.fixture
def sqlite_store_uninitialized(mock_app_config):
    """Returns an SqliteVectorStore instance created with mock_app_config, uninitialized."""
    # The SqliteVectorStore.create_from_config is a classmethod that should
    # correctly instantiate the store using the provided app_config.
    # No direct mocking of __init__ parameters is needed here if create_from_config works.
    store = SqliteVectorStore.create_from_config(app_config=mock_app_config)
    return store


@pytest.fixture
def sqlite_store(
    sqlite_store_uninitialized: SqliteVectorStore,
    mock_sqlalchemy_engine: dict,  # This will trigger its setup
    mock_sqlite_vec: MagicMock,  # This will trigger its setup
):
    """
    Initializes the sqlite_store_uninitialized and returns it.
    Relies on the mocks for sqlalchemy and sqlite_vec to be active.
    """
    sqlite_store_uninitialized.initialize()
    return sqlite_store_uninitialized


class TestSqliteVectorStoreFixtures:
    """Placeholder for tests using the new fixtures."""

    def test_mock_app_config_creation(self, mock_app_config: AppConfig):
        """Test that the mock_app_config fixture creates an AppConfig instance correctly."""
        assert isinstance(mock_app_config, AppConfig)
        assert mock_app_config.db.provider == "sqlite"
        assert mock_app_config.db.path == TEST_DB_PATH
        assert mock_app_config.db.collection_name == TEST_TABLE_NAME
        assert mock_app_config.db.content_field == TEST_CONTENT_FIELD
        assert mock_app_config.embedding.dimension == TEST_EMBEDDING_DIM

    def test_sqlite_store_uninitialized_creation(
        self, sqlite_store_uninitialized: SqliteVectorStore
    ):
        """Test that the sqlite_store_uninitialized fixture creates a store instance."""
        assert isinstance(sqlite_store_uninitialized, SqliteVectorStore)
        assert sqlite_store_uninitialized.db_path == TEST_DB_PATH
        assert sqlite_store_uninitialized.table_name == TEST_TABLE_NAME
        assert sqlite_store_uninitialized.embedding_dim == TEST_EMBEDDING_DIM
        assert sqlite_store_uninitialized.content_field == TEST_CONTENT_FIELD
        # Check that it's not yet initialized (e.g., engine is None)
        assert sqlite_store_uninitialized._engine is None
        assert sqlite_store_uninitialized._session_factory is None

    def test_sqlite_store_initialized_creation_and_initialization(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
        mock_app_config: AppConfig,
    ):
        """
        Test that the sqlite_store fixture initializes the store,
        calling relevant sqlalchemy and sqlite_vec mocks.
        """
        assert isinstance(sqlite_store, SqliteVectorStore)
        assert sqlite_store._engine is not None
        assert sqlite_store._session_factory is not None

        # Check that sqlalchemy.create_engine was called
        mock_sqlalchemy_engine["create_engine"].assert_called_once_with(
            mock_app_config.db.path
        )

        # Check that the engine's connect method was called (indirectly via session or direct use)
        # This depends on how initialize() uses the engine. If it just creates tables
        # and loads extensions, connect might be called by sqlalchemy internally or by sqlite_vec.
        # For now, let's assume initialize sets up the engine and session factory.
        # A more specific check would be for `mock_engine_instance.connect()` if called directly,
        # or for `mock_session_factory()` if a session is created and used.

        # Check that sqlite_vec.load was called on the connection from the engine
        # The actual connection object might be tricky to get if it's hidden inside session.
        # However, if initialize directly uses engine.connect() for sqlite_vec.load:
        mock_engine = mock_sqlalchemy_engine["engine"]
        # The `initialize` method in the SUT should look like:
        # with self._engine.connect() as conn:
        #     sqlite_vec.load(conn.connection) # conn.connection is the raw DBAPI conn
        # So, `mock_engine.connect().__enter__().connection` is what `sqlite_vec.load` gets.

        # This part needs to align with the actual implementation of initialize()
        # For now, let's assume `sqlite_vec.load` is called.
        # We might need to refine this assertion based on the SUT's `initialize()` method.
        # A simple check that load was called:
        mock_sqlite_vec.load.assert_called()

        # Check if tables were created (this implies execute was called on a connection/session)
        # This would involve checking calls on mock_sqlalchemy_engine["session"].execute(...)
        # or mock_sqlalchemy_engine["connection"].execute(...)
        # For example, if using a session:
        mock_session = mock_sqlalchemy_engine["session"]
        # Check for table creation SQL (highly dependent on SUT, placeholder here)
        # Example: mock_session.execute.assert_any_call(sqlalchemy.text("CREATE TABLE..."))
        # This needs to be more specific based on actual SQL generated by initialize.
        # For now, we'll assume initialization implies some execution happened.
        # If initialize uses `metadata.create_all(self._engine)`, then:
        # We'd need to mock the `Base.metadata.create_all` call.
        # This test focuses on fixture setup, detailed interaction tests will follow.
        # assert mock_session.execute.call_count > 0 # Placeholder for table creation checks
        # Instead, we will check Base.metadata.create_all and connection.execute for virtual table

        # 1. Check Base.metadata.create_all
        # Need to patch where `Base` is defined: knowlang.vector_stores.sqlite.Base
        with patch("knowlang.vector_stores.sqlite.Base") as mock_base:
            # Re-initialize the store to capture the call to create_all
            # This is a bit tricky as the `sqlite_store` fixture already calls initialize.
            # For a more isolated test of initialize(), one might call it directly here
            # on sqlite_store_uninitialized and pass mocks.
            # However, let's verify the state after `sqlite_store` fixture's initialization.
            # We need to ensure the mock_base was in place *before* initialize was called.
            # This means this patch should ideally be part of mock_sqlalchemy_engine or a separate fixture.

            # For now, let's assume we can check the call on the mock_base if it was
            # patched *before* sqlite_store called initialize. This requires careful ordering
            # or a different test structure.

            # A simpler approach for THIS test: re-run initialize on an uninitialized store
            # with mocks in place.

            # Let's refine the test structure for this specific check later if needed.
            # For now, assume `sqlite_store` fixture has done its job.
            # The `mock_base` needs to be active when `initialize` is called by `sqlite_store`.
            # This can be achieved by patching it in `mock_sqlalchemy_engine` or make this test
            # call initialize itself.

            # To proceed, we'll assume the patch is active during `sqlite_store`'s `initialize()` call.
            # This would mean `mock_base` needs to be passed into `sqlite_store` or `mock_sqlalchemy_engine`.
            # This test structure is becoming complex.

            # Let's try a direct check on the mocks assuming they were correctly configured by fixtures:
            # The `Base.metadata.create_all` would have been called on `mock_sqlalchemy_engine["engine"]`
            # We need to mock `Base.metadata.create_all` in the SUT's module.

            # Re-evaluating: The `sqlite_store` fixture calls `initialize()`.
            # So, the mocks need to be set up *before* that fixture runs.
            # `mock_sqlalchemy_engine` sets up engine mocks.
            # We need to add mock for `Base.metadata.create_all`.

            # The easiest way is to patch 'knowlang.vector_stores.sqlite.Base' for the duration of this test
            # and then call initialize again, or check the results from the fixture's call.

            # Let's assume the `sqlite_store` fixture has already run `initialize()`.
            # We need to check the mock_sqlalchemy_engine's mocks.
            # The actual `Base.metadata.create_all` is not directly part of `mock_sqlalchemy_engine`'s returns.
            # It's a call that *uses* the mock_engine.

            # The most straightforward way to test this is to patch `Base.metadata.create_all`
            # where it's used (`knowlang.vector_stores.sqlite.Base.metadata.create_all`)
            # and ensure it was called.

            # This test will verify that `initialize` was called by `sqlite_store` by checking its effects.
            # Patching `Base.metadata.create_all` for the scope of this test:
            with patch("knowlang.vector_stores.sqlite.Base.metadata") as mock_sqlalchemy_base_metadata:
                # Because sqlite_store fixture already calls initialize,
                # we need to call it again on a fresh uninitialized store
                # or ensure the patch was active during the fixture's initialization.
                # Let's create a new store and initialize it here for clarity.

                temp_store = SqliteVectorStore.create_from_config(app_config=mock_app_config)
                # Manually set the mocked engine and session factory as initialize would expect them
                temp_store._engine = mock_sqlalchemy_engine["engine"]
                temp_store._Session = mock_sqlalchemy_engine["session_factory"]
                # Mock the _setup_sqlite_vec_extension to prevent its direct call if not already handled by mock_sqlite_vec
                with patch.object(temp_store, "_setup_sqlite_vec_extension"):
                    temp_store.initialize() # Now call initialize

                mock_sqlalchemy_base_metadata.create_all.assert_called_once_with(mock_sqlalchemy_engine["engine"])

        # 2. Check virtual table creation SQL
        # This uses engine.connect() and connection.execute()
        mock_connection = mock_sqlalchemy_engine["connection"]

        # Construct the expected SQL for virtual table creation
        # Note: SqliteVectorStore uses `self.virtual_table` which is `f"{self.table_name}_vec_idx"`
        # and `self.embedding_dim`. These come from `mock_app_config`.
        expected_virtual_table_sql = f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {mock_app_config.db.collection_name}_vec_idx USING vec0(
                        id TEXT PRIMARY KEY,
                        embedding FLOAT[{mock_app_config.embedding.dimension}]
                    )
                """

        # Check if execute was called with a string containing the core part of this SQL.
        # sqlalchemy.text() is used, so we need to check the string passed to it.
        # The mock_connection.execute is called.

        # We need to iterate through calls if there are multiple `execute` calls.
        # Or, if we know the order, check a specific call.
        # `initialize` calls `Base.metadata.create_all` first, then `engine.connect()` for virtual table.

        # The `mock_connection` is from `mock_engine_instance.connect().__enter__()`.
        # Let's check its `execute` and `commit` calls.

        # Find the call to execute that matches the virtual table creation.
        call_found = False
        for call_args in mock_connection.execute.call_args_list:
            args, _ = call_args
            if args and isinstance(args[0], sqlalchemy.sql.elements.TextClause):
                executed_sql = str(args[0]).strip()
                # Normalize whitespace for comparison
                normalized_executed_sql = " ".join(executed_sql.split())
                normalized_expected_sql = " ".join(expected_virtual_table_sql.split())
                if normalized_expected_sql == normalized_executed_sql:
                    call_found = True
                    break
        assert call_found, f"Expected virtual table SQL not executed. Expected:\n{expected_virtual_table_sql}"

        # 3. Check that commit was called on the connection for virtual table
        mock_connection.commit.assert_called_once()


class TestSqliteVectorStoreCreateFromConfig:
    """Tests for the SqliteVectorStore.create_from_config method."""

    def test_create_from_config_success(self, mock_app_config: AppConfig):
        """Test successful creation of SqliteVectorStore from AppConfig."""
        store = SqliteVectorStore.create_from_config(mock_app_config)

        assert isinstance(store, SqliteVectorStore)
        # db_path in store is taken from connection_url in config
        assert store.db_path == mock_app_config.db.connection_url # or .path if validator ensures sync
        assert store.table_name == mock_app_config.db.collection_name
        assert store.embedding_dim == mock_app_config.embedding.dimension
        assert store.content_field == mock_app_config.db.content_field
        assert store.similarity_metric == mock_app_config.db.similarity_metric
        # Also check app_config attribute if it's set
        assert store.app_config == mock_app_config

    def test_create_from_config_missing_path(self, mock_app_config: AppConfig):
        """Test VectorStoreInitError if db.connection_url (derived from db.path) is missing."""
        # Set both to None to be sure, as connection_url is what's directly checked by SUT
        mock_app_config.db.path = None
        mock_app_config.db.connection_url = None

        with pytest.raises(
            VectorStoreInitError, match="Database path not set for SqliteVectorStore."
        ):
            SqliteVectorStore.create_from_config(mock_app_config)


class TestSqliteVectorStoreInitialize:
    """Tests for the SqliteVectorStore.initialize method."""

    @patch("knowlang.vector_stores.sqlite.Base.metadata")
    @patch("knowlang.vector_stores.sqlite.event") # Patch event where SUT uses it
    def test_initialize_success(
        self,
        mock_sut_event_module, # sqlalchemy.event mock used by SUT
        mock_base_metadata,    # knowlang.vector_stores.sqlite.Base.metadata mock
        sqlite_store_uninitialized: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
        # mock_app_config is not directly used but implicitly by sqlite_store_uninitialized
    ):
        """Test successful initialization of the SqliteVectorStore."""
        store = sqlite_store_uninitialized
        mock_engine = mock_sqlalchemy_engine["engine"]
        mock_connection = mock_sqlalchemy_engine["connection"]

        # Ensure create_engine returns our specific mock_engine
        mock_sqlalchemy_engine["create_engine"].return_value = mock_engine

        # Capture the listener registered by _setup_sqlite_vec_extension
        # _setup_sqlite_vec_extension calls: event.listen(self.engine, "connect", self._load_sqlite_vec_listener)
        # So, mock_sut_event_module.listen will be called.
        captured_listeners = {"connect": []}
        def mock_listen_side_effect(target, event_name, listener_func):
            if target == mock_engine and event_name == "connect":
                captured_listeners["connect"].append(listener_func)
            else: # Fallback to default MagicMock behavior if needed elsewhere
                return MagicMock()
        mock_sut_event_module.listen = MagicMock(side_effect=mock_listen_side_effect)

        store.initialize()

        # 1. Assert create_engine was called
        mock_sqlalchemy_engine["create_engine"].assert_called_once_with(store.db_path)
        assert store._engine == mock_engine

        # 2. Assert event.listen was called (indirectly via mock_listen_side_effect)
        # Check that our mock_listen was called trying to register on mock_engine for 'connect'
        assert len(captured_listeners["connect"]) > 0, "Listener for 'connect' event was not registered."

        # 3. Simulate the 'connect' event by calling the captured listener
        actual_listener_from_sut = captured_listeners["connect"][0]

        mock_dbapi_connection = MagicMock(spec=sqlalchemy.engine.interfaces.DBAPIConnection)
        # The SUT's listener uses conn.connection, but it receives a raw DBAPI connection directly.
        # So, we pass mock_dbapi_connection as `conn` to the listener.

        actual_listener_from_sut(mock_dbapi_connection, None) # Second arg (conn_record) is not used by SUT's listener

        mock_sqlite_vec.load.assert_called_once_with(mock_dbapi_connection)
        mock_dbapi_connection.enable_load_extension.assert_any_call(True)
        mock_dbapi_connection.enable_load_extension.assert_any_call(False)

        # 4. Assert Base.metadata.create_all was called
        mock_base_metadata.create_all.assert_called_once_with(mock_engine)

        # 5. Assert virtual table creation SQL
        # This comes from store.engine.connect() which is mock_engine.connect()
        expected_virtual_table_sql = f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {store.virtual_table} USING vec0(
                        id TEXT PRIMARY KEY,
                        embedding FLOAT[{store.embedding_dim}]
                    )
                """
        call_found = False
        # mock_connection is what mock_engine.connect().__enter__() returns
        for call_args in mock_connection.execute.call_args_list:
            args, _ = call_args
            if args and isinstance(args[0], sqlalchemy.sql.elements.TextClause):
                executed_sql = str(args[0]).strip()
                normalized_executed_sql = " ".join(executed_sql.split())
                normalized_expected_sql = " ".join(expected_virtual_table_sql.split())
                if normalized_expected_sql == normalized_executed_sql:
                    call_found = True
                    break
        assert call_found, f"Expected virtual table SQL not executed. Executed calls: {mock_connection.execute.call_args_list}"

        mock_connection.commit.assert_called_once() # From the `with engine.connect()` block

        # 6. Assert store.engine and store.Session (session factory) are set
        assert store._engine is not None
        assert store.Session is not None

    def test_initialize_create_engine_fails(
        self,
        sqlite_store_uninitialized: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test VectorStoreInitError if sqlalchemy.create_engine fails."""
        store = sqlite_store_uninitialized
        error_message = "DB connection failed"
        mock_sqlalchemy_engine["create_engine"].side_effect = sqlalchemy.exc.SQLAlchemyError(error_message)

        with pytest.raises(
            VectorStoreInitError,
            match=f"Failed to initialize SqliteVectorStore: {error_message}",
        ):
            store.initialize()

        assert store._engine is None
        assert store.Session is None # SUT uses self.Session for factory

    @patch("knowlang.vector_stores.sqlite.Base.metadata") # Mock to prevent issues if create_all is reached
    def test_initialize_sqlite_vec_load_fails(
        self,
        mock_base_metadata_unused, # Patched to prevent its execution if error occurs earlier
        sqlite_store_uninitialized: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
    ):
        """Test VectorStoreInitError if sqlite_vec.load fails during a connection event."""
        store = sqlite_store_uninitialized
        mock_engine = mock_sqlalchemy_engine["engine"]

        mock_sqlalchemy_engine["create_engine"].return_value = mock_engine

        sqlite_vec_error_message = "sqlite-vec load error"
        mock_sqlite_vec.load.side_effect = Exception(sqlite_vec_error_message)

        # The SUT's _setup_sqlite_vec_extension uses `@event.listens_for(self.engine, "connect")`
        # This effectively calls `sqlalchemy.event.listen(self.engine, "connect", listener_function)`.
        # The listener_function, when called, will execute `mock_sqlite_vec.load()`.
        # We need to ensure that when `mock_engine.connect()` is invoked (e.g., by `create_all` or
        # the `with store.engine.connect()` block in `initialize`), this listener gets called.

        # Capture the listener that _setup_sqlite_vec_extension registers on mock_engine.
        # We need to ensure `sqlalchemy.event.listen` (as used by SUT) is not fully mocked away,
        # or if it is, that it correctly interacts with our mock_engine's dispatch mechanism.

        # Assuming `knowlang.vector_stores.sqlite.event` is the path to `event` module used by SUT.
        with patch("knowlang.vector_stores.sqlite.event") as mock_sut_event_module:
            actual_listeners_on_mock_engine = []
            def capture_listener_side_effect(target, event_name, listener_func):
                if target == mock_engine and event_name == "connect":
                    actual_listeners_on_mock_engine.append(listener_func)
            mock_sut_event_module.listen = MagicMock(side_effect=capture_listener_side_effect)

            # Now, we modify how mock_engine.connect() works.
            # When its context manager is entered, it should call these captured listeners.
            original_connect_cm = mock_engine.connect.return_value # This is the mock context manager
            original_enter = original_connect_cm.__enter__

            mock_dbapi_connection_for_listener = MagicMock(spec=sqlalchemy.engine.interfaces.DBAPIConnection)
            mock_sqlalchemy_engine["connection"].connection = mock_dbapi_connection_for_listener


            def new_enter_that_triggers_listeners(*args, **kwargs):
                # Call listeners before returning the connection
                for listener in actual_listeners_on_mock_engine:
                    # This call to the SUT's actual listener will trigger the error
                    # because mock_sqlite_vec.load is rigged to fail.
                    # The listener itself should raise VectorStoreInitError.
                    listener(mock_dbapi_connection_for_listener, None)
                return original_enter(*args, **kwargs) # Proceed to return the mock connection

            original_connect_cm.__enter__ = new_enter_that_triggers_listeners

            expected_match = f"Failed to load sqlite-vec extension. Ensure it's installed and accessible. {sqlite_vec_error_message}"
            with pytest.raises(VectorStoreInitError, match=expected_match):
                store.initialize()

            # Restore original __enter__ if other tests use the same mock_engine instance from fixture
            original_connect_cm.__enter__ = original_enter


        # As per previous analysis, if VectorStoreInitError from listener is not caught by
        # the SQLAlchemyError block, engine and Session will not be None.
        # The task implies they should be None, suggesting a robust cleanup.
        # If the SUT does not do this, these assertions will fail and indicate a potential SUT improvement area.
        assert store._engine is None
        assert store.Session is None


class TestSqliteVectorStoreAddDocuments:
    """Tests for the SqliteVectorStore.add_documents method."""

    @pytest.mark.asyncio
    async def test_add_documents_success_with_ids(
        self,
        sqlite_store: SqliteVectorStore, # Initialized store
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
    ):
        """Test successful addition of documents with provided IDs."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        docs = ["doc1 content", "doc2 content"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]] # Assuming embedding_dim=2 for simplicity in test
        metadatas = [{"source": "s1"}, {"source": "s2"}]
        ids = ["id1", "id2"]

        # Mock serialize_float32 to return predictable bytes
        serialized_embedding1 = b"emb1_bytes"
        serialized_embedding2 = b"emb2_bytes"
        mock_sqlite_vec.serialize_float32.side_effect = [serialized_embedding1, serialized_embedding2]

        # Mock session.execute().scalar() which is used to get the rowid (actually primary key id)
        # In the SUT, it does: session.execute(select(VectorDocumentModel.id).where(VectorDocumentModel.id == doc_id)).scalar()
        # This returns the doc_id itself because the virtual table uses the main table's PK.
        mock_session.execute.return_value.scalar.side_effect = ids # Returns id1, then id2

        await store.add_documents(docs, embeddings, metadatas, ids)

        # 1. Assert mock_sqlite_vec.serialize_float32 was called
        assert mock_sqlite_vec.serialize_float32.call_count == len(docs)
        mock_sqlite_vec.serialize_float32.assert_any_call(embeddings[0])
        mock_sqlite_vec.serialize_float32.assert_any_call(embeddings[1])

        # 2. Assert mock_session.add was called with VectorDocumentModel instances
        assert mock_session.add.call_count == len(docs)

        # Check call arguments for session.add
        # First call
        call_args_doc1 = mock_session.add.call_args_list[0][0][0]
        assert isinstance(call_args_doc1, VectorDocumentModel)
        assert call_args_doc1.id == ids[0]
        assert call_args_doc1.content == docs[0] # Assuming content_field in metadata is not used here
        assert call_args_doc1.embedding == serialized_embedding1
        assert call_args_doc1.doc_metadata == json.dumps(metadatas[0])

        # Second call
        call_args_doc2 = mock_session.add.call_args_list[1][0][0]
        assert isinstance(call_args_doc2, VectorDocumentModel)
        assert call_args_doc2.id == ids[1]
        assert call_args_doc2.content == docs[1]
        assert call_args_doc2.embedding == serialized_embedding2
        assert call_args_doc2.doc_metadata == json.dumps(metadatas[1])

        # 3. Assert mock_session.flush was called (once per document in current SUT loop)
        assert mock_session.flush.call_count == len(docs)

        # 4. Assert mock_session.execute was called for select(VectorDocumentModel.id)
        # This is called by the execute().scalar() mock setup.
        # The actual `select` statement object might be hard to match directly.
        # We can check that execute was called and its scalar attribute was accessed.
        # The side_effect on scalar already confirms it was called as expected.
        # We can check the number of execute calls: 2 for select, 2 for virtual table insert

        # 5. Assert mock_session.execute for virtual table insert
        virtual_table_insert_calls = 0
        expected_sql_virtual_insert_pattern = f"INSERT INTO {store.virtual_table} (id, embedding) VALUES (:id, :embedding)"

        for call in mock_session.execute.call_args_list:
            args, kwargs_execute = call
            if args and isinstance(args[0], sqlalchemy.sql.elements.TextClause):
                sql_text = str(args[0]).strip()
                if expected_sql_virtual_insert_pattern in sql_text:
                    virtual_table_insert_calls += 1
                    # Check parameters for each insert
                    if kwargs_execute["params"]["id"] == ids[0]: # scalar() returned id1
                        assert kwargs_execute["params"]["embedding"] == serialized_embedding1
                    elif kwargs_execute["params"]["id"] == ids[1]: # scalar() returned id2
                        assert kwargs_execute["params"]["embedding"] == serialized_embedding2
        assert virtual_table_insert_calls == len(docs)

        # 6. Assert mock_session.commit was called once at the end
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch("knowlang.vector_stores.sqlite.uuid") # Patch where uuid is used in SUT
    async def test_add_documents_success_auto_generated_ids(
        self,
        mock_uuid,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
    ):
        """Test successful addition of documents with auto-generated UUIDs."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        docs = ["doc1 content"]
        embeddings = [[0.5, 0.6]]
        metadatas = [{"source": "s3"}]

        # Setup mock_uuid
        mock_generated_id = "test-uuid-123"
        mock_uuid.uuid4.return_value = MagicMock(hex=mock_generated_id, __str__=lambda: mock_generated_id)

        serialized_embedding = b"emb_auto_id_bytes"
        mock_sqlite_vec.serialize_float32.return_value = serialized_embedding

        # Mock session.execute().scalar() to return the generated ID
        mock_session.execute.return_value.scalar.return_value = mock_generated_id

        await store.add_documents(docs, embeddings, metadatas, ids=None)

        mock_uuid.uuid4.assert_called_once()

        # Assert session.add with generated ID
        call_args_doc = mock_session.add.call_args_list[0][0][0]
        assert isinstance(call_args_doc, VectorDocumentModel)
        assert call_args_doc.id == mock_generated_id
        assert call_args_doc.content == docs[0]
        assert call_args_doc.embedding == serialized_embedding
        assert call_args_doc.doc_metadata == json.dumps(metadatas[0])

        # Assert virtual table insert with generated ID
        expected_sql_virtual_insert_pattern = f"INSERT INTO {store.virtual_table} (id, embedding) VALUES (:id, :embedding)"
        virtual_insert_called_correctly = False
        for call in mock_session.execute.call_args_list:
            args, kwargs_execute = call
            if args and isinstance(args[0], sqlalchemy.sql.elements.TextClause):
                sql_text = str(args[0]).strip()
                if expected_sql_virtual_insert_pattern in sql_text:
                    assert kwargs_execute["params"]["id"] == mock_generated_id
                    assert kwargs_execute["params"]["embedding"] == serialized_embedding
                    virtual_insert_called_correctly = True
                    break
        assert virtual_insert_called_correctly

        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_documents_content_field_logic(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock, # Needed for serialize_float32
    ):
        """Test content extraction logic (from metadata vs. from document string)."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        content_field_name = store.content_field # e.g., "document_content"

        docs_ref = ["doc ref content1", "doc ref content2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        ids = ["id_cf1", "id_cf2"]

        # Case 1: Content from metadata
        metadata_with_content_field = [{content_field_name: "content from meta1", "other": "meta"}, {"other": "meta2"}]
        docs_content_expected_case1 = ["content from meta1", docs_ref[1]] # 2nd doc has no content_field in meta

        # Reset mocks for clean state if needed, or use different store instances
        mock_session.reset_mock()
        mock_sqlite_vec.reset_mock()
        mock_sqlite_vec.serialize_float32.return_value = b"serialized_emb"
        mock_session.execute.return_value.scalar.side_effect = ids # For simplicity

        await store.add_documents(docs_ref, embeddings, metadata_with_content_field, ids)

        added_doc_models = [call[0][0] for call in mock_session.add.call_args_list]
        assert added_doc_models[0].content == docs_content_expected_case1[0]
        assert added_doc_models[1].content == docs_content_expected_case1[1] # Falls back to docs_ref[1]

        # Case 2: Content from document string (content_field not in metadata)
        mock_session.reset_mock() # Reset for the second part of the test
        mock_sqlite_vec.reset_mock()
        mock_sqlite_vec.serialize_float32.return_value = b"serialized_emb_case2"
        mock_session.execute.return_value.scalar.side_effect = ids

        metadata_without_content_field = [{"other": "meta1"}, {"other_meta": "val2"}]
        docs_content_expected_case2 = [docs_ref[0], docs_ref[1]]

        await store.add_documents(docs_ref, embeddings, metadata_without_content_field, ids)

        added_doc_models_case2 = [call[0][0] for call in mock_session.add.call_args_list]
        assert added_doc_models_case2[0].content == docs_content_expected_case2[0]
        assert added_doc_models_case2[1].content == docs_content_expected_case2[1]

    @pytest.mark.asyncio
    async def test_add_documents_not_initialized(
        self,
        sqlite_store_uninitialized: SqliteVectorStore,
    ):
        """Test VectorStoreError if add_documents is called on an uninitialized store."""
        store = sqlite_store_uninitialized
        with pytest.raises(
            VectorStoreError,
            match="Vector store is not initialized. Call initialize() first.",
        ):
            await store.add_documents([], [], [])

    @pytest.mark.asyncio
    async def test_add_documents_mismatched_lengths(
        self,
        sqlite_store: SqliteVectorStore,
    ):
        """Test ValueError for mismatched input list lengths."""
        store = sqlite_store
        docs1 = ["d1"]
        embed1 = [[0.1]]
        meta1 = [{"s":1}]
        ids1 = ["id1"]

        with pytest.raises(ValueError, match="documents, embeddings, and metadatas lists must have the same length."):
            await store.add_documents(docs1, [embed1[0], [0.2]], meta1, ids1) # embeddings too long

        with pytest.raises(ValueError, match="documents, embeddings, and metadatas lists must have the same length."):
            await store.add_documents(docs1, embed1, [meta1[0], {"s":2}], ids1) # metadatas too long

        with pytest.raises(ValueError, match="If provided, ids list must have the same length as documents."):
            await store.add_documents(docs1, embed1, meta1, [ids1[0], "id2"]) # ids too long

    @pytest.mark.asyncio
    async def test_add_documents_sqlalchemy_error_on_add(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test VectorStoreError if session.add raises SQLAlchemyError."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        db_error_msg = "Simulated DB error on add"
        mock_session.add.side_effect = sqlalchemy.exc.SQLAlchemyError(db_error_msg)

        with pytest.raises(VectorStoreError, match=f"Failed to add documents: {db_error_msg}"):
            await store.add_documents(["d1"], [[0.1]], [{"s":1}], ["id1"])

    @pytest.mark.asyncio
    async def test_add_documents_sqlalchemy_error_on_commit(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock, # Needed for serialization
    ):
        """Test VectorStoreError if session.commit raises SQLAlchemyError."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        db_error_msg = "Simulated DB error on commit"
        mock_session.commit.side_effect = sqlalchemy.exc.SQLAlchemyError(db_error_msg)

        # Need serialize_float32 and scalar for calls before commit
        mock_sqlite_vec.serialize_float32.return_value = b"any_bytes"
        mock_session.execute.return_value.scalar.return_value = "any_id"


        with pytest.raises(VectorStoreError, match=f"Failed to add documents: {db_error_msg}"):
            await store.add_documents(["d1"], [[0.1]], [{"s":1}], ["id1"])

    @pytest.mark.asyncio
    async def test_add_documents_unexpected_error_serialization(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict, # Session is used
        mock_sqlite_vec: MagicMock,
    ):
        """Test VectorStoreError if serialize_float32 raises a non-SQLAlchemyError."""
        store = sqlite_store
        # mock_session = mock_sqlalchemy_engine["session"] # Not strictly needed if error is before session use

        serialization_error_msg = "Serialization boom"
        mock_sqlite_vec.serialize_float32.side_effect = Exception(serialization_error_msg)

        with pytest.raises(
            VectorStoreError,
            match=f"An unexpected error occurred while adding documents: {serialization_error_msg}"
        ):
            await store.add_documents(["d1"], [[0.1]], [{"s":1}], ["id1"])


class TestSqliteVectorStoreQuery:
    """Tests for the SqliteVectorStore.query method."""

    @pytest.mark.asyncio
    async def test_query_success(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
    ):
        """Test successful querying of documents."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        query_embedding = [0.1, 0.2, 0.3]
        top_k = 2
        serialized_query_embedding = b"query_emb_bytes"
        mock_sqlite_vec.serialize_float32.return_value = serialized_query_embedding

        # Sample raw results: (id, content, doc_metadata_json, distance)
        raw_results = [
            ("id1", "content1", json.dumps({"meta": "val1"}), 0.1),
            ("id2", "content2", json.dumps({"meta": "val2"}), 0.2),
        ]
        mock_session.execute.return_value.fetchall.return_value = raw_results

        results = await store.query(query_embedding, top_k=top_k)

        # 1. Assert mock_sqlite_vec.serialize_float32 was called
        mock_sqlite_vec.serialize_float32.assert_called_once_with(query_embedding)

        # 2. Assert mock_session.execute was called with correct SQL and params
        expected_sql_query_pattern = f"""
                    SELECT
                        m.id,
                        m.{store.content_field},
                        m.doc_metadata,
                        v.distance
                    FROM
                        {store.virtual_table} v
                    JOIN
                        {VectorDocumentModel.__tablename__} m ON v.id = m.id
                    WHERE
                        v.embedding MATCH :query_embedding
                    AND
                        v.k = :top_k
                    ORDER BY
                        v.distance ASC
                """

        call_found = False
        for call_args_tuple in mock_session.execute.call_args_list:
            args, kwargs_execute = call_args_tuple # call_args is a tuple itself if using call_args
            if args and isinstance(args[0], sqlalchemy.sql.elements.TextClause):
                executed_sql = str(args[0]).strip()
                normalized_executed_sql = " ".join(executed_sql.split())
                normalized_expected_sql = " ".join(expected_sql_query_pattern.split())

                if normalized_executed_sql == normalized_expected_sql:
                    assert kwargs_execute["params"]["query_embedding"] == serialized_query_embedding
                    assert kwargs_execute["params"]["top_k"] == top_k
                    call_found = True
                    break
        assert call_found, f"Expected SQL query not executed or params incorrect. Executed: {mock_session.execute.call_args_list}"


        # 3. Verify returned List[SearchResult]
        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[0].document == "content1"
        assert results[0].metadata == {"meta": "val1"}
        assert results[0].score == pytest.approx(1.0 - 0.1)

        assert results[1].id == "id2"
        assert results[1].document == "content2"
        assert results[1].metadata == {"meta": "val2"}
        assert results[1].score == pytest.approx(1.0 - 0.2)

    @pytest.mark.asyncio
    async def test_query_success_with_score_threshold(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
    ):
        """Test querying with a score threshold."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        mock_sqlite_vec.serialize_float32.return_value = b"query_emb_bytes"

        raw_results = [
            ("id1", "content1", json.dumps({"meta": "val1"}), 0.1), # score 0.9
            ("id2", "content2", json.dumps({"meta": "val2"}), 0.25),# score 0.75
            ("id3", "content3", json.dumps({"meta": "val3"}), 0.4), # score 0.6
        ]
        mock_session.execute.return_value.fetchall.return_value = raw_results

        score_threshold = 0.7
        results = await store.query([0.1,0.2], top_k=3, score_threshold=score_threshold)

        assert len(results) == 2
        assert results[0].id == "id1" # score 0.9
        assert results[1].id == "id2" # score 0.75
        # id3 with score 0.6 should be filtered out

    @pytest.mark.asyncio
    async def test_query_success_with_filter(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
    ):
        """Test querying with a metadata filter."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        mock_sqlite_vec.serialize_float32.return_value = b"query_emb_bytes"

        raw_results = [
            ("id1", "content1", json.dumps({"meta_key": "target_value", "other": "a"}), 0.1),
            ("id2", "content2", json.dumps({"meta_key": "other_value", "other": "b"}), 0.2),
            ("id3", "content3", json.dumps({"meta_key": "target_value", "other": "c"}), 0.3),
        ]
        mock_session.execute.return_value.fetchall.return_value = raw_results

        query_filter = {"meta_key": "target_value"}
        results = await store.query([0.1,0.2], top_k=3, filter=query_filter)

        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[0].metadata["meta_key"] == "target_value"
        assert results[1].id == "id3"
        assert results[1].metadata["meta_key"] == "target_value"

    @pytest.mark.asyncio
    async def test_query_not_initialized(
        self,
        sqlite_store_uninitialized: SqliteVectorStore,
    ):
        """Test VectorStoreError if query is called on an uninitialized store."""
        store = sqlite_store_uninitialized
        with pytest.raises(
            VectorStoreError,
            match="Vector store is not initialized. Call initialize() first.",
        ):
            await store.query([])

    @pytest.mark.asyncio
    async def test_query_sqlalchemy_error(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock, # For serialize_float32
    ):
        """Test VectorStoreError if session.execute raises SQLAlchemyError."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        mock_sqlite_vec.serialize_float32.return_value = b"query_emb_bytes"
        db_error_msg = "DB query failed"
        mock_session.execute.side_effect = sqlalchemy.exc.SQLAlchemyError(db_error_msg)

        with pytest.raises(VectorStoreError, match=f"Failed to query: {db_error_msg}"):
            await store.query([0.1, 0.2])

    @pytest.mark.asyncio
    async def test_query_struct_error_on_serialize(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlite_vec: MagicMock,
        # mock_sqlalchemy_engine is not directly used if serialization fails first
    ):
        """Test VectorStoreError if serialize_float32 raises struct.error."""
        store = sqlite_store
        import struct # Required for struct.error

        serialization_error_msg = "Packing failed"
        mock_sqlite_vec.serialize_float32.side_effect = struct.error(serialization_error_msg)

        with pytest.raises(
            VectorStoreError,
            match=f"Failed to pack query embedding: {serialization_error_msg}"
        ):
            await store.query([0.1, 0.2])


class TestSqliteVectorStoreDelete:
    """Tests for the SqliteVectorStore.delete method."""

    @pytest.mark.asyncio
    async def test_delete_success(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test successful deletion of documents."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        ids_to_delete = ["id1", "id2"]

        # Mock VectorDocumentModel instances
        mock_doc1 = MagicMock(spec=VectorDocumentModel)
        mock_doc1.id = "id1"
        mock_doc2 = MagicMock(spec=VectorDocumentModel)
        mock_doc2.id = "id2"

        # Mock for the select statement
        # session.execute(select(VectorDocumentModel).where(...)).scalars().all()
        select_execute_mock = MagicMock()
        select_execute_mock.scalars.return_value.all.return_value = [mock_doc1, mock_doc2]

        # Mock for delete statements (virtual and main)
        delete_execute_mock = MagicMock()
        delete_execute_mock.rowcount = 1 # Example rowcount

        # Configure session.execute side_effect based on statement type
        def execute_side_effect(statement, *args, **kwargs):
            if isinstance(statement, sqlalchemy.sql.selectable.Select):
                return select_execute_mock
            elif isinstance(statement, sqlalchemy.sql.dml.Delete) or isinstance(statement, sqlalchemy.sql.elements.TextClause):
                # This covers delete(VectorDocumentModel) and text("DELETE FROM virtual_table")
                return delete_execute_mock
            raise ValueError(f"Unexpected statement type for execute mock: {type(statement)}")

        mock_session.execute.side_effect = execute_side_effect

        await store.delete(ids_to_delete)

        # 1. Assert select was called
        # Check that execute was called with a select statement
        select_call_found = False
        for call in mock_session.execute.call_args_list:
            stmt = call[0][0]
            if isinstance(stmt, sqlalchemy.sql.selectable.Select) and \
               stmt.whereclause is not None: # Further check on whereclause if needed
                select_call_found = True
                # Example detailed check: str(stmt.whereclause) == "vector_documents.id IN (:id_1, :id_2)"
                # This can be fragile. Checking the type is often enough for mocked logic.
                break
        assert select_call_found, "Select statement for documents to delete was not executed."
        select_execute_mock.scalars.assert_called_once()
        select_execute_mock.scalars.return_value.all.assert_called_once()


        # 2. Assert delete from virtual table for each found ID
        # DELETE FROM {store.virtual_table} WHERE id = :id
        virtual_delete_calls = 0
        for call in mock_session.execute.call_args_list:
            stmt = call[0][0]
            params = call[1] if len(call[0]) > 1 else call[1].get("params", {}) if len(call) > 1 else {}
            if isinstance(stmt, sqlalchemy.sql.elements.TextClause) and f"DELETE FROM {store.virtual_table}" in str(stmt):
                virtual_delete_calls +=1
                assert params["id"] in ids_to_delete
        assert virtual_delete_calls == len(ids_to_delete)

        # 3. Assert delete from main table
        # delete(VectorDocumentModel).where(VectorDocumentModel.id.in_(ids_to_delete))
        main_delete_call_found = False
        for call in mock_session.execute.call_args_list:
            stmt = call[0][0]
            if isinstance(stmt, sqlalchemy.sql.dml.Delete) and stmt.table.name == VectorDocumentModel.__tablename__:
                # Further check on whereclause if needed, e.g., involve stmt.whereclause.right.clauses
                main_delete_call_found = True
                break
        assert main_delete_call_found, "Delete statement for main table was not executed."

        # 4. Assert commit was called
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_ids_not_found(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test deletion when specified IDs are not found."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        ids_to_delete = ["id_not_found1", "id_not_found2"]

        # Mock for the select statement to return no documents
        select_execute_mock = MagicMock()
        select_execute_mock.scalars.return_value.all.return_value = []

        # Mock for the main table delete (SUT calls this even if no docs found by select)
        main_delete_execute_mock = MagicMock()
        main_delete_execute_mock.rowcount = 0

        original_execute_side_effect = mock_session.execute.side_effect
        execute_calls = []
        def track_execute_side_effect(statement, *args, **kwargs):
            execute_calls.append({"statement_type": type(statement), "sql": str(statement)})
            if isinstance(statement, sqlalchemy.sql.selectable.Select):
                return select_execute_mock
            if isinstance(statement, sqlalchemy.sql.dml.Delete): # Main table delete
                return main_delete_execute_mock
            # Virtual table delete should not happen
            if isinstance(statement, sqlalchemy.sql.elements.TextClause) and "DELETE FROM" in str(statement):
                pytest.fail("Virtual table delete should not be called when no documents are found.")
            return MagicMock() # Fallback for other execute calls if any

        mock_session.execute.side_effect = track_execute_side_effect

        await store.delete(ids_to_delete)

        # Assert select was called
        assert any(call["statement_type"] == sqlalchemy.sql.selectable.Select for call in execute_calls)
        select_execute_mock.scalars.assert_called_once()
        select_execute_mock.scalars.return_value.all.assert_called_once()

        # Assert main table delete was called (SUT behavior)
        assert any(call["statement_type"] == sqlalchemy.sql.dml.Delete for call in execute_calls)

        # Assert commit IS called (SUT commits the session regardless of whether rows were affected, if no error)
        mock_session.commit.assert_called_once()

        # Restore side_effect if it was specific to this test and other tests use the same session instance from fixture
        mock_session.execute.side_effect = original_execute_side_effect


    @pytest.mark.asyncio
    async def test_delete_no_ids_provided(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test delete operation when no IDs are provided."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"] # To check it's not used

        await store.delete([])

        mock_session.execute.assert_not_called()
        mock_session.commit.assert_not_called()


class TestSqliteVectorStoreGetDocument:
    """Tests for the SqliteVectorStore.get_document method."""

    @pytest.mark.asyncio
    async def test_get_document_success(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test successfully retrieving a document."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        doc_id = "test_doc_id"
        content = "Test document content"
        metadata_dict = {"source": "test_source", "page": 1}
        metadata_json = json.dumps(metadata_dict)

        mock_doc_instance = MagicMock(spec=VectorDocumentModel)
        mock_doc_instance.id = doc_id
        mock_doc_instance.content = content # Assuming store.content_field is 'content'
        mock_doc_instance.doc_metadata = metadata_json

        # Mock for session.execute(...).scalar_one_or_none()
        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_doc_instance

        result = await store.get_document(doc_id)

        # Assert execute was called with the correct select statement
        call_found = False
        for call_args_tuple in mock_session.execute.call_args_list:
            stmt = call_args_tuple[0][0] # The statement is the first arg
            if isinstance(stmt, sqlalchemy.sql.selectable.Select) and \
               stmt.whereclause is not None and \
               str(stmt.whereclause) == f"{VectorDocumentModel.__tablename__}.id = :id_1": # Example check
                # Further check if stmt.columns contains expected columns like VectorDocumentModel.id etc.
                # And that the parameter passed to execute for :id_1 would be doc_id
                # This part of assertion can be tricky and fragile.
                # A simpler check is that execute was called, and the mock chain for scalar_one_or_none was hit.
                call_found = True
                break
        assert call_found, "Select statement for get_document was not correctly called."
        mock_session.execute.return_value.scalar_one_or_none.assert_called_once()

        assert result is not None
        assert isinstance(result, SearchResult)
        assert result.id == doc_id
        assert result.document == content
        assert result.metadata == metadata_dict
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_get_document_success_with_content_field(
        self,
        sqlite_store: SqliteVectorStore, # Uses default store, we'll modify its content_field for this test
        mock_sqlalchemy_engine: dict,
    ):
        """Test retrieving document content using a custom content_field."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        # Modify content_field for this test
        original_content_field = store.content_field
        store.content_field = "custom_text_field" # Ensure SUT uses this

        doc_id = "doc_custom_content"
        custom_content = "Content from custom field"
        default_content_attr_val = "This is from the default 'content' attribute"
        metadata_dict = {"source": "custom"}
        metadata_json = json.dumps(metadata_dict)

        mock_doc_instance = MagicMock(spec=VectorDocumentModel)
        mock_doc_instance.id = doc_id
        # Set both the custom field and the default 'content' field to different values
        setattr(mock_doc_instance, store.content_field, custom_content)
        mock_doc_instance.content = default_content_attr_val
        mock_doc_instance.doc_metadata = metadata_json

        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_doc_instance

        result = await store.get_document(doc_id)

        assert result is not None
        assert result.document == custom_content # Should fetch from store.content_field

        # Restore original content_field if other tests depend on it from the same fixture instance
        store.content_field = original_content_field


    @pytest.mark.asyncio
    async def test_get_document_metadata_is_none(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test document retrieval when its metadata is None in DB."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        doc_id = "doc_no_meta"

        mock_doc_instance = MagicMock(spec=VectorDocumentModel)
        mock_doc_instance.id = doc_id
        mock_doc_instance.content = "content"
        mock_doc_instance.doc_metadata = None # Metadata is None

        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_doc_instance
        result = await store.get_document(doc_id)

        assert result is not None
        assert result.metadata == {}

    @pytest.mark.asyncio
    async def test_get_document_metadata_invalid_json(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test document retrieval when its metadata is invalid JSON."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        doc_id = "doc_bad_meta"

        mock_doc_instance = MagicMock(spec=VectorDocumentModel)
        mock_doc_instance.id = doc_id
        mock_doc_instance.content = "content"
        mock_doc_instance.doc_metadata = "this is not json" # Invalid JSON

        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_doc_instance
        result = await store.get_document(doc_id)

        assert result is not None
        assert result.metadata == {} # Should default to empty dict

    @pytest.mark.asyncio
    async def test_get_document_not_found(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test retrieving a document that is not found."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        mock_session.execute.return_value.scalar_one_or_none.return_value = None # Simulate not found

        result = await store.get_document("non_existent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_document_not_initialized(
        self,
        sqlite_store_uninitialized: SqliteVectorStore,
    ):
        """Test VectorStoreError if get_document is called on an uninitialized store."""
        store = sqlite_store_uninitialized
        with pytest.raises(
            VectorStoreError,
            match="Vector store is not initialized. Call initialize() first.",
        ):
            await store.get_document("some_id")

    @pytest.mark.asyncio
    async def test_get_document_sqlalchemy_error(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test VectorStoreError if session.execute raises SQLAlchemyError."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        db_error_msg = "DB error on get"

        mock_session.execute.side_effect = sqlalchemy.exc.SQLAlchemyError(db_error_msg)

        with pytest.raises(VectorStoreError, match=f"Failed to get document: {db_error_msg}"):
            await store.get_document("some_id")


class TestSqliteVectorStoreUpdateDocument:
    """Tests for the SqliteVectorStore.update_document method."""

    @pytest.mark.asyncio
    async def test_update_document_success(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
    ):
        """Test successful updating of a document."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        doc_id = "doc_to_update"
        new_document_content = "updated content"
        new_embedding = [0.7, 0.8, 0.9]
        new_metadata = {"tag": "updated"}

        serialized_new_embedding = b"new_emb_bytes"
        mock_sqlite_vec.serialize_float32.return_value = serialized_new_embedding

        mock_doc_found = MagicMock(spec=VectorDocumentModel)
        mock_doc_found.id = doc_id # Used for virtual table update id
        # Other attributes (content, embedding, doc_metadata) will be set by SUT on this mock object.

        # Mock for session.execute(...).scalar_one_or_none() for finding the doc
        # Mock for session.execute() for the virtual table INSERT
        # We need to distinguish these calls if they happen on the same mock_session.execute

        execute_call_count = 0
        virtual_table_insert_params = None

        def execute_side_effect_update(statement, *args, **kwargs):
            nonlocal execute_call_count, virtual_table_insert_params
            execute_call_count += 1
            if isinstance(statement, sqlalchemy.sql.selectable.Select): # First call: select document
                return MagicMock(scalar_one_or_none=MagicMock(return_value=mock_doc_found))
            elif isinstance(statement, sqlalchemy.sql.elements.TextClause) and \
                 f"INSERT INTO {store.virtual_table}" in str(statement): # Second call: virtual table insert
                virtual_table_insert_params = kwargs.get("params", args[0] if args else {}) # In SUT, params is in kwargs
                return MagicMock() # DML usually returns a result proxy
            raise ValueError(f"Unexpected statement for execute mock in update: {statement}")

        mock_session.execute.side_effect = execute_side_effect_update

        await store.update_document(doc_id, new_document_content, new_embedding, new_metadata)

        # 1. Assert serialize_float32 called
        mock_sqlite_vec.serialize_float32.assert_called_once_with(new_embedding)

        # 2. Assert select was called to find the document
        # The side_effect checks this implicitly by type. We can check call count if needed.
        # And scalar_one_or_none was called on its result.
        assert mock_session.execute.call_args_list[0][0][0].__class__ == sqlalchemy.sql.selectable.Select

        # 3. Assert attributes of mock_doc_found were updated
        assert mock_doc_found.content == new_document_content
        assert mock_doc_found.embedding == serialized_new_embedding
        assert mock_doc_found.doc_metadata == json.dumps(new_metadata)

        # 4. Assert virtual table insert
        assert virtual_table_insert_params is not None
        assert virtual_table_insert_params["id"] == mock_doc_found.id
        assert virtual_table_insert_params["embedding"] == serialized_new_embedding

        # 5. Assert commit was called
        mock_session.commit.assert_called_once()


    @pytest.mark.asyncio
    async def test_update_document_not_found(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock, # For serialization if it gets that far
    ):
        """Test VectorStoreError if document to update is not found."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        # Ensure serialize_float32 doesn't fail if called before not_found error
        mock_sqlite_vec.serialize_float32.return_value = b"some_embedding"

        mock_session.execute.return_value.scalar_one_or_none.return_value = None # Simulate not found

        doc_id_not_found = "non_existent_id"
        with pytest.raises(
            VectorStoreError,
            match=f"Document with id {doc_id_not_found} not found for update."
        ):
            await store.update_document(doc_id_not_found, "doc", [0.1], {})

        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_document_not_initialized(
        self,
        sqlite_store_uninitialized: SqliteVectorStore,
    ):
        """Test VectorStoreError if update_document is called on an uninitialized store."""
        store = sqlite_store_uninitialized
        with pytest.raises(
            VectorStoreError,
            match="Vector store is not initialized. Call initialize() first.",
        ):
            await store.update_document("id", "doc", [0.1], {})

    @pytest.mark.asyncio
    async def test_update_document_sqlalchemy_error_on_select(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
    ):
        """Test VectorStoreError if select fails during update."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        mock_sqlite_vec.serialize_float32.return_value = b"some_embedding"

        error_msg = "DB error on select during update"
        # Make the first execute call (the select) fail
        mock_session.execute.side_effect = sqlalchemy.exc.SQLAlchemyError(error_msg)

        with pytest.raises(VectorStoreError, match=f"Failed to update document: {error_msg}"):
            await store.update_document("id1", "doc", [0.1], {})
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_document_sqlalchemy_error_on_virtual_table_insert(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
    ):
        """Test VectorStoreError if virtual table insert fails."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        mock_sqlite_vec.serialize_float32.return_value = b"new_emb_bytes"

        mock_doc_found = MagicMock(spec=VectorDocumentModel)
        mock_doc_found.id = "doc_id_for_virtual_fail"
        error_msg = "DB error on virtual insert"

        def execute_side_effect_virtual_fail(statement, *args, **kwargs):
            if isinstance(statement, sqlalchemy.sql.selectable.Select):
                return MagicMock(scalar_one_or_none=MagicMock(return_value=mock_doc_found))
            elif isinstance(statement, sqlalchemy.sql.elements.TextClause) and \
                 f"INSERT INTO {store.virtual_table}" in str(statement):
                raise sqlalchemy.exc.SQLAlchemyError(error_msg)
            raise ValueError("Unexpected execute call in virtual_fail test")

        mock_session.execute.side_effect = execute_side_effect_virtual_fail

        with pytest.raises(VectorStoreError, match=f"Failed to update document: {error_msg}"):
            await store.update_document(mock_doc_found.id, "doc", [0.1], {})
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_document_sqlalchemy_error_on_commit(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
    ):
        """Test VectorStoreError if session.commit fails."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        mock_sqlite_vec.serialize_float32.return_value = b"new_emb_bytes"

        mock_doc_found = MagicMock(spec=VectorDocumentModel)
        mock_doc_found.id = "doc_id_for_commit_fail"

        # Let select and virtual table insert succeed
        select_mock_result = MagicMock(scalar_one_or_none=MagicMock(return_value=mock_doc_found))
        virtual_insert_mock_result = MagicMock()

        call_count = 0
        def execute_side_effect_commit_fail(statement, *args, **kwargs):
            nonlocal call_count
            call_count +=1
            if call_count == 1 and isinstance(statement, sqlalchemy.sql.selectable.Select): # First call: select document
                return select_mock_result
            elif call_count == 2 and isinstance(statement, sqlalchemy.sql.elements.TextClause): # Second call: virtual table insert
                 return virtual_insert_mock_result
            raise ValueError(f"Unexpected execute call ({call_count}) in commit_fail test: {statement}")

        mock_session.execute.side_effect = execute_side_effect_commit_fail

        commit_error_msg = "DB error on commit during update"
        mock_session.commit.side_effect = sqlalchemy.exc.SQLAlchemyError(commit_error_msg)

        with pytest.raises(VectorStoreError, match=f"Failed to update document: {commit_error_msg}"):
            await store.update_document(mock_doc_found.id, "doc", [0.1], {})

        mock_session.commit.assert_called_once() # Commit was called and it failed

    @pytest.mark.asyncio
    async def test_update_document_struct_error_on_serialize(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlite_vec: MagicMock,
        # mock_sqlalchemy_engine not strictly needed if serialization fails first
    ):
        """Test VectorStoreError if serialize_float32 raises struct.error."""
        store = sqlite_store
        import struct # Required for struct.error

        serialization_error_msg = "Packing failed for update"
        mock_sqlite_vec.serialize_float32.side_effect = struct.error(serialization_error_msg)

        with pytest.raises(
            VectorStoreError,
            match=f"Failed to pack embedding for update: {serialization_error_msg}"
        ):
            await store.update_document("id1", "doc", [0.1], {})


class TestSqliteVectorStoreGetAll:
    """Tests for the SqliteVectorStore.get_all method."""

    @pytest.mark.asyncio
    async def test_get_all_success(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test successfully retrieving all documents."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        # Prepare mock VectorDocumentModel objects
        mock_doc1 = MagicMock(spec=VectorDocumentModel)
        mock_doc1.id = "id1"
        setattr(mock_doc1, store.content_field, "content1") # Use store.content_field
        mock_doc1.doc_metadata = json.dumps({"meta": "val1"})

        mock_doc2 = MagicMock(spec=VectorDocumentModel)
        mock_doc2.id = "id2"
        setattr(mock_doc2, store.content_field, "content2")
        mock_doc2.doc_metadata = json.dumps({"meta": "val2"})

        mock_documents_from_db = [mock_doc1, mock_doc2]
        mock_session.execute.return_value.scalars.return_value.all.return_value = mock_documents_from_db

        results = await store.get_all()

        # 1. Assert mock_session.execute was called with select(VectorDocumentModel)
        call_found = False
        for call_args_tuple in mock_session.execute.call_args_list:
            stmt = call_args_tuple[0][0]
            # Check if it's a SELECT statement targeting the VectorDocumentModel table without a WHERE clause.
            if isinstance(stmt, sqlalchemy.sql.selectable.Select) and \
               str(stmt.froms[0]) == VectorDocumentModel.__tablename__ and \
               stmt.whereclause is None:
                call_found = True
                break
        assert call_found, "Expected select(VectorDocumentModel) not executed or has a whereclause."
        mock_session.execute.return_value.scalars.assert_called_once()
        mock_session.execute.return_value.scalars.return_value.all.assert_called_once()

        # 2. Assert results is a list of SearchResult objects
        assert isinstance(results, list)
        assert len(results) == len(mock_documents_from_db)
        for result in results:
            assert isinstance(result, SearchResult)

        # 3. Verify content of SearchResult objects
        # Result 1
        assert results[0].id == mock_doc1.id
        assert results[0].document == getattr(mock_doc1, store.content_field)
        assert results[0].metadata == json.loads(mock_doc1.doc_metadata)
        assert results[0].score == 0.0

        # Result 2
        assert results[1].id == mock_doc2.id
        assert results[1].document == getattr(mock_doc2, store.content_field)
        assert results[1].metadata == json.loads(mock_doc2.doc_metadata)
        assert results[1].score == 0.0

    @pytest.mark.asyncio
    async def test_get_all_empty_store(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test retrieving all documents when the store is empty."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        mock_session.execute.return_value.scalars.return_value.all.return_value = [] # Empty list

        results = await store.get_all()

        assert results == []

    @pytest.mark.asyncio
    async def test_get_all_metadata_handling(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test metadata handling: None or invalid JSON should result in empty dict."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]

        mock_doc1_none_meta = MagicMock(spec=VectorDocumentModel)
        mock_doc1_none_meta.id = "id_none_meta"
        setattr(mock_doc1_none_meta, store.content_field, "content_none")
        mock_doc1_none_meta.doc_metadata = None

        mock_doc2_invalid_json = MagicMock(spec=VectorDocumentModel)
        mock_doc2_invalid_json.id = "id_invalid_json"
        setattr(mock_doc2_invalid_json, store.content_field, "content_invalid")
        mock_doc2_invalid_json.doc_metadata = "this is not valid json"

        mock_documents_from_db = [mock_doc1_none_meta, mock_doc2_invalid_json]
        mock_session.execute.return_value.scalars.return_value.all.return_value = mock_documents_from_db

        results = await store.get_all()

        assert len(results) == 2
        assert results[0].id == "id_none_meta"
        assert results[0].metadata == {}

        assert results[1].id == "id_invalid_json"
        assert results[1].metadata == {}


    @pytest.mark.asyncio
    async def test_get_all_not_initialized(
        self,
        sqlite_store_uninitialized: SqliteVectorStore,
    ):
        """Test VectorStoreError if get_all is called on an uninitialized store."""
        store = sqlite_store_uninitialized
        with pytest.raises(
            VectorStoreError,
            match="Vector store is not initialized. Call initialize() first.",
        ):
            await store.get_all()

    @pytest.mark.asyncio
    async def test_get_all_sqlalchemy_error(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test VectorStoreError if session.execute raises SQLAlchemyError."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        db_error_msg = "DB error on get_all"

        mock_session.execute.side_effect = sqlalchemy.exc.SQLAlchemyError(db_error_msg)

        with pytest.raises(VectorStoreError, match=f"Failed to get all documents: {db_error_msg}"):
            await store.get_all()

    @pytest.mark.asyncio
    async def test_delete_not_initialized(
        self,
        sqlite_store_uninitialized: SqliteVectorStore,
    ):
        """Test VectorStoreError if delete is called on an uninitialized store."""
        store = sqlite_store_uninitialized
        with pytest.raises(
            VectorStoreError,
            match="Vector store is not initialized. Call initialize() first.",
        ):
            await store.delete(["some_id"])

    @pytest.mark.asyncio
    async def test_delete_sqlalchemy_error_on_select(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test VectorStoreError if the select statement fails."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        error_message = "DB error on select"

        mock_session.execute.side_effect = sqlalchemy.exc.SQLAlchemyError(error_message)

        with pytest.raises(VectorStoreError, match=f"Failed to delete documents: {error_message}"):
            await store.delete(["id1"])

        # In SQLAlchemy, with a session context manager, rollback is often implicit on error.
        # Explicit check for rollback might be useful if SUT handles it manually.
        # mock_session.rollback.assert_called_once() # If SUT has explicit rollback
        mock_session.commit.assert_not_called()


    @pytest.mark.asyncio
    async def test_delete_sqlalchemy_error_on_delete_virtual(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test VectorStoreError if deleting from virtual table fails."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        ids_to_delete = ["id1"]
        error_message = "DB error on virtual delete"

        mock_doc1 = MagicMock(spec=VectorDocumentModel)
        mock_doc1.id = "id1"

        select_execute_mock = MagicMock()
        select_execute_mock.scalars.return_value.all.return_value = [mock_doc1]

        def execute_side_effect_for_virtual_fail(statement, *args, **kwargs):
            if isinstance(statement, sqlalchemy.sql.selectable.Select):
                return select_execute_mock
            # Fail on virtual table delete
            if isinstance(statement, sqlalchemy.sql.elements.TextClause) and f"DELETE FROM {store.virtual_table}" in str(statement):
                raise sqlalchemy.exc.SQLAlchemyError(error_message)
            return MagicMock()

        mock_session.execute.side_effect = execute_side_effect_for_virtual_fail

        with pytest.raises(VectorStoreError, match=f"Failed to delete documents: {error_message}"):
            await store.delete(ids_to_delete)
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_sqlalchemy_error_on_delete_main(
        self,
        sqlite_store: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test VectorStoreError if deleting from main table fails."""
        store = sqlite_store
        mock_session = mock_sqlalchemy_engine["session"]
        ids_to_delete = ["id1"]
        error_message = "DB error on main delete"

        mock_doc1 = MagicMock(spec=VectorDocumentModel)
        mock_doc1.id = "id1"

        select_execute_mock = MagicMock()
        select_execute_mock.scalars.return_value.all.return_value = [mock_doc1]

        virtual_delete_execute_mock = MagicMock() # Successful virtual table delete

        def execute_side_effect_for_main_fail(statement, *args, **kwargs):
            if isinstance(statement, sqlalchemy.sql.selectable.Select):
                return select_execute_mock
            if isinstance(statement, sqlalchemy.sql.elements.TextClause) and f"DELETE FROM {store.virtual_table}" in str(statement):
                return virtual_delete_execute_mock
            # Fail on main table delete
            if isinstance(statement, sqlalchemy.sql.dml.Delete) and statement.table.name == VectorDocumentModel.__tablename__:
                raise sqlalchemy.exc.SQLAlchemyError(error_message)
            return MagicMock() # Should not be reached if logic is tight

        mock_session.execute.side_effect = execute_side_effect_for_main_fail

        with pytest.raises(VectorStoreError, match=f"Failed to delete documents: {error_message}"):
            await store.delete(ids_to_delete)
        mock_session.commit.assert_not_called()


class TestSqliteVectorStoreInitialize:
    """Tests for the SqliteVectorStore.initialize method."""

    @patch("knowlang.vector_stores.sqlite.Base.metadata")
    @patch("sqlalchemy.event")
    def test_initialize_success(
        self,
        mock_sqlalchemys_event, # sqlalchemy.event mock
        mock_base_metadata,    # knowlang.vector_stores.sqlite.Base.metadata mock
        sqlite_store_uninitialized: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
        mock_app_config: AppConfig,
    ):
        """Test successful initialization of the SqliteVectorStore."""
        store = sqlite_store_uninitialized
        mock_engine = mock_sqlalchemy_engine["engine"]
        mock_connection = mock_sqlalchemy_engine["connection"]

        # Configure create_engine to return our specific mock_engine when called by initialize
        mock_sqlalchemy_engine["create_engine"].return_value = mock_engine

        store.initialize()

        # 1. Assert create_engine was called
        mock_sqlalchemy_engine["create_engine"].assert_called_once_with(store.db_path)
        assert store._engine == mock_engine # check if the engine is assigned

        # 2. Assert event.listen was called for _setup_sqlite_vec_extension
        # sqlalchemy.event.listen(self.engine, "connect", self.load_sqlite_vec_listener)
        # The actual listener function `load_sqlite_vec` is an inner function of `_setup_sqlite_vec_extension`.
        # We check that `listen` was called on the correct engine and event.
        assert mock_sqlalchemys_event.listen.call_count > 0
        listen_call_args = None
        for call in mock_sqlalchemys_event.listen.call_args_list:
            if call[0][0] == mock_engine and call[0][1] == "connect":
                listen_call_args = call
                break
        assert listen_call_args is not None, "sqlalchemy.event.listen was not called correctly for 'connect' event."

        # 3. Simulate the 'connect' event to test sqlite_vec.load
        # The listener is the third argument in the listen_call_args: listen_call_args[0][2]
        listener_function = listen_call_args[0][2]

        # The listener expects a DBAPI connection, not a SQLAlchemy Connection object.
        # The SQLAlchemy Connection object (mock_connection) has a `connection` attribute
        # that usually points to the raw DBAPI connection.
        # Our mock_connection needs this `connection` attribute.
        mock_dbapi_connection = MagicMock(spec=sqlalchemy.engine.interfaces.DBAPIConnection)
        mock_connection.connection = mock_dbapi_connection # type: ignore

        # Call the listener
        listener_function(mock_dbapi_connection, None) # Second arg is connection_record, can be None for this test
        mock_sqlite_vec.load.assert_called_once_with(mock_dbapi_connection)
        mock_dbapi_connection.enable_load_extension.assert_any_call(True) # type: ignore
        mock_dbapi_connection.enable_load_extension.assert_any_call(False) # type: ignore


        # 4. Assert Base.metadata.create_all was called
        mock_base_metadata.create_all.assert_called_once_with(mock_engine)

        # 5. Assert virtual table creation SQL
        expected_virtual_table_sql = f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {store.virtual_table} USING vec0(
                        id TEXT PRIMARY KEY,
                        embedding FLOAT[{store.embedding_dim}]
                    )
                """
        call_found = False
        for call_args in mock_connection.execute.call_args_list:
            args, _ = call_args
            if args and isinstance(args[0], sqlalchemy.sql.elements.TextClause):
                executed_sql = str(args[0]).strip()
                normalized_executed_sql = " ".join(executed_sql.split())
                normalized_expected_sql = " ".join(expected_virtual_table_sql.split())
                if normalized_expected_sql == normalized_executed_sql:
                    call_found = True
                    break
        assert call_found, f"Expected virtual table SQL not executed. Executed calls: {mock_connection.execute.call_args_list}"

        mock_connection.commit.assert_called_once()

        # 6. Assert store.engine and store.Session are set
        assert store._engine is not None
        assert store._session_factory is not None # In SUT it's self.Session

    def test_initialize_create_engine_fails(
        self,
        sqlite_store_uninitialized: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
    ):
        """Test VectorStoreInitError if sqlalchemy.create_engine fails."""
        store = sqlite_store_uninitialized
        error_message = "DB connection failed"
        mock_sqlalchemy_engine["create_engine"].side_effect = sqlalchemy.exc.SQLAlchemyError(error_message)

        with pytest.raises(
            VectorStoreInitError,
            match=f"Failed to initialize SqliteVectorStore: {error_message}",
        ):
            store.initialize()

        assert store._engine is None
        assert store._session_factory is None

    @patch("sqlalchemy.event") # Mock event.listen to avoid trying to attach to a failing engine setup
    def test_initialize_sqlite_vec_load_fails(
        self,
        mock_sqlalchemys_event_unused, # Patched but not directly asserted here, just to prevent real attach
        sqlite_store_uninitialized: SqliteVectorStore,
        mock_sqlalchemy_engine: dict,
        mock_sqlite_vec: MagicMock,
    ):
        """Test VectorStoreInitError if sqlite_vec.load fails during the 'connect' event."""
        store = sqlite_store_uninitialized
        mock_engine = mock_sqlalchemy_engine["engine"]
        mock_connection = mock_sqlalchemy_engine["connection"]

        # Ensure create_engine returns the mock engine
        mock_sqlalchemy_engine["create_engine"].return_value = mock_engine

        # Configure sqlite_vec.load to fail
        sqlite_vec_error_message = "sqlite-vec load error"
        mock_sqlite_vec.load.side_effect = Exception(sqlite_vec_error_message)

        # To trigger the listener that calls mock_sqlite_vec.load, we need to simulate the event dispatch.
        # The event listener is setup by _setup_sqlite_vec_extension.
        # We can directly call the method that would be registered as a listener.
        # Or, we can try to make initialize run up to the point of failure.

        # The listener `load_sqlite_vec` is an inner function.
        # We will patch `_setup_sqlite_vec_extension` to manually call the listener part
        # after configuring the mocks. This is a bit white-boxy but necessary for this specific failure.

        # Let's try a different approach: allow `initialize` to run.
        # The `_setup_sqlite_vec_extension` will register the listener.
        # The subsequent `Base.metadata.create_all(self.engine)` or `with self.engine.connect()`
        # *might* trigger the 'connect' event on the *mock_engine*.
        # The mock_engine's `connect` method is `mock_engine_instance.connect`.
        # Its event dispatch mechanism `mock_engine.dispatch.connect` needs to be sophisticated enough.
        # This is often not the case with standard MagicMocks.

        # Simplest way: Assume the listener is correctly registered by `_setup_sqlite_vec_extension`.
        # The error should be caught by the try-except block in `_setup_sqlite_vec_extension`'s listener
        # or by the main try-except in `initialize`.
        # The SUT's listener:
        # @event.listens_for(self.engine, "connect")
        # def load_sqlite_vec(conn: sqlite3.Connection, connection_record):
        #     try:
        #         ... sqlite_vec.load(conn) ...
        #     except Exception as e:
        #         raise VectorStoreInitError(...)

        # So, if `sqlite_vec.load` fails, the listener itself should raise VectorStoreInitError.
        # This error should then cause the `initialize` method to fail.

        # The `initialize` method will call `create_engine`, then `_setup_sqlite_vec_extension`.
        # Then it calls `Base.metadata.create_all(self.engine)`. This *might* be the first point
        # a connection is actually made, thus firing the event.

        # If `mock_engine.connect()` is called (e.g. by `create_all` or the subsequent `with store.engine.connect()`),
        # then the listener attached via `event.listen(mock_engine, ...)` should be triggered if
        # `sqlalchemy.event.listen` was not fully mocked away from its real implementation.
        # Our `mock_sqlalchemys_event` is a full mock of the `sqlalchemy.event` module.
        # So, the real `event.listen` is not called.
        # This means `mock_engine.dispatch.connect` will not have our listener.

        # We need to re-think this.
        # The `test_initialize_success` manually invoked the listener. We should do the same here.

        # Let's assume _setup_sqlite_vec_extension is called.
        # We can get the listener from the mock_sqlalchemys_event.listen call.

        with patch.object(store, '_setup_sqlite_vec_extension') as mock_setup_ext:
            # We define a side effect for _setup_sqlite_vec_extension
            # that *would* register the listener, and then we can simulate the event.
            # OR, we let the actual _setup_sqlite_vec_extension run, which uses the *real* sqlalchemy.event.listen
            # if we don't mock sqlalchemy.event itself for this part of the test.

            # For this test, we mock `sqlalchemy.event` as `mock_sqlalchemys_event_unused`
            # just to pass it to the test signature if other parts of `initialize` use it.
            # The actual SUT calls `sqlalchemy.event.listen`.

            # The listener is established in `_setup_sqlite_vec_extension`.
            # If `sqlite_vec.load` (called by the listener) raises an Exception,
            # that Exception is caught by the listener and re-raised as VectorStoreInitError.
            # This VectorStoreInitError should then be caught by the main try-catch in `initialize`.

            # The key is that the listener must be executable.
            # The `initialize()` method:
            # 1. self.engine = create_engine(...)
            # 2. self._setup_sqlite_vec_extension() -> this calls REAL event.listen on self.engine
            # 3. self.Session = sessionmaker(...)
            # 4. Base.metadata.create_all(self.engine) -> this may trigger connect event
            # 5. with self.engine.connect() as conn: -> this WILL trigger connect event

            # So, the event WILL be triggered on `mock_engine` if `event.listen` was called on it.
            # We need to ensure `mock_engine.dispatch.connect` calls our listener.
            # The `mock_engine` is a MagicMock. Its `dispatch` attribute is another MagicMock.
            # We can make `mock_engine.dispatch.connect` call the listener.

            # Let's refine mock_sqlalchemy_engine to make dispatch usable.
            # This is getting too complex for a fixture.

            # Alternative for this specific test:
            # Mock `_setup_sqlite_vec_extension` itself to directly raise the error,
            # as if the listener failed.

            # Or, more targeted:
            # The listener is:
            # @event.listens_for(self.engine, "connect")
            # def load_sqlite_vec(conn, record):
            #   try: sqlite_vec.load()
            #   except Exception as e: raise VectorStoreInitError("Failed to load sqlite-vec extension...")
            # This VectorStoreInitError, if raised when a connection is attempted,
            # should make initialize fail.

            # The `initialize` method has a broad try-except SQLAlchemyError.
            # The error from the listener is `VectorStoreInitError`.
            # If `VectorStoreInitError` is not a `SQLAlchemyError`, it might pass through
            # that specific handler. The SUT has:
            # except SQLAlchemyError as e: -> handles DB specific issues
            # The listener raises VectorStoreInitError.
            # The `initialize` method does not have a try-except for generic Exception or VectorStoreInitError
            # *directly* around the parts that trigger connection events (create_all, engine.connect()).
            # This means if the listener raises VectorStoreInitError, it should propagate out of initialize().

            # Let's assume `event.listen` successfully registers our listener on `mock_engine`.
            # Then, when `mock_engine.connect()` is called (by `create_all` or `with self.engine.connect()`),
            # we need `mock_engine.dispatch.connect` (which is `MagicMock().connect`) to call the listener.
            # This is the tricky part.

            # Instead of making `mock_engine.dispatch.connect` smart, let's use the approach from `test_initialize_success`:
            # 1. Call `initialize()`. This will try to set up everything.
            # 2. `_setup_sqlite_vec_extension` is called. It uses the *real* `sqlalchemy.event.listen` (because
            #    `mock_sqlalchemys_event_unused` only mocks it for *this test's scope*, not for SUT's scope unless specified).
            #    To control SUT's `sqlalchemy.event.listen`, we need to patch it at `knowlang.vector_stores.sqlite.event`.

            # Let's re-patch `sqlalchemy.event.listen` at the SUT's import location.
            with patch("knowlang.vector_stores.sqlite.event") as mock_sut_event_module:
                # This listener_wrapper will stand in for the real listener.
                # It will call our mock_sqlite_vec.load which is rigged to fail.
                def listener_wrapper_that_fails(dbapi_conn, conn_record):
                    # This is the body of the SUT's `load_sqlite_vec` listener
                    try:
                        dbapi_conn.enable_load_extension(True)
                        mock_sqlite_vec.load(dbapi_conn) # This will raise Exception("sqlite-vec load error")
                        dbapi_conn.enable_load_extension(False)
                    except Exception as e: # pylint: disable=broad-except
                        # This is what the SUT's listener does
                        raise VectorStoreInitError(f"Failed to load sqlite-vec extension. Ensure it's installed and accessible. {e}") from e

                # Make `event.listen` capture the target and listener.
                # When `_setup_sqlite_vec_extension` calls `event.listen(self.engine, "connect", ...)`,
                # we want `mock_sut_event_module.listen` to be called.
                # And we want the actual listener that `_setup_sqlite_vec_extension` defines
                # to use our `mock_sqlite_vec.load` that's configured to fail.

                # The SUT's `_setup_sqlite_vec_extension` defines its own listener `load_sqlite_vec`.
                # That listener calls `import sqlite_vec; sqlite_vec.load()`.
                # Our `mock_sqlite_vec` fixture patches `knowlang.vector_stores.sqlite.sqlite_vec`.
                # So the listener *will* call `mock_sqlite_vec.load()`.

                # So, no need for complex listener simulation here.
                # The `mock_sqlite_vec.load.side_effect` is already set.
                # The `initialize()` method will call `_setup_sqlite_vec_extension`.
                # This will attach a listener to `mock_engine`.
                # Then, `Base.metadata.create_all(mock_engine)` or `with mock_engine.connect()` will be called.
                # This is where we need `mock_engine` to actually execute its 'connect' listeners.
                # `mock_engine.connect()` returns a context manager.
                # The `__enter__` of that manager should trigger `mock_engine.dispatch.connect`.

                # Let's make `mock_engine.connect().__enter__` trigger the listeners.
                original_connect_enter = mock_sqlalchemy_engine["engine"].connect.return_value.__enter__

                def new_connect_enter(*args, **kwargs):
                    # Call registered 'connect' listeners on mock_engine
                    # This is a simplification; real dispatch is more complex.
                    # We assume listeners are stored in a way that `mock_engine.dispatch.connect` can access them.
                    # However, `mock_engine.dispatch` is just a MagicMock.
                    # So we need to find the listener that `_setup_sqlite_vec_extension` registered.

                    # The listener is registered on `store.engine` (which is `mock_engine`).
                    # The `event.listen` call from SUT would have used `sqlalchemy.event.listen`.
                    # If `sqlalchemy.event` itself is not mocked, then the listener is truly attached to `mock_engine`.

                    # For this test, `mock_sqlalchemys_event_unused` is patching `sqlalchemy.event` at global level.
                    # This means `_setup_sqlite_vec_extension` will call the mocked `event.listen`.
                    # The listener won't be *really* on `mock_engine.dispatch`.

                    # This test's setup for event handling is tricky.
                    # The `_setup_sqlite_vec_extension` uses `@event.listens_for(self.engine, "connect")`.
                    # This decorator translates to `event.listen(self.engine, "connect", decorated_function)`.
                    # If `sqlalchemy.event` is mocked (as it is by `mock_sqlalchemys_event_unused`),
                    # then `event.listen` is that mock.

                    # Let's get the listener from the mocked `event.listen`.
                    listener_to_call = None
                    for call in mock_sqlalchemys_event_unused.listen.call_args_list:
                        if call[0][0] == mock_engine and call[0][1] == "connect":
                            listener_to_call = call[0][2]
                            break

                    if listener_to_call:
                        mock_dbapi_conn = MagicMock(spec=sqlalchemy.engine.interfaces.DBAPIConnection)
                        mock_sqlalchemy_engine["connection"].connection = mock_dbapi_conn
                        # This call to the listener should trigger the mock_sqlite_vec.load error
                        # and the listener should raise VectorStoreInitError
                        listener_to_call(mock_dbapi_conn, None)

                    return original_connect_enter(*args, **kwargs)

                # Apply this new __enter__ behavior
                mock_sqlalchemy_engine["engine"].connect.return_value.__enter__ = new_connect_enter

        # Now, call initialize and expect the error from the listener.
        with pytest.raises(
            VectorStoreInitError,
            match=f"Failed to load sqlite-vec extension. Ensure it's installed and accessible. {sqlite_vec_error_message}",
        ):
            store.initialize()

        # If the listener failed, initialize should have bailed out.
        # The SUT's initialize catches SQLAlchemyError. VectorStoreInitError is not one.
        # So the VectorStoreInitError from the listener should propagate.
        # The store's engine would be set, but Session might not if error occurs before Session init.
        # SUT: self.engine set, then _setup, then self.Session, then create_all.
        # If listener fails during create_all's connect, Session is already set.
        # The question is whether initialize cleans up self.engine/self.Session.
        # SUT's main try-except block:
        # except SQLAlchemyError as e:
        #    self.engine = None
        #    self.Session = None
        # Since VectorStoreInitError is not a SQLAlchemyError, this cleanup won't run.
        # So engine and Session might remain assigned. This might be a bug in SUT.
        # For now, let's test current SUT behavior.
        # Based on SUT, if listener (VectorStoreInitError) fails, it propagates.
        # So, self.engine and self.Session would have been assigned.
        # Let's assume the goal is to check they are None if *initialize overall fails*.
        # If an unhandled error like VectorStoreInitError from listener propagates,
        # the state of store.engine and store.Session depends on where it failed.
        # If it failed during a connect event (e.g. during create_all or with engine.connect()),
        # then self.engine and self.Session would have been set.

        # However, the problem statement implies engine/session should be None.
        # This suggests VectorStoreInitError from listener *should* cause cleanup.
        # Let's assume VectorStoreInitError *is* caught by a broader except block or should be.
        # The SUT's listener *itself* raises VectorStoreInitError.
        # If this specific error is to be tested as causing full cleanup, the SUT might need adjustment.

        # Let's assume the current task is to test the specified match message,
        # and the state of engine/Session as None (implying cleanup).
        # This means the VectorStoreInitError raised by the listener *must* be caught by initialize()
        # and lead to cleanup.
        # The current SUT's main catch block is `except SQLAlchemyError`.
        # This implies the test might fail on engine/Session being None if the error is not SQLAlchemyError.
        # Let's proceed with the expectation that they *should* be None.
        # This might reveal a need to refine SUT's error handling.

        assert store._engine is None # This will fail if VectorStoreInitError doesn't lead to cleanup
        assert store._session_factory is None # Same as above
