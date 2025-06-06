from __future__ import annotations

import json
import sqlite3
import struct
import uuid
from unittest.mock import MagicMock, patch

import pytest

from knowlang.configs import AppConfig, DBConfig, EmbeddingConfig
from knowlang.vector_stores.base import (SearchResult, VectorStoreError,
                                         VectorStoreInitError)
from knowlang.vector_stores.sqlite import SqliteVectorStore

# Default values for testing
TEST_DB_PATH = ":memory:"
TEST_COLLECTION_NAME = "test_collection"
TEST_EMBEDDING_DIM = 3
TEST_CONTENT_FIELD = "text_content"
DEFAULT_SIMILARITY_METRIC = "cosine"


@pytest.fixture
def mock_app_config():
    """Fixture for a mock AppConfig."""
    return AppConfig(
        db=DBConfig(
            provider="sqlite",
            path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION_NAME,
            content_field=TEST_CONTENT_FIELD,
            similarity_metric=DEFAULT_SIMILARITY_METRIC,
        ),
        embedding=EmbeddingConfig(
            provider="fake",  # Not directly used by store, but part of AppConfig
            dimension=TEST_EMBEDDING_DIM,
        ),
    )


@pytest.fixture
def mock_sqlite3_conn():
    """Fixture for a mock sqlite3 connection."""
    mock_conn = MagicMock(spec=sqlite3.Connection)
    mock_cursor = MagicMock(spec=sqlite3.Cursor)
    mock_conn.cursor.return_value = mock_cursor
    # Mock methods that should return self or specific values
    mock_cursor.fetchone.return_value = None
    mock_cursor.fetchall.return_value = []
    mock_cursor.lastrowid = None
    return mock_conn, mock_cursor


@pytest.fixture
def sqlite_store(mock_app_config):
    """Fixture for an SqliteVectorStore instance."""
    return SqliteVectorStore(
        app_config=mock_app_config,
        db_path=mock_app_config.db.path,
        table_name=mock_app_config.db.collection_name,
        embedding_dim=mock_app_config.embedding.dimension,
        content_field=mock_app_config.db.content_field,
        similarity_metric=mock_app_config.db.similarity_metric,
    )


class TestSqliteVectorStore:
    def test_create_from_config(self, mock_app_config):
        """Test creating SqliteVectorStore from AppConfig."""
        store = SqliteVectorStore.create_from_config(mock_app_config)
        assert isinstance(store, SqliteVectorStore)
        assert store.db_path == TEST_DB_PATH
        assert store.table_name == TEST_COLLECTION_NAME
        assert store.embedding_dim == TEST_EMBEDDING_DIM
        assert store.content_field == TEST_CONTENT_FIELD

    def test_create_from_config_missing_path(self, mock_app_config):
        """Test VectorStoreInitError if db_path is missing."""
        mock_app_config.db.path = None
        with pytest.raises(VectorStoreInitError, match="Database path not set"):
            SqliteVectorStore.create_from_config(mock_app_config)

    @patch("sqlite3.connect")
    def test_initialize_success(self, mock_connect, sqlite_store, mock_sqlite3_conn):
        """Test successful initialization of the vector store."""
        mock_db_conn, mock_db_cursor = mock_sqlite3_conn
        mock_connect.return_value = mock_db_conn

        sqlite_store.initialize()

        mock_connect.assert_called_once_with(TEST_DB_PATH)
        mock_db_conn.enable_load_extension.assert_called_once_with(True)
        # TODO: Add more specific checks for load_extension calls if needed, including fallbacks
        mock_db_conn.load_extension.assert_called() # Basic check

        expected_table_create_sql = f"""
                CREATE TABLE IF NOT EXISTS {TEST_COLLECTION_NAME} (
                    id TEXT PRIMARY KEY,
                    {TEST_CONTENT_FIELD} TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT -- Store metadata as JSON string
                )
            """
        expected_virtual_table_create_sql = f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {TEST_COLLECTION_NAME}_vec_idx USING vec_f32(
                    embedding({TEST_EMBEDDING_DIM}) /* affinity: BLOB */
                )
            """

        self.assert_sql_executed_multiple(
            mock_db_cursor,
            [
                expected_table_create_sql,
                expected_virtual_table_create_sql,
            ],
        )
        mock_db_conn.commit.assert_called_once()
        assert sqlite_store.conn == mock_db_conn
        assert sqlite_store.cursor == mock_db_cursor

    @patch("sqlite3.connect")
    def test_initialize_connection_error(self, mock_connect, sqlite_store):
        """Test VectorStoreInitError on sqlite3.Error during connect."""
        mock_connect.side_effect = sqlite3.Error("Connection failed")
        with pytest.raises(VectorStoreInitError, match="Failed to initialize SqliteVectorStore: Connection failed"):
            sqlite_store.initialize()
        assert sqlite_store.conn is None
        assert sqlite_store.cursor is None

    @patch("sqlite3.connect")
    def test_initialize_load_extension_error(self, mock_connect, sqlite_store, mock_sqlite3_conn):
        """Test VectorStoreInitError if sqlite-vec extension fails to load."""
        mock_db_conn, _ = mock_sqlite3_conn
        mock_connect.return_value = mock_db_conn
        mock_db_conn.load_extension.side_effect = sqlite3.OperationalError("Cannot load extension")

        with pytest.raises(VectorStoreInitError, match="Failed to load sqlite-vec extension"):
            sqlite_store.initialize()
        mock_db_conn.enable_load_extension.assert_called_once_with(True)
        # Ensure all fallback paths were attempted if that's the logic
        # For now, just check that load_extension was called multiple times if fallbacks are in place.
        assert mock_db_conn.load_extension.call_count > 0


    @pytest.mark.asyncio
    @patch("sqlite3.connect") # Keep patching connect for store.initialize if called
    async def test_add_documents_success(self, mock_connect_unused, sqlite_store, mock_sqlite3_conn, mock_app_config):
        """Test adding documents successfully."""
        # Setup mock connection and cursor directly on the store instance for this test
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        docs = ["doc1 content", "doc2 content"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadatas = [
            {mock_app_config.db.content_field: "doc1 content", "other": "meta1"},
            {mock_app_config.db.content_field: "doc2 content", "other": "meta2"},
        ]
        ids = ["id1", "id2"]

        # Mock lastrowid behavior
        mock_cursor.lastrowid = 101 # First insert

        await sqlite_store.add_documents(docs, embeddings, metadatas, ids)

        actual_content_field = mock_app_config.db.content_field
        virtual_table_name = f"{mock_app_config.db.collection_name}_vec_idx"

        # Expected calls to cursor.execute()
        # Call 1: Insert into main table (doc1)
        packed_embedding1 = struct.pack(f'{TEST_EMBEDDING_DIM}f', *embeddings[0])
        meta_json1 = json.dumps(metadatas[0])
        expected_sql_main1 = f"INSERT INTO {sqlite_store.table_name} (id, {actual_content_field}, embedding, metadata) VALUES (?, ?, ?, ?)"
        mock_cursor.execute.assert_any_call(expected_sql_main1, ("id1", metadatas[0][actual_content_field], packed_embedding1, meta_json1))

        # Call 2: Insert into virtual table (doc1)
        expected_sql_virtual1 = f"INSERT INTO {virtual_table_name} (rowid, embedding) VALUES (?, ?)"
        mock_cursor.execute.assert_any_call(expected_sql_virtual1, (101, packed_embedding1))

        # Mock lastrowid for the second document
        # We need to control the sequence of return values for lastrowid if it's accessed multiple times
        # For simplicity, if add_documents iterates and accesses lastrowid after each main table insert:
        mock_cursor.lastrowid = 102 # Second insert

        packed_embedding2 = struct.pack(f'{TEST_EMBEDDING_DIM}f', *embeddings[1])
        meta_json2 = json.dumps(metadatas[1])
        expected_sql_main2 = f"INSERT INTO {sqlite_store.table_name} (id, {actual_content_field}, embedding, metadata) VALUES (?, ?, ?, ?)"
        mock_cursor.execute.assert_any_call(expected_sql_main2, ("id2", metadatas[1][actual_content_field], packed_embedding2, meta_json2))

        expected_sql_virtual2 = f"INSERT INTO {virtual_table_name} (rowid, embedding) VALUES (?, ?)"
        mock_cursor.execute.assert_any_call(expected_sql_virtual2, (102, packed_embedding2)) # Assuming lastrowid was updated to 102

        mock_conn.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_documents_mismatched_lengths(self, sqlite_store):
        """Test ValueError if input lists have mismatched lengths."""
        sqlite_store.conn = MagicMock() # Initialized enough to pass the first check
        sqlite_store.cursor = MagicMock()

        with pytest.raises(ValueError, match="documents, embeddings, and metadatas lists must have the same length"):
            await sqlite_store.add_documents(["doc1"], [[0.1]], [], None)

        with pytest.raises(ValueError, match="If provided, ids list must have the same length as documents"):
            await sqlite_store.add_documents(["doc1"], [[0.1]], [{"f":1}], ["id1", "id2"])


    @pytest.mark.asyncio
    async def test_query_success(self, sqlite_store, mock_sqlite3_conn, mock_app_config):
        """Test querying documents successfully."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        query_embedding = [0.1, 0.2, 0.3]
        top_k = 2

        # Sample data returned by cursor.fetchall()
        # (id, content, metadata_json, distance)
        db_results = [
            ("doc1_id", "Document 1 content", json.dumps({"meta": "val1"}), 0.1),
            ("doc2_id", "Document 2 content", json.dumps({"meta": "val2"}), 0.2),
        ]
        mock_cursor.fetchall.return_value = db_results

        results = await sqlite_store.query(query_embedding, top_k=top_k)

        packed_query_embedding = struct.pack(f'{TEST_EMBEDDING_DIM}f', *query_embedding)
        virtual_table_name = f"{mock_app_config.db.collection_name}_vec_idx"
        actual_content_field = mock_app_config.db.content_field

        expected_query_sql_pattern = f"""
            SELECT
                m.id,
                m.{actual_content_field},
                m.metadata,
                v.distance
            FROM
                {virtual_table_name} v
            JOIN
                {sqlite_store.table_name} m ON v.rowid = m.rowid
            WHERE
                v.embedding MATCH ?1
            LIMIT ?2;
        """
        # Check if the execute call matches the pattern (ignoring minor whitespace)
        executed_sql = mock_cursor.execute.call_args[0][0].strip()
        params = mock_cursor.execute.call_args[0][1]

        assert expected_query_sql_pattern.strip() in executed_sql
        assert params[0] == packed_query_embedding
        assert params[1] == top_k

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "doc1_id"
        assert results[0].document == "Document 1 content" # 'document' as per current accumulate_result
        assert results[0].metadata == {"meta": "val1"}
        assert results[0].score == pytest.approx(1.0 - 0.1) # score = 1.0 - distance

        assert results[1].id == "doc2_id"
        assert results[1].score == pytest.approx(1.0 - 0.2)

    # TODO: Add more tests:
    # - test_initialize with default content_field
    # - test_add_documents without content_field in metadata (uses docs list directly)
    # - test_add_documents with auto-generated IDs
    # - test_add_documents_sqlite_error
    # - test_query_with_filter (basic post-filtering)
    # - test_query_sqlite_error
    # - test_delete_success
    # - test_delete_not_found (should not error)
    # - test_delete_sqlite_error
    # - test_get_document_success
    # - test_get_document_not_found
    # - test_get_document_sqlite_error
    # - test_update_document_success
    # - test_update_document_not_found
    # - test_update_document_sqlite_error
    # - test_get_all_success
    # - test_get_all_empty
    # - test_get_all_sqlite_error
    # - test_close_success
    # - test_close_error (if applicable)
    # - test_del_closes_connection (harder to test reliably)

    # Helper for asserting SQL calls, robust to whitespace
    def assert_sql_executed(self, mock_cursor, expected_sql):
        """ Asserts if a similar SQL was executed, ignoring whitespace issues. """
        executed_sqls = [' '.join(call_args[0][0].split()) for call_args in mock_cursor.execute.call_args_list]
        normalized_expected_sql = ' '.join(expected_sql.split())
        assert normalized_expected_sql in executed_sqls, \
            f"Expected SQL '{normalized_expected_sql}' not found in executed SQLs: {executed_sqls}"

    # Example of how to use assert_sql_executed in a test:
    # self.assert_sql_executed(mock_db_cursor, expected_table_create_sql)
    # instead of direct string comparison of stripped SQLs in test_initialize.
    # This helper can be refined or used as a model for more complex assertions.

    def assert_sql_executed_multiple(self, mock_cursor, expected_sqls):
        """ Asserts if all expected SQLs were executed, ignoring whitespace. """
        executed_sqls_normalized = [' '.join(call_args[0][0].split()) for call_args in mock_cursor.execute.call_args_list]
        for expected_sql in expected_sqls:
            normalized_expected = ' '.join(expected_sql.split())
            assert normalized_expected in executed_sqls_normalized, \
                f"Expected SQL '{normalized_expected}' not found in executed SQLs: {executed_sqls_normalized}"


    @patch("sqlite3.connect")
    def test_initialize_default_content_field(self, mock_connect, sqlite_store, mock_sqlite3_conn, mock_app_config):
        """Test successful initialization with default content_field ('content')."""
        mock_db_conn, mock_db_cursor = mock_sqlite3_conn
        mock_connect.return_value = mock_db_conn

        # Modify store to use default content_field
        mock_app_config.db.content_field = None # Simulate it not being set
        store_default_cf = SqliteVectorStore.create_from_config(mock_app_config)
        # Re-assign mocked connection to this new store instance for the test
        store_default_cf.conn = mock_db_conn
        store_default_cf.cursor = mock_db_cursor

        store_default_cf.initialize() # Initialize this specific store

        mock_connect.assert_called_once_with(TEST_DB_PATH)

        # Default content field is 'content'
        expected_table_create_sql = f"""
                CREATE TABLE IF NOT EXISTS {TEST_COLLECTION_NAME} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT -- Store metadata as JSON string
                )
            """
        # Virtual table creation remains the same
        expected_virtual_table_create_sql = f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {TEST_COLLECTION_NAME}_vec_idx USING vec_f32(
                    embedding({TEST_EMBEDDING_DIM}) /* affinity: BLOB */
                )
            """
        self.assert_sql_executed_multiple(
            mock_db_cursor,
            [
                expected_table_create_sql,
                expected_virtual_table_create_sql,
            ],
        )
        mock_db_conn.commit.assert_called_once()


    @pytest.mark.asyncio
    @patch("sqlite3.connect")
    async def test_add_documents_no_content_field_in_metadata(self, mock_connect_unused, sqlite_store, mock_sqlite3_conn, mock_app_config):
        """Test adding docs where content_field is not in metadata, so doc string is used."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        # 'text_content' (from mock_app_config.db.content_field) is NOT in metadatas
        docs_content = ["doc1 actual content", "doc2 actual content"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadatas = [ {"other_meta": "value1"}, {"other_meta": "value2"}] # No content_field here
        ids = ["id1", "id2"]

        mock_cursor.lastrowid = 101 # For first doc

        await sqlite_store.add_documents(docs_content, embeddings, metadatas, ids)

        actual_content_field_name_in_db = mock_app_config.db.content_field # This is the COLUMN NAME

        # Verify doc1 - content comes from docs_content[0]
        packed_embedding1 = struct.pack(f'{TEST_EMBEDDING_DIM}f', *embeddings[0])
        meta_json1 = json.dumps(metadatas[0])
        # SQL uses the configured content_field name for the column
        expected_sql_main1 = f"INSERT INTO {sqlite_store.table_name} (id, {actual_content_field_name_in_db}, embedding, metadata) VALUES (?, ?, ?, ?)"
        # The content stored is from docs_content[0]
        mock_cursor.execute.assert_any_call(expected_sql_main1, (ids[0], docs_content[0], packed_embedding1, meta_json1))

        # Verify doc2 - content comes from docs_content[1]
        mock_cursor.lastrowid = 102 # For second doc
        packed_embedding2 = struct.pack(f'{TEST_EMBEDDING_DIM}f', *embeddings[1])
        meta_json2 = json.dumps(metadatas[1])
        expected_sql_main2 = f"INSERT INTO {sqlite_store.table_name} (id, {actual_content_field_name_in_db}, embedding, metadata) VALUES (?, ?, ?, ?)"
        mock_cursor.execute.assert_any_call(expected_sql_main2, (ids[1], docs_content[1], packed_embedding2, meta_json2))

        mock_conn.commit.assert_called_once()


    @pytest.mark.asyncio
    @patch("sqlite3.connect")
    async def test_add_documents_auto_generated_ids(self, mock_connect_unused, sqlite_store, mock_sqlite3_conn, mock_app_config):
        """Test adding documents with auto-generated UUIDs for IDs."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        docs = ["doc1 content"]
        embeddings = [[0.1, 0.2, 0.3]]
        metadatas = [{mock_app_config.db.content_field: "doc1 content"}]
        # ids = None, so they should be auto-generated

        mock_cursor.lastrowid = 201

        with patch.object(uuid, 'uuid4', return_value=MagicMock(hex='testuuid123')) as mock_uuid:
            await sqlite_store.add_documents(docs, embeddings, metadatas, ids=None)

        mock_uuid.assert_called_once() # Ensure uuid4 was called to generate an ID

        actual_content_field = mock_app_config.db.content_field
        virtual_table_name = f"{mock_app_config.db.collection_name}_vec_idx"

        packed_embedding = struct.pack(f'{TEST_EMBEDDING_DIM}f', *embeddings[0])
        meta_json = json.dumps(metadatas[0])

        # Check main table insert
        # The ID will be the hex representation of the mock_uuid's return value if str(uuid.uuid4()) is used
        # If uuid.uuid4().hex is used, it's 'testuuid123'. SqliteVectorStore uses str(uuid.uuid4())
        # So we need to mock str(mock_uuid_object)
        mock_uuid_obj = uuid.UUID('12345678-1234-5678-1234-567812345678')
        mock_uuid.return_value = mock_uuid_obj # Make uuid.uuid4() return a specific UUID object

        # Re-run add_documents with this specific UUID mock
        await sqlite_store.add_documents(docs, embeddings, metadatas, ids=None)

        expected_id = str(mock_uuid_obj)
        expected_sql_main = f"INSERT INTO {sqlite_store.table_name} (id, {actual_content_field}, embedding, metadata) VALUES (?, ?, ?, ?)"
        # This assertion will be made on the *second* call to add_documents in this test,
        # so we need to check call_args_list or ensure the mock is reset if this test is isolated.
        # For simplicity, let's assume this is the dominant call or check specifically.

        # We need to find the call that matches the SQL and check its params.
        found_call = False
        for call in mock_cursor.execute.call_args_list:
            sql, params = call[0]
            if sql == expected_sql_main and params[0] == expected_id:
                assert params[1] == metadatas[0][actual_content_field]
                assert params[2] == packed_embedding
                assert params[3] == meta_json
                found_call = True
                break
        assert found_call, "Call to insert with auto-generated ID not found or params incorrect."

        # Check virtual table insert
        expected_sql_virtual = f"INSERT INTO {virtual_table_name} (rowid, embedding) VALUES (?, ?)"
        # This also depends on the call order. If it's the second set of calls:
        found_virtual_call = False
        for call in mock_cursor.execute.call_args_list:
            sql, params = call[0]
            if sql == expected_sql_virtual and params[0] == mock_cursor.lastrowid: # lastrowid would be 201
                 assert params[1] == packed_embedding
                 found_virtual_call = True
                 break
        assert found_virtual_call, "Call to insert into virtual table with auto-generated ID not found."

        # Commit should be called (potentially multiple times if not reset)
        assert mock_conn.commit.call_count > 0


    @pytest.mark.asyncio
    async def test_add_documents_sqlite_error(self, sqlite_store, mock_sqlite3_conn):
        """Test VectorStoreError on sqlite3.Error during add_documents."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        mock_cursor.execute.side_effect = sqlite3.Error("Insert failed")

        docs = ["doc1"]
        embeddings = [[0.1]]
        metadatas = [{"c": "doc1"}]

        with pytest.raises(VectorStoreError, match="Failed to add documents: Insert failed"):
            await sqlite_store.add_documents(docs, embeddings, metadatas, ["id1"])

        mock_conn.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document_success(self, sqlite_store, mock_sqlite3_conn, mock_app_config):
        """Test successfully retrieving a document."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        doc_id = "test_doc_id"
        expected_content = "This is the document content."
        expected_metadata = {"field": "value", "source": "test_source"}
        # get_document query: SELECT id, {actual_content_field}, metadata FROM {table_name} WHERE id = ?
        mock_cursor.fetchone.return_value = (doc_id, expected_content, json.dumps(expected_metadata))

        result = await sqlite_store.get_document(doc_id)

        actual_content_field_name = mock_app_config.db.content_field
        expected_sql = f"SELECT id, {actual_content_field_name}, metadata FROM {sqlite_store.table_name} WHERE id = ?"
        mock_cursor.execute.assert_called_once_with(expected_sql, (doc_id,))

        assert result is not None
        assert result.id == doc_id
        assert result.document == expected_content # 'document' based on accumulate_result
        assert result.metadata == expected_metadata
        assert result.score == 0.0 # Default score for get_document

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, sqlite_store, mock_sqlite3_conn):
        """Test retrieving a document that is not found."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        mock_cursor.fetchone.return_value = None # Simulate document not found

        result = await sqlite_store.get_document("non_existent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_document_sqlite_error(self, sqlite_store, mock_sqlite3_conn):
        """Test VectorStoreError on sqlite3.Error during get_document."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        mock_cursor.execute.side_effect = sqlite3.Error("Get failed")
        with pytest.raises(VectorStoreError, match="Failed to get document: Get failed"):
            await sqlite_store.get_document("any_id")

    @pytest.mark.asyncio
    async def test_update_document_success(self, sqlite_store, mock_sqlite3_conn, mock_app_config):
        """Test successfully updating a document."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        doc_id = "doc_to_update"
        new_content = "updated content"
        new_embedding = [0.7, 0.8, 0.9]
        new_metadata = {"updated_field": "new_value"}

        # Mock rowid lookup
        mock_cursor.fetchone.return_value = (123,) # Simulate rowid found

        await sqlite_store.update_document(doc_id, new_content, new_embedding, new_metadata)

        actual_content_field = mock_app_config.db.content_field
        virtual_table_name = f"{mock_app_config.db.collection_name}_vec_idx"
        packed_new_embedding = struct.pack(f'{TEST_EMBEDDING_DIM}f', *new_embedding)
        meta_json = json.dumps(new_metadata)

        # 1. Check SELECT rowid
        mock_cursor.execute.assert_any_call(f"SELECT rowid FROM {sqlite_store.table_name} WHERE id = ?", (doc_id,))

        # 2. Check UPDATE main table
        expected_update_sql = f"UPDATE {sqlite_store.table_name} SET {actual_content_field} = ?, embedding = ?, metadata = ? WHERE id = ?"
        mock_cursor.execute.assert_any_call(expected_update_sql, (new_content, packed_new_embedding, meta_json, doc_id))

        # 3. Check DELETE from virtual table
        mock_cursor.execute.assert_any_call(f"DELETE FROM {virtual_table_name} WHERE rowid = ?", (123,))

        # 4. Check INSERT into virtual table
        mock_cursor.execute.assert_any_call(f"INSERT INTO {virtual_table_name} (rowid, embedding) VALUES (?, ?)", (123, packed_new_embedding))

        mock_conn.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_document_not_found(self, sqlite_store, mock_sqlite3_conn):
        """Test VectorStoreError if document to update is not found."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        mock_cursor.fetchone.return_value = None # Simulate document (rowid) not found

        with pytest.raises(VectorStoreError, match="Document with id missing_doc not found for update."):
            await sqlite_store.update_document("missing_doc", "content", [0.1], {})

    @pytest.mark.asyncio
    async def test_update_document_sqlite_error_on_update(self, sqlite_store, mock_sqlite3_conn):
        """Test VectorStoreError and rollback on sqlite3.Error during UPDATE operation."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        mock_cursor.fetchone.return_value = (123,) # Rowid found
        # Make the UPDATE statement fail
        def execute_side_effect(sql, params):
            if "UPDATE" in sql:
                raise sqlite3.Error("Update failed")
            return MagicMock() # For other calls like SELECT, DELETE, INSERT rowid
        mock_cursor.execute.side_effect = execute_side_effect

        with pytest.raises(VectorStoreError, match="Failed to update document: Update failed"):
            await sqlite_store.update_document("doc_id", "content", [0.1,0.2,0.3], {})

        mock_conn.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_success(self, sqlite_store, mock_sqlite3_conn, mock_app_config):
        """Test successfully retrieving all documents."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        db_data = [
            ("id1", "content1", json.dumps({"meta": "val1"})),
            ("id2", "content2", json.dumps({"meta": "val2"})),
        ]
        mock_cursor.fetchall.return_value = db_data

        results = await sqlite_store.get_all()

        actual_content_field = mock_app_config.db.content_field
        expected_sql = f"SELECT id, {actual_content_field}, metadata FROM {sqlite_store.table_name}"
        mock_cursor.execute.assert_called_once_with(expected_sql)

        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[0].document == "content1"
        assert results[0].metadata == {"meta": "val1"}
        assert results[0].score == 0.0

        assert results[1].id == "id2"
        assert results[1].document == "content2"
        assert results[1].metadata == {"meta": "val2"}

    @pytest.mark.asyncio
    async def test_get_all_empty(self, sqlite_store, mock_sqlite3_conn):
        """Test retrieving all documents when the store is empty."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor
        mock_cursor.fetchall.return_value = []
        results = await sqlite_store.get_all()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_all_sqlite_error(self, sqlite_store, mock_sqlite3_conn):
        """Test VectorStoreError on sqlite3.Error during get_all."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor
        mock_cursor.execute.side_effect = sqlite3.Error("Get all failed")
        with pytest.raises(VectorStoreError, match="Failed to get all documents: Get all failed"):
            await sqlite_store.get_all()

    @pytest.mark.asyncio
    async def test_close_success(self, sqlite_store, mock_sqlite3_conn):
        """Test successfully closing the connection."""
        mock_conn, _ = mock_sqlite3_conn
        sqlite_store.conn = mock_conn # Simulate connection is open

        await sqlite_store.close()

        mock_conn.close.assert_called_once()
        assert sqlite_store.conn is None
        assert sqlite_store.cursor is None

    @pytest.mark.asyncio
    async def test_close_error(self, sqlite_store, mock_sqlite3_conn):
        """Test VectorStoreError on sqlite3.Error during close."""
        mock_conn, _ = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        mock_conn.close.side_effect = sqlite3.Error("Close failed")

        with pytest.raises(VectorStoreError, match="Failed to close connection: Close failed"):
            await sqlite_store.close()

        # Connection should still be reset locally even if close fails on DB side
        assert sqlite_store.conn is None
        assert sqlite_store.cursor is None

    @pytest.mark.asyncio
    async def test_query_with_filter_success(self, sqlite_store, mock_sqlite3_conn, mock_app_config):
        """Test querying with basic post-filtering."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        query_embedding = [0.1, 0.2, 0.3]
        top_k = 3 # Request more to allow filtering

        db_results = [
            ("id1", "content1", json.dumps({"meta_field": "valueA", "common": "yes"}), 0.1),
            ("id2", "content2", json.dumps({"meta_field": "valueB", "common": "yes"}), 0.2),
            ("id3", "content3", json.dumps({"meta_field": "valueA", "common": "no"}), 0.3),
        ]
        mock_cursor.fetchall.return_value = db_results

        # Filter for meta_field == "valueA"
        query_filter = {"meta_field": "valueA"}
        results = await sqlite_store.query(query_embedding, top_k=top_k, filter=query_filter)

        assert len(results) == 2 # id1 and id3 should match
        assert results[0].id == "id1"
        assert results[0].metadata["meta_field"] == "valueA"
        assert results[1].id == "id3"
        assert results[1].metadata["meta_field"] == "valueA"

        # Filter for common == "yes"
        query_filter_common = {"common": "yes"}
        results_common = await sqlite_store.query(query_embedding, top_k=top_k, filter=query_filter_common)
        assert len(results_common) == 2 # id1 and id2
        assert results_common[0].id == "id1"
        assert results_common[1].id == "id2"

    @pytest.mark.asyncio
    async def test_query_sqlite_error(self, sqlite_store, mock_sqlite3_conn):
        """Test VectorStoreError on sqlite3.Error during query."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        mock_cursor.execute.side_effect = sqlite3.Error("Query failed")
        query_embedding = [0.1, 0.2, 0.3]

        with pytest.raises(VectorStoreError, match="Failed to query: Query failed"):
            await sqlite_store.query(query_embedding)

    @pytest.mark.asyncio
    async def test_delete_success(self, sqlite_store, mock_sqlite3_conn, mock_app_config):
        """Test successful deletion of documents."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        ids_to_delete = ["id1", "id2"]
        # Mock fetchone to return different rowids for different IDs
        mock_cursor.fetchone.side_effect = [(101,), (102,)]

        await sqlite_store.delete(ids_to_delete)

        virtual_table_name = f"{mock_app_config.db.collection_name}_vec_idx"

        # Check select rowid calls
        mock_cursor.execute.assert_any_call(f"SELECT rowid FROM {sqlite_store.table_name} WHERE id = ?", ("id1",))
        mock_cursor.execute.assert_any_call(f"SELECT rowid FROM {sqlite_store.table_name} WHERE id = ?", ("id2",))

        # Check delete from main table (uses IN clause)
        expected_delete_main_sql = f"DELETE FROM {sqlite_store.table_name} WHERE id IN (?,?)"
        # Check delete from virtual table (uses IN clause)
        expected_delete_virtual_sql = f"DELETE FROM {virtual_table_name} WHERE rowid IN (?,?)"

        # Check calls more carefully
        main_delete_called_correctly = False
        virtual_delete_called_correctly = False

        for call_args in mock_cursor.execute.call_args_list:
            sql = call_args[0][0]
            params = call_args[0][1]
            if "DELETE FROM " + sqlite_store.table_name in sql and sql.endswith("IN (?,?)"):
                assert sorted(params) == sorted(ids_to_delete) # Order in IN clause might vary
                main_delete_called_correctly = True
            if "DELETE FROM " + virtual_table_name in sql and sql.endswith("IN (?,?)"):
                assert sorted(params) == sorted([101, 102]) # Row IDs
                virtual_delete_called_correctly = True

        assert main_delete_called_correctly, "Main table delete not called correctly"
        assert virtual_delete_called_correctly, "Virtual table delete not called correctly"

        mock_conn.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_ids_not_found(self, sqlite_store, mock_sqlite3_conn):
        """Test delete operation when some or all IDs are not found (should not error)."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        # Simulate one ID found, one not
        mock_cursor.fetchone.side_effect = [(101,), None]
        ids_to_delete = ["id1_found", "id2_not_found"]

        await sqlite_store.delete(ids_to_delete)

        # Only one rowid (101) should be in the list for deleting from virtual table
        # And only "id1_found" for the main table (as per current implementation)
        # The implementation deletes all provided ids from main table and found rowids from virtual.

        main_delete_called = False
        virtual_delete_called_for_101 = False

        for call_args in mock_cursor.execute.call_args_list:
            sql = call_args[0][0]
            params = call_args[0][1]
            if "DELETE FROM " + sqlite_store.table_name in sql and "IN (?,?)" in sql:
                 assert sorted(params) == sorted(ids_to_delete)
                 main_delete_called = True
            if "DELETE FROM " + sqlite_store.table_name + "_vec_idx" in sql and "IN (?)" in sql : # Only one rowid found
                 assert params == (101,)
                 virtual_delete_called_for_101 = True

        assert main_delete_called
        assert virtual_delete_called_for_101
        mock_conn.commit.assert_called_once()


    @pytest.mark.asyncio
    async def test_delete_sqlite_error(self, sqlite_store, mock_sqlite3_conn):
        """Test VectorStoreError on sqlite3.Error during delete."""
        mock_conn, mock_cursor = mock_sqlite3_conn
        sqlite_store.conn = mock_conn
        sqlite_store.cursor = mock_cursor

        mock_cursor.execute.side_effect = sqlite3.Error("Delete failed")

        with pytest.raises(VectorStoreError, match="Failed to delete documents: Delete failed"):
            await sqlite_store.delete(["id1"])

        mock_conn.rollback.assert_called_once()
```
