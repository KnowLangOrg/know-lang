"""Unit tests for CLI command implementations."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from knowlang.core.types import CodeChunk
from knowlang.vector_stores.mock import MockVectorStore


@pytest.fixture
def mock_parser_factory():
    with patch("knowlang.cli.commands.parse.CodeParserFactory") as factory:
        # Set up the mock parser behavior
        mock_parser = Mock()
        # Mock parse_file as an async method that returns a list of CodeChunk
        mock_parser.parse_file = AsyncMock(return_value=[MagicMock(spec=CodeChunk)])
        factory.return_value.get_parser.return_value = mock_parser
        yield factory


@pytest.fixture
def mock_codebase_manager():
    with patch("knowlang.cli.commands.parse.CodebaseManager") as manager:
        mock_instance = AsyncMock()
        mock_instance.get_current_files = AsyncMock(return_value={Path("test.py")})
        manager.return_value = mock_instance
        yield manager


@pytest.fixture
def mock_incremental_updater():
    with patch("knowlang.cli.commands.parse.IncrementalUpdater") as updater:
        mock_instance = AsyncMock()
        mock_instance.update_codebase = AsyncMock()
        updater.return_value = mock_instance
        yield updater


@pytest.fixture
def mock_formatter():
    with patch("knowlang.cli.commands.parse.get_formatter") as formatter_func:
        mock_formatter = Mock()
        mock_formatter.display_chunks = Mock()
        formatter_func.return_value = mock_formatter
        yield formatter_func


@pytest.fixture
def mock_vector_store():
    """Mock vector store instance"""
    store = MockVectorStore()
    return store


@pytest.fixture
def mock_chatbot():
    with patch("knowlang.chat_bot.chat_interface.create_chatbot") as chatbot:
        mock_demo = Mock()
        mock_demo.launch = Mock()
        chatbot.return_value = mock_demo
        yield chatbot


class TestParseCommand:
    # TODO: parse command should be tested with end-to-end integration tests
    pass


class TestChatCommand:
    # TODO: chat command should be tested with end-to-end integration tests
    pass
