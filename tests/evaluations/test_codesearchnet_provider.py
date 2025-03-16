import json
import pytest
from unittest import mock
from knowlang.evaluations.providers.codesearchnet_provider import CodeSearchNetProvider

class TestCodeSearchNetProvider:
    """Tests for the CodeSearchNetProvider."""
    
    @pytest.fixture
    def setup_dataset_dir(self, temp_dir):
        """Set up a mock dataset directory."""
        dataset_dir = temp_dir / "codesearchnet"
        
        # Create language directories and test files
        python_dir = dataset_dir / "python"
        python_dir.mkdir(parents=True)
        
        # Create a test file with sample data
        test_file = python_dir / "test.jsonl"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "docstring": "Sort a list in ascending order",
                "code": "def sort_list(lst):\n    return sorted(lst)",
                "url": "python_test_0_code",
                "repo": "test_repo",
                "path": "test_path",
                "func_name": "sort_list",
                "original_string": "def sort_list(lst):\n    return sorted(lst)"
            }) + "\n")
            f.write(json.dumps({
                "docstring": "Find an element in a list",
                "code": "def find_element(lst, element):\n    return element in lst",
                "url": "python_test_1_code",
                "repo": "test_repo",
                "path": "test_path",
                "func_name": "find_element",
                "original_string": "def find_element(lst, element):\n    return element in lst"
            }) + "\n")
        
        return dataset_dir
    
    @pytest.mark.asyncio
    async def test_load(self, setup_dataset_dir):
        """Test loading the dataset."""
        provider = CodeSearchNetProvider(setup_dataset_dir)
        
        # Mock rich.progress.track to avoid progress bar in tests
        with mock.patch("knowlang.evaluations.providers.codesearchnet_provider.track") as mock_track:
            mock_track.side_effect = lambda x, **kwargs: x
            pairs = await provider.load(languages=["python"], split="test")
        
        assert len(pairs) == 2
        
        # Check first pair
        assert pairs[0].query == "Sort a list in ascending order"
        assert pairs[0].code == "def sort_list(lst):\n    return sorted(lst)"
        assert pairs[0].language == "python"
        assert pairs[0].code_id == "python_test_0_code"
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_language(self, setup_dataset_dir):
        """Test loading a language that doesn't exist."""
        provider = CodeSearchNetProvider(setup_dataset_dir)
        
        # Mock rich.progress.track to avoid progress bar in tests
        with mock.patch("knowlang.evaluations.providers.codesearchnet_provider.track") as mock_track:
            mock_track.side_effect = lambda x, **kwargs: x
            pairs = await provider.load(languages=["java"], split="test")
        
        assert len(pairs) == 0