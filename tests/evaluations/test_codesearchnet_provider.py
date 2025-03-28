import json
import gzip
import pytest
from unittest import mock
from knowlang.evaluations.providers.codesearchnet_provider import CodeSearchNetProvider
from knowlang.evaluations.types import DatasetSplitType

class TestCodeSearchNetProvider:
    """Tests for the CodeSearchNetProvider."""
    
    @pytest.fixture
    def setup_dataset_dir(self, temp_dir):
        """Set up a mock dataset directory."""
        dataset_dir = temp_dir / "codesearchnet"
        
        # Create language directories and test files for test split
        python_test_dir = dataset_dir / "python" / "final" / "jsonl" / "test"
        python_test_dir.mkdir(parents=True)
        
        # Create a test file with sample data
        test_file = python_test_dir / "python_test_0.jsonl.gz"
        with gzip.open(test_file, "wt", encoding="utf-8") as f:
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
        
        # Create train directory with sample data
        python_train_dir = dataset_dir / "python" / "final" / "jsonl" / "train"
        python_train_dir.mkdir(parents=True)
        
        train_file = python_train_dir / "python_train_0.jsonl.gz"
        with gzip.open(train_file, "wt", encoding="utf-8") as f:
            f.write(json.dumps({
                "docstring": "Generate a random number",
                "code": "def random_number():\n    import random\n    return random.randint(0, 100)",
                "url": "python_train_0_code",
                "repo": "test_repo",
                "path": "test_path",
                "func_name": "random_number",
                "original_string": "def random_number():\n    import random\n    return random.randint(0, 100)"
            }) + "\n")
        
        return dataset_dir
    
    @pytest.mark.asyncio
    async def test_load_specific_split(self, setup_dataset_dir):
        """Test loading a specific dataset split."""
        provider = CodeSearchNetProvider(setup_dataset_dir)
        
        # Mock rich.progress.track to avoid progress bar in tests
        with mock.patch("knowlang.evaluations.providers.codesearchnet_provider.track") as mock_track:
            mock_track.side_effect = lambda x, **kwargs: x
            pairs = await provider.load(
                languages=["python"], 
                splits=[DatasetSplitType.TEST]
            )
        
        assert len(pairs) == 2
        
        # Check first pair
        assert pairs[0].query == "Sort a list in ascending order"
        assert pairs[0].code == "def sort_list(lst):\n    return sorted(lst)"
        assert pairs[0].language == "python"
        assert pairs[0].code_id == "python_test_0_code"
        assert pairs[0].dataset_split == "test"
    
    @pytest.mark.asyncio
    async def test_load_multiple_splits(self, setup_dataset_dir):
        """Test loading multiple dataset splits."""
        provider = CodeSearchNetProvider(setup_dataset_dir)
        
        # Mock rich.progress.track to avoid progress bar in tests
        with mock.patch("knowlang.evaluations.providers.codesearchnet_provider.track") as mock_track:
            mock_track.side_effect = lambda x, **kwargs: x
            pairs = await provider.load(
                languages=["python"], 
                splits=[DatasetSplitType.TEST, DatasetSplitType.TRAIN]
            )
        
        assert len(pairs) == 3  # 2 test + 1 train
        
        # Verify we have both test and train data
        splits = [pair.dataset_split for pair in pairs]
        assert "test" in splits
        assert "train" in splits
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_language(self, setup_dataset_dir):
        """Test loading a language that doesn't exist."""
        provider = CodeSearchNetProvider(setup_dataset_dir)
        
        # Mock rich.progress.track to avoid progress bar in tests
        with mock.patch("knowlang.evaluations.providers.codesearchnet_provider.track") as mock_track:
            mock_track.side_effect = lambda x, **kwargs: x
            pairs = await provider.load(
                languages=["java"], 
                splits=[DatasetSplitType.TEST]
            )
        
        assert len(pairs) == 0
        
    @pytest.mark.asyncio
    async def test_load_all_splits_by_default(self, setup_dataset_dir):
        """Test that all splits are loaded by default if none specified."""
        provider = CodeSearchNetProvider(setup_dataset_dir)
        
        # Mock rich.progress.track to avoid progress bar in tests
        with mock.patch("knowlang.evaluations.providers.codesearchnet_provider.track") as mock_track:
            mock_track.side_effect = lambda x, **kwargs: x
            pairs = await provider.load(languages=["python"])
        
        assert len(pairs) == 3  # Should load all splits: 2 test + 1 train
        
        # Verify we have both test and train data
        splits = set(pair.dataset_split for pair in pairs)
        assert splits == {"test", "train"}  # We didn't create valid data in the fixture