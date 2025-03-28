import pytest
from unittest import mock
from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import MagicMock

from knowlang.configs import AppConfig
from knowlang.evaluations.evaluation_runner import CodeSearchEvaluator
from knowlang.evaluations.base import QueryCodePair, QueryEvaluationResult, EvaluationRun, SearchConfiguration
from knowlang.evaluations.types import DatasetSplitType
from knowlang.search.base import SearchResult

class TestCodeSearchEvaluator:
    """Tests for the CodeSearchEvaluator."""
    
    @pytest.fixture
    def test_dirs(self, temp_dir: Path) -> Tuple[Path, Path]:
        """Set up test directories."""
        data_dir = temp_dir / "data"
        output_dir = temp_dir / "output"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return data_dir, output_dir
    
    @pytest.fixture
    def mock_vector_store(self) -> MagicMock:
        """Create a mock vector store."""
        return MagicMock()
    
    @pytest.fixture
    def code_search_evaluator(self, mock_app_config: AppConfig, test_dirs: Tuple[Path, Path], mock_vector_store: MagicMock) -> CodeSearchEvaluator:
        """Set up a CodeSearchEvaluator with mocks."""
        data_dir, output_dir = test_dirs
        
        # Mock the VectorStoreFactory
        with mock.patch("knowlang.evaluations.evaluation_runner.VectorStoreFactory") as mock_factory:
            mock_factory.get.return_value = mock_vector_store
            
            evaluator = CodeSearchEvaluator(mock_app_config, data_dir, output_dir)
            yield evaluator
    
    @pytest.mark.asyncio
    async def test_initialize(self, code_search_evaluator: CodeSearchEvaluator, mock_vector_store: MagicMock) -> None:
        """Test initializing the evaluator."""
        await code_search_evaluator.initialize()
        
        # Check that initialize was called on the vector store
        mock_vector_store.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_with_configuration(
        self, 
        code_search_evaluator: CodeSearchEvaluator, 
        sample_search_configuration: SearchConfiguration, 
        sample_search_results: List[SearchResult]
    ) -> None:
        """Test searching with a configuration."""
        # Mock the search graph run
        with mock.patch("knowlang.evaluations.evaluation_runner.search_graph") as mock_graph:
            mock_result = MagicMock()
            mock_result.search_results = sample_search_results
            mock_graph.run = mock.AsyncMock()
            mock_graph.run.return_value = mock_result, None
            
            results, query_time = await code_search_evaluator.search_with_configuration(
                "test query", sample_search_configuration
            )
        
        # Check the results
        assert results == sample_search_results
        assert query_time > 0
    
    @pytest.mark.asyncio
    async def test_evaluate_query(
        self, 
        code_search_evaluator: CodeSearchEvaluator, 
        sample_search_configuration: SearchConfiguration, 
        sample_search_results: List[SearchResult]
    ) -> None:
        """Test evaluating a query."""
        # Mock the search_with_configuration method
        with mock.patch.object(code_search_evaluator, "search_with_configuration") as mock_search:
            mock_search.return_value = (sample_search_results, 0.1)
            
            result = await code_search_evaluator.evaluate_query(
                query_id="query1",
                query="test query",
                relevant_code_ids=["code1", "code3"],
                config=sample_search_configuration
            )
        
        # Check the result
        assert isinstance(result, QueryEvaluationResult)
        assert result.query_id == "query1"
        assert result.query == "test query"
        assert result.results == sample_search_results
        assert result.query_time == 0.1
        assert result.mrr > 0  # Should be 1.0 since code1 is the first result
    
    @pytest.mark.asyncio
    async def test_evaluate_dataset(
        self, 
        code_search_evaluator: CodeSearchEvaluator, 
        sample_search_configuration: SearchConfiguration, 
        sample_search_results: List[SearchResult]
    ) -> None:
        """Test evaluating a dataset."""
        # Create a mock query map
        query_map = {
            "query1": QueryCodePair(
                query_id="query1",
                query="test query 1",
                code_id="code1",
                code="def test_func(): pass",
                language="python",
                is_relevant=True,
                metadata={"repo": "sample_repo", "path": "sample_path"},
                dataset_split=DatasetSplitType.TEST.value
            ),
        
            "query2": QueryCodePair(
                query_id="query2",
                query="test query 2",
                code_id="code2",
                code="def another_func(): pass",
                language="python",
                is_relevant=True,
                metadata={"repo": "sample_repo", "path": "sample_path"},
                dataset_split=DatasetSplitType.TEST.value
            )
        }
        
        # Mock methods
        with mock.patch.object(code_search_evaluator.query_manager, "load_query_mappings", return_value=query_map), \
             mock.patch.object(code_search_evaluator, "initialize"), \
             mock.patch.object(code_search_evaluator, "evaluate_query"), \
             mock.patch.object(code_search_evaluator, "save_evaluation_run"):
            
            # Create mock query results
            code_search_evaluator.evaluate_query.side_effect = [
                QueryEvaluationResult(
                    query_id="query1",
                    query="test query 1",
                    relevant_code_ids=["code1"],
                    results=sample_search_results,
                    query_time=0.1,
                    mrr=1.0,
                    recall_at_1=1.0,
                    recall_at_5=1.0,
                    recall_at_10=1.0,
                    recall_at_100=1.0,
                    ndcg_at_10=1.0
                ),
                QueryEvaluationResult(
                    query_id="query2",
                    query="test query 2",
                    relevant_code_ids=["code2"],
                    results=sample_search_results,
                    query_time=0.2,
                    mrr=0.5,
                    recall_at_1=0.0,
                    recall_at_5=1.0,
                    recall_at_10=1.0,
                    recall_at_100=1.0,
                    ndcg_at_10=0.5
                )
            ]
            
            run = await code_search_evaluator.evaluate_dataset(
                dataset_name="test_dataset",
                language="python",
                config=sample_search_configuration
            )
        
        # Check the run
        assert isinstance(run, EvaluationRun)
        assert run.dataset_name == "test_dataset"
        assert run.language == "python"
        assert run.num_queries == 2
        assert run.mrr == 0.75  # Average of 1.0 and 0.5
    