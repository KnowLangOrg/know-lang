import pytest
from unittest import mock
from unittest.mock import MagicMock
from typing import List, Tuple

from knowlang.evaluations.grid_search import EvaluationGridSearch
from knowlang.evaluations.base import EvaluationRun, SearchConfiguration
from knowlang.evaluations.config_manager import SearchConfigurationManager
from knowlang.evaluations.evaluation_runner import CodeSearchEvaluator

class TestEvaluationGridSearch:
    """Tests for the EvaluationGridSearch."""
    
    @pytest.fixture
    def mock_evaluator(self) -> CodeSearchEvaluator:
        """Create a mock evaluator."""
        return MagicMock(spec=CodeSearchEvaluator)
    
    @pytest.fixture
    def mock_config_manager(self) -> SearchConfigurationManager:
        """Create a mock configuration manager."""
        return MagicMock(spec=SearchConfigurationManager)
    
    @pytest.fixture
    def grid_search(self, mock_evaluator: CodeSearchEvaluator, mock_config_manager: SearchConfigurationManager) -> EvaluationGridSearch:
        """Set up an EvaluationGridSearch with mocks."""
        return EvaluationGridSearch(mock_evaluator, mock_config_manager)
    
    def test_generate_grid_configurations(
        self, 
        grid_search: EvaluationGridSearch, 
        mock_config_manager: MagicMock
    ) -> None:
        """Test generating grid configurations."""
        configs = grid_search.generate_grid_configurations()
        
        # Check that configs were generated
        assert len(configs) > 0
        
        # Check that configurations have expected properties
        for config in configs:
            assert isinstance(config, SearchConfiguration)
            assert config.name.startswith("grid_")
            assert hasattr(config, "keyword_search_enabled")
            assert hasattr(config, "vector_search_enabled")
            assert hasattr(config, "reranking_enabled")
            
        # Check that save_configuration was called for each config
        assert mock_config_manager.save_configuration.call_count == len(configs)
    
    @pytest.mark.asyncio
    async def test_run_grid_search(
        self, 
        grid_search: EvaluationGridSearch, 
        mock_evaluator: MagicMock, 
        sample_search_configuration: SearchConfiguration
    ) -> None:
        """Test running a grid search."""
        # Use a simpler set of configurations for testing
        mock_configs = [
            sample_search_configuration,
            SearchConfiguration(
                name="test_config_2",
                description="Test configuration 2",
                keyword_search_enabled=False,
                vector_search_enabled=True,
                reranking_enabled=False
            )
        ]
        
        # Mock the generate_grid_configurations method
        with mock.patch.object(grid_search, "generate_grid_configurations", return_value=mock_configs):
            # Mock evaluate_dataset to return evaluation runs
            mock_evaluator.evaluate_dataset.side_effect = [
                EvaluationRun(
                    configuration=mock_configs[0],
                    dataset_name="test_dataset",
                    language="python",
                    mrr=0.75
                ),
                EvaluationRun(
                    configuration=mock_configs[1],
                    dataset_name="test_dataset", 
                    language="python",
                    mrr=0.5
                )
            ]
            
            runs = await grid_search.run_grid_search(
                dataset_name="test_dataset",
                language="python",
                limit=50
            )
        
        # Check that evaluate_dataset was called for each configuration
        assert mock_evaluator.evaluate_dataset.call_count == 2
        mock_evaluator.evaluate_dataset.assert_any_call(
            dataset_name="test_dataset",
            language="python",
            config=mock_configs[0],
            limit=50
        )
        
        # Check the runs
        assert len(runs) == 2
        assert runs[0].mrr > runs[1].mrr  # Sorted by MRR (descending)
        
        # The first run should be the one with higher MRR
        assert runs[0].mrr == 0.75
        assert runs[1].mrr == 0.5