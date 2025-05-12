from typing import List, Optional
from rich.console import Console
from rich.table import Table

from knowlang.evaluations.base import EvaluationRun, SearchConfiguration
from knowlang.evaluations.config_manager import SearchConfigurationManager
from knowlang.evaluations.evaluation_runner import CodeSearchEvaluator
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)
console = Console()


class EvaluationGridSearch:
    """Run a grid search over search configurations."""
    
    def __init__(self, evaluator : CodeSearchEvaluator, config_manager: SearchConfigurationManager):
        # Use late binding for evaluator to avoid circular imports
        self.evaluator = evaluator
        self.config_manager = config_manager
    
    def generate_grid_configurations(self) -> List[SearchConfiguration]:
        """
        Generate a grid of search configurations to evaluate.
        
        Returns:
            List of search configurations
        """
        configs = []
        
        # Define parameter grids
        name_prefix = "grid"
        keyword_enabled = [True, False]
        vector_enabled = [True, False]
        reranking_enabled = [True, False]
        keyword_thresholds = [0.0, 0.2, 0.4]
        vector_thresholds = [0.5, 0.7, 0.9]
        reranker_thresholds = [0.3, 0.5, 0.7]
        
        # Skip invalid combinations (need at least one search method)
        # and limit the total number of configurations
        index = 0
        for k_enabled in keyword_enabled:
            for v_enabled in vector_enabled:
                # Skip if both search methods are disabled
                if not k_enabled and not v_enabled:
                    continue
                
                for r_enabled in reranking_enabled:
                    # Skip if reranking is enabled but no search methods are enabled
                    if r_enabled and not (k_enabled or v_enabled):
                        continue
                    
                    # For basic grid, just use default thresholds
                    name = f"{name_prefix}_{index}"
                    description = f"Grid search: keyword={k_enabled}, vector={v_enabled}, rerank={r_enabled}"
                    
                    config = SearchConfiguration(
                        name=name,
                        description=description,
                        keyword_search_enabled=k_enabled,
                        vector_search_enabled=v_enabled,
                        reranking_enabled=r_enabled,
                        keyword_search_threshold=0.0 if k_enabled else 0.0,
                        vector_search_threshold=0.7 if v_enabled else 0.0,
                        reranker_threshold=0.5 if r_enabled else 0.0,
                    )
                    
                    configs.append(config)
                    index += 1
        
        # Add more fine-grained configs for the most promising combinations
        # (This would depend on prior knowledge or preliminary results)
        # For example, if keyword+vector+reranking is promising:
        for k_thresh in keyword_thresholds:
            for v_thresh in vector_thresholds:
                for r_thresh in reranker_thresholds:
                    name = f"{name_prefix}_fine_{index}"
                    description = f"Fine grid: k_thresh={k_thresh}, v_thresh={v_thresh}, r_thresh={r_thresh}"
                    
                    config = SearchConfiguration(
                        name=name,
                        description=description,
                        keyword_search_enabled=True,
                        vector_search_enabled=True,
                        reranking_enabled=True,
                        keyword_search_threshold=k_thresh,
                        vector_search_threshold=v_thresh,
                        reranker_threshold=r_thresh,
                    )
                    
                    configs.append(config)
                    index += 1
        
        # Save all configurations
        for config in configs:
            self.config_manager.save_configuration(config)
        
        return configs
    
    async def run_grid_search(
        self,
        dataset_name: str,
        language: str,
        limit: Optional[int] = 50
    ) -> List[EvaluationRun]:
        """
        Run grid search on a dataset.
        
        Args:
            dataset_name: Name of the dataset
            language: Programming language to evaluate
            limit: Optional limit on number of queries per configuration
            
        Returns:
            List of evaluation runs, sorted by MRR
        """
        configs = self.generate_grid_configurations()
        LOG.info(f"Running grid search with {len(configs)} configurations")
        
        runs : List[EvaluationRun] = []
        for i, config in enumerate(configs):
            LOG.info(f"Evaluating configuration {i+1}/{len(configs)}: {config.name}")
            
            try:
                run = await self.evaluator.evaluate_dataset(
                    dataset_name=dataset_name,
                    language=language,
                    config=config,
                    limit=limit
                )
                
                runs.append(run)
                
                # Print interim results
                self.evaluator.print_evaluation_summary(run)
                
            except Exception as e:
                LOG.error(f"Error evaluating configuration {config.name}: {e}")
                continue
        
        # Sort runs by MRR (descending)
        runs.sort(key=lambda r: r.mrr, reverse=True)
        
        # Print best configuration
        if runs:
            LOG.info(f"Best configuration: {runs[0].configuration.name}")
            self.evaluator.print_evaluation_summary(runs[0])

        return runs