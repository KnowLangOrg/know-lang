from pathlib import Path
from knowlang.cli.types import RunEvaluationCommandArgs
from knowlang.cli.utils import create_config
from knowlang.configs import AppConfig
from knowlang.evaluations.config_manager import SearchConfigurationManager
from knowlang.evaluations.evaluation_runner import CodeSearchEvaluator
from knowlang.evaluations.grid_search import EvaluationGridSearch
from knowlang.utils import FancyLogger
from knowlang.vector_stores import VectorStoreError
from knowlang.vector_stores.factory import VectorStoreFactory

LOG = FancyLogger(__name__)


def list_configurations(config_dir: Path):
    """
    List available search configurations.
    
    Args:
        config_dir: Directory containing search configurations
    """
    config_manager = SearchConfigurationManager(config_dir)
    configs = config_manager.list_configurations()
    
    if not configs:
        LOG.info("No configurations found. Creating defaults...")
        config_manager.create_default_configurations()
        configs = config_manager.list_configurations()
    
    LOG.info("Available search configurations:")
    for i, name in enumerate(configs):
        config = config_manager.load_configuration(name)
        if config:
            LOG.info(f"{i+1}. {name}: {config.description}")


async def run_evaluation(
    config: AppConfig,
    args: RunEvaluationCommandArgs,
):
    """
    Run code search evaluation.
    
    Args:
        config: Application configuration
        args: Command-line arguments
    """
    # Create evaluator and configuration manager
    evaluator = CodeSearchEvaluator(config, args.data_dir, args.output_dir)
    config_manager = SearchConfigurationManager(args.config_dir)
    
    # Ensure we have some default configurations
    if not config_manager.list_configurations():
        config_manager.create_default_configurations()
    
    if args.grid_search:
        # Run grid search
        grid_search = EvaluationGridSearch(evaluator, config_manager)
        await grid_search.run_grid_search(args.dataset, args.language, args.limit)
    else:
        # Single configuration evaluation
        search_config = config_manager.load_configuration(args.configuration)
        if not search_config:
            LOG.error(f"Configuration not found: {args.configuration}")
            return
        
        run = await evaluator.evaluate_dataset(
            dataset_name=args.dataset,
            language=args.language,
            config=search_config,
            limit=args.limit
        )
        
        evaluator.print_evaluation_summary(run)


async def run_evaluation_command(args: RunEvaluationCommandArgs) -> None:
    """Execute the run-evaluation command.
    
    Args:
        args: Typed command line arguments
    """
    # If we just want to list configurations, do that and exit
    if args.list_configurations:
        list_configurations(args.config_dir)
        return
    
    # Load configuration
    config = create_config(args.config)
    
    # Initialize vector store
    try:
        VectorStoreFactory.get(config.db, config.embedding)
    except VectorStoreError as e:
        LOG.error(
            "Vector store initialization failed. Please run 'knowlang evaluate prepare' first to index benchmark datasets."
            f"\nError: {str(e)}"
        )
        return
    
    try:
        # Run evaluation
        await run_evaluation(config, args)
    except Exception as e:
        LOG.error(f"Error running evaluation: {e}")