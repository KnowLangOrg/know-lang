from typing import Dict

from knowlang.cli.types import PrepareDatasetCommandArgs
from knowlang.cli.utils import create_config
from knowlang.configs import AppConfig
from knowlang.utils import FancyLogger
from knowlang.evaluations.base import DatasetType
from knowlang.evaluations.dataset_manager import DatasetManager

LOG = FancyLogger(__name__)


async def _prepare_datasets(
    config: AppConfig,
    args: PrepareDatasetCommandArgs
) -> Dict:
    """
    Prepare datasets for evaluation.
    
    Args:
        config: Application configuration
        args: Command line arguments
        
    Returns:
        Dictionary with preparation results
    """
    # Create dataset manager
    manager = DatasetManager(config, args.data_dir, args.output_dir)
    
    # Determine which datasets to prepare
    datasets_to_prepare = []
    if args.dataset == "all":
        datasets_to_prepare = [DatasetType.CODESEARCHNET, DatasetType.COSQA]
    elif args.dataset == "codesearchnet":
        datasets_to_prepare = [DatasetType.CODESEARCHNET]
    elif args.dataset == "cosqa":
        datasets_to_prepare = [DatasetType.COSQA]
    else:
        LOG.error(f"Unknown dataset: {args.dataset}")
        return {"error": f"Unknown dataset: {args.dataset}"}
    
    results = {}
    
    # Prepare each dataset
    for dataset_type in datasets_to_prepare:
        LOG.info(f"Preparing dataset: {dataset_type}")
        result = await manager.prepare_dataset(
            dataset_type=dataset_type,
            languages=args.languages,
            split=args.split
        )
        results[str(dataset_type)] = result
    
    return results

async def prepare_dataset_command(args: PrepareDatasetCommandArgs) -> None:
    """Execute the prepare-dataset command.
    
    Args:
        args: Typed command line arguments
    """
    # Load configuration
    config = create_config(args.config)
    
    try:
        # Prepare datasets
        results = await _prepare_datasets(config, args)
        
        # Check for errors
        errors = [
            f"{dataset}: {result['error']}" 
            for dataset, result in results.items() 
            if "error" in result
        ]
        
        if errors:
            LOG.error("Errors occurred during dataset preparation:")
            for error in errors:
                LOG.error(f"  - {error}")
        else:
            # Log success
            LOG.info("Dataset preparation complete")
            for dataset, result in results.items():
                LOG.info(f"  - {dataset}: {result['pairs_loaded']} pairs loaded, "
                         f"{result['indexed_snippets']} snippets indexed")
                         
    except Exception as e:
        LOG.error(f"Error preparing datasets: {e}")