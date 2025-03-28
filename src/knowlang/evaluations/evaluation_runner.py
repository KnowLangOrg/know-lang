import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.progress import track
from rich.table import Table

from knowlang.configs import AppConfig, RerankerConfig
from knowlang.configs.retrieval_config import (MultiStageRetrievalConfig,
                                               SearchConfig)
from knowlang.evaluations.base import (EvaluationRun, QueryEvaluationResult,
                                       SearchConfiguration)
from knowlang.evaluations.indexer import QueryManager
from knowlang.evaluations.metrics import MetricsCalculator
from knowlang.evaluations.types import DatasetSplitType, DatasetType
from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.search.search_graph.base import SearchDeps, SearchState
from knowlang.search.search_graph.graph import FirstStageNode, search_graph
from knowlang.utils import FancyLogger
from knowlang.vector_stores.factory import VectorStoreFactory

LOG = FancyLogger(__name__)
console = Console()


class CodeSearchEvaluator:
    """
    Run code search evaluations against indexed datasets.
    
    This class handles running queries and computing standard
    retrieval metrics like MRR and Recall@K.
    """
    
    def __init__(self, config: AppConfig, data_dir: Path, output_dir: Path):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.vector_store = VectorStoreFactory.get(config)
        
        # Make sure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize query manager for loading mappings
        self.query_manager = QueryManager(data_dir)
    
    async def initialize(self):
        """Initialize components."""
        try:
            if hasattr(self.vector_store, 'initialize'):
                self.vector_store.initialize()
            LOG.info(f"Vector store initialized: {type(self.vector_store).__name__}")
        except Exception as e:
            LOG.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def search_with_configuration(self, query: str, config: SearchConfiguration) -> Tuple[List[SearchResult], float]:
        """
        Run search with a specific configuration.
        
        Args:
            query: Search query string
            config: Search configuration
            
        Returns:
            Tuple of (results, query_time)
        """
        start_time = time.time()
        
        try:
            # Create search state
            state = SearchState(
                query=query,
                search_results=[],
                refined_queries={
                    SearchMethodology.KEYWORD: [],
                    SearchMethodology.VECTOR: []
                }
            )
            
            # Create retrieval config from the evaluation configuration
            retrieval_config = MultiStageRetrievalConfig(
                keyword_search=SearchConfig(
                    enabled=config.keyword_search_enabled,
                    top_k=config.keyword_search_top_k,
                    score_threshold=config.keyword_search_threshold,
                    filter=config.filter,
                ),
                vector_search=SearchConfig(
                    enabled=config.vector_search_enabled,
                    top_k=config.vector_search_top_k,
                    score_threshold=config.vector_search_threshold,
                    filter=config.filter,
                )
            )
            
            # Create reranker config
            reranker_config = RerankerConfig(
                model_name=config.reranker_model_name,
                enabled=config.reranking_enabled,
                top_k=config.reranker_top_k,
                relevance_threshold=config.reranker_threshold,
            )
            
            # Use the application's existing model configs, but override retrieval and reranker settings
            eval_config = AppConfig(
                llm=self.config.llm,  
                embedding=self.config.embedding,
                retrieval=retrieval_config,
                reranker=reranker_config,
                db=self.config.db  # Keep the same DB config to use the same store
            )
            
            # Create search dependencies
            deps = SearchDeps(
                store=self.vector_store,
                config=eval_config
            )
            
            # Run the search graph
            result, history = await search_graph.run(FirstStageNode(), state=state, deps=deps)
            
            query_time = time.time() - start_time
            return result.search_results, query_time
            
        except Exception as e:
            LOG.error(f"Error searching with configuration: {e}")
            return [], time.time() - start_time
    
    async def evaluate_query(
        self,
        query_id: str,
        query: str,
        relevant_code_ids: List[str],
        config: SearchConfiguration
    ) -> QueryEvaluationResult:
        """
        Evaluate a single query.
        
        Args:
            query_id: Query identifier
            query: Query text
            relevant_code_ids: List of relevant code IDs
            config: Search configuration
            
        Returns:
            QueryEvaluationResult with metrics
        """
        try:
            # Run search
            results, query_time = await self.search_with_configuration(query, config)
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_metrics(results, relevant_code_ids)
            
            # Create result object
            return QueryEvaluationResult(
                query_id=query_id,
                query=query,
                relevant_code_ids=relevant_code_ids,
                results=results,
                query_time=query_time,
                mrr=metrics.get("mrr", 0.0),
                recall_at_1=metrics.get("recall_at_1", 0.0),
                recall_at_5=metrics.get("recall_at_5", 0.0),
                recall_at_10=metrics.get("recall_at_10", 0.0),
                recall_at_100=metrics.get("recall_at_100", 0.0),
                ndcg_at_10=metrics.get("ndcg_at_10", 0.0)
            )
            
        except Exception as e:
            LOG.error(f"Error evaluating query {query_id}: {e}")
            return QueryEvaluationResult(
                query_id=query_id,
                query=query,
                relevant_code_ids=relevant_code_ids,
                results=[],
                query_time=0.0
            )
    
    async def evaluate_dataset(
        self,
        dataset_name: str,
        language: str,
        config: SearchConfiguration,
        limit: Optional[int] = None
    ) -> EvaluationRun:
        """
        Evaluate a dataset with a specific configuration.
        
        Args:
            dataset_name: Name of the dataset
            language: Programming language to evaluate
            config: Search configuration
            limit: Optional limit on number of queries to evaluate
            
        Returns:
            EvaluationRun with aggregated metrics
        """
        await self.initialize()
        
        # Load query mappings
        query_map = self.query_manager.load_query_mappings(dataset_name)
        if not query_map:
            LOG.error(f"No query mappings found for {dataset_name}")
            return EvaluationRun(
                configuration=config,
                dataset_name=dataset_name,
                language=language,
                num_queries=0
            )
        
        # Filter queries by language if specified
        language_queries = {
            qid: data for qid, data in query_map.items()
            if data.language == language or language == "all"
        }
        
        if not language_queries:
            LOG.warning(f"No queries found for language {language}")
            return EvaluationRun(
                configuration=config,
                dataset_name=dataset_name,
                language=language,
                num_queries=0
            )
        
        # Limit the dataset split type
        selected_queries = {
            qid: data for qid, data in language_queries.items()
            if data.dataset_split == config.dataset_split
        }
        
        # Limit number of queries if specified
        if limit and limit < len(language_queries):
            # Take a random sample for more representative results
            import random
            query_ids = list(language_queries.keys())
            random.shuffle(query_ids)
            selected_queries = {qid: language_queries[qid] for qid in query_ids[:limit]}
        else:
            selected_queries = language_queries
        
        LOG.info(f"Evaluating {len(selected_queries)} queries for {dataset_name} ({language})")
        
        # Evaluate each query
        query_results : List[QueryEvaluationResult] = []
        total_time = 0.0
        
        for query_id, data in track(selected_queries.items(), description=f"Evaluating {dataset_name}"):
            query = data.query
            relevant_code = [data.code_id]
            
            result = await self.evaluate_query(
                query_id=query_id,
                query=query,
                relevant_code_ids=relevant_code,
                config=config
            )
            LOG.debug(f"Query Evaluation Results: \n{result.model_dump(exclude={'results', 'query_id'})}")
            
            query_results.append(result)
            total_time += result.query_time
        
        # Aggregate metrics
        num_queries = len(query_results)
        if num_queries == 0:
            return EvaluationRun(
                configuration=config,
                dataset_name=dataset_name,
                language=language,
                num_queries=0
            )
        
        mrr = sum(r.mrr for r in query_results) / num_queries
        recall_at_1 = sum(r.recall_at_1 for r in query_results) / num_queries
        recall_at_5 = sum(r.recall_at_5 for r in query_results) / num_queries
        recall_at_10 = sum(r.recall_at_10 for r in query_results) / num_queries
        recall_at_100 = sum(r.recall_at_100 for r in query_results) / num_queries
        ndcg_at_10 = sum(r.ndcg_at_10 for r in query_results) / num_queries
        avg_query_time = total_time / num_queries
        
        # Create evaluation run
        run = EvaluationRun(
            configuration=config,
            dataset_name=dataset_name,
            language=language,
            mrr=mrr,
            recall_at_1=recall_at_1,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            recall_at_100=recall_at_100,
            ndcg_at_10=ndcg_at_10,
            avg_query_time=avg_query_time,
            num_queries=num_queries
        )
        
        # Save results
        self.save_evaluation_run(run)
        
        return run
    
    def save_evaluation_run(self, run: EvaluationRun) -> None:
        """
        Save evaluation run to a file.
        
        Args:
            run: Evaluation run to save
        """
        file_name = f"{run.dataset_name}_{run.language}_{run.configuration.name}_{run.timestamp.replace(' ', '_').replace(':', '-')}.json"
        file_path = self.output_dir / file_name
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(run.model_dump(), f, indent=2)
        
        LOG.info(f"Saved evaluation run to {file_path}")
    
    def print_evaluation_summary(self, run: EvaluationRun) -> None:
        """
        Print a human-readable summary of evaluation results.
        
        Args:
            run: Evaluation run to summarize
        """
        table = Table(title=f"Evaluation Summary: {run.dataset_name} ({run.language})")
        
        table.add_column("Metric", justify="right", style="cyan")
        table.add_column("Value", justify="center", style="green")
        
        table.add_row("Configuration", run.configuration.name)
        table.add_row("Description", run.configuration.description)
        table.add_row("Queries", str(run.num_queries))
        table.add_row("MRR", f"{run.mrr:.4f}")
        table.add_row("Recall@1", f"{run.recall_at_1:.4f}")
        table.add_row("Recall@5", f"{run.recall_at_5:.4f}")
        table.add_row("Recall@10", f"{run.recall_at_10:.4f}")
        table.add_row("NDCG@10", f"{run.ndcg_at_10:.4f}")
        table.add_row("Avg Query Time", f"{run.avg_query_time:.4f} seconds")
        
        console.print(table)
        
    async def generate_training_data(
        self,
        dataset_name: str = DatasetType.CODESEARCHNET,
        language: str = 'python',
        vector_top_k: int = 100,
        neg_examples: int = 2,
        limit: Optional[int] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, int]:
        """
        Generate training data for reranker model from search results.
        
        This method:
        1. Runs queries through the first-stage retrieval (GraphCodeBERT)
        2. Identifies ground truth (relevant) results (label=1)
        3. Identifies hard negatives (high similarity, non-relevant results) (label=0)
        4. Saves the data to JSONL files split by dataset_split (train/valid/test)
        
        Args:
            dataset_name: Name of the dataset
            language: Programming language filter
            vector_top_k: Number of results to retrieve from vector search
            neg_examples: Number of negative examples to include per query (default: 3)
            limit: Optional limit on number of queries to process
            output_dir: Directory to save JSONL files (defaults to self.output_dir)
            
        Returns:
            Dictionary with counts of examples by split
        """
        await self.initialize()
        
        # Set output directory
        output_dir = output_dir or self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load query mappings
        query_map = self.query_manager.load_query_mappings(dataset_name)
        if not query_map:
            LOG.error(f"No query mappings found for {dataset_name}")
            return {"error": "No query mappings found"}
        
        # Filter queries by language if specified
        language_queries = {
            qid: data for qid, data in query_map.items()
            if data.language == language or language == "all"
        }
        
        if not language_queries:
            LOG.warning(f"No queries found for language {language}")
            return {"error": "No queries found for language"}
        
        # Limit number of queries if specified
        if limit and limit < len(language_queries):
            import random
            query_ids = list(language_queries.keys())
            random.shuffle(query_ids)
            selected_queries = {qid: language_queries[qid] for qid in query_ids[:limit]}
        else:
            selected_queries = language_queries
        
        LOG.info(f"Generating training data for {len(selected_queries)} queries for {dataset_name} ({language})")
        
        # Create a configuration for vector search only (first-stage retrieval)
        vector_config = SearchConfiguration(
            name="vector_only_for_training",
            description="Vector search only for generating training data",
            keyword_search_enabled=False,
            vector_search_enabled=True,
            reranking_enabled=False,
            vector_search_top_k=vector_top_k,
            vector_search_threshold=0.0  # No threshold to get all results
        )
        
        # Initialize counters for each split
        examples_count = {
            DatasetSplitType.TRAIN.value: 0,
            DatasetSplitType.VALID.value: 0,
            DatasetSplitType.TEST.value: 0,
            "unknown": 0
        }
        
        # Create output files for each split
        file_paths = {}
        for split in [DatasetSplitType.TRAIN.value, DatasetSplitType.VALID.value, DatasetSplitType.TEST.value, "unknown"]:
            file_path = output_dir / f"{dataset_name}_{language}_{split}_reranker_training.jsonl"
            file_paths[split] = file_path
            # Create or truncate the file
            with open(file_path, "w", encoding="utf-8") as f:
                pass
        
        # Process each query
        for query_id, data in track(selected_queries.items(), description=f"Generating data for {dataset_name}"):
            query = data.query
            ground_truth_id = data.code_id
            
            # Get the split from query mapping
            query_split = data.dataset_split.value
            
            # Configure the Search Filter
            vector_config.filter = {"dataset_split" : {"$eq": query_split}}
            
            # Run vector search to get candidates (hard negatives will come from these results)
            results, _ = await self.search_with_configuration(query, vector_config)
            
            # Group examples by split
            split_examples = {
                DatasetSplitType.TRAIN.value: [],
                DatasetSplitType.VALID.value: [],
                DatasetSplitType.TEST.value: [],
                "unknown": []
            }
            
            # Check if we have enough results for this query (positive + neg_examples)
            if len(results) < neg_examples:
                LOG.warning(f"Not enough search results for query {query_id}. Found {len(results)}, need at least {neg_examples}.")
                continue
            
            # First, find the positive example (ground truth)
            positive_example = None
            negative_examples = []
            
            for result in results:
                code_id = result.metadata.get("id")
                code = result.document
                
                # Get dataset_split from result metadata or fall back to query split
                dataset_split = result.metadata.get("dataset_split", query_split)
                
                # If we still don't have a valid split, use "unknown"
                if dataset_split not in [s.value for s in DatasetSplitType]:
                    dataset_split = "unknown"
                
                # Create JSON entry
                json_entry = {
                    "label": 1 if code_id == ground_truth_id else 0,
                    "query_id": query_id,
                    "url": code_id,
                    "query": query,
                    "code": code
                }
                
                # Separate positive and negative examples
                if code_id == ground_truth_id:
                    positive_example = (json_entry, dataset_split)
                else:
                    negative_examples.append((json_entry, dataset_split))
            
            # Skip if no positive example found
            if positive_example is None:
                LOG.warning(f"No positive example found for query {query_id}. Skipping.")
                continue
            
            # Limit negative examples to the desired number
            negative_examples = negative_examples[:neg_examples]
            
            # Add positive example first
            json_entry, dataset_split = positive_example
            json_line = json.dumps(json_entry) + "\n"
            split_examples[dataset_split].append(json_line)
            examples_count[dataset_split] += 1
            
            # Then add negative examples
            for json_entry, dataset_split in negative_examples:
                json_line = json.dumps(json_entry) + "\n"
                split_examples[dataset_split].append(json_line)
                examples_count[dataset_split] += 1
            
            # Write examples to appropriate files
            for split, examples in split_examples.items():
                if examples:  # Only write if we have examples for this split
                    with open(file_paths[split], "a", encoding="utf-8") as f:
                        f.writelines(examples)
        
        LOG.info(f"Generated training data: {examples_count}")
        return examples_count