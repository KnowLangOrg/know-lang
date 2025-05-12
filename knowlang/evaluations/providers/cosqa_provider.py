import json
from pathlib import Path
from typing import List, Optional

from rich.progress import track

from knowlang.evaluations.base import QueryCodePair, DatasetStats
from knowlang.evaluations.dataset_provider import DatasetProvider
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)


class CoSQAProvider(DatasetProvider):
    """Provider for the CoSQA dataset."""
    
    async def load(self, languages: Optional[List[str]] = None, 
                 split: str = "test") -> List[QueryCodePair]:
        """
        Load CoSQA dataset.
        
        Args:
            languages: Filter for programming languages (only 'python' available)
            split: Dataset split to use (defaults to 'test')
            
        Returns:
            List of QueryCodePair objects
        """
        # CoSQA only has Python
        if languages and "python" not in languages:
            LOG.warning("CoSQA only contains Python code snippets")
            return []
        
        pairs = []
        stats = DatasetStats(dataset_name="CoSQA")
        
        try:
            file_path = self.dataset_dir / f"{split}.jsonl"
            if not file_path.exists():
                LOG.warning(f"CoSQA file not found: {file_path}")
                return []
            
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(track(
                    list(f), description=f"Loading CoSQA {split}")):
                    try:
                        data = json.loads(line)
                        
                        # CoSQA uses natural language queries
                        query = data.get("query", "").strip()
                        code = data.get("code", "").strip()
                        
                        # Skip empty entries
                        if not query or not code:
                            continue
                            
                        pair = QueryCodePair(
                            query_id=f"cosqa_{split}_{i}",
                            query=query,
                            code_id=data.get("id", f"cosqa_{split}_{i}_code"),
                            code=code,
                            language="python",
                            metadata={
                                "url": data.get("url", ""),
                                "is_answer": data.get("label", 1) == 1
                            },
                            is_relevant=data.get("label", 1) == 1
                        )
                        pairs.append(pair)
                        stats.update_for_pair(pair)
                    except Exception as e:
                        LOG.error(f"Error processing line {i} in {file_path}: {e}")
                        continue
        except Exception as e:
            LOG.error(f"Error loading CoSQA data: {e}")
            
        LOG.info(stats.summary())
        return pairs