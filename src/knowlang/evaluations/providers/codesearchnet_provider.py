import json
import gzip
from pathlib import Path
from typing import List, Optional

from rich.progress import track

from knowlang.evaluations.base import QueryCodePair, DatasetStats
from knowlang.evaluations.dataset_provider import DatasetProvider
from knowlang.evaluations.types import DatasetSplitType, DatasetType
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)


class CodeSearchNetProvider(DatasetProvider):
    """Provider for the CodeSearchNet dataset."""
    
    async def load(self, 
                 languages: Optional[List[str]] = None, 
                 splits: Optional[List[DatasetSplitType]] = None) -> List[QueryCodePair]:
        """
        Load CodeSearchNet dataset.
        
        Args:
            languages: Filter for programming languages (default: all available)
            splits: Dataset splits to load (default: all splits)
            
        Returns:
            List of QueryCodePair objects
        """
        if not languages:
            # CodeSearchNet has these 6 languages
            languages = ["python", "java", "javascript", "go", "ruby", "php"]
            
        if not splits:
            # If no splits provided, use all
            splits = [DatasetSplitType.TRAIN, DatasetSplitType.VALID, DatasetSplitType.TEST]
        
        pairs = []
        stats = DatasetStats(dataset_name=DatasetType.CODESEARCHNET.value)
        
        for language in languages:
            try:
                for split in splits:
                    split_value = split.value
                    lang_dir = self.dataset_dir / language / "final" / "jsonl" / split_value
                    if not lang_dir.exists():
                        LOG.warning(f"Language/split directory not found: {lang_dir}")
                        continue
                    
                    # Find all gzipped jsonl files in the split directory
                    file_pattern = f"{language}_{split_value}_*.jsonl.gz"
                    files = list(lang_dir.glob(file_pattern))
                    
                    if not files:
                        LOG.warning(f"No files found matching pattern {file_pattern} in {lang_dir}")
                        continue
                    
                    for file_path in files:
                        try:
                            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                                lines = list(f)
                                for i, line in enumerate(track(
                                    lines, description=f"Loading {language} {split_value} from {file_path.name}")):
                                    try:
                                        data = json.loads(line)
                                        # CodeSearchNet uses docstrings as queries
                                        docstring = data.get("docstring", "").strip()
                                        code = data.get("code", "").strip()
                                        
                                        # Skip empty entries
                                        if not docstring or not code:
                                            continue
                                            
                                        pair = QueryCodePair(
                                            query_id=f"{language}_{split_value}_{i}",
                                            query=docstring,
                                            code_id=data.get("url", f"{language}_{split_value}_{i}_code"),
                                            code=code,
                                            language=language,
                                            metadata={
                                                "repo": data.get("repo", ""),
                                                "path": data.get("path", ""),
                                                "func_name": data.get("func_name", ""),
                                                "original_string": data.get("original_string", ""),
                                            },
                                            dataset_split=split_value
                                        )
                                        pairs.append(pair)
                                        stats.update_for_pair(pair)
                                    except Exception as e:
                                        LOG.error(f"Error processing line {i} in {file_path}: {e}")
                                        continue
                        except Exception as e:
                            LOG.error(f"Error loading file {file_path}: {e}")
                            continue
            except Exception as e:
                LOG.error(f"Error loading {language} data: {e}")
                continue
        
        LOG.info(stats.summary())
        return pairs