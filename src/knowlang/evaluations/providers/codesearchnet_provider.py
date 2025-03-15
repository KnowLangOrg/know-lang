import json
from pathlib import Path
from typing import List, Optional

from rich.progress import track

from knowlang.evaluations.base import QueryCodePair, DatasetStats
from knowlang.evaluations.dataset_provider import DatasetProvider
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)


class CodeSearchNetProvider(DatasetProvider):
    """Provider for the CodeSearchNet dataset."""
    
    async def load(self, languages: Optional[List[str]] = None, 
                 split: str = "test") -> List[QueryCodePair]:
        """
        Load CodeSearchNet dataset.
        
        Args:
            languages: Filter for programming languages (default: all available)
            split: Dataset split to use (train, valid, test)
            
        Returns:
            List of QueryCodePair objects
        """
        if not languages:
            # CodeSearchNet has these 6 languages
            languages = ["python", "java", "javascript", "go", "ruby", "php"]
        
        pairs = []
        stats = DatasetStats(dataset_name="CodeSearchNet")
        
        for language in languages:
            try:
                lang_dir = self.dataset_dir / language
                if not lang_dir.exists():
                    LOG.warning(f"Language directory not found: {lang_dir}")
                    continue
                
                file_path = lang_dir / f"{split}.jsonl"
                if not file_path.exists():
                    LOG.warning(f"Split file not found: {file_path}")
                    continue
                
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(track(
                        list(f), description=f"Loading {language} {split}")):
                        try:
                            data = json.loads(line)
                            # CodeSearchNet uses docstrings as queries
                            docstring = data.get("docstring", "").strip()
                            code = data.get("code", "").strip()
                            
                            # Skip empty entries
                            if not docstring or not code:
                                continue
                                
                            pair = QueryCodePair(
                                query_id=f"{language}_{split}_{i}",
                                query=docstring,
                                code_id=data.get("url", f"{language}_{split}_{i}_code"),
                                code=code,
                                language=language,
                                metadata={
                                    "repo": data.get("repo", ""),
                                    "path": data.get("path", ""),
                                    "func_name": data.get("func_name", ""),
                                    "original_string": data.get("original_string", ""),
                                }
                            )
                            pairs.append(pair)
                            stats.update_for_pair(pair)
                        except Exception as e:
                            LOG.error(f"Error processing line {i} in {file_path}: {e}")
                            continue
            except Exception as e:
                LOG.error(f"Error loading {language} data: {e}")
                continue
        
        LOG.info(stats.summary())
        return pairs