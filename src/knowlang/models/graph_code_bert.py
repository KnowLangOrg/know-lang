from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from knowlang.models.types import EmbeddingVector

# Global model cache
_MODEL_CACHE: Dict[str, Tuple[Any, Any, str]] = {}

class GraphCodeBertMode(str, Enum):
    """Operational modes for GraphCodeBERT"""
    BI_ENCODER = "bi-encoder"
    CROSS_ENCODER = "cross-encoder"

@lru_cache(maxsize=8)
def _get_model_and_tokenizer(
    model_path: str, 
    mode: GraphCodeBertMode = GraphCodeBertMode.BI_ENCODER,
    device: Optional[str] = None
) -> Tuple[Any, Any, str]:
    """
    Load model and tokenizer from HuggingFace, with caching.
    Returns cached version if already loaded.
    
    Args:
        model_path: Path or identifier of the model in HuggingFace hub
        mode: Mode of operation (bi-encoder or cross-encoder)
        device: Device to use (cuda/cpu)
        
    Returns:
        Tuple of (model, tokenizer, device)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cache_key = f"{model_path}_{mode}_{device}"
    
    if cache_key not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if mode == GraphCodeBertMode.BI_ENCODER:
            model = AutoModel.from_pretrained(model_path).to(device)
        elif mode == GraphCodeBertMode.CROSS_ENCODER:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=1  # Regression task for scoring
            ).to(device)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        _MODEL_CACHE[cache_key] = (model, tokenizer, device)
    
    return _MODEL_CACHE[cache_key]

def generate_embeddings(
    inputs: List[str], 
) -> List[EmbeddingVector]:
    """
    Generate embeddings using GraphCodeBERT in bi-encoder mode.
    
    Args:
        inputs: List of text inputs to embed
    
    Returns:
        List of embedding vectors
    """
    model, tokenizer, device = _get_model_and_tokenizer(
        "microsoft/graphcodebert-base",
        GraphCodeBertMode.BI_ENCODER,
        None
    )
    
    embeddings = []
    
    with torch.no_grad():
        for text in inputs:
            # Tokenize input
            tokens = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            # Get model output
            outputs = model(**tokens)
            
            # Use CLS token embedding as the document embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0].tolist())
    
    return embeddings

def calculate_relevance_scores(
    query: str,
    code_snippets: List[str],
) -> List[float]:
    """
    Score query-code pairs using GraphCodeBERT in cross-encoder mode.
    
    Args:
        query: Natural language query
        code_snippets: List of code snippets to score
        
    Returns:
        List of similarity scores for each query-code pair
    """
    model, tokenizer, device = _get_model_and_tokenizer(
        "microsoft/graphcodebert-base",
        GraphCodeBertMode.CROSS_ENCODER,
        None
    )
    
    scores = []
    
    with torch.no_grad():
        for code in code_snippets:
            # Tokenize query and code as a pair
            inputs = tokenizer(
                query,
                code,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            # Get model output
            outputs = model(**inputs)
            
            # Get score (logit)
            score = outputs.logits.item()
            scores.append(score)
    
    return scores