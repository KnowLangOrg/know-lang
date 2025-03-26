from __future__ import annotations
from functools import lru_cache
import os
from typing import List, Any
from knowlang.configs import RerankerConfig
from knowlang.configs.config import AppConfig
from knowlang.search.base import SearchResult


from enum import Enum
from typing import Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from transformers import RobertaConfig, RobertaModel, RobertaForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from knowlang.models.utils import get_device
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

_RERANKER_CACHE: Dict[str, Tuple[Any, Any, str]] = {}



class RerankerType(str, Enum):
    """Enum for reranker types."""
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"


@lru_cache(maxsize=4)
def _get_tokenizer_and_model(
    model_name: str,
    config: Optional[RobertaConfig] = None,
    reranker_type: RerankerType = RerankerType.PAIRWISE,
    device: str = get_device()
) -> Tuple[Any, Any, str]:
    """
    Load tokenizer and model with caching.
    
    Args:
        model_name: Name of the model to load
        config: Optional model configuration
        reranker_type: Type of reranker (pointwise or pairwise)
        device: Device to load the model on
        
    Returns:
        Tuple of (tokenizer, model, device)
    """
    cache_key = f"{model_name}_{reranker_type}_{device}"
    
    if cache_key not in _RERANKER_CACHE:
        LOG.info(f"Loading reranker model and tokenizer for {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Create model config if not provided
        if config is None:
            model_config = RobertaConfig.from_pretrained(
                model_name,
                num_labels=2,
                finetuning_task="reranking"
            )
        else:
            model_config = config
            
        # Load model
        model = CodeBERTReranker.from_pretrained(
            model_path=model_name,
            config=model_config,
            reranker_type=reranker_type,
            device=device
        )
        
        # Cache model and tokenizer
        _RERANKER_CACHE[cache_key] = (tokenizer, model, device)
        LOG.info(f"Reranker model and tokenizer loaded successfully")
    else:
        LOG.debug(f"Using cached reranker model and tokenizer for {model_name}")
    
    return _RERANKER_CACHE[cache_key]


class KnowLangReranker:
    """Base class for KnowLang rerankers."""
    def __init__(self, config: RerankerConfig):
        """Initialize the reranker with a configuration."""
        self.config = config

        if not config.enabled:
            # Skip loading model and tokenizer if reranking is disabled
            self.tokenizer = None
            self.reranker = None
            self.device = None
            return
            
        # Get tokenizer and model from cache
        self.tokenizer, self.reranker, self.device = _get_tokenizer_and_model(
            model_name=config.model_name,
            reranker_type=RerankerType.PAIRWISE,  # Default to pairwise reranking
            device=get_device()
        )
    
    def _get_score(self, query_tokens, search_result: SearchResult) -> float:
        """Get the relevance score for a search result."""
        code_tokens = self.tokenizer.tokenize(search_result.document)
        
        # Account for [CLS], [SEP], [SEP] with "- 3"
        max_tokens = self.config.max_sequence_length - 3
        
        # Truncate or pad sequences
        if len(query_tokens) + len(code_tokens) > max_tokens:
            # Prioritize code by allocating more tokens to it
            query_max = min(64, int(max_tokens * 0.2))
            code_max = max_tokens - min(len(query_tokens), query_max)
            
            if len(query_tokens) > query_max:
                query_tokens = query_tokens[:query_max]
            if len(code_tokens) > code_max:
                code_tokens = code_tokens[:code_max]
        
        # Build token sequence
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        tokens = [cls_token] + query_tokens + [sep_token] + code_tokens + [sep_token]
        token_type_ids = [0] + [0] * (len(query_tokens) + 1) + [1] * (len(code_tokens) + 1)
        
        # Convert tokens to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # Pad sequences
        padding_length = self.config.max_sequence_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        token_type_ids = token_type_ids + [0] * padding_length
        
        score = self.reranker.get_score(
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device),
            attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device),
            token_type_ids = torch.tensor([token_type_ids], dtype=torch.long, device=self.device),
        )

        return score.item()

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results based on the query."""

        if not self.config.enabled:
            LOG.debug("Reranking disabled, returning original results")
            return results

        query_tokens = self.tokenizer.tokenize(query)
        ranked_results = results.copy()
        for result in ranked_results:
            result.score = self._get_score(query_tokens, result)
        
        # Sort results by score in descending order
        ranked_results.sort(key=lambda x: x.score, reverse=True)

        return ranked_results



class CodeBERTReranker(nn.Module):
    """
    CodeBERT-based reranker for code search
    
    This model takes a query and code candidate as input and outputs a relevance score.
    For fine-tuning, it can be trained with pointwise or pairwise ranking approaches.
    """

    @classmethod
    def from_pretrained(cls, 
        model_path: str, 
        config: Optional[RobertaConfig] = None, 
        margin: float = 0.3, 
        reranker_type: Union[str, RerankerType] = RerankerType.PAIRWISE,
        device: Optional[str] = get_device()
    ):
        """
        Create a CodeBERTReranker instance from a pretrained state dictionary
        
        Args:
            model_path: Path to the saved model state dict
            config: Optional RobertaConfig object
            margin: Margin for pairwise ranking loss
            reranker_type: 'pointwise' or 'pairwise' reranking approach
            device: Device to load the model on (defaults to CPU if None)
            
        Returns:
            Initialized CodeBERTReranker with loaded weights
        """
        # Create a dummy model name that won't be used for loading weights
        placeholder_model_name = 'microsoft/codebert-base'
        
        # Initialize the model structure
        model = cls(
            model_name_or_path=placeholder_model_name,
            config=config,
            margin=margin,
            reranker_type=reranker_type
        )
        
        if not os.path.exists(model_path):
            LOG.info(f"Model path not found: {model_path}, attempting to download from Hugging Face Hub")
            from huggingface_hub import hf_hub_download

            # Download the pytorch_model.bin file
            model_path = hf_hub_download(
                repo_id=model_path,
                filename="pytorch_model.bin",
            )

            LOG.info(f"Model downloaded from {model_path}")

        # Load state dict
        model_bin_path = os.path.join(model_path, 'pytorch_model.bin')
        if not os.path.exists(model_bin_path):
            raise ValueError(f"Model path not found: {model_path}")
        state_dict = torch.load(model_bin_path)
        
        
        # Handle loading based on reranker type
        if model.reranker_type == RerankerType.PAIRWISE:
            # Extract classifier weights from state dict
            classifier_weights = {}
            model_weights = {}
            
            for key, value in state_dict.items():
                if key.startswith('classifier'):
                    # Remove the 'classifier.' prefix for the classifier layers
                    classifier_key = key
                    if key.startswith('classifier.'):
                        classifier_key = key[len('classifier.'):]
                    classifier_weights[classifier_key] = value
                else:
                    if key.startswith('model.'):
                        key = key[len('model.'):]
                    model_weights[key] = value
            
            # Load base model weights
            model.model.load_state_dict(model_weights)
            
            # Load classifier weights if they exist
            if classifier_weights:
                model.classifier.load_state_dict(classifier_weights)
        elif model.reranker_type == RerankerType.POINTWISE:
            # The model is already a RobertaForSequenceClassification
            model.model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Invalid reranker type: {model.reranker_type}")
        
        # Set model to eval mode by default
        model.eval()
        model.to(device)
        
        return model
    
    def __init__(self, 
                 model_name_or_path: str = 'microsoft/codebert-base', 
                 config: Optional[RobertaConfig] = None, 
                 margin: float = 0.3, 
                 reranker_type: Union[str, RerankerType] = RerankerType.POINTWISE):
        """
        Initialize the reranker model
        
        Args:
            model_name_or_path: Path to pretrained model or model name (e.g., 'microsoft/codebert-base')
            config: Model configuration (RobertaConfig)
            margin: Margin for pairwise ranking loss
            reranker_type: 'pointwise' or 'pairwise' reranking approach
        """
        super(CodeBERTReranker, self).__init__()
        
        # Validate and convert string to enum if needed
        if isinstance(reranker_type, str):
            reranker_type = RerankerType(reranker_type)
        
        self.reranker_type = reranker_type
        self.margin = margin
        
        if config is None:
            self.config = RobertaConfig.from_pretrained(model_name_or_path)
        else:
            self.config = config
            
        # For pointwise reranking, we use the standard sequence classification model
        if reranker_type == RerankerType.POINTWISE:
            self.model = RobertaForSequenceClassification.from_pretrained(
                model_name_or_path, 
                config=self.config
            )
        # For pairwise reranking, we use the base model and add our own classification head
        else:
            self.model = RobertaModel.from_pretrained(model_name_or_path)
            self.classifier = nn.Linear(self.config.hidden_size, 1)
            
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                token_type_ids: Optional[torch.Tensor] = None, 
                position_ids: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None, 
                pos_input_ids: Optional[torch.Tensor] = None, 
                pos_attention_mask: Optional[torch.Tensor] = None, 
                pos_token_type_ids: Optional[torch.Tensor] = None,
                neg_input_ids: Optional[torch.Tensor] = None, 
                neg_attention_mask: Optional[torch.Tensor] = None, 
                neg_token_type_ids: Optional[torch.Tensor] = None
               ) -> Union[SequenceClassifierOutput, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        For pointwise reranking:
            - Input: query-code pair
            - Output: relevance score
        
        For pairwise reranking:
            - Input: query with positive code example and negative code example
            - Output: margin ranking loss
        """
        
        if self.reranker_type == RerankerType.POINTWISE:
            # Standard sequence classification
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                labels=labels
            )
            
            return outputs  # (loss), logits, (hidden_states), (attentions)
            
        else:  # pairwise reranking
            # Process positive example
            pos_outputs = self.model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                token_type_ids=pos_token_type_ids
            )
            
            # Process negative example
            neg_outputs = self.model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
                token_type_ids=neg_token_type_ids
            )
            
            # Get CLS token representation for both examples
            pos_pooled = pos_outputs.last_hidden_state[:, 0]
            neg_pooled = neg_outputs.last_hidden_state[:, 0]
            
            # Calculate scores
            pos_score = self.classifier(pos_pooled)
            neg_score = self.classifier(neg_pooled)
            
            # Calculate loss
            loss = None
            if labels is not None:
                # Margin ranking loss
                loss_fn = nn.MarginRankingLoss(margin=self.margin)
                # All labels should be 1 as positive should be ranked higher
                target = torch.ones_like(pos_score)
                loss = loss_fn(pos_score, neg_score, target)
            
            return {
                'loss': loss,
                'pos_score': pos_score,
                'neg_score': neg_score
            }
    
    def get_score(self, 
                  input_ids: torch.Tensor,
                  attention_mask: torch.Tensor,
                  token_type_ids: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        """
        Get relevance score for a query-code pair
        Used for inference in both pointwise and pairwise reranking
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)
            
        Returns:
            Relevance scores
        """
        if self.reranker_type == RerankerType.POINTWISE:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits = outputs.logits
            # For binary classification, return the positive class score
            if logits.shape[-1] == 2:
                return logits[:, 1]
            return logits
            
        else:  # pairwise reranking
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            pooled = outputs.last_hidden_state[:, 0]
            score = self.classifier(pooled)
            return score