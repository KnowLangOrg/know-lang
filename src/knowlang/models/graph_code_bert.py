import torch
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
from transformers import RobertaModel, RobertaTokenizer
from tree_sitter import Language, Parser


from knowlang.models.types import EmbeddingVector, EmbeddingInputType
from knowlang.models.GraphCodeBERT_search import (
    DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript,
    extract_dataflow
)

# Global caches
_MODEL_CACHE: Dict[str, Tuple[Any, Any, str]] = {}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class GraphCodeBertMode(str, Enum):
    """Operational modes for GraphCodeBERT"""
    BI_ENCODER = "bi-encoder"
    CROSS_ENCODER = "cross-encoder"

# GraphCodeBERT model parameters
CODE_LENGTH = 256
DATA_FLOW_LENGTH = 64
NL_LENGTH = 128
MAX_LENGTH = CODE_LENGTH + DATA_FLOW_LENGTH

class Model(torch.nn.Module):
    """Custom GraphCodeBERT model implementation"""
    
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
    
    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None):
        if code_inputs is not None:
            # Process code with graph information
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)
            inputs_embeddings = self.encoder.embeddings.word_embeddings(code_inputs)
            
            # Calculate node-to-token attention
            nodes_to_token_mask = nodes_mask[:,:,None] & token_mask[:,None,:] & attn_mask
            nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:,:,None]
            
            # Blend embeddings based on the DFG
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings * (~nodes_mask)[:,:,None] + avg_embeddings * nodes_mask[:,:,None]
            
            # Get the final embedding from the encoder
            return self.encoder(
                inputs_embeds=inputs_embeddings, 
                attention_mask=attn_mask,
                position_ids=position_idx
            )[1]  # Use the pooler output (CLS token representation)
        else:
            # Process natural language query
            return self.encoder(
                nl_inputs, 
                attention_mask=nl_inputs.ne(1)
            )[1]  # Use the pooler output (CLS token representation)

def setup_parsers():
    """Setup language parsers for data flow extraction"""
    
    # Define DFG functions for different languages
    dfg_function = {
        'python': DFG_python,
        # 'java': DFG_java,
        # 'ruby': DFG_ruby,
        # 'go': DFG_go,
        # 'php': DFG_php,
        # 'javascript': DFG_javascript
    }
    
    # Load parsers for each language
    parsers = {}
    for lang in dfg_function:
        if lang == 'python':
            import tree_sitter_python
            LANGUAGE = Language(tree_sitter_python.language())
            parser = Parser(LANGUAGE)
        else:        
            raise NotImplementedError("language not supported yet")
        parser = [parser,dfg_function[lang]]    
        parsers[lang]= parser
        
        return parsers

def prepare_code_inputs(code, tokenizer, parsers, lang='python', device=None):
    """Prepare code inputs for the GraphCodeBERT model"""
    if device is None:
        device = get_device()
            
    """Prepare code inputs for the GraphCodeBERT model"""
    parser = parsers.get(lang)
    if not parser:
        raise ValueError(f"No parser available for language {lang}")
    
    # Extract data flow
    code_tokens, dfg = extract_dataflow(code, parser, lang)
    
    # Tokenize code tokens
    code_tokens = [
        tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) 
        for idx, x in enumerate(code_tokens)
    ]
    
    # Build position map
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1] + len(code_tokens[i]))
    
    # Flatten token list
    code_tokens = [y for x in code_tokens for y in x]
    
    # Truncate and add special tokens
    code_tokens = code_tokens[:CODE_LENGTH + DATA_FLOW_LENGTH - 2 - min(len(dfg), DATA_FLOW_LENGTH)]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    
    # Build position index
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    
    # Truncate DFG if necessary
    dfg = dfg[:CODE_LENGTH + DATA_FLOW_LENGTH - len(code_tokens)]
    
    # Add DFG nodes to tokens and position index
    code_tokens += [x[0] for x in dfg]
    position_idx += [0 for _ in dfg]
    code_ids += [tokenizer.unk_token_id for _ in dfg]
    
    # Pad sequences
    padding_length = CODE_LENGTH + DATA_FLOW_LENGTH - len(code_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    code_ids += [tokenizer.pad_token_id] * padding_length
    
    # Reindex DFG nodes
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    
    # Map DFG to code and to other DFG nodes
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
    
    # Create attention mask
    attn_mask = np.zeros((CODE_LENGTH + DATA_FLOW_LENGTH, CODE_LENGTH + DATA_FLOW_LENGTH), dtype=np.bool)
    
    # Calculate indices
    node_index = sum([i > 1 for i in position_idx])
    max_length = sum([i != tokenizer.pad_token_id for i in position_idx])
    
    # Set up attention mask:
    # 1. Sequence can attend to sequence
    attn_mask[:node_index, :node_index] = True
    
    # 2. Special tokens attend to all tokens
    for idx, i in enumerate(code_ids):
        if i in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
            attn_mask[idx, :max_length] = True
    
    # 3. Nodes attend to code tokens they are identified from
    for idx, (a, b) in enumerate(dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True
    
    # 4. Nodes attend to adjacent nodes
    for idx, nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(position_idx):
                attn_mask[idx + node_index, a + node_index] = True
    
    return (
        # Convert to numpy arrays first, then to tensors with batch dimension
        torch.tensor(np.array(code_ids), device=device).unsqueeze(0),
        torch.tensor(np.array(attn_mask), device=device).unsqueeze(0),
        torch.tensor(np.array(position_idx), device=device).unsqueeze(0)
    )

def prepare_nl_inputs(query, tokenizer):
    """Prepare natural language inputs for the GraphCodeBERT model"""
    nl_tokens = tokenizer.tokenize(query)[:NL_LENGTH - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    
    # Pad to fixed length
    padding_length = NL_LENGTH - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length
    
    return torch.tensor([nl_ids])

@lru_cache(maxsize=4)
def _get_model_and_tokenizer(
    model_path: str,
    mode: GraphCodeBertMode = GraphCodeBertMode.BI_ENCODER,
    device: Optional[str] = None
) -> Tuple[Any, Any, str, Dict[str, Any]]:
    """
    Load model, tokenizer, and parsers with caching.
    """
    if device is None:
        device = get_device()
    
    cache_key = f"{model_path}_{mode}_{device}"
    
    if cache_key not in _MODEL_CACHE:
        # Load tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        
        # Load base model
        base_model = RobertaModel.from_pretrained(model_path)
        
        # Create custom model
        if mode == GraphCodeBertMode.BI_ENCODER:
            model = Model(base_model).to(device)
        else:
            raise ValueError(f"Mode {mode} not implemented yet")
        
        # Setup parsers
        parsers = setup_parsers()
        
        _MODEL_CACHE[cache_key] = (model, tokenizer, device, parsers)
    
    return _MODEL_CACHE[cache_key]

def generate_code_embedding(code: str, model_name: str, lang: str = 'python') -> EmbeddingVector:
    """
    Generate embedding for a code snippet using GraphCodeBERT with data flow graph.
    
    Args:
        code: Source code to embed
        lang: Programming language of the code
    
    Returns:
        Embedding vector
    """
    model, tokenizer, device, parsers = _get_model_and_tokenizer(
        model_name,
        GraphCodeBertMode.BI_ENCODER
    )
    
    code_inputs, attn_mask, position_idx = prepare_code_inputs(code, tokenizer, parsers, lang)
    
    with torch.no_grad():
        code_inputs = code_inputs.to(device)
        attn_mask = attn_mask.to(device)
        position_idx = position_idx.to(device)
        
        # Get code vector
        code_vec = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
        
        # Return embedding as list
        return code_vec.cpu().numpy()[0].tolist()

def generate_query_embedding(query: str, model_name: str) -> EmbeddingVector:
    """
    Generate embedding for a natural language query using GraphCodeBERT.
    
    Args:
        query: Natural language query
    
    Returns:
        Embedding vector
    """
    model, tokenizer, device, _ = _get_model_and_tokenizer(
        model_name,
        GraphCodeBertMode.BI_ENCODER
    )
    
    nl_inputs = prepare_nl_inputs(query, tokenizer)
    
    with torch.no_grad():
        nl_inputs = nl_inputs.to(device)
        
        # Get query vector
        nl_vec = model(nl_inputs=nl_inputs)
        
        # Return embedding as list
        return nl_vec.cpu().numpy()[0].tolist()

def generate_embeddings(
    inputs: List[str],
    input_type: EmbeddingInputType,
    model_name: str,
    lang: str = 'python'
) -> List[EmbeddingVector]:
    """
    Generate embeddings for a list of inputs.
    
    Args:
        inputs: List of text/code inputs to embed
        input_type: Type of input (query/code)
        lang: If is_code is True, the programming language of the code
    
    Returns:
        List of embedding vectors
    """
    embeddings = []
    
    for text in inputs:
        if input_type == EmbeddingInputType.DOCUMENT:
            embedding = generate_code_embedding(code=text, lang=lang, model_name=model_name)
        else:
            embedding = generate_query_embedding(query=text, model_name=model_name)
        embeddings.append(embedding)
    
    return embeddings

def calculate_similarity(query: str, code: str, lang: str = 'python') -> float:
    """
    Calculate similarity between a query and a code snippet.
    
    Args:
        query: Natural language query
        code: Code snippet
        lang: Programming language of the code
    
    Returns:
        Similarity score between query and code
    """
    query_embedding = generate_query_embedding(query)
    code_embedding = generate_code_embedding(code, lang)
    
    # Convert to numpy arrays
    query_vec = np.array(query_embedding)
    code_vec = np.array(code_embedding)
    
    # Calculate cosine similarity
    similarity = np.dot(query_vec, code_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(code_vec))
    
    return float(similarity)