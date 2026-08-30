import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, CLIPImageProcessor

from src.model import LaTeXOCRConfig

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_tokenizer_and_processor(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(cfg["language_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    image_processor = CLIPImageProcessor.from_pretrained(cfg["vision_model_name"])
    return tokenizer, image_processor

def build_model_config(cfg: Dict[str, Any]) -> LaTeXOCRConfig:
    return LaTeXOCRConfig(
        vision_model_name = cfg["vision_model_name"],
        language_model_name = cfg["language_model_name"],
        projector_hidden_dim = cfg["projector_hidden_dim"],
        freeze_vision_model = cfg["freeze_vision_model"],
        freeze_language_model = cfg["freeze_language_model"]
    )

def resize_for_added_tokens(model, tokenizer):
    model.language_model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)