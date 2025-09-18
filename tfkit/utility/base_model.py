"""Base model class for all TFKit tasks."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseTFKitModel(nn.Module, ABC):
    """Base class for all TFKit task models.
    
    Provides common functionality for all TFKit models including:
    - Consistent initialization patterns
    - Predictor setup
    - Cache management
    - Utility methods for model dimensions
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, pretrained: PreTrainedModel, 
                 maxlen: int = 512, **kwargs) -> None:
        """Initialize the base model.
        
        Args:
            tokenizer: The tokenizer for text processing
            pretrained: The pretrained transformer model
            maxlen: Maximum sequence length
            **kwargs: Additional arguments passed to subclasses
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.maxlen = maxlen
        self.vocab_size = max(pretrained.config.vocab_size, tokenizer.__len__())
        
        # Initialize predictor - to be implemented by subclasses
        self.predictor: Optional[Any] = None
        self.predict: Optional[Callable] = None
        
    def _setup_predictor(self, predictor_class: type, preprocessor_class: type) -> None:
        """Setup predictor and prediction method.
        
        Args:
            predictor_class: The predictor class to instantiate
            preprocessor_class: The preprocessor class to use with the predictor
        """
        predictor = predictor_class(self, preprocessor_class)
        self.predictor = predictor
        self.predict = predictor.predict
        
    def clean_cache(self) -> None:
        """Clean model cache - default implementation."""
        if hasattr(self, 'encoder_outputs'):
            self.encoder_outputs = None
        if hasattr(self, 'past_key_values'):
            self.past_key_values = None
            
    @abstractmethod
    def forward(self, batch_data: Dict[str, Any], eval: bool = False, 
                **kwargs) -> Union[torch.Tensor, Dict[str, Any]]:
        """Forward pass - must be implemented by subclasses.
        
        Args:
            batch_data: Dictionary containing batch data
            eval: Whether in evaluation mode
            **kwargs: Additional arguments
            
        Returns:
            Loss tensor during training or results dictionary during evaluation
        """
        pass
        
    def get_hidden_size(self) -> int:
        """Get the hidden size of the pretrained model.
        
        Returns:
            Hidden size dimension
        """
        return self.pretrained.config.hidden_size
        
    def get_vocab_size(self) -> int:
        """Get the vocabulary size.
        
        Returns:
            Vocabulary size
        """
        return self.vocab_size
