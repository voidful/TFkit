"""Tests for the base model class."""

import pytest
import torch
from torch import nn
from typing import Dict, Any, Union

from tfkit.utility.base_model import BaseTFKitModel


class MockPredictor:
    """Mock predictor for testing."""
    
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
    
    def predict(self, **kwargs):
        return ["mock_prediction"]


class MockPreprocessor:
    """Mock preprocessor for testing."""
    pass


class TestModel(BaseTFKitModel):
    """Test implementation of BaseTFKitModel."""
    
    def __init__(self, tokenizer, pretrained, maxlen=512, **kwargs):
        super().__init__(tokenizer, pretrained, maxlen, **kwargs)
        self.test_layer = nn.Linear(self.get_hidden_size(), 2)
        self._setup_predictor(MockPredictor, MockPreprocessor)
    
    def forward(self, batch_data: Dict[str, Any], eval: bool = False, 
                **kwargs) -> Union[torch.Tensor, Dict[str, Any]]:
        """Mock forward implementation."""
        if eval:
            return {"mock": "result"}
        return torch.tensor(1.0, requires_grad=True)


class TestBaseTFKitModel:
    """Test cases for BaseTFKitModel."""
    
    def test_initialization(self, mock_tokenizer, mock_pretrained):
        """Test model initialization."""
        model = TestModel(mock_tokenizer, mock_pretrained, maxlen=256)
        
        assert model.tokenizer == mock_tokenizer
        assert model.pretrained == mock_pretrained
        assert model.maxlen == 256
        assert model.vocab_size == max(mock_pretrained.config.vocab_size, len(mock_tokenizer))
        assert model.predictor is not None
        assert model.predict is not None
    
    def test_predictor_setup(self, mock_tokenizer, mock_pretrained):
        """Test predictor setup functionality."""
        model = TestModel(mock_tokenizer, mock_pretrained)
        
        assert isinstance(model.predictor, MockPredictor)
        assert callable(model.predict)
        assert model.predict() == ["mock_prediction"]
    
    def test_get_hidden_size(self, mock_tokenizer, mock_pretrained):
        """Test hidden size retrieval."""
        model = TestModel(mock_tokenizer, mock_pretrained)
        
        assert model.get_hidden_size() == mock_pretrained.config.hidden_size
    
    def test_get_vocab_size(self, mock_tokenizer, mock_pretrained):
        """Test vocabulary size retrieval."""
        model = TestModel(mock_tokenizer, mock_pretrained)
        
        expected_vocab_size = max(mock_pretrained.config.vocab_size, len(mock_tokenizer))
        assert model.get_vocab_size() == expected_vocab_size
    
    def test_clean_cache_with_attributes(self, mock_tokenizer, mock_pretrained):
        """Test cache cleaning when attributes exist."""
        model = TestModel(mock_tokenizer, mock_pretrained)
        
        # Add cache attributes
        model.encoder_outputs = torch.tensor([1, 2, 3])
        model.past_key_values = torch.tensor([4, 5, 6])
        
        model.clean_cache()
        
        assert model.encoder_outputs is None
        assert model.past_key_values is None
    
    def test_clean_cache_without_attributes(self, mock_tokenizer, mock_pretrained):
        """Test cache cleaning when attributes don't exist."""
        model = TestModel(mock_tokenizer, mock_pretrained)
        
        # Should not raise an error
        model.clean_cache()
    
    def test_forward_training_mode(self, mock_tokenizer, mock_pretrained, mock_batch_data):
        """Test forward pass in training mode."""
        model = TestModel(mock_tokenizer, mock_pretrained)
        
        result = model.forward(mock_batch_data, eval=False)
        
        assert isinstance(result, torch.Tensor)
        assert result.requires_grad
    
    def test_forward_eval_mode(self, mock_tokenizer, mock_pretrained, mock_batch_data):
        """Test forward pass in evaluation mode."""
        model = TestModel(mock_tokenizer, mock_pretrained)
        
        result = model.forward(mock_batch_data, eval=True)
        
        assert isinstance(result, dict)
        assert "mock" in result
    
    def test_model_parameters(self, mock_tokenizer, mock_pretrained):
        """Test that model has learnable parameters."""
        model = TestModel(mock_tokenizer, mock_pretrained)
        
        params = list(model.parameters())
        assert len(params) > 0
        
        # Test that some parameters require gradients
        trainable_params = [p for p in params if p.requires_grad]
        assert len(trainable_params) > 0
    
    def test_model_device_placement(self, mock_tokenizer, mock_pretrained):
        """Test model device placement."""
        model = TestModel(mock_tokenizer, mock_pretrained)
        
        # Test CPU placement (default)
        for param in model.parameters():
            assert param.device.type == 'cpu'
        
        # Test GPU placement if available
        if torch.cuda.is_available():
            model = model.cuda()
            for param in model.parameters():
                assert param.device.type == 'cuda'
    
    def test_model_mode_switching(self, mock_tokenizer, mock_pretrained):
        """Test switching between train and eval modes."""
        model = TestModel(mock_tokenizer, mock_pretrained)
        
        # Default should be training mode
        assert model.training
        
        # Switch to eval mode
        model.eval()
        assert not model.training
        
        # Switch back to training mode
        model.train()
        assert model.training
    
    def test_kwargs_passing(self, mock_tokenizer, mock_pretrained):
        """Test that kwargs are properly passed."""
        custom_arg = "test_value"
        model = TestModel(mock_tokenizer, mock_pretrained, custom_arg=custom_arg)
        
        # The base class should accept kwargs without error
        assert model.maxlen == 512  # default value should still be set


class TestAbstractMethods:
    """Test abstract method enforcement."""
    
    def test_cannot_instantiate_base_class(self, mock_tokenizer, mock_pretrained):
        """Test that BaseTFKitModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTFKitModel(mock_tokenizer, mock_pretrained)
    
    def test_must_implement_forward(self, mock_tokenizer, mock_pretrained):
        """Test that subclasses must implement forward method."""
        
        class IncompleteModel(BaseTFKitModel):
            pass
        
        with pytest.raises(TypeError):
            IncompleteModel(mock_tokenizer, mock_pretrained) 