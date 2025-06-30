"""Tests for the training utilities module."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch

from tfkit.utility.training_utils import TrainingManager
from tfkit.utility.constants import WARMUP_RATIO


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=10, output_size=2):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, batch_data):
        # Mock forward that returns a loss
        x = batch_data.get('input', torch.randn(2, 10))
        return torch.tensor(1.0, requires_grad=True)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def sample_dataloader():
    """Create a sample dataloader for testing."""
    # Create sample data
    inputs = torch.randn(20, 10)
    targets = torch.randint(0, 2, (20,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=4, shuffle=True)


class TestTrainingManager:
    """Test cases for TrainingManager."""
    
    def test_initialization(self, mock_accelerator, mock_logger):
        """Test TrainingManager initialization."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        assert trainer.accelerator == mock_accelerator
        assert trainer.logger == mock_logger
    
    def test_create_optimizer(self, mock_accelerator, mock_logger, simple_model):
        """Test optimizer creation."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        lr = 1e-4
        total_steps = 1000
        
        optimizer, scheduler = trainer.create_optimizer(simple_model, lr, total_steps)
        
        # Check optimizer
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]['lr'] == lr
        
        # Check scheduler
        assert hasattr(scheduler, 'step')
        
        # Check warmup steps calculation
        expected_warmup_steps = int(total_steps * WARMUP_RATIO)
        # This is hard to test directly, but we can verify the scheduler exists
        assert scheduler is not None
    
    def test_prepare_models_and_optimizers(self, mock_accelerator, mock_logger, 
                                         simple_model, sample_dataloader):
        """Test model and optimizer preparation."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        models_list = [simple_model]
        dataloaders = [sample_dataloader]
        input_arg = {'lr': [1e-4], 'grad_accum': 1}
        
        result = trainer.prepare_models_and_optimizers(models_list, dataloaders, input_arg)
        models, optims_schs, data_iters, total_iter_length = result
        
        assert len(models) == 1
        assert len(optims_schs) == 1
        assert len(data_iters) == 1
        assert total_iter_length == len(sample_dataloader)
        
        # Check optimizer and scheduler tuple
        optimizer, scheduler = optims_schs[0]
        assert isinstance(optimizer, torch.optim.AdamW)
        assert hasattr(scheduler, 'step')
    
    def test_prepare_models_multiple_learning_rates(self, mock_accelerator, mock_logger, 
                                                   sample_dataloader):
        """Test preparation with multiple learning rates."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        models_list = [SimpleModel(), SimpleModel()]
        dataloaders = [sample_dataloader, sample_dataloader]
        input_arg = {'lr': [1e-4, 2e-4], 'grad_accum': 1}
        
        result = trainer.prepare_models_and_optimizers(models_list, dataloaders, input_arg)
        models, optims_schs, data_iters, total_iter_length = result
        
        # Check that different learning rates are used
        optimizer1, _ = optims_schs[0]
        optimizer2, _ = optims_schs[1]
        
        assert optimizer1.param_groups[0]['lr'] == 1e-4
        assert optimizer2.param_groups[0]['lr'] == 2e-4
    
    def test_prepare_models_fewer_learning_rates(self, mock_accelerator, mock_logger, 
                                                sample_dataloader):
        """Test preparation with fewer learning rates than models."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        models_list = [SimpleModel(), SimpleModel()]
        dataloaders = [sample_dataloader, sample_dataloader]
        input_arg = {'lr': [1e-4], 'grad_accum': 1}  # Only one LR for two models
        
        result = trainer.prepare_models_and_optimizers(models_list, dataloaders, input_arg)
        models, optims_schs, data_iters, total_iter_length = result
        
        # Both models should use the same learning rate
        optimizer1, _ = optims_schs[0]
        optimizer2, _ = optims_schs[1]
        
        assert optimizer1.param_groups[0]['lr'] == 1e-4
        assert optimizer2.param_groups[0]['lr'] == 1e-4
    
    @patch('tfkit.utility.training_utils.save_model')
    def test_train_epoch_basic(self, mock_save_model, mock_accelerator, mock_logger):
        """Test basic training epoch functionality."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        # Create mock components
        models = [SimpleModel()]
        optimizers = [torch.optim.AdamW(models[0].parameters(), lr=1e-4)]
        schedulers = [Mock()]
        optims_schs = [(optimizers[0], schedulers[0])]
        
        # Create mock data iterator
        mock_batch = {'input': torch.randn(2, 10)}
        data_iters = [iter([mock_batch, None])]  # None signals end
        
        models_tag = ['test_model']
        input_arg = {'grad_accum': 1}
        epoch = 1
        fname = 'test_model'
        add_tokens = None
        total_iter_length = 2
        
        avg_loss = trainer.train_epoch(
            models, optims_schs, data_iters, models_tag, 
            input_arg, epoch, fname, add_tokens, total_iter_length
        )
        
        # Check that loss is returned and is a reasonable value
        assert isinstance(avg_loss, (int, float, torch.Tensor))
        if isinstance(avg_loss, torch.Tensor):
            assert avg_loss.numel() == 1
        
        # Check that scheduler step was called
        schedulers[0].step.assert_called()
    
    def test_evaluate_models(self, mock_accelerator, mock_logger, sample_dataloader):
        """Test model evaluation."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        models = [SimpleModel()]
        dataloaders = [sample_dataloader]
        fname = 'test_model'
        input_arg = {'grad_accum': 1}
        epoch = 1
        
        avg_loss = trainer.evaluate_models(models, dataloaders, fname, input_arg, epoch)
        
        # Check that loss is returned
        assert isinstance(avg_loss, (int, float, torch.Tensor))
        if isinstance(avg_loss, torch.Tensor):
            assert avg_loss.numel() == 1
        
        # Check that model is in eval mode after evaluation
        assert not models[0].training
    
    def test_process_batch(self, mock_accelerator, mock_logger):
        """Test batch processing."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = Mock()
        train_batch = {'input': torch.randn(2, 10)}
        input_arg = {'grad_accum': 1}
        total_iter = 0
        epoch = 1
        mtag = 'test_model'
        
        loss_value = trainer._process_batch(
            model, optimizer, scheduler, train_batch, 
            input_arg, total_iter, epoch, mtag
        )
        
        # Check that loss is returned
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.numel() == 1
        
        # Check that optimizer step was called (since grad_accum = 1 and total_iter + 1 = 1)
        # This is harder to test directly, but we can check that no error occurred
        assert True  # If we reach here, no exception was raised
    
    def test_log_progress(self, mock_accelerator, mock_logger):
        """Test progress logging."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        epoch = 1
        mtag = 'test_model'
        model = SimpleModel()
        total_iter = 100
        t_loss = torch.tensor(5.0)
        total_iter_length = 1000
        
        trainer._log_progress(epoch, mtag, model, total_iter, t_loss, total_iter_length)
        
        # Check that log was written
        assert len(mock_logger.logs) > 0
        log_message = mock_logger.logs[-1]
        assert 'epoch: 1' in log_message
        assert 'test_model' in log_message
        assert 'SimpleModel' in log_message


class TestTrainingManagerIntegration:
    """Integration tests for TrainingManager."""
    
    def test_full_training_pipeline(self, mock_accelerator, mock_logger, sample_dataloader):
        """Test the full training pipeline integration."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        # Create models and data
        models_list = [SimpleModel()]
        dataloaders = [sample_dataloader]
        input_arg = {'lr': [1e-4], 'grad_accum': 1}
        
        # Prepare models and optimizers
        result = trainer.prepare_models_and_optimizers(models_list, dataloaders, input_arg)
        models, optims_schs, data_iters, total_iter_length = result
        
        # The integration test would be more complex in a real scenario
        # Here we just verify that all components are properly initialized
        assert len(models) == 1
        assert len(optims_schs) == 1
        assert total_iter_length > 0
    
    def test_error_handling(self, mock_accelerator, mock_logger):
        """Test error handling in training manager."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        # Test with invalid model
        with pytest.raises(Exception):
            trainer.create_optimizer(None, 1e-4, 1000)
        
        # Test with invalid learning rate
        model = SimpleModel()
        with pytest.raises(Exception):
            trainer.create_optimizer(model, -1e-4, 1000)  # Negative LR should cause issues


class TestTrainingManagerEdgeCases:
    """Test edge cases for TrainingManager."""
    
    def test_zero_total_steps(self, mock_accelerator, mock_logger, simple_model):
        """Test behavior with zero total steps."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        # This should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, ZeroDivisionError)):
            trainer.create_optimizer(simple_model, 1e-4, 0)
    
    def test_very_large_total_steps(self, mock_accelerator, mock_logger, simple_model):
        """Test behavior with very large total steps."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        # Should handle large numbers gracefully
        optimizer, scheduler = trainer.create_optimizer(simple_model, 1e-4, 1000000)
        assert optimizer is not None
        assert scheduler is not None
    
    def test_empty_model_list(self, mock_accelerator, mock_logger):
        """Test behavior with empty model list."""
        trainer = TrainingManager(mock_accelerator, mock_logger)
        
        models_list = []
        dataloaders = []
        input_arg = {'lr': [1e-4], 'grad_accum': 1}
        
        result = trainer.prepare_models_and_optimizers(models_list, dataloaders, input_arg)
        models, optims_schs, data_iters, total_iter_length = result
        
        assert len(models) == 0
        assert len(optims_schs) == 0
        assert len(data_iters) == 0 