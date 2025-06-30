"""Tests for the configuration system."""

import pytest
import json
import yaml
from pathlib import Path
from tempfile import NamedTemporaryFile

from tfkit.utility.config import (
    TrainingConfig, EvaluationConfig, TFKitConfig, ConfigManager, create_example_config
)
from tfkit.utility.constants import DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE


class TestTrainingConfig:
    """Test cases for TrainingConfig."""
    
    def test_default_initialization(self):
        """Test default initialization of TrainingConfig."""
        config = TrainingConfig()
        
        assert config.batch_size == DEFAULT_BATCH_SIZE
        assert config.learning_rate == [DEFAULT_LEARNING_RATE]
        assert config.epochs > 0
        assert config.max_length > 0
        assert isinstance(config.train_files, list)
        assert isinstance(config.test_files, list)
        assert isinstance(config.task_types, list)
    
    def test_custom_initialization(self):
        """Test custom initialization of TrainingConfig."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=[1e-4, 2e-4],
            epochs=10,
            train_files=["train.csv"],
            test_files=["test.csv"],
            task_types=["clas"]
        )
        
        assert config.batch_size == 32
        assert config.learning_rate == [1e-4, 2e-4]
        assert config.epochs == 10
        assert config.train_files == ["train.csv"]
        assert config.test_files == ["test.csv"]
        assert config.task_types == ["clas"]


class TestEvaluationConfig:
    """Test cases for EvaluationConfig."""
    
    def test_default_initialization(self):
        """Test default initialization of EvaluationConfig."""
        config = EvaluationConfig()
        
        assert isinstance(config.model_paths, list)
        assert config.metric == "clas"
        assert isinstance(config.validation_files, list)
        assert config.task_tag is None
        assert config.print_results is False
    
    def test_custom_initialization(self):
        """Test custom initialization of EvaluationConfig."""
        config = EvaluationConfig(
            model_paths=["model.pt"],
            metric="nlg",
            validation_files=["val.csv"],
            print_results=True
        )
        
        assert config.model_paths == ["model.pt"]
        assert config.metric == "nlg"
        assert config.validation_files == ["val.csv"]
        assert config.print_results is True


class TestTFKitConfig:
    """Test cases for TFKitConfig."""
    
    def test_default_initialization(self):
        """Test default initialization of TFKitConfig."""
        config = TFKitConfig()
        
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
        assert config.name == "tfkit_experiment"
        assert isinstance(config.description, str)
        assert isinstance(config.environment, dict)
    
    def test_custom_initialization(self):
        """Test custom initialization of TFKitConfig."""
        training = TrainingConfig(batch_size=16)
        evaluation = EvaluationConfig(metric="emf1")
        
        config = TFKitConfig(
            training=training,
            evaluation=evaluation,
            name="test_experiment",
            description="Test description"
        )
        
        assert config.training.batch_size == 16
        assert config.evaluation.metric == "emf1"
        assert config.name == "test_experiment"
        assert config.description == "Test description"


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_initialization_without_path(self):
        """Test ConfigManager initialization without config path."""
        manager = ConfigManager()
        
        assert manager.config_path is None
        assert isinstance(manager.config, TFKitConfig)
    
    def test_save_and_load_yaml(self, temp_dir):
        """Test saving and loading YAML configuration."""
        manager = ConfigManager()
        manager.config.name = "test_config"
        manager.config.training.batch_size = 64
        
        config_path = Path(temp_dir) / "test_config.yaml"
        manager.save_config(config_path, format='yaml')
        
        assert config_path.exists()
        
        # Load the configuration
        new_manager = ConfigManager()
        loaded_config = new_manager.load_config(config_path)
        
        assert loaded_config.name == "test_config"
        assert loaded_config.training.batch_size == 64
    
    def test_save_and_load_json(self, temp_dir):
        """Test saving and loading JSON configuration."""
        manager = ConfigManager()
        manager.config.name = "test_config_json"
        manager.config.evaluation.metric = "nlg"
        
        config_path = Path(temp_dir) / "test_config.json"
        manager.save_config(config_path, format='json')
        
        assert config_path.exists()
        
        # Load the configuration
        new_manager = ConfigManager()
        loaded_config = new_manager.load_config(config_path)
        
        assert loaded_config.name == "test_config_json"
        assert loaded_config.evaluation.metric == "nlg"
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent configuration file."""
        manager = ConfigManager()
        
        with pytest.raises(FileNotFoundError):
            manager.load_config("nonexistent.yaml")
    
    def test_unsupported_format(self, temp_dir):
        """Test loading unsupported file format."""
        manager = ConfigManager()
        config_path = Path(temp_dir) / "test_config.txt"
        config_path.write_text("some content")
        
        with pytest.raises(ValueError, match="Unsupported configuration format"):
            manager.load_config(config_path)
    
    def test_update_from_training_args(self):
        """Test updating configuration from training arguments."""
        manager = ConfigManager()
        
        args = {
            'batch': 32,
            'lr': [1e-4],
            'epoch': 5,
            'task': ['clas', 'tag'],
            'train': ['train1.csv', 'train2.csv'],
            'test': ['test1.csv'],
            'cache': True
        }
        
        manager.update_from_args(args, section='training')
        
        assert manager.config.training.batch_size == 32
        assert manager.config.training.learning_rate == [1e-4]
        assert manager.config.training.epochs == 5
        assert manager.config.training.task_types == ['clas', 'tag']
        assert manager.config.training.train_files == ['train1.csv', 'train2.csv']
        assert manager.config.training.test_files == ['test1.csv']
        assert manager.config.training.cache_data is True
    
    def test_update_from_evaluation_args(self):
        """Test updating configuration from evaluation arguments."""
        manager = ConfigManager()
        
        args = {
            'model': ['model1.pt', 'model2.pt'],
            'metric': 'emf1',
            'valid': ['valid.csv'],
            'print': True
        }
        
        manager.update_from_args(args, section='evaluation')
        
        assert manager.config.evaluation.model_paths == ['model1.pt', 'model2.pt']
        assert manager.config.evaluation.metric == 'emf1'
        assert manager.config.evaluation.validation_files == ['valid.csv']
        assert manager.config.evaluation.print_results is True
    
    def test_validation_valid_config(self, temp_dir):
        """Test validation of valid configuration."""
        # Create temporary files
        train_file = Path(temp_dir) / "train.csv"
        test_file = Path(temp_dir) / "test.csv"
        train_file.write_text("input,target\ntest,label")
        test_file.write_text("input,target\ntest,label")
        
        manager = ConfigManager()
        manager.config.training.train_files = [str(train_file)]
        manager.config.training.test_files = [str(test_file)]
        manager.config.training.task_types = ['clas']
        
        errors = manager.validate_config()
        assert len(errors) == 0
    
    def test_validation_invalid_config(self):
        """Test validation of invalid configuration."""
        manager = ConfigManager()
        manager.config.training.batch_size = -1  # Invalid
        manager.config.training.learning_rate = [-1e-4]  # Invalid
        manager.config.training.epochs = 0  # Invalid
        manager.config.training.dropout = 1.5  # Invalid
        # Missing required fields (train_files, test_files, task_types)
        
        errors = manager.validate_config()
        assert len(errors) > 0
        
        # Check specific error messages
        error_messages = ' '.join(errors)
        assert 'batch_size must be positive' in error_messages
        assert 'learning_rate[0] must be positive' in error_messages
        assert 'epochs must be positive' in error_messages
        assert 'dropout must be between 0 and 1' in error_messages
        assert 'train_files cannot be empty' in error_messages
    
    def test_get_training_args(self):
        """Test getting training arguments."""
        manager = ConfigManager()
        manager.config.training.batch_size = 16
        manager.config.training.learning_rate = [2e-5]
        manager.config.training.task_types = ['clas']
        manager.config.training.train_files = ['train.csv']
        manager.config.training.test_files = ['test.csv']
        
        args = manager.get_training_args()
        
        assert args['batch'] == 16
        assert args['lr'] == [2e-5]
        assert args['task'] == ['clas']
        assert args['train'] == ['train.csv']
        assert args['test'] == ['test.csv']
    
    def test_get_evaluation_args(self):
        """Test getting evaluation arguments."""
        manager = ConfigManager()
        manager.config.evaluation.model_paths = ['model.pt']
        manager.config.evaluation.metric = 'nlg'
        manager.config.evaluation.validation_files = ['val.csv']
        manager.config.evaluation.print_results = True
        
        args = manager.get_evaluation_args()
        
        assert args['model'] == ['model.pt']
        assert args['metric'] == 'nlg'
        assert args['valid'] == ['val.csv']
        assert args['print'] is True
    
    def test_invalid_section_update(self):
        """Test updating with invalid section name."""
        manager = ConfigManager()
        
        with pytest.raises(ValueError, match="Unknown configuration section"):
            manager.update_from_args({}, section='invalid')


class TestConfigManagerEdgeCases:
    """Test edge cases for ConfigManager."""
    
    def test_save_without_path(self):
        """Test saving configuration without specifying path."""
        manager = ConfigManager()
        
        with pytest.raises(ValueError, match="No configuration path specified"):
            manager.save_config()
    
    def test_unsupported_save_format(self, temp_dir):
        """Test saving with unsupported format."""
        manager = ConfigManager()
        config_path = Path(temp_dir) / "test.txt"
        
        with pytest.raises(ValueError, match="Unsupported format"):
            manager.save_config(config_path, format='txt')
    
    def test_corrupted_yaml_file(self, temp_dir):
        """Test loading corrupted YAML file."""
        config_path = Path(temp_dir) / "corrupted.yaml"
        config_path.write_text("invalid: yaml: content: [")
        
        manager = ConfigManager()
        with pytest.raises(yaml.YAMLError):
            manager.load_config(config_path)
    
    def test_corrupted_json_file(self, temp_dir):
        """Test loading corrupted JSON file."""
        config_path = Path(temp_dir) / "corrupted.json"
        config_path.write_text('{"invalid": json}')
        
        manager = ConfigManager()
        with pytest.raises(json.JSONDecodeError):
            manager.load_config(config_path)


class TestExampleConfig:
    """Test example configuration creation."""
    
    def test_create_example_config(self, temp_dir):
        """Test creating example configuration."""
        config_path = Path(temp_dir) / "example.yaml"
        create_example_config(config_path)
        
        assert config_path.exists()
        
        # Load and verify the example config
        manager = ConfigManager()
        config = manager.load_config(config_path)
        
        assert config.name == "example_experiment"
        assert "Example TFKit configuration" in config.description
        assert config.training.batch_size == 16
        assert config.training.learning_rate == [5e-5]
        assert config.training.epochs == 5
        assert config.training.use_tensorboard is True
        assert config.evaluation.metric == "clas"
        assert config.evaluation.print_results is True


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_full_workflow(self, temp_dir):
        """Test full configuration workflow."""
        # Create configuration
        manager = ConfigManager()
        manager.config.name = "integration_test"
        manager.config.training.batch_size = 8
        manager.config.training.task_types = ['clas']
        
        # Save configuration
        config_path = Path(temp_dir) / "integration.yaml"
        manager.save_config(config_path)
        
        # Load configuration in new manager
        new_manager = ConfigManager(config_path)
        assert new_manager.config.name == "integration_test"
        assert new_manager.config.training.batch_size == 8
        
        # Update from args
        args = {'batch': 16, 'epoch': 20}
        new_manager.update_from_args(args, section='training')
        assert new_manager.config.training.batch_size == 16
        assert new_manager.config.training.epochs == 20
        
        # Get training args
        training_args = new_manager.get_training_args()
        assert training_args['batch'] == 16
        assert training_args['epoch'] == 20
    
    def test_config_override_priority(self):
        """Test that command line args override config file values."""
        manager = ConfigManager()
        
        # Set initial config values
        manager.config.training.batch_size = 32
        manager.config.training.epochs = 10
        
        # Override with command line args
        args = {'batch': 64}  # Only override batch size
        manager.update_from_args(args, section='training')
        
        # Check that batch size was overridden but epochs remained
        assert manager.config.training.batch_size == 64
        assert manager.config.training.epochs == 10 