"""Configuration management for TFKit."""

import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict, field

from tfkit.utility.constants import (
    DEFAULT_MAXLEN, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, 
    DEFAULT_EPOCHS, DEFAULT_DROPOUT, DEFAULT_SEED, DEFAULT_WORKER_COUNT,
    DEFAULT_GRADIENT_ACCUMULATION, DEFAULT_PRETRAINED_MODEL, DEFAULT_CHECKPOINT_DIR
)


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Training parameters
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: Union[float, List[float]] = field(default_factory=lambda: [DEFAULT_LEARNING_RATE])
    epochs: int = DEFAULT_EPOCHS
    max_length: int = DEFAULT_MAXLEN
    gradient_accumulation: int = DEFAULT_GRADIENT_ACCUMULATION
    
    # Model parameters
    model_config: str = DEFAULT_PRETRAINED_MODEL
    tokenizer_config: Optional[str] = None
    dropout: float = DEFAULT_DROPOUT
    
    # Data parameters
    train_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    task_types: List[str] = field(default_factory=list)
    task_tags: Optional[List[str]] = None
    
    # System parameters
    seed: int = DEFAULT_SEED
    worker_count: int = DEFAULT_WORKER_COUNT
    save_directory: str = DEFAULT_CHECKPOINT_DIR
    
    # Training options
    no_evaluation: bool = False
    cache_data: bool = False
    use_tensorboard: bool = False
    use_wandb: bool = False
    resume_from: Optional[str] = None
    
    # Token handling
    add_tokens_freq: int = 0
    add_tokens_file: Optional[str] = None
    add_tokens_config: Optional[str] = None
    
    # Advanced options
    handle_exceed: Optional[str] = None
    panel_mode: bool = False


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    
    model_paths: List[str] = field(default_factory=list)
    config_path: Optional[str] = None
    metric: str = "clas"
    validation_files: List[str] = field(default_factory=list)
    task_tag: Optional[str] = None
    print_results: bool = False
    panel_mode: bool = False


@dataclass
class TFKitConfig:
    """Main TFKit configuration."""
    
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Metadata
    name: str = "tfkit_experiment"
    description: str = ""
    version: str = "1.0"
    author: str = ""
    
    # Environment
    environment: Dict[str, str] = field(default_factory=dict)


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = TFKitConfig()
        
        if self.config_path and self.config_path.exists():
            self.load_config(self.config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> TFKitConfig:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
            
        Raises:
            ValueError: If configuration format is not supported
            FileNotFoundError: If configuration file doesn't exist
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
        
        self.config = self._dict_to_config(config_data)
        self.config_path = config_path
        return self.config
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None, 
                   format: str = 'yaml') -> None:
        """Save configuration to file.
        
        Args:
            config_path: Path to save configuration (optional, uses current path if None)
            format: Configuration format ('yaml' or 'json')
        """
        if config_path is None:
            if self.config_path is None:
                raise ValueError("No configuration path specified")
            save_path = self.config_path
        else:
            save_path = Path(config_path)
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._config_to_dict(self.config)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'yaml':
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            elif format.lower() == 'json':
                json.dump(config_dict, f, indent=2, sort_keys=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def update_from_args(self, args: Dict[str, Any], section: str = 'training') -> None:
        """Update configuration from command line arguments.
        
        Args:
            args: Dictionary of arguments
            section: Configuration section to update ('training' or 'evaluation')
        """
        if section == 'training':
            self._update_training_config(args)
        elif section == 'evaluation':
            self._update_evaluation_config(args)
        else:
            raise ValueError(f"Unknown configuration section: {section}")
    
    def validate_config(self) -> List[str]:
        """Validate the current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate training config
        training = self.config.training
        
        if training.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if isinstance(training.learning_rate, list):
            for i, lr in enumerate(training.learning_rate):
                if lr <= 0:
                    errors.append(f"learning_rate[{i}] must be positive")
        elif training.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if training.epochs <= 0:
            errors.append("epochs must be positive")
        
        if training.max_length <= 0:
            errors.append("max_length must be positive")
        
        if not (0 <= training.dropout < 1):
            errors.append("dropout must be between 0 and 1")
        
        if not training.train_files:
            errors.append("train_files cannot be empty")
        
        if not training.test_files:
            errors.append("test_files cannot be empty")
        
        if not training.task_types:
            errors.append("task_types cannot be empty")
        
        # Validate file paths
        for train_file in training.train_files:
            if not os.path.exists(train_file):
                errors.append(f"Training file not found: {train_file}")
        
        for test_file in training.test_files:
            if not os.path.exists(test_file):
                errors.append(f"Test file not found: {test_file}")
        
        return errors
    
    def get_training_args(self) -> Dict[str, Any]:
        """Get training arguments in the format expected by the training script.
        
        Returns:
            Dictionary of training arguments
        """
        training = self.config.training
        
        return {
            'batch': training.batch_size,
            'lr': training.learning_rate if isinstance(training.learning_rate, list) else [training.learning_rate],
            'epoch': training.epochs,
            'maxlen': training.max_length,
            'grad_accum': training.gradient_accumulation,
            'config': training.model_config,
            'tok_config': training.tokenizer_config,
            'task': training.task_types,
            'tag': training.task_tags,
            'train': training.train_files,
            'test': training.test_files,
            'savedir': training.save_directory,
            'seed': training.seed,
            'worker': training.worker_count,
            'no_eval': training.no_evaluation,
            'cache': training.cache_data,
            'tensorboard': training.use_tensorboard,
            'wandb': training.use_wandb,
            'resume': training.resume_from,
            'add_tokens_freq': training.add_tokens_freq,
            'add_tokens_file': training.add_tokens_file,
            'add_tokens_config': training.add_tokens_config,
            'handle_exceed': training.handle_exceed,
            'panel': training.panel_mode
        }
    
    def get_evaluation_args(self) -> Dict[str, Any]:
        """Get evaluation arguments in the format expected by the evaluation script.
        
        Returns:
            Dictionary of evaluation arguments
        """
        evaluation = self.config.evaluation
        
        return {
            'model': evaluation.model_paths,
            'config': evaluation.config_path,
            'metric': evaluation.metric,
            'valid': evaluation.validation_files,
            'tag': evaluation.task_tag,
            'print': evaluation.print_results,
            'panel': evaluation.panel_mode
        }
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> TFKitConfig:
        """Convert dictionary to TFKitConfig object."""
        # Extract sections
        training_dict = config_dict.get('training', {})
        evaluation_dict = config_dict.get('evaluation', {})
        
        # Convert to dataclass instances
        training_config = TrainingConfig(**{
            k: v for k, v in training_dict.items() 
            if k in TrainingConfig.__dataclass_fields__
        })
        
        evaluation_config = EvaluationConfig(**{
            k: v for k, v in evaluation_dict.items() 
            if k in EvaluationConfig.__dataclass_fields__
        })
        
        # Create main config
        main_config = TFKitConfig(
            training=training_config,
            evaluation=evaluation_config
        )
        
        # Set metadata
        for key in ['name', 'description', 'version', 'author', 'environment']:
            if key in config_dict:
                setattr(main_config, key, config_dict[key])
        
        return main_config
    
    def _config_to_dict(self, config: TFKitConfig) -> Dict[str, Any]:
        """Convert TFKitConfig object to dictionary."""
        return asdict(config)
    
    def _update_training_config(self, args: Dict[str, Any]) -> None:
        """Update training configuration from arguments."""
        training = self.config.training
        
        # Map argument names to config fields
        arg_mapping = {
            'batch': 'batch_size',
            'lr': 'learning_rate',
            'epoch': 'epochs',
            'maxlen': 'max_length',
            'grad_accum': 'gradient_accumulation',
            'config': 'model_config',
            'tok_config': 'tokenizer_config',
            'task': 'task_types',
            'tag': 'task_tags',
            'train': 'train_files',
            'test': 'test_files',
            'savedir': 'save_directory',
            'seed': 'seed',
            'worker': 'worker_count',
            'no_eval': 'no_evaluation',
            'cache': 'cache_data',
            'tensorboard': 'use_tensorboard',
            'wandb': 'use_wandb',
            'resume': 'resume_from',
            'add_tokens_freq': 'add_tokens_freq',
            'add_tokens_file': 'add_tokens_file',
            'add_tokens_config': 'add_tokens_config',
            'handle_exceed': 'handle_exceed',
            'panel': 'panel_mode'
        }
        
        for arg_name, config_field in arg_mapping.items():
            if arg_name in args and args[arg_name] is not None:
                setattr(training, config_field, args[arg_name])
    
    def _update_evaluation_config(self, args: Dict[str, Any]) -> None:
        """Update evaluation configuration from arguments."""
        evaluation = self.config.evaluation
        
        # Map argument names to config fields
        arg_mapping = {
            'model': 'model_paths',
            'config': 'config_path',
            'metric': 'metric',
            'valid': 'validation_files',
            'tag': 'task_tag',
            'print': 'print_results',
            'panel': 'panel_mode'
        }
        
        for arg_name, config_field in arg_mapping.items():
            if arg_name in args and args[arg_name] is not None:
                setattr(evaluation, config_field, args[arg_name])


def create_example_config(output_path: Union[str, Path]) -> None:
    """Create an example configuration file.
    
    Args:
        output_path: Path to save the example configuration
    """
    config = TFKitConfig(
        name="example_experiment",
        description="Example TFKit configuration for text classification",
        author="TFKit User",
        training=TrainingConfig(
            batch_size=16,
            learning_rate=[5e-5],
            epochs=5,
            max_length=512,
            model_config="bert-base-uncased",
            train_files=["data/train.csv"],
            test_files=["data/test.csv"],
            task_types=["clas"],
            save_directory="experiments/example",
            use_tensorboard=True
        ),
        evaluation=EvaluationConfig(
            model_paths=["experiments/example/1.pt"],
            metric="clas",
            validation_files=["data/test.csv"],
            print_results=True
        )
    )
    
    manager = ConfigManager()
    manager.config = config
    manager.save_config(output_path, format='yaml')


if __name__ == "__main__":
    # Example usage
    create_example_config("example_config.yaml") 