#!/usr/bin/env python3
"""Command-line interface for TFKit configuration management."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from tfkit.utility.config import ConfigManager, create_example_config


def create_example_command(args):
    """Create an example configuration file."""
    output_path = args.output or "tfkit_config.yaml"
    
    try:
        create_example_config(output_path)
        print(f"Example configuration created at: {output_path}")
        print(f"Edit this file to customize your training setup.")
    except Exception as e:
        print(f"Error creating example configuration: {e}")
        return 1
    
    return 0


def validate_command(args):
    """Validate a configuration file."""
    config_path = args.config
    
    if not Path(config_path).exists():
        print(f"Configuration file not found: {config_path}")
        return 1
    
    try:
        manager = ConfigManager(config_path)
        errors = manager.validate_config()
        
        if not errors:
            print(f"✓ Configuration is valid: {config_path}")
            return 0
        else:
            print(f"✗ Configuration validation failed: {config_path}")
            print("\nValidation errors:")
            for error in errors:
                print(f"  - {error}")
            return 1
            
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1


def show_command(args):
    """Show configuration details."""
    config_path = args.config
    
    if not Path(config_path).exists():
        print(f"Configuration file not found: {config_path}")
        return 1
    
    try:
        manager = ConfigManager(config_path)
        config = manager.config
        
        print(f"Configuration: {config_path}")
        print("=" * 50)
        print(f"Name: {config.name}")
        print(f"Description: {config.description}")
        print(f"Author: {config.author}")
        print(f"Version: {config.version}")
        print()
        
        # Training configuration
        print("Training Configuration:")
        print("-" * 25)
        training = config.training
        print(f"  Batch size: {training.batch_size}")
        print(f"  Learning rate: {training.learning_rate}")
        print(f"  Epochs: {training.epochs}")
        print(f"  Max length: {training.max_length}")
        print(f"  Model config: {training.model_config}")
        print(f"  Task types: {training.task_types}")
        print(f"  Train files: {training.train_files}")
        print(f"  Test files: {training.test_files}")
        print(f"  Save directory: {training.save_directory}")
        print()
        
        # Evaluation configuration
        print("Evaluation Configuration:")
        print("-" * 28)
        evaluation = config.evaluation
        print(f"  Model paths: {evaluation.model_paths}")
        print(f"  Metric: {evaluation.metric}")
        print(f"  Validation files: {evaluation.validation_files}")
        print(f"  Print results: {evaluation.print_results}")
        
        return 0
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1


def convert_command(args):
    """Convert configuration between formats."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Input configuration file not found: {input_path}")
        return 1
    
    try:
        # Load configuration
        manager = ConfigManager(input_path)
        
        # Determine output format
        output_format = args.format
        if output_format is None:
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                output_format = 'yaml'
            elif output_path.suffix.lower() == '.json':
                output_format = 'json'
            else:
                print(f"Cannot determine output format from extension: {output_path.suffix}")
                print("Please specify --format yaml or --format json")
                return 1
        
        # Save in new format
        manager.save_config(output_path, format=output_format)
        print(f"Configuration converted from {input_path} to {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error converting configuration: {e}")
        return 1


def update_command(args):
    """Update configuration with new values."""
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return 1
    
    try:
        manager = ConfigManager(config_path)
        
        # Update fields based on arguments
        updates = {}
        if args.batch_size is not None:
            updates['batch'] = args.batch_size
        if args.learning_rate is not None:
            updates['lr'] = [args.learning_rate]
        if args.epochs is not None:
            updates['epoch'] = args.epochs
        if args.model_config is not None:
            updates['config'] = args.model_config
        
        if updates:
            manager.update_from_args(updates, section='training')
            manager.save_config()
            print(f"Configuration updated: {config_path}")
            
            # Show what was updated
            print("Updated fields:")
            for field, value in updates.items():
                print(f"  {field}: {value}")
        else:
            print("No updates specified. Use --help to see available options.")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error updating configuration: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TFKit Configuration Management CLI",
        prog="tfkit-config"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create example command
    create_parser = subparsers.add_parser(
        'create-example', 
        help='Create an example configuration file'
    )
    create_parser.add_argument(
        '--output', '-o', 
        type=str, 
        help='Output path for example configuration (default: tfkit_config.yaml)'
    )
    create_parser.set_defaults(func=create_example_command)
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate', 
        help='Validate a configuration file'
    )
    validate_parser.add_argument(
        'config', 
        type=str, 
        help='Path to configuration file'
    )
    validate_parser.set_defaults(func=validate_command)
    
    # Show command
    show_parser = subparsers.add_parser(
        'show', 
        help='Show configuration details'
    )
    show_parser.add_argument(
        'config', 
        type=str, 
        help='Path to configuration file'
    )
    show_parser.set_defaults(func=show_command)
    
    # Convert command
    convert_parser = subparsers.add_parser(
        'convert', 
        help='Convert configuration between formats'
    )
    convert_parser.add_argument(
        'input', 
        type=str, 
        help='Input configuration file'
    )
    convert_parser.add_argument(
        'output', 
        type=str, 
        help='Output configuration file'
    )
    convert_parser.add_argument(
        '--format', 
        choices=['yaml', 'json'], 
        help='Output format (auto-detected from extension if not specified)'
    )
    convert_parser.set_defaults(func=convert_command)
    
    # Update command
    update_parser = subparsers.add_parser(
        'update', 
        help='Update configuration values'
    )
    update_parser.add_argument(
        'config', 
        type=str, 
        help='Path to configuration file'
    )
    update_parser.add_argument(
        '--batch-size', 
        type=int, 
        help='Update batch size'
    )
    update_parser.add_argument(
        '--learning-rate', 
        type=float, 
        help='Update learning rate'
    )
    update_parser.add_argument(
        '--epochs', 
        type=int, 
        help='Update number of epochs'
    )
    update_parser.add_argument(
        '--model-config', 
        type=str, 
        help='Update model configuration'
    )
    update_parser.set_defaults(func=update_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    # Execute command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main()) 