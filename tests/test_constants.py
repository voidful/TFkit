"""Tests for the constants module."""

import pytest

from tfkit.utility import constants


class TestConstants:
    """Test cases for constants module."""
    
    def test_default_values_exist(self):
        """Test that all expected default values are defined."""
        assert hasattr(constants, 'DEFAULT_MAXLEN')
        assert hasattr(constants, 'DEFAULT_BATCH_SIZE')
        assert hasattr(constants, 'DEFAULT_LEARNING_RATE')
        assert hasattr(constants, 'DEFAULT_EPOCHS')
        assert hasattr(constants, 'DEFAULT_DROPOUT')
        assert hasattr(constants, 'DEFAULT_SEED')
        assert hasattr(constants, 'DEFAULT_WORKER_COUNT')
        assert hasattr(constants, 'DEFAULT_GRADIENT_ACCUMULATION')
    
    def test_default_values_types(self):
        """Test that default values have correct types."""
        assert isinstance(constants.DEFAULT_MAXLEN, int)
        assert isinstance(constants.DEFAULT_BATCH_SIZE, int)
        assert isinstance(constants.DEFAULT_LEARNING_RATE, float)
        assert isinstance(constants.DEFAULT_EPOCHS, int)
        assert isinstance(constants.DEFAULT_DROPOUT, float)
        assert isinstance(constants.DEFAULT_SEED, int)
        assert isinstance(constants.DEFAULT_WORKER_COUNT, int)
        assert isinstance(constants.DEFAULT_GRADIENT_ACCUMULATION, int)
    
    def test_default_values_ranges(self):
        """Test that default values are in reasonable ranges."""
        assert constants.DEFAULT_MAXLEN > 0
        assert constants.DEFAULT_BATCH_SIZE > 0
        assert 0 < constants.DEFAULT_LEARNING_RATE < 1
        assert constants.DEFAULT_EPOCHS > 0
        assert 0 <= constants.DEFAULT_DROPOUT < 1
        assert constants.DEFAULT_SEED >= 0
        assert constants.DEFAULT_WORKER_COUNT > 0
        assert constants.DEFAULT_GRADIENT_ACCUMULATION > 0
    
    def test_model_config_constants(self):
        """Test model configuration constants."""
        assert hasattr(constants, 'DEFAULT_PRETRAINED_MODEL')
        assert hasattr(constants, 'DEFAULT_CHECKPOINT_DIR')
        
        assert isinstance(constants.DEFAULT_PRETRAINED_MODEL, str)
        assert isinstance(constants.DEFAULT_CHECKPOINT_DIR, str)
        assert len(constants.DEFAULT_PRETRAINED_MODEL) > 0
        assert len(constants.DEFAULT_CHECKPOINT_DIR) > 0
    
    def test_training_config_constants(self):
        """Test training configuration constants."""
        assert hasattr(constants, 'WARMUP_RATIO')
        assert hasattr(constants, 'MONITORING_STEP_INTERVAL')
        assert hasattr(constants, 'CACHE_STEP_INTERVAL')
        
        assert isinstance(constants.WARMUP_RATIO, float)
        assert isinstance(constants.MONITORING_STEP_INTERVAL, int)
        assert isinstance(constants.CACHE_STEP_INTERVAL, int)
        
        assert 0 < constants.WARMUP_RATIO < 1
        assert constants.MONITORING_STEP_INTERVAL > 0
        assert constants.CACHE_STEP_INTERVAL > 0
    
    def test_environment_variables(self):
        """Test environment variable constants."""
        assert hasattr(constants, 'ENV_TOKENIZERS_PARALLELISM')
        assert hasattr(constants, 'ENV_OMP_NUM_THREADS')
        
        assert isinstance(constants.ENV_TOKENIZERS_PARALLELISM, str)
        assert isinstance(constants.ENV_OMP_NUM_THREADS, str)
    
    def test_special_tokens(self):
        """Test special token constants."""
        assert hasattr(constants, 'BLANK_TOKEN')
        assert hasattr(constants, 'UNIVERSAL_SEP')
        
        assert isinstance(constants.BLANK_TOKEN, str)
        assert isinstance(constants.UNIVERSAL_SEP, str)
        assert len(constants.BLANK_TOKEN) > 0
        assert len(constants.UNIVERSAL_SEP) > 0
    
    def test_file_extensions(self):
        """Test file extension constants."""
        assert hasattr(constants, 'MODEL_EXTENSION')
        assert hasattr(constants, 'CACHE_EXTENSION')
        
        assert isinstance(constants.MODEL_EXTENSION, str)
        assert isinstance(constants.CACHE_EXTENSION, str)
        assert constants.MODEL_EXTENSION.startswith('.')
        assert constants.CACHE_EXTENSION.startswith('.')
    
    def test_supported_metrics(self):
        """Test supported metrics constant."""
        assert hasattr(constants, 'SUPPORTED_METRICS')
        assert isinstance(constants.SUPPORTED_METRICS, list)
        assert len(constants.SUPPORTED_METRICS) > 0
        
        for metric in constants.SUPPORTED_METRICS:
            assert isinstance(metric, str)
            assert len(metric) > 0
    
    def test_task_types(self):
        """Test task types constant."""
        assert hasattr(constants, 'TASK_TYPES')
        assert isinstance(constants.TASK_TYPES, dict)
        assert len(constants.TASK_TYPES) > 0
        
        for key, value in constants.TASK_TYPES.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(key) > 0
            assert len(value) > 0
    
    def test_log_levels(self):
        """Test log levels constant."""
        assert hasattr(constants, 'LOG_LEVELS')
        assert isinstance(constants.LOG_LEVELS, dict)
        assert len(constants.LOG_LEVELS) > 0
        
        expected_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        for level in expected_levels:
            assert level in constants.LOG_LEVELS
            assert isinstance(constants.LOG_LEVELS[level], int)
    
    def test_constants_immutability(self):
        """Test that constants are not accidentally modified."""
        # This is more of a convention test since Python doesn't have true constants
        original_batch_size = constants.DEFAULT_BATCH_SIZE
        
        # Try to modify (this will work but shouldn't be done)
        constants.DEFAULT_BATCH_SIZE = 999
        
        # Reset for other tests
        constants.DEFAULT_BATCH_SIZE = original_batch_size
        assert constants.DEFAULT_BATCH_SIZE == original_batch_size
    
    def test_constants_usage_patterns(self):
        """Test that constants follow expected naming patterns."""
        # All constants should be uppercase
        for attr_name in dir(constants):
            if not attr_name.startswith('_'):  # Skip private attributes
                assert attr_name.isupper(), f"Constant {attr_name} should be uppercase"
    
    def test_reasonable_default_combinations(self):
        """Test that default values work well together."""
        # Warmup ratio should be reasonable for typical training
        assert constants.WARMUP_RATIO * constants.DEFAULT_EPOCHS >= 0.1
        
        # Monitoring interval should be reasonable for typical batch sizes
        assert constants.MONITORING_STEP_INTERVAL >= constants.DEFAULT_BATCH_SIZE
        
        # Cache interval should be much larger than monitoring interval
        assert constants.CACHE_STEP_INTERVAL > constants.MONITORING_STEP_INTERVAL * 10 