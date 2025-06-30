"""Pytest configuration and fixtures for TFKit testing."""

import os
import tempfile
from typing import Dict, List, Any

import pytest
import torch
from transformers import AutoTokenizer, AutoModel

from tfkit.utility.constants import DEFAULT_MAXLEN, DEFAULT_BATCH_SIZE


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    return AutoTokenizer.from_pretrained('bert-base-uncased')


@pytest.fixture
def mock_pretrained():
    """Create a mock pretrained model for testing."""
    return AutoModel.from_pretrained('bert-base-uncased')


@pytest.fixture
def mock_batch_data():
    """Create mock batch data for testing."""
    return {
        'input': torch.randint(0, 1000, (2, 10)),
        'mask': torch.ones(2, 10),
        'target': torch.randint(0, 2, (2, 1)),
        'task': [b'test_task', b'test_task']
    }


@pytest.fixture
def mock_tasks_detail():
    """Create mock tasks detail for classification testing."""
    return {
        'test_task': ['label1', 'label2', 'label3']
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_training_args():
    """Create sample training arguments for testing."""
    return {
        'batch': DEFAULT_BATCH_SIZE,
        'lr': [5e-5],
        'epoch': 2,
        'maxlen': DEFAULT_MAXLEN,
        'grad_accum': 1,
        'task': ['clas'],
        'config': 'bert-base-uncased',
        'train': ['dummy_train.csv'],
        'test': ['dummy_test.csv'],
        'savedir': 'test_checkpoints',
        'seed': 42,
        'worker': 1,
        'no_eval': True
    }


@pytest.fixture
def mock_csv_data():
    """Create mock CSV data for testing."""
    return """input,target
"This is a test sentence",label1
"Another test sentence",label2
"Third test sentence",label1
"""


class MockLogger:
    """Mock logger for testing."""
    
    def __init__(self):
        self.logs = []
        self.metrics = []
    
    def write_log(self, message: str) -> None:
        self.logs.append(message)
    
    def write_metric(self, name: str, value: Any, step: int) -> None:
        self.metrics.append((name, value, step))
    
    def write_config(self, config: Dict[str, Any]) -> None:
        self.logs.append(f"Config: {config}")


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MockLogger()


class MockAccelerator:
    """Mock accelerator for testing."""
    
    def __init__(self):
        self.state = type('State', (), {'backend': None})()
    
    def prepare(self, *args):
        if len(args) == 1:
            return args[0]
        return args
    
    def backward(self, loss):
        loss.backward()
    
    def print(self, *args, **kwargs):
        print(*args, **kwargs)
    
    def wait_for_everyone(self):
        pass
    
    def get_state_dict(self, model):
        return model.state_dict()


@pytest.fixture
def mock_accelerator():
    """Create a mock accelerator for testing."""
    return MockAccelerator()


@pytest.fixture(autouse=True)
def set_test_environment():
    """Set up test environment variables."""
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    yield
    # Cleanup is automatic 