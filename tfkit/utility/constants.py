"""Constants used throughout TFKit."""

# Default configuration values
DEFAULT_MAXLEN = 512
DEFAULT_BATCH_SIZE = 20
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_EPOCHS = 10
DEFAULT_DROPOUT = 0.1
DEFAULT_SEED = 609
DEFAULT_WORKER_COUNT = 8
DEFAULT_GRADIENT_ACCUMULATION = 1

# Model configuration
DEFAULT_PRETRAINED_MODEL = 'bert-base-multilingual-cased'
DEFAULT_CHECKPOINT_DIR = 'checkpoints/'

# Training configuration
WARMUP_RATIO = 0.05
MONITORING_STEP_INTERVAL = 100
CACHE_STEP_INTERVAL = 50000

# Environment variables
ENV_TOKENIZERS_PARALLELISM = "TOKENIZERS_PARALLELISM"
ENV_OMP_NUM_THREADS = "OMP_NUM_THREADS"

# Special tokens
BLANK_TOKEN = "<BLANK>"
UNIVERSAL_SEP = "///"

# File extensions
MODEL_EXTENSION = ".pt"
CACHE_EXTENSION = ".cache"

# Evaluation metrics
SUPPORTED_METRICS = ['emf1', 'nlg', 'clas', 'er']

# Task types
TASK_TYPES = {
    'CLASSIFICATION': 'clas',
    'QUESTION_ANSWERING': 'qa',
    'SEQUENCE_TO_SEQUENCE': 'seq2seq',
    'CAUSAL_LANGUAGE_MODEL': 'clm',
    'ONCE_GENERATION': 'once',
    'ONCE_CTC': 'oncectc',
    'TAGGING': 'tag'
}

# Logging levels
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
} 