"""
Configuration parameters for the fraud detection system.
"""

# Data parameters
MAX_SEQUENCE_LENGTH = 60
EMBEDDING_SIZE = 50
BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.2
RANDOM_SEED = 42

# Model parameters
TRANSFORMER_PARAMS = {
    'num_heads': 8,
    'num_layers': 6,
    'd_model': 512,
    'dff': 2048,
    'dropout_rate': 0.1
}

LSTM_PARAMS = {
    'units': 128,
    'dropout': 0.2,
    'recurrent_dropout': 0.2
}

# Training parameters
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5

# Feature engineering
TIME_BINS = {
    'stay_time': [0, 1, 5, 10, 30, 60, 120, float('inf')],
    'lag_time': [0, 1, 5, 10, 30, 60, 120, float('inf')]
}

# File paths
DATA_DIR = 'data/'
MODEL_DIR = 'models/'
LOG_DIR = 'logs/' 