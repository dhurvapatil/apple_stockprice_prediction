import torch
from pathlib import Path
import os


class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    # Create directories if they don't exist
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # GPU Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_MIXED_PRECISION = torch.cuda.is_available()

    # Data Parameters
    SYMBOL = "AAPL"
    START_DATE = "2018-01-01"
    END_DATE = "2024-12-31"

    # Split dates - Updated for better statistical significance
    # This gives you ~500 test samples instead of 51
    TRAIN_START = "2018-01-01"  # 3 years (~750 samples)
    TRAIN_END = "2020-12-31"
    VAL_START = "2021-01-01"  # 2 years (~500 samples)
    VAL_END = "2022-12-31"
    TEST_START = "2023-01-01"  # 2 years (~500 samples)
    TEST_END = "2024-12-31"

    # Model parameters
    SEQUENCE_LENGTH = 60
    BATCH_SIZE = 64 if torch.cuda.is_available() else 32
    EPOCHS = 30
    LEARNING_RATE = 0.001

    # Feature engineering windows
    SMA_WINDOWS = [7, 21, 50, 200]
    EMA_WINDOWS = [7, 21, 50, 200]
    RETURN_PERIODS = [1, 3, 7]

    # Random seeds for reproducibility
    RANDOM_STATE = 42

    # Additional validation parameters
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_DELTA = 1e-4

    # Hyperparameter optimization settings
    OPTUNA_N_TRIALS = 75
    OPTUNA_TIMEOUT = 1800  # 30 minutes per model

    # Walk-forward validation parameters
    WALK_FORWARD_WINDOW = 750  # ~3 years training window
    WALK_FORWARD_TEST = 90  # ~3 months test window
    WALK_FORWARD_STEP = 30  # ~1 month step size
