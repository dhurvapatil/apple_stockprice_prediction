import pandas as pd
import numpy as np
import torch
import joblib
import logging
from pathlib import Path
from config import Config


def setup_logging(log_file="training.log"):
    """Setup logging configuration"""
    log_path = Config.RESULTS_DIR / log_file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def set_random_seeds(seed=Config.RANDOM_STATE):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_gpu_availability():
    """Check GPU availability and print information"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1e9

        print(f"GPU Available: Yes")
        print(f"GPU Count: {gpu_count}")
        print(f"Current GPU: {gpu_name}")
        print(f"GPU Memory: {memory_total:.1f} GB")

        return True
    else:
        print("GPU Available: No - using CPU")
        return False


def load_processed_data():
    """Load processed data from CSV files"""
    try:
        train_path = Config.PROCESSED_DATA_DIR / "train.csv"
        val_path = Config.PROCESSED_DATA_DIR / "val.csv"
        test_path = Config.PROCESSED_DATA_DIR / "test.csv"

        train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
        test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)

        return train_df, val_df, test_df

    except Exception as e:
        logging.error(f"Error loading processed data: {str(e)}")
        raise


def load_scaler():
    """Load the fitted scaler"""
    try:
        scaler_path = Config.PROCESSED_DATA_DIR / "scaler.joblib"
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        logging.error(f"Error loading scaler: {str(e)}")
        raise


def memory_usage_check():
    """Check current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        print("No GPU available for memory check")


class EarlyStopping:
    """Early stopping utility class"""

    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self, model):
        if hasattr(model, 'state_dict'):
            self.best_weights = model.state_dict().copy()
