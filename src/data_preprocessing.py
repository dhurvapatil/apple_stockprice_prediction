import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import logging
from pathlib import Path
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataPreprocessor:
    def __init__(self):
        self.raw_dir = Config.RAW_DATA_DIR
        self.processed_dir = Config.PROCESSED_DATA_DIR
        self.scaler = None

    def load_raw_data(self, symbol=Config.SYMBOL):
        """Load raw data from CSV file and properly handle timezone-aware datetime strings"""
        try:
            file_path = self.raw_dir / f"{symbol}_raw.csv"

            # Load CSV without automatic date parsing first
            df = pd.read_csv(file_path, index_col=0)

            # Manually convert the timezone-aware string index to datetime
            df.index = pd.to_datetime(df.index, utc=True)  # Parse timezone-aware strings
            df.index = df.index.tz_localize(None)  # Remove timezone info

            logging.info(f"Loaded raw data: {df.shape}")
            logging.info(f"Index type: {type(df.index)}, dtype: {df.index.dtype}")
            logging.info(f"Date range: {df.index.min()} to {df.index.max()}")

            return df
        except Exception as e:
            logging.error(f"Error loading raw data: {str(e)}")
            raise

    def clean_data(self, df):
        """
        Clean the raw data

        Args:
            df: Raw dataframe

        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        logging.info("Starting data cleaning...")

        # Remove any rows with zero volume (non-trading days)
        initial_len = len(df)
        df = df[df['Volume'] > 0].copy()
        removed_rows = initial_len - len(df)
        if removed_rows > 0:
            logging.info(f"Removed {removed_rows} rows with zero volume")

        # Handle missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            logging.warning(f"Found {missing_before} missing values")
            # Forward fill missing values (common for financial data)
            df = df.fillna(method='ffill')
            # If still missing (at the beginning), backward fill
            df = df.fillna(method='bfill')

        missing_after = df.isnull().sum().sum()
        logging.info(f"Missing values after cleaning: {missing_after}")

        # Check for any remaining missing values
        if df.isnull().sum().sum() > 0:
            logging.warning("Still have missing values after cleaning")
            df = df.dropna()

        # Ensure proper data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Basic data validation
        if len(df) < 100:
            raise ValueError(f"Insufficient data after cleaning: {len(df)} rows")

        # Check for price anomalies (prices should be positive)
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close']
        for col in price_cols:
            if col in df.columns:
                if (df[col] <= 0).any():
                    logging.warning(f"Found non-positive prices in {col}")
                    df = df[df[col] > 0]

        logging.info(f"Data cleaning completed. Final shape: {df.shape}")
        return df

    def create_time_splits(self, df):
        """
        Create time-based train/validation/test splits

        Args:
            df: Cleaned dataframe

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        logging.info("Creating time-based splits...")

        try:
            # Ensure the index is a proper DatetimeIndex (should be fixed in load_raw_data now)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                logging.info("Converted index to timezone-naive DatetimeIndex")

            # Convert config dates to datetime objects
            train_start = pd.to_datetime(Config.TRAIN_START)
            train_end = pd.to_datetime(Config.TRAIN_END)
            val_start = pd.to_datetime(Config.VAL_START)
            val_end = pd.to_datetime(Config.VAL_END)
            test_start = pd.to_datetime(Config.TEST_START)
            test_end = pd.to_datetime(Config.TEST_END)

            # Debug info
            logging.info(f"Index type: {type(df.index)}, dtype: {df.index.dtype}")
            logging.info(f"Data date range: {df.index.min()} to {df.index.max()}")

            # Use boolean indexing for robust filtering
            train_mask = (df.index >= train_start) & (df.index <= train_end)
            val_mask = (df.index >= val_start) & (df.index <= val_end)
            test_mask = (df.index >= test_start) & (df.index <= test_end)

            train_df = df[train_mask].copy()
            val_df = df[val_mask].copy()
            test_df = df[test_mask].copy()

            logging.info(f"Train set: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
            logging.info(f"Validation set: {len(val_df)} samples ({val_df.index.min()} to {val_df.index.max()})")
            logging.info(f"Test set: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")

            # Ensure we have sufficient data in each split
            min_samples = 50
            if len(train_df) < min_samples:
                raise ValueError(f"Insufficient training data: {len(train_df)} samples")
            if len(val_df) < min_samples:
                raise ValueError(f"Insufficient validation data: {len(val_df)} samples")
            if len(test_df) < min_samples:
                logging.warning(f"Limited test data: {len(test_df)} samples")

            return train_df, val_df, test_df

        except Exception as e:
            logging.error(f"Error creating time splits: {str(e)}")
            raise

    def normalize_features(self, train_df, val_df, test_df):
        """
        Normalize features using training data statistics

        Args:
            train_df, val_df, test_df: DataFrames to normalize

        Returns:
            tuple: Normalized dataframes and fitted scaler
        """
        logging.info("Normalizing features...")

        # Define which columns to normalize
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        columns_to_scale = [col for col in numeric_columns if col in train_df.columns]

        # Use RobustScaler to handle outliers better than StandardScaler
        self.scaler = RobustScaler()

        # Fit scaler on training data only
        train_scaled = train_df.copy()
        train_scaled[columns_to_scale] = self.scaler.fit_transform(train_df[columns_to_scale])

        # Transform validation and test data using training statistics
        val_scaled = val_df.copy()
        val_scaled[columns_to_scale] = self.scaler.transform(val_df[columns_to_scale])

        test_scaled = test_df.copy()
        test_scaled[columns_to_scale] = self.scaler.transform(test_df[columns_to_scale])

        # Save the scaler
        scaler_path = self.processed_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logging.info(f"Scaler saved to: {scaler_path}")

        logging.info("Feature normalization completed")
        return train_scaled, val_scaled, test_scaled

    def save_processed_data(self, train_df, val_df, test_df):
        """Save processed data to CSV files"""
        logging.info("Saving processed data...")

        train_path = self.processed_dir / "train.csv"
        val_path = self.processed_dir / "val.csv"
        test_path = self.processed_dir / "test.csv"

        train_df.to_csv(train_path)
        val_df.to_csv(val_path)
        test_df.to_csv(test_path)

        logging.info(f"Processed data saved:")
        logging.info(f"  Train: {train_path}")
        logging.info(f"  Validation: {val_path}")
        logging.info(f"  Test: {test_path}")

    def process_pipeline(self, symbol=Config.SYMBOL):
        """
        Complete preprocessing pipeline

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        logging.info("Starting preprocessing pipeline...")

        # Load raw data
        df = self.load_raw_data(symbol)

        # Clean data
        df_clean = self.clean_data(df)

        # Create time splits
        train_df, val_df, test_df = self.create_time_splits(df_clean)

        # Normalize features
        train_norm, val_norm, test_norm = self.normalize_features(train_df, val_df, test_df)

        # Save processed data
        self.save_processed_data(train_norm, val_norm, test_norm)

        logging.info("Preprocessing pipeline completed successfully!")

        return train_norm, val_norm, test_norm


def main():
    """Main function to run preprocessing"""
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.process_pipeline()

    print(f"\nPreprocessing completed!")
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"\nSample of processed training data:")
    print(train_df.head())


if __name__ == "__main__":
    main()
