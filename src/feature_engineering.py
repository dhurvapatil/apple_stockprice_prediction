import pandas as pd
import numpy as np
import ta
import logging
from pathlib import Path
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FeatureEngineer:
    def __init__(self):
        self.processed_dir = Config.PROCESSED_DATA_DIR
        self.features_created = []

    def load_processed_data(self):
        """Load the preprocessed data from Phase 1"""
        try:
            train_path = self.processed_dir / "train.csv"
            val_path = self.processed_dir / "val.csv"
            test_path = self.processed_dir / "test.csv"

            train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
            val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
            test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)

            logging.info(
                f"Loaded preprocessed data - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
            return train_df, val_df, test_df

        except Exception as e:
            logging.error(f"Error loading processed data: {str(e)}")
            raise

    def create_price_features(self, df):
        """Create price-based technical features"""
        logging.info("Creating price-based features...")

        # Use Adj_Close for calculations to handle splits/dividends
        close = df['Adj_Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Simple Moving Averages
        for window in Config.SMA_WINDOWS:
            df[f'SMA_{window}'] = ta.trend.SMAIndicator(close, window).sma_indicator()
            df[f'Close_SMA_{window}_ratio'] = close / df[f'SMA_{window}']
            self.features_created.append(f'SMA_{window}')
            self.features_created.append(f'Close_SMA_{window}_ratio')

        # Exponential Moving Averages
        for window in Config.EMA_WINDOWS:
            df[f'EMA_{window}'] = ta.trend.EMAIndicator(close, window).ema_indicator()
            df[f'Close_EMA_{window}_ratio'] = close / df[f'EMA_{window}']
            self.features_created.append(f'EMA_{window}')
            self.features_created.append(f'Close_EMA_{window}_ratio')

        # Price Returns (percentage changes)
        for period in Config.RETURN_PERIODS:
            df[f'Return_{period}d'] = close.pct_change(period)
            self.features_created.append(f'Return_{period}d')

        # High-Low ratios
        df['HL_ratio'] = high / low
        df['Close_High_ratio'] = close / high
        df['Close_Low_ratio'] = close / low

        # Price position within daily range
        df['Price_position'] = (close - low) / (high - low)

        self.features_created.extend(['HL_ratio', 'Close_High_ratio', 'Close_Low_ratio', 'Price_position'])

        logging.info(
            f"Created {len([f for f in self.features_created if 'SMA' in f or 'EMA' in f or 'Return' in f or 'ratio' in f])} price features")
        return df

    def validate_no_lookahead_bias(self, df):
        """Validate that no features use future information"""
        logging.info("Validating features for look-ahead bias...")

        # Check that all features are properly lagged
        close = df['Adj_Close']

        # Verify target is properly shifted
        future_close = close.shift(-1)
        current_close = close

        # Check some critical features
        suspicious_features = []

        for feature in self.features_created:
            if feature in df.columns:
                feature_series = df[feature]

                # Check correlation with future prices (should be lower than current prices)
                if len(feature_series.dropna()) > 50:  # Enough data points
                    corr_future = feature_series.corr(future_close)
                    corr_current = feature_series.corr(current_close)

                    # If correlation with future is much higher than current, flag it
                    if abs(corr_future) > abs(corr_current) + 0.1:
                        suspicious_features.append({
                            'feature': feature,
                            'corr_current': corr_current,
                            'corr_future': corr_future
                        })

        if suspicious_features:
            logging.warning(f"Found {len(suspicious_features)} potentially leaky features:")
            for item in suspicious_features:
                logging.warning(
                    f"  {item['feature']}: current={item['corr_current']:.3f}, future={item['corr_future']:.3f}")
        else:
            logging.info("✓ No obvious look-ahead bias detected")

        return suspicious_features

    def create_bollinger_bands(self, df):
        """Create Bollinger Bands features"""
        logging.info("Creating Bollinger Bands features...")

        close = df['Adj_Close']

        # Bollinger Bands (20-day window, 2 standard deviations)
        bb_indicator = ta.volatility.BollingerBands(close, window=20, window_dev=2)

        df['BB_upper'] = bb_indicator.bollinger_hband()
        df['BB_middle'] = bb_indicator.bollinger_mavg()
        df['BB_lower'] = bb_indicator.bollinger_lband()

        # Bollinger Band position (0 = at lower band, 1 = at upper band)
        df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # Bollinger Band width (volatility measure)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

        # Distance from bands
        df['BB_upper_dist'] = (df['BB_upper'] - close) / close
        df['BB_lower_dist'] = (close - df['BB_lower']) / close

        bb_features = ['BB_upper', 'BB_middle', 'BB_lower', 'BB_position', 'BB_width', 'BB_upper_dist', 'BB_lower_dist']
        self.features_created.extend(bb_features)

        logging.info(f"Created {len(bb_features)} Bollinger Band features")
        return df

    def create_momentum_indicators(self, df):
        """Create momentum-based technical indicators"""
        logging.info("Creating momentum indicators...")

        close = df['Adj_Close']
        high = df['High']
        low = df['Low']

        # RSI (Relative Strength Index)
        df['RSI_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        df['RSI_30'] = ta.momentum.RSIIndicator(close, window=30).rsi()

        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        df['Stoch_k'] = stoch.stoch()
        df['Stoch_d'] = stoch.stoch_signal()

        # Williams %R
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()

        # Rate of Change (ROC)
        df['ROC_10'] = ta.momentum.ROCIndicator(close, window=10).roc()
        df['ROC_20'] = ta.momentum.ROCIndicator(close, window=20).roc()

        momentum_features = ['RSI_14', 'RSI_30', 'MACD', 'MACD_signal', 'MACD_histogram',
                             'Stoch_k', 'Stoch_d', 'Williams_R', 'ROC_10', 'ROC_20']
        self.features_created.extend(momentum_features)

        logging.info(f"Created {len(momentum_features)} momentum indicators")
        return df

    def create_volume_features(self, df):
        """Create volume-based features"""
        logging.info("Creating volume features...")

        volume = df['Volume']
        close = df['Adj_Close']
        high = df['High']
        low = df['Low']

        # Volume moving averages (using pandas rolling instead of ta)
        df['Volume_SMA_10'] = volume.rolling(window=10).mean()
        df['Volume_SMA_20'] = volume.rolling(window=20).mean()

        # Volume ratios
        df['Volume_ratio_10'] = volume / df['Volume_SMA_10']
        df['Volume_ratio_20'] = volume / df['Volume_SMA_20']

        # On-Balance Volume (OBV) - manual calculation
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        # Volume Price Trend (VPT) - manual calculation
        df['VPT'] = ta.volume.VolumePriceTrendIndicator(close, volume).volume_price_trend()

        # Accumulation/Distribution Line
        df['AD_Line'] = ta.volume.AccDistIndexIndicator(high=high, low=low,
                                                        close=close, volume=volume).acc_dist_index()

        # Additional volume features using pandas
        df['Volume_std_10'] = volume.rolling(window=10).std()
        df['Volume_price_trend'] = (close.pct_change() * volume).rolling(window=5).mean()

        volume_features = ['Volume_SMA_10', 'Volume_SMA_20', 'Volume_ratio_10', 'Volume_ratio_20',
                           'OBV', 'VPT', 'AD_Line', 'Volume_std_10', 'Volume_price_trend']
        self.features_created.extend(volume_features)

        logging.info(f"Created {len(volume_features)} volume features")
        return df

    def create_volatility_features(self, df):
        """Create volatility-based features"""
        logging.info("Creating volatility features...")

        close = df['Adj_Close']
        high = df['High']
        low = df['Low']

        # Average True Range (ATR)
        df['ATR_14'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        df['ATR_20'] = ta.volatility.AverageTrueRange(high, low, close, window=20).average_true_range()

        # Normalized ATR (as percentage of price)
        df['ATR_14_pct'] = df['ATR_14'] / close
        df['ATR_20_pct'] = df['ATR_20'] / close

        # Rolling volatility (standard deviation of returns)
        df['Volatility_10'] = close.pct_change().rolling(window=10).std()
        df['Volatility_20'] = close.pct_change().rolling(window=20).std()
        df['Volatility_50'] = close.pct_change().rolling(window=50).std()

        volatility_features = ['ATR_14', 'ATR_20', 'ATR_14_pct', 'ATR_20_pct',
                               'Volatility_10', 'Volatility_20', 'Volatility_50']
        self.features_created.extend(volatility_features)

        logging.info(f"Created {len(volatility_features)} volatility features")
        return df

    def create_time_features(self, df):
        """Create time-based features"""
        logging.info("Creating time-based features...")

        # Day of week (0 = Monday, 6 = Sunday)
        df['DayOfWeek'] = df.index.dayofweek

        # Month (1-12)
        df['Month'] = df.index.month

        # Quarter (1-4)
        df['Quarter'] = df.index.quarter

        # Year
        df['Year'] = df.index.year

        # Days since market highs/lows
        close = df['Adj_Close']
        df['Days_since_high_20'] = self._days_since_extreme(close, 20, 'high')
        df['Days_since_low_20'] = self._days_since_extreme(close, 20, 'low')
        df['Days_since_high_50'] = self._days_since_extreme(close, 50, 'high')
        df['Days_since_low_50'] = self._days_since_extreme(close, 50, 'low')

        # Distance from recent highs/lows
        df['Dist_from_high_20'] = (close.rolling(20).max() - close) / close
        df['Dist_from_low_20'] = (close - close.rolling(20).min()) / close

        time_features = ['DayOfWeek', 'Month', 'Quarter', 'Year',
                         'Days_since_high_20', 'Days_since_low_20', 'Days_since_high_50', 'Days_since_low_50',
                         'Dist_from_high_20', 'Dist_from_low_20']
        self.features_created.extend(time_features)

        logging.info(f"Created {len(time_features)} time-based features")
        return df

    def _days_since_extreme(self, series, window, extreme_type):
        """Helper function to calculate days since high/low"""
        if extreme_type == 'high':
            rolling_extreme = series.rolling(window).max()
        else:
            rolling_extreme = series.rolling(window).min()

        days_since = []
        for i in range(len(series)):
            if i < window:
                days_since.append(np.nan)
            else:
                extreme_value = rolling_extreme.iloc[i]
                # Find the most recent occurrence of this extreme
                recent_window = series.iloc[max(0, i - window + 1):i + 1]
                if extreme_type == 'high':
                    extreme_idx = recent_window[recent_window == extreme_value].index[-1]
                else:
                    extreme_idx = recent_window[recent_window == extreme_value].index[-1]

                days_since.append(i - recent_window.index.get_loc(extreme_idx))

        return pd.Series(days_since, index=series.index)

    def create_target_variables(self, df):
        """Create target variables for prediction"""
        logging.info("Creating target variables...")

        close = df['Adj_Close']

        # Binary classification target: 1 if next day's close > today's close, 0 otherwise
        df['Target_binary'] = (close.shift(-1) > close).astype(int)

        # Regression target: next day's percentage change
        df['Target_return'] = close.pct_change().shift(-1)

        # Multi-class target: direction with magnitude
        returns = df['Target_return']
        df['Target_multiclass'] = pd.cut(returns,
                                         bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                                         labels=[0, 1, 2, 3, 4]).astype(float)

        logging.info("Created target variables: binary, return, and multiclass")
        return df

    def remove_outliers(self, df, method='iqr', threshold=3):
        """Remove outliers from features (but keep all target data)"""
        logging.info(f"Removing outliers using {method} method...")

        # Get feature columns (exclude original OHLCV and target columns)
        feature_cols = [col for col in df.columns if col not in
                        ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'Dividends', 'Stock_Splits',
                         'Target_binary', 'Target_return', 'Target_multiclass']]

        initial_shape = df.shape

        if method == 'iqr':
            # Interquartile Range method
            for col in feature_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower, upper=upper)

        elif method == 'zscore':
            # Z-score method
            for col in feature_cols:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(lower=mean - threshold * std, upper=mean + threshold * std)

        logging.info(f"Outlier removal completed. Shape: {initial_shape} -> {df.shape}")
        return df

    def engineer_features(self, df):
        """Apply all feature engineering steps to a dataframe"""
        logging.info("Starting complete feature engineering pipeline...")

        # Create all feature types
        df = self.create_price_features(df)
        df = self.create_bollinger_bands(df)
        df = self.create_momentum_indicators(df)
        df = self.create_volume_features(df)
        df = self.create_volatility_features(df)
        df = self.create_time_features(df)
        df = self.create_target_variables(df)

        # Remove outliers
        df = self.remove_outliers(df)

        # Drop rows with NaN values (from indicators that need lookback)
        initial_len = len(df)
        df = df.dropna()
        final_len = len(df)
        dropped = initial_len - final_len

        logging.info(f"Dropped {dropped} rows with NaN values (from indicator lookback)")
        logging.info(f"Final dataset shape: {df.shape}")
        logging.info(f"Total features created: {len(self.features_created)}")

        return df

    def save_engineered_data(self, train_df, val_df, test_df):
        """Save feature-engineered data"""
        logging.info("Saving feature-engineered data...")

        # Create features directory
        features_dir = Config.PROCESSED_DATA_DIR / "features"
        features_dir.mkdir(exist_ok=True)

        # Save datasets
        train_df.to_csv(features_dir / "train_features.csv")
        val_df.to_csv(features_dir / "val_features.csv")
        test_df.to_csv(features_dir / "test_features.csv")

        # Save feature list
        feature_info = {
            'total_features': len(self.features_created),
            'feature_names': self.features_created,
            'train_shape': train_df.shape,
            'val_shape': val_df.shape,
            'test_shape': test_df.shape
        }

        import json
        with open(features_dir / "feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2, default=str)

        logging.info(f"Feature-engineered data saved to: {features_dir}")
        return features_dir

    def run_feature_engineering_pipeline(self):
        """Complete feature engineering pipeline with validation"""
        logging.info("=" * 60)
        logging.info("PHASE 2: FEATURE ENGINEERING PIPELINE (WITH VALIDATION)")
        logging.info("=" * 60)

        # Load preprocessed data from Phase 1
        train_df, val_df, test_df = self.load_processed_data()

        # Apply feature engineering to each dataset
        logging.info("\nEngineering features for training set...")
        train_featured = self.engineer_features(train_df.copy())

        # Validate for look-ahead bias using training data
        suspicious_features = self.validate_no_lookahead_bias(train_featured)

        logging.info("\nEngineering features for validation set...")
        val_featured = self.engineer_features(val_df.copy())

        logging.info("\nEngineering features for test set...")
        test_featured = self.engineer_features(test_df.copy())

        # Additional validation step
        if suspicious_features:
            logging.warning("⚠️  Potential data leakage detected - review features carefully")
        else:
            logging.info("✓ Feature validation passed")

        # Save results
        features_dir = self.save_engineered_data(train_featured, val_featured, test_featured)

        return train_featured, val_featured, test_featured


def main():
    """Main function to run feature engineering"""
    engineer = FeatureEngineer()
    train_df, val_df, test_df = engineer.run_feature_engineering_pipeline()


if __name__ == "__main__":
    main()
