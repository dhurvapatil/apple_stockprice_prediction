#!/usr/bin/env python3
"""
Stock Price Prediction Project - Updated for Phase 4
Main entry point for the project
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collection import DataCollector
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from models import ModelTrainer
from hyperparameter_optimization import OptunaTuner
from utils import setup_logging, set_random_seeds, check_gpu_availability
from config import Config


def main():
    parser = argparse.ArgumentParser(description="Stock Price Prediction Project")
    parser.add_argument("--mode", choices=["collect", "preprocess", "features", "train", "optimize", "all"],
                        default="all", help="Operation mode")
    parser.add_argument("--symbol", default=Config.SYMBOL,
                        help="Stock symbol to process")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of optimization trials per model")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    # Setup
    setup_logging()
    set_random_seeds()

    print("=" * 60)
    print("STOCK PRICE PREDICTION PROJECT")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Mode: {args.mode}")

    # Check GPU
    gpu_available = check_gpu_availability()

    try:
        if args.mode in ["collect", "all"]:
            print("\n" + "-" * 40)
            print("PHASE 1A: DATA COLLECTION")
            print("-" * 40)

            collector = DataCollector(args.symbol)
            collector.get_stock_info()
            df = collector.download_stock_data()

            print(f"✓ Data collection completed: {len(df)} records")

        if args.mode in ["preprocess", "all"]:
            print("\n" + "-" * 40)
            print("PHASE 1B: DATA PREPROCESSING")
            print("-" * 40)

            preprocessor = DataPreprocessor()
            train_df, val_df, test_df = preprocessor.process_pipeline(args.symbol)

            print(f"✓ Data preprocessing completed")
            print(f"  Train: {len(train_df)} samples")
            print(f"  Validation: {len(val_df)} samples")
            print(f"  Test: {len(test_df)} samples")

        if args.mode in ["features", "all"]:
            print("\n" + "-" * 40)
            print("PHASE 2: FEATURE ENGINEERING")
            print("-" * 40)

            engineer = FeatureEngineer()
            train_featured, val_featured, test_featured = engineer.run_feature_engineering_pipeline()

            print(f"✓ Feature engineering completed")
            print(f"  Features created: {len(engineer.features_created)}")

        if args.mode in ["train", "all"]:
            print("\n" + "-" * 40)
            print("PHASE 3: MODEL TRAINING")
            print("-" * 40)

            trainer = ModelTrainer()
            results = trainer.run_model_training_pipeline()

            print(f"✓ Model training completed")
            print(f"  Models trained: {len(results)}")

        if args.mode in ["optimize", "all"]:
            print("\n" + "-" * 40)
            print("PHASE 4: HYPERPARAMETER OPTIMIZATION")
            print("-" * 40)

            tuner = OptunaTuner()
            optimized_results = tuner.run_optimization_pipeline(
                rf_trials=args.trials,
                xgb_trials=args.trials,
                lstm_trials=max(25, args.trials // 2)  # Fewer LSTM trials due to time
            )

            print(f"✓ Hyperparameter optimization completed")
            print(f"  Optimized models: {len(optimized_results)}")

        print("\n" + "=" * 60)
        print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        if args.mode == "all":
            print("\nProject complete! Check the following directories:")
            print("- models/trained_models/ - Trained and optimized models")
            print("- results/optuna_studies/ - Optimization studies and results")
            print("- data/processed/features/ - Feature-engineered data")
        elif args.mode == "optimize":
            print("\nOptimization complete! Check results/optuna_studies/ for detailed results")

    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
