#!/usr/bin/env python3
"""Quick script to test overfitting fixes"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from models import ModelTrainer


def test_regularized_models():
    """Test models with stronger regularization"""
    trainer = ModelTrainer()

    print("Testing regularized XGBoost...")

    # Test with heavy regularization
    val_acc = trainer.train_xgboost(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=10,
        reg_lambda=10
    )

    print(f"Regularized XGBoost Results:")
    print(f"Train Acc: {trainer.results['xgboost']['train_accuracy']:.4f}")
    print(f"Val Acc: {trainer.results['xgboost']['val_accuracy']:.4f}")
    print(f"Test Acc: {trainer.results['xgboost']['test_accuracy']:.4f}")

    # Check if overfitting is reduced
    train_val_gap = trainer.results['xgboost']['train_accuracy'] - trainer.results['xgboost']['val_accuracy']
    print(f"Train-Val Gap: {train_val_gap:.4f}")

    if train_val_gap < 0.15:  # Less than 15% gap
        print("✓ Overfitting significantly reduced!")
    elif train_val_gap < 0.30:
        print("⚠️  Overfitting reduced but still present")
    else:
        print("❌ Overfitting still severe")

    # Run walk-forward validation
    print("\nRunning walk-forward validation...")
    results, avg_acc, std_acc = trainer.walk_forward_validation('xgboost')

    if std_acc < 0.05:  # Standard deviation less than 5%
        print("✓ Model shows consistent performance across time periods")
    else:
        print("⚠️  Model performance varies significantly across time periods")


if __name__ == "__main__":
    test_regularized_models()
