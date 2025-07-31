import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import logging
import json
import time
from pathlib import Path
from config import Config
from models import StockDataset, LSTMPredictor, ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OptunaTuner:
    """Hyperparameter optimization using Optuna for all three models"""

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.results_dir = Config.RESULTS_DIR / "optuna_studies"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.trainer = ModelTrainer()
        self.best_params = {}
        self.best_scores = {}

    def optimize_random_forest(self, n_trials=100):
        """Optimize Random Forest hyperparameters"""
        logging.info("Starting Random Forest optimization...")

        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }

            # Train model with suggested parameters
            model = RandomForestClassifier(
                **params,
                class_weight='balanced',
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            )

            model.fit(self.trainer.X_train, self.trainer.y_train)

            # Evaluate on validation set
            val_pred = model.predict(self.trainer.X_val)
            val_accuracy = accuracy_score(self.trainer.y_val, val_pred)

            return val_accuracy

        # Create study with TPE sampler and median pruner
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=Config.RANDOM_STATE, n_startup_trials=10),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30 minutes max

        self.best_params['random_forest'] = study.best_params
        self.best_scores['random_forest'] = study.best_value

        # Save study
        joblib.dump(study, self.results_dir / "rf_study.pkl")

        logging.info(f"RF Best validation accuracy: {study.best_value:.4f}")
        logging.info(f"RF Best parameters: {study.best_params}")

        return study.best_params, study.best_value

    def optimize_xgboost(self, n_trials=100):
        """Optimize XGBoost hyperparameters"""
        logging.info("Starting XGBoost optimization...")

        def objective(trial):
            # Suggest hyperparameters with regularization focus
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 25, 200),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 20),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 20),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5)
            }

            # Train model
            model = xgb.XGBClassifier(
                **params,
                objective='binary:logistic',
                eval_metric='auc',
                tree_method='hist',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            )

            model.fit(
                self.trainer.X_train, self.trainer.y_train,
                eval_set=[(self.trainer.X_val, self.trainer.y_val)],
                verbose=False
            )

            # Evaluate
            val_pred = model.predict(self.trainer.X_val)
            val_accuracy = accuracy_score(self.trainer.y_val, val_pred)

            # Check for overfitting
            train_pred = model.predict(self.trainer.X_train)
            train_accuracy = accuracy_score(self.trainer.y_train, train_pred)

            # Penalize severe overfitting
            overfitting_penalty = max(0, train_accuracy - val_accuracy - 0.15)

            return val_accuracy - overfitting_penalty

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=Config.RANDOM_STATE, n_startup_trials=15),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=15)
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30 minutes max

        self.best_params['xgboost'] = study.best_params
        self.best_scores['xgboost'] = study.best_value

        # Save study
        joblib.dump(study, self.results_dir / "xgb_study.pkl")

        logging.info(f"XGB Best validation accuracy: {study.best_value:.4f}")
        logging.info(f"XGB Best parameters: {study.best_params}")

        return study.best_params, study.best_value

    def optimize_lstm(self, n_trials=50):
        """Optimize LSTM hyperparameters"""
        logging.info("Starting LSTM optimization...")

        def optimize_lstm(self, n_trials=50):
            """Optimize LSTM hyperparameters"""
            logging.info("Starting LSTM optimization...")

            def objective(trial):
                # Suggest hyperparameters
                params = {
                    'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
                    'num_layers': trial.suggest_int('num_layers', 1, 4),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.6),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                    'sequence_length': trial.suggest_int('sequence_length', 20, 80)
                }

                # Prepare data
                train_features = self.trainer.X_train.values
                val_features = self.trainer.X_val.values
                train_targets = self.trainer.y_train.values
                val_targets = self.trainer.y_val.values

                # Create datasets
                train_dataset = StockDataset(train_features, train_targets, params['sequence_length'])
                val_dataset = StockDataset(val_features, val_targets, params['sequence_length'])

                if len(train_dataset) < 10 or len(val_dataset) < 5:
                    return 0.0  # Not enough data for this sequence length

                # Create data loaders
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=params['batch_size'],
                    shuffle=True,
                    pin_memory=torch.cuda.is_available(),
                    num_workers=0  # Avoid multiprocessing issues in Optuna
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=params['batch_size'],
                    shuffle=False,
                    pin_memory=torch.cuda.is_available(),
                    num_workers=0
                )

                # Initialize model
                input_size = len(self.trainer.feature_cols)
                model = LSTMPredictor(
                    input_size=input_size,
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout']
                ).to(self.device)

                # Training setup
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

                # Mixed precision - FIXED API
                scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

                # Training loop with early stopping
                best_val_acc = 0
                patience_counter = 0
                max_patience = 5
                max_epochs = 30

                for epoch in range(max_epochs):
                    # Training phase
                    model.train()
                    train_correct = 0
                    train_total = 0

                    for data, targets in train_loader:
                        data, targets = data.to(self.device), targets.to(self.device)

                        optimizer.zero_grad()

                        if scaler:
                            with torch.amp.autocast('cuda'):  # FIXED
                                outputs = model(data).squeeze()
                                loss = criterion(outputs, targets)

                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(data).squeeze()
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()

                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        train_total += targets.size(0)
                        train_correct += (predicted == targets).sum().item()

                    # Validation phase
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    val_loss = 0

                    with torch.no_grad():
                        for data, targets in val_loader:
                            data, targets = data.to(self.device), targets.to(self.device)

                            if scaler:
                                with torch.amp.autocast('cuda'):  # FIXED
                                    outputs = model(data).squeeze()
                                    loss = criterion(outputs, targets)
                            else:
                                outputs = model(data).squeeze()
                                loss = criterion(outputs, targets)

                            val_loss += loss.item()
                            predicted = (torch.sigmoid(outputs) > 0.5).float()
                            val_total += targets.size(0)
                            val_correct += (predicted == targets).sum().item()

                    val_acc = val_correct / val_total if val_total > 0 else 0.0

                    # Learning rate scheduling
                    scheduler.step(val_loss)

                    # Early stopping
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Report to Optuna for pruning
                    trial.report(val_acc, epoch)

                    # Check if trial should be pruned
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    if patience_counter >= max_patience:
                        break

                return best_val_acc

    def train_optimized_models(self):
        """Train final models with optimized hyperparameters"""
        logging.info("Training final models with optimized parameters...")

        final_results = {}

        # Train optimized Random Forest
        if 'random_forest' in self.best_params:
            logging.info("Training optimized Random Forest...")
            rf_model = RandomForestClassifier(
                **self.best_params['random_forest'],
                class_weight='balanced',
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            )
            rf_model.fit(self.trainer.X_train, self.trainer.y_train)

            # Evaluate
            train_pred = rf_model.predict(self.trainer.X_train)
            val_pred = rf_model.predict(self.trainer.X_val)
            test_pred = rf_model.predict(self.trainer.X_test)

            final_results['optimized_random_forest'] = {
                'train_accuracy': accuracy_score(self.trainer.y_train, train_pred),
                'val_accuracy': accuracy_score(self.trainer.y_val, val_pred),
                'test_accuracy': accuracy_score(self.trainer.y_test, test_pred),
                'best_params': self.best_params['random_forest']
            }

            # Save model
            joblib.dump(rf_model, self.trainer.models_dir / "optimized_random_forest.pkl")

        # Train optimized XGBoost
        if 'xgboost' in self.best_params:
            logging.info("Training optimized XGBoost...")
            xgb_model = xgb.XGBClassifier(
                **self.best_params['xgboost'],
                objective='binary:logistic',
                eval_metric='auc',
                tree_method='hist',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            )
            xgb_model.fit(self.trainer.X_train, self.trainer.y_train)

            # Evaluate
            train_pred = xgb_model.predict(self.trainer.X_train)
            val_pred = xgb_model.predict(self.trainer.X_val)
            test_pred = xgb_model.predict(self.trainer.X_test)

            final_results['optimized_xgboost'] = {
                'train_accuracy': accuracy_score(self.trainer.y_train, train_pred),
                'val_accuracy': accuracy_score(self.trainer.y_val, val_pred),
                'test_accuracy': accuracy_score(self.trainer.y_test, test_pred),
                'best_params': self.best_params['xgboost']
            }

            # Save model
            joblib.dump(xgb_model, self.trainer.models_dir / "optimized_xgboost.pkl")

        # Train optimized LSTM
        if 'lstm' in self.best_params:
            logging.info("Training optimized LSTM...")
            # Use the trainer's LSTM training method with optimized parameters
            val_acc = self.trainer.train_lstm(**self.best_params['lstm'])

            final_results['optimized_lstm'] = {
                'train_accuracy': self.trainer.results['lstm']['train_accuracy'],
                'val_accuracy': self.trainer.results['lstm']['val_accuracy'],
                'test_accuracy': self.trainer.results['lstm']['test_accuracy'],
                'best_params': self.best_params['lstm']
            }

        return final_results

    def run_optimization_pipeline(self, rf_trials=100, xgb_trials=100, lstm_trials=50):
        """Run complete hyperparameter optimization pipeline"""
        logging.info("=" * 60)
        logging.info("PHASE 4: HYPERPARAMETER OPTIMIZATION PIPELINE")
        logging.info("=" * 60)

        start_time = time.time()

        # Optimize each model
        print("\n" + "-" * 40)
        print("OPTIMIZING RANDOM FOREST")
        print("-" * 40)
        self.optimize_random_forest(rf_trials)

        print("\n" + "-" * 40)
        print("OPTIMIZING XGBOOST")
        print("-" * 40)
        self.optimize_xgboost(xgb_trials)

        print("\n" + "-" * 40)
        print("OPTIMIZING LSTM")
        print("-" * 40)
        self.optimize_lstm(lstm_trials)

        # Train final optimized models
        print("\n" + "-" * 40)
        print("TRAINING OPTIMIZED MODELS")
        print("-" * 40)
        final_results = self.train_optimized_models()

        total_time = time.time() - start_time

        # Save optimization results
        optimization_summary = {
            'best_parameters': self.best_params,
            'best_validation_scores': self.best_scores,
            'final_results': final_results,
            'optimization_time': total_time,
            'trials': {
                'random_forest': rf_trials,
                'xgboost': xgb_trials,
                'lstm': lstm_trials
            }
        }

        with open(self.results_dir / "optimization_summary.json", 'w') as f:
            json.dump(optimization_summary, f, indent=2, default=str)

        # Print summary
        self.print_optimization_summary(final_results, total_time)

        return final_results

    def print_optimization_summary(self, final_results, total_time):
        """Print optimization results summary"""
        print(f"\n" + "=" * 60)
        print("PHASE 4 OPTIMIZATION RESULTS")
        print("=" * 60)

        print(f"Total optimization time: {total_time / 60:.1f} minutes")
        print(f"Total trials executed: {sum([100, 100, 50])}")  # rf + xgb + lstm trials

        print(f"\n" + "=" * 50)
        print("OPTIMIZED MODEL COMPARISON")
        print("=" * 50)

        comparison_data = []
        for model_name, results in final_results.items():
            comparison_data.append({
                'Model': model_name.replace('optimized_', '').replace('_', ' ').title(),
                'Train Acc': f"{results['train_accuracy']:.4f}",
                'Val Acc': f"{results['val_accuracy']:.4f}",
                'Test Acc': f"{results['test_accuracy']:.4f}"
            })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Find best model
        if final_results:
            best_model = max(final_results.keys(), key=lambda x: final_results[x]['test_accuracy'])
            best_test_acc = final_results[best_model]['test_accuracy']

            print(f"\n🏆 Best Optimized Model: {best_model.replace('optimized_', '').replace('_', ' ').title()}")
            print(f"🎯 Best Test Accuracy: {best_test_acc:.4f}")

            # Show improvement
            if best_test_acc > 0.7273:  # Previous best LSTM result
                improvement = (best_test_acc - 0.7273) * 100
                print(f"📈 Improvement over baseline: +{improvement:.2f}%")

            print(f"\n💡 Best Parameters:")
            for param, value in final_results[best_model]['best_params'].items():
                print(f"  {param}: {value}")


def main():
    """Main function to run hyperparameter optimization"""
    tuner = OptunaTuner()
    final_results = tuner.run_optimization_pipeline(
        rf_trials=100,  # Random Forest trials
        xgb_trials=100,  # XGBoost trials
        lstm_trials=50  # LSTM trials (fewer due to longer training time)
    )
    return final_results


if __name__ == "__main__":
    main()
