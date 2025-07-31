import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from config import Config
import optuna
import json
import time
from torch.cuda.amp import GradScaler, autocast


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class StockDataset(Dataset):
    """PyTorch Dataset for stock sequence data"""

    def __init__(self, features, targets, sequence_length=Config.SEQUENCE_LENGTH):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Get sequence of features
        x = self.features[idx:idx + self.sequence_length]
        # Get corresponding target (last day of sequence)
        y = self.targets[idx + self.sequence_length - 1]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMPredictor(nn.Module):
    """LSTM Neural Network for Stock Price Prediction"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, output_size=1):
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layers - Remove sigmoid (for BCEWithLogitsLoss)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        # Removed sigmoid layer

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Get last output
        last_output = lstm_out[:, -1, :]

        # Apply dropout
        dropped = self.dropout(last_output)

        # Fully connected layers - return logits (no sigmoid)
        out = self.relu(self.fc1(dropped))
        out = self.fc2(out)

        return out  # Return raw logits


class ModelTrainer:
    """Main class for training all three models"""

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.models_dir = Config.MODELS_DIR / "trained_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Load feature-engineered data
        self.load_data()

        # Results storage
        self.results = {}

    def load_data(self):
        """Load feature-engineered data from Phase 2"""
        try:
            features_dir = Config.PROCESSED_DATA_DIR / "features"

            self.train_df = pd.read_csv(features_dir / "train_features.csv", index_col=0, parse_dates=True)
            self.val_df = pd.read_csv(features_dir / "val_features.csv", index_col=0, parse_dates=True)
            self.test_df = pd.read_csv(features_dir / "test_features.csv", index_col=0, parse_dates=True)

            logging.info(
                f"Loaded feature data - Train: {self.train_df.shape}, Val: {self.val_df.shape}, Test: {self.test_df.shape}")

            # Separate features and targets
            self.prepare_model_data()

        except Exception as e:
            logging.error(f"Error loading feature data: {str(e)}")
            raise

    def prepare_model_data(self):
        """Prepare data for different model types"""
        # Define feature columns (exclude targets and original OHLCV)
        target_cols = ['Target_binary', 'Target_return', 'Target_multiclass']
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'Dividends',
                        'Stock_Splits'] + target_cols

        self.feature_cols = [col for col in self.train_df.columns if col not in exclude_cols]

        # Prepare data for sklearn models (Random Forest, XGBoost)
        self.X_train = self.train_df[self.feature_cols].fillna(0)
        self.X_val = self.val_df[self.feature_cols].fillna(0)
        self.X_test = self.test_df[self.feature_cols].fillna(0)

        # Handle infinity values
        self.X_train = self.X_train.replace([np.inf, -np.inf], 0)
        self.X_val = self.X_val.replace([np.inf, -np.inf], 0)
        self.X_test = self.X_test.replace([np.inf, -np.inf], 0)

        # Clip extreme values
        for col in self.feature_cols:
            q99 = self.X_train[col].quantile(0.99)
            q01 = self.X_train[col].quantile(0.01)
            self.X_train[col] = self.X_train[col].clip(q01, q99)
            self.X_val[col] = self.X_val[col].clip(q01, q99)
            self.X_test[col] = self.X_test[col].clip(q01, q99)

        self.y_train = self.train_df['Target_binary'].fillna(0)
        self.y_val = self.val_df['Target_binary'].fillna(0)
        self.y_test = self.test_df['Target_binary'].fillna(0)

        logging.info(f"Features prepared: {len(self.feature_cols)} features")
        logging.info(f"Target distribution - Train: {self.y_train.value_counts().to_dict()}")

        # Verify no NaN/inf values remain
        assert not self.X_train.isin([np.nan, np.inf, -np.inf]).any().any(), "Training data contains NaN/inf"
        assert not self.X_val.isin([np.nan, np.inf, -np.inf]).any().any(), "Validation data contains NaN/inf"

    def walk_forward_validation(self, model_type='random_forest', n_windows=12):
        """
        Professional walk-forward validation with multiple test periods

        Returns multiple independent test results instead of one tiny test set
        """
        # Load all your processed data
        all_data = pd.concat([self.train_df, self.val_df, self.test_df]).sort_index()

        # Define window sizes
        train_window = 750  # ~3 years
        test_window = 60  # ~3 months per test
        step_size = 30  # Roll forward monthly

        results = []
        all_predictions = []
        all_actual = []

        # Walk through time
        for i in range(n_windows):
            start_idx = i * step_size
            train_end = start_idx + train_window
            test_end = train_end + test_window

            if test_end > len(all_data):
                break

            # Get train/test data for this window
            X_train = all_data.iloc[start_idx:train_end][self.feature_cols]
            y_train = all_data.iloc[start_idx:train_end]['Target_binary']
            X_test = all_data.iloc[train_end:test_end][self.feature_cols]
            y_test = all_data.iloc[train_end:test_end]['Target_binary']

            # Train optimized model
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=242, max_depth=15,
                    min_samples_split=16, min_samples_leaf=6,
                    max_features=None, bootstrap=True,
                    class_weight='balanced', random_state=42
                )
            elif model_type == 'xgboost':
                model = xgb.XGBClassifier(
                    n_estimators=97, max_depth=5,
                    learning_rate=0.0158, subsample=0.656,
                    colsample_bytree=0.643, reg_alpha=0.224,
                    reg_lambda=13.67, random_state=42
                )

            # Fit and predict
            model.fit(X_train.fillna(0), y_train.fillna(0))
            y_pred = model.predict(X_test.fillna(0))

            # Store results
            accuracy = accuracy_score(y_test, y_pred)
            results.append({
                'window': i + 1,
                'test_period': f"{X_test.index[0].date()} to {X_test.index[-1].date()}",
                'accuracy': accuracy,
                'n_samples': len(X_test)
            })

            all_predictions.extend(y_pred)
            all_actual.extend(y_test)

        # Calculate overall statistics
        overall_accuracy = accuracy_score(all_actual, all_predictions)
        accuracies = [r['accuracy'] for r in results]

        return {
            'overall_accuracy': overall_accuracy,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'total_samples': len(all_predictions),
            'n_windows': len(results),
            'window_results': results
        }
    def walk_forward_validation(self, model_type='xgboost', window_size=252):
        """Implement walk-forward validation to test model robustness"""
        logging.info(f"Running walk-forward validation for {model_type}...")

        # Combine train and validation for walk-forward
        combined_X = pd.concat([self.X_train, self.X_val])
        combined_y = pd.concat([self.y_train, self.y_val])

        # Walk-forward parameters
        min_train_size = 500  # Minimum training samples
        step_size = 30  # Retrain every 30 days (monthly)

        results = []

        for i in range(min_train_size, len(combined_X) - step_size, step_size):
            # Split data
            train_X = combined_X.iloc[:i]
            train_y = combined_y.iloc[:i]
            test_X = combined_X.iloc[i:i + step_size]
            test_y = combined_y.iloc[i:i + step_size]

            if len(test_X) == 0:
                continue

            # Train model
            if model_type == 'xgboost':
                model = xgb.XGBClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.05,
                    subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=5, reg_lambda=5,
                    random_state=Config.RANDOM_STATE
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=5,
                    min_samples_split=10, class_weight='balanced',
                    random_state=Config.RANDOM_STATE
                )

            model.fit(train_X, train_y)

            # Predict and evaluate
            pred = model.predict(test_X)
            acc = accuracy_score(test_y, pred)

            results.append({
                'train_end': train_X.index[-1],
                'test_start': test_X.index[0],
                'test_end': test_X.index[-1],
                'accuracy': acc,
                'n_train': len(train_X),
                'n_test': len(test_X)
            })

            logging.info(f"Period {test_X.index[0]} to {test_X.index[-1]}: {acc:.4f}")

        # Summary statistics
        accuracies = [r['accuracy'] for r in results]
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        logging.info(f"Walk-forward validation results:")
        logging.info(f"  Average accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        logging.info(f"  Min accuracy: {min(accuracies):.4f}")
        logging.info(f"  Max accuracy: {max(accuracies):.4f}")
        logging.info(f"  Periods tested: {len(results)}")

        return results, avg_acc, std_acc

    def train_random_forest(self, **params):
        """Train Random Forest model"""
        logging.info("Training Random Forest model...")

        # Default parameters
        rf_params = {
            'n_estimators': params.get('n_estimators', 200),
            'max_depth': params.get('max_depth', 10),
            'min_samples_split': params.get('min_samples_split', 5),
            'min_samples_leaf': params.get('min_samples_leaf', 2),
            'class_weight': 'balanced',
            'random_state': Config.RANDOM_STATE,
            'n_jobs': -1
        }

        # Train model
        start_time = time.time()
        model = RandomForestClassifier(**rf_params)
        model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        # Evaluate
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        test_pred = model.predict(self.X_test)

        train_acc = accuracy_score(self.y_train, train_pred)
        val_acc = accuracy_score(self.y_val, val_pred)
        test_acc = accuracy_score(self.y_test, test_pred)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Store results
        self.results['random_forest'] = {
            'model': model,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'training_time': training_time,
            'feature_importance': feature_importance,
            'predictions': {
                'train': train_pred,
                'val': val_pred,
                'test': test_pred
            }
        }

        # Save model
        joblib.dump(model, self.models_dir / "random_forest.pkl")
        feature_importance.to_csv(self.models_dir / "rf_feature_importance.csv", index=False)

        logging.info(f"Random Forest - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        logging.info(f"Training time: {training_time:.2f} seconds")

        return val_acc

    def train_xgboost(self, **params):
        """Train XGBoost model with regularization"""
        logging.info("Training XGBoost model with regularization...")

        # Heavily regularized XGBoost parameters
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            # Reduced complexity parameters
            'n_estimators': params.get('n_estimators', 50),  # Much fewer trees
            'max_depth': params.get('max_depth', 3),  # Shallow trees
            'learning_rate': params.get('learning_rate', 0.05),  # Slow learning
            # Strong regularization
            'subsample': params.get('subsample', 0.7),  # Sample 70% of data
            'colsample_bytree': params.get('colsample_bytree', 0.7),  # Sample 70% of features
            'reg_alpha': params.get('reg_alpha', 5),  # L1 regularization
            'reg_lambda': params.get('reg_lambda', 5),  # L2 regularization
            'min_child_weight': params.get('min_child_weight', 3),  # Minimum samples per leaf
            'gamma': params.get('gamma', 1),  # Minimum split loss
            'random_state': Config.RANDOM_STATE,
            'n_jobs': -1,
            # Add early stopping parameters directly to the model
            'early_stopping_rounds': 10,
            'eval_metric': 'auc'
        }

        # Train model
        start_time = time.time()
        model = xgb.XGBClassifier(**xgb_params)

        # Fit with eval_set but without callbacks parameter
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )
        training_time = time.time() - start_time

        # Rest of the method remains the same...
        # Evaluate
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        test_pred = model.predict(self.X_test)

        train_acc = accuracy_score(self.y_train, train_pred)
        val_acc = accuracy_score(self.y_val, val_pred)
        test_acc = accuracy_score(self.y_test, test_pred)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Store results
        self.results['xgboost'] = {
            'model': model,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'training_time': training_time,
            'feature_importance': feature_importance,
            'predictions': {
                'train': train_pred,
                'val': val_pred,
                'test': test_pred
            }
        }

        # Save model
        joblib.dump(model, self.models_dir / "xgboost.pkl")
        feature_importance.to_csv(self.models_dir / "xgb_feature_importance.csv", index=False)

        logging.info(f"XGBoost - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        logging.info(f"Training time: {training_time:.2f} seconds")

        return val_acc

    def train_lstm(self, **params):
        """Train LSTM model with GPU acceleration"""
        logging.info("Training LSTM model with GPU acceleration...")

        # LSTM parameters
        hidden_size = params.get('hidden_size', 64)
        num_layers = params.get('num_layers', 2)
        dropout = params.get('dropout', 0.3)
        learning_rate = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', 32)
        epochs = params.get('epochs', 50)
        sequence_length = params.get('sequence_length', 30)

        # Prepare sequence data
        train_features = self.X_train.values
        val_features = self.X_val.values
        test_features = self.X_test.values

        train_targets = self.y_train.values
        val_targets = self.y_val.values
        test_targets = self.y_test.values

        # Create datasets
        train_dataset = StockDataset(train_features, train_targets, sequence_length)
        val_dataset = StockDataset(val_features, val_targets, sequence_length)
        test_dataset = StockDataset(test_features, test_targets, sequence_length)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=2
        )

        # Initialize model
        input_size = len(self.feature_cols)
        model = LSTMPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)

        # Use BCEWithLogitsLoss instead of BCELoss (safe for mixed precision)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Mixed precision training (updated API)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

        # Training loop
        start_time = time.time()
        best_val_acc = 0
        patience_counter = 0
        max_patience = 10

        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                if scaler:
                    with autocast('cuda'):  # Updated autocast API
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

                train_loss += loss.item()
                # Use sigmoid for predictions since we removed it from model
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
                        with torch.amp.autocast('cuda'):
                            outputs = model(data).squeeze()
                            loss = criterion(outputs, targets)
                    else:
                        outputs = model(data).squeeze()
                        loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()

            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)

            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'model_params': {
                        'input_size': input_size,
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'dropout': dropout,
                        'sequence_length': sequence_length
                    }
                }, self.models_dir / "lstm_best.pt")
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logging.info(
                    f'Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

            if patience_counter >= max_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

        training_time = time.time() - start_time

        # Load best model for final evaluation
        checkpoint = torch.load(self.models_dir / "lstm_best.pt")
        model.load_state_dict(checkpoint['model_state_dict'])

        # Final evaluation
        def evaluate_model(data_loader):
            model.eval()
            correct = 0
            total = 0
            predictions = []

            with torch.no_grad():
                for data, targets in data_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = model(data).squeeze()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    predictions.extend(predicted.cpu().numpy())
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            return correct / total, np.array(predictions)

        train_acc, train_pred = evaluate_model(train_loader)
        val_acc, val_pred = evaluate_model(val_loader)
        test_acc, test_pred = evaluate_model(test_loader)

        # Store results
        self.results['lstm'] = {
            'model': model,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'training_time': training_time,
            'best_epoch': checkpoint['epoch'],
            'training_history': {
                'train_losses': train_losses,
                'val_accuracies': val_accuracies
            },
            'predictions': {
                'train': train_pred,
                'val': val_pred,
                'test': test_pred
            }
        }

        logging.info(f"LSTM - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        logging.info(f"Training time: {training_time:.2f} seconds")
        logging.info(f"Best epoch: {checkpoint['epoch']}")

        return val_acc

    def run_model_training_pipeline(self):
        """Run complete model training pipeline"""
        logging.info("=" * 60)
        logging.info("PHASE 3: MODEL TRAINING PIPELINE")
        logging.info("=" * 60)

        print(f"Device: {self.device}")
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Features: {len(self.feature_cols)}")

        # Train all models
        models_to_train = ['random_forest', 'xgboost', 'lstm']

        for model_name in models_to_train:
            print(f"\n" + "-" * 40)
            print(f"TRAINING {model_name.upper()}")
            print("-" * 40)

            try:
                if model_name == 'random_forest':
                    self.train_random_forest()
                elif model_name == 'xgboost':
                    self.train_xgboost()
                elif model_name == 'lstm':
                    self.train_lstm()

                print(f"✓ {model_name} training completed")

            except Exception as e:
                logging.error(f"Error training {model_name}: {str(e)}")
                print(f"❌ {model_name} training failed: {str(e)}")

        # Compare results
        self.compare_models()

        # Save results summary
        self.save_results_summary()

        print(f"\n" + "=" * 60)
        print("PHASE 3 COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        return self.results

    def compare_models(self):
        """Compare model performance"""
        print(f"\n" + "=" * 50)
        print("MODEL COMPARISON RESULTS")
        print("=" * 50)

        if not self.results:
            print("No models were successfully trained.")
            return pd.DataFrame()

        comparison_data = []

        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train Acc': f"{results['train_accuracy']:.4f}",
                'Val Acc': f"{results['val_accuracy']:.4f}",
                'Test Acc': f"{results['test_accuracy']:.4f}",
                'Training Time': f"{results['training_time']:.1f}s"
            })

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Find best model
        if self.results:
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['val_accuracy'])
            best_val_acc = self.results[best_model]['val_accuracy']

            print(f"\n🏆 Best Model: {best_model.replace('_', ' ').title()}")
            print(f"🎯 Best Validation Accuracy: {best_val_acc:.4f}")

        return comparison_df

    def save_results_summary(self):
        """Save training results summary"""
        results_summary = {}

        for model_name, results in self.results.items():
            # Create serializable summary (excluding model objects)
            summary = {
                'train_accuracy': results['train_accuracy'],
                'val_accuracy': results['val_accuracy'],
                'test_accuracy': results['test_accuracy'],
                'training_time': results['training_time']
            }

            # Add model-specific info
            if model_name == 'lstm' and 'best_epoch' in results:
                summary['best_epoch'] = results['best_epoch']

            results_summary[model_name] = summary

        # Save to JSON
        with open(self.models_dir / "training_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2)

        logging.info(f"Results summary saved to: {self.models_dir}/training_results.json")


def main():
    """Main function to run model training"""
    trainer = ModelTrainer()
    results = trainer.run_model_training_pipeline()
    return results


if __name__ == "__main__":
    main()
