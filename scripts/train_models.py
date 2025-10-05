"""
Train ML Models for Optical Network Routing

Trains Random Forest, XGBoost, and Neural Network models on the generated dataset.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.routing.ml_routing import RandomForestRouter, XGBoostRouter, NeuralNetworkRouter


def load_dataset(data_path='data/processed/training_data.csv'):
    """Load and prepare dataset."""
    print("Loading dataset...")
    df = pd.read_csv(data_path)

    print(f"Dataset loaded: {len(df)} samples")
    print(f"Columns: {list(df.columns)}")

    return df


def prepare_features_labels(df):
    """Prepare features and labels for training."""
    print("\nPreparing features and labels...")

    # Load feature and label columns
    with open('data/processed/feature_columns.json', 'r') as f:
        feature_cols = json.load(f)

    with open('data/processed/label_columns.json', 'r') as f:
        label_cols = json.load(f)

    print(f"Features: {len(feature_cols)}")
    print(f"Labels: {len(label_cols)}")

    # Extract features and primary label (path_cost)
    X = df[feature_cols].values
    y = df['path_cost'].values

    # Handle any missing values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y, feature_cols


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split and scale the dataset."""
    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model."""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)

    rf_router = RandomForestRouter(
        n_estimators=100,
        max_depth=20,
        random_state=42
    )

    # Train model
    results = rf_router.train(X_train, y_train, X_test, y_test)

    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf_router, f)

    print(f"\n✓ Random Forest saved to models/random_forest.pkl")

    return results


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model."""
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)

    xgb_router = XGBoostRouter(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    )

    # Train model
    results = xgb_router.train(X_train, y_train, X_test, y_test)

    # Save model
    with open('models/xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_router, f)

    print(f"\n✓ XGBoost saved to models/xgboost.pkl")

    return results


def train_neural_network(X_train, y_train, X_test, y_test):
    """Train Neural Network model."""
    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK")
    print("="*60)

    nn_router = NeuralNetworkRouter(
        hidden_sizes=[64, 32],
        learning_rate=0.001,
        epochs=50,
        batch_size=32
    )

    # Train model
    results = nn_router.train(X_train, y_train, X_test, y_test)

    # Save model
    with open('models/neural_network.pkl', 'wb') as f:
        pickle.dump(nn_router, f)

    print(f"\n✓ Neural Network saved to models/neural_network.pkl")

    return results


def save_training_results(results_list):
    """Save training results to JSON."""
    print("\n" + "="*60)
    print("SAVING TRAINING RESULTS")
    print("="*60)

    results_dict = {
        'models': {},
        'summary': {}
    }

    for result in results_list:
        results_dict['models'][result.model_name] = {
            'train_mse': float(result.train_mse),
            'test_mse': float(result.test_mse),
            'train_r2': float(result.train_r2),
            'test_r2': float(result.test_r2),
            'train_mae': float(result.train_mae),
            'test_mae': float(result.test_mae),
            'training_time': float(result.training_time)
        }

    # Summary statistics
    results_dict['summary'] = {
        'best_test_r2': max(r.test_r2 for r in results_list),
        'best_test_mae': min(r.test_mae for r in results_list),
        'total_training_time': sum(r.training_time for r in results_list)
    }

    # Save to file
    os.makedirs('results', exist_ok=True)
    with open('results/training_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print("✓ Training results saved to results/training_results.json")

    # Print summary
    print("\nTRAINING SUMMARY:")
    print("-" * 60)
    for result in results_list:
        print(f"\n{result.model_name}:")
        print(f"  Test R²: {result.test_r2:.4f}")
        print(f"  Test MAE: {result.test_mae:,.2f}")
        print(f"  Training Time: {result.training_time:.2f}s")

    print(f"\nBest Test R²: {results_dict['summary']['best_test_r2']:.4f}")
    print(f"Best Test MAE: {results_dict['summary']['best_test_mae']:,.2f}")
    print(f"Total Training Time: {results_dict['summary']['total_training_time']:.2f}s")


def main():
    """Main training pipeline."""
    print("="*60)
    print("ML MODEL TRAINING PIPELINE")
    print("="*60)

    # Load dataset
    df = load_dataset('data/processed/training_data.csv')

    # Prepare features and labels
    X, y, feature_cols = prepare_features_labels(df)

    # Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

    # Save scaler
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("\n✓ Scaler saved to models/scaler.pkl")

    # Train all models
    results = []

    try:
        rf_results = train_random_forest(X_train, y_train, X_test, y_test)
        results.append(rf_results)
    except Exception as e:
        print(f"✗ Random Forest training failed: {e}")

    try:
        xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
        results.append(xgb_results)
    except Exception as e:
        print(f"✗ XGBoost training failed: {e}")

    try:
        nn_results = train_neural_network(X_train, y_train, X_test, y_test)
        results.append(nn_results)
    except Exception as e:
        print(f"✗ Neural Network training failed: {e}")

    # Save results
    if results:
        save_training_results(results)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n✓ {len(results)} models trained successfully")
    print("✓ Models saved to models/")
    print("✓ Results saved to results/training_results.json")


if __name__ == '__main__':
    main()
