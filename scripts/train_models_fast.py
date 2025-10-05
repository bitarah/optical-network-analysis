"""
Fast ML Model Training - Tree-based models only

Trains Random Forest and XGBoost models (faster than Neural Networks).
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

from src.routing.ml_routing import RandomForestRouter, XGBoostRouter


def main():
    """Fast training pipeline."""
    print("="*60)
    print("FAST ML MODEL TRAINING (Tree-based models)")
    print("="*60)

    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv('data/processed/training_data.csv')
    print(f"✓ Dataset loaded: {len(df)} samples")

    # Load feature columns
    with open('data/processed/feature_columns.json', 'r') as f:
        feature_cols = json.load(f)

    # Prepare data
    print("\nPreparing data...")
    X = df[feature_cols].values
    y = df['path_cost'].values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    results = []

    # Train Random Forest
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    rf_router = RandomForestRouter(n_estimators=50, max_depth=15, random_state=42)
    rf_router.feature_columns = feature_cols  # Set feature columns for prediction
    rf_results = rf_router.train(X_train, y_train, X_test, y_test)
    results.append(rf_results)

    with open('models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf_router, f)
    print("✓ Saved to models/random_forest.pkl")

    # Train XGBoost
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    xgb_router = XGBoostRouter(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_router.feature_columns = feature_cols  # Set feature columns for prediction
    xgb_results = xgb_router.train(X_train, y_train, X_test, y_test)
    results.append(xgb_results)

    with open('models/xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_router, f)
    print("✓ Saved to models/xgboost.pkl")

    # Save results
    results_dict = {
        'models': {
            r.model_name: {
                'test_r2': float(r.test_r2),
                'test_mae': float(r.test_mae),
                'training_time': float(r.training_time)
            } for r in results
        }
    }

    os.makedirs('results', exist_ok=True)
    with open('results/training_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    for r in results:
        print(f"\n{r.model_name}:")
        print(f"  Test R²: {r.test_r2:.4f}")
        print(f"  Test MAE: {r.test_mae:,.2f}")
        print(f"  Time: {r.training_time:.2f}s")


if __name__ == '__main__':
    main()
