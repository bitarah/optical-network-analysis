#!/usr/bin/env python3
"""
Dataset Generation Script

Generates training dataset for ML-based optical network routing.
Creates synthetic networks and downloads real topologies, then generates
routing problems and extracts features/labels for ML training.
"""

import sys
import os

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from src.data_generation.dataset_builder import DatasetBuilder


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def generate_dataset():
    """Main function to generate the complete training dataset."""

    print_section("OPTICAL NETWORK ROUTING - DATASET GENERATION")

    # Configuration
    N_SYNTHETIC = 50  # Reduced from 100 for speed
    PROBLEMS_PER_NETWORK = 50  # Routing problems per network
    OUTPUT_DIR = 'data/processed'
    SEED = 42

    print("\nConfiguration:")
    print(f"  Synthetic networks: {N_SYNTHETIC}")
    print(f"  Problems per network: {PROBLEMS_PER_NETWORK}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Random seed: {SEED}")

    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('data/synthetic', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)

    # Initialize dataset builder
    print_section("INITIALIZING DATASET BUILDER")
    builder = DatasetBuilder(seed=SEED)
    print("Dataset builder initialized successfully!")

    # Generate synthetic networks
    print_section("GENERATING SYNTHETIC NETWORKS")
    print(f"Generating {N_SYNTHETIC} diverse network topologies...")

    try:
        networks = builder.network_generator.generate_diverse_dataset(
            n_networks=N_SYNTHETIC,
            output_dir='data/synthetic'
        )
        print(f"Successfully generated {len(networks)} synthetic networks")
    except Exception as e:
        print(f"Error generating synthetic networks: {e}")
        print("Continuing with fallback...")
        networks = []

    # Build dataset from synthetic networks
    print_section("BUILDING DATASET FROM SYNTHETIC NETWORKS")
    all_data = []
    successful_networks = 0
    failed_networks = 0

    for i, G in enumerate(tqdm(networks, desc="Processing networks")):
        try:
            df = builder.build_dataset_from_network(G, PROBLEMS_PER_NETWORK)
            df['network_id'] = i
            df['network_type'] = 'synthetic'
            all_data.append(df)
            successful_networks += 1
        except Exception as e:
            print(f"\n  Warning: Error processing synthetic network {i}: {e}")
            failed_networks += 1
            continue

    print(f"\nSynthetic network processing complete:")
    print(f"  Success: {successful_networks}")
    print(f"  Failed: {failed_networks}")

    # Load real networks
    print_section("LOADING REAL NETWORK TOPOLOGIES")
    real_networks_to_load = ['abilene', 'geant']
    real_networks = {}

    try:
        print(f"Attempting to download: {', '.join(real_networks_to_load)}")
        real_networks = builder.topology_parser.load_real_networks(
            real_networks_to_load,
            output_dir='data/raw'
        )
        print(f"Successfully loaded {len(real_networks)} real networks")
    except Exception as e:
        print(f"Warning: Could not download real networks: {e}")
        print("Falling back to simplified topologies...")

        # Create simplified fallback networks if download fails
        try:
            import networkx as nx

            # Simplified Abilene (US research network)
            G_abilene = nx.Graph()
            G_abilene.add_edges_from([
                (0, 1), (0, 2), (1, 3), (2, 3), (2, 4),
                (3, 5), (4, 5), (4, 6), (5, 7), (6, 7)
            ])
            for u, v in G_abilene.edges():
                G_abilene[u][v]['distance'] = 500
                G_abilene[u][v]['total_cost'] = 100
                G_abilene[u][v]['n_regenerators'] = 1
            G_abilene.graph['network_id'] = 'abilene'
            real_networks['abilene'] = G_abilene

            # Simplified GEANT (European research network)
            G_geant = nx.Graph()
            G_geant.add_edges_from([
                (0, 1), (0, 2), (1, 3), (1, 4), (2, 5),
                (3, 6), (4, 6), (5, 6), (6, 7), (7, 8)
            ])
            for u, v in G_geant.edges():
                G_geant[u][v]['distance'] = 600
                G_geant[u][v]['total_cost'] = 120
                G_geant[u][v]['n_regenerators'] = 1
            G_geant.graph['network_id'] = 'geant'
            real_networks['geant'] = G_geant

            print(f"Created {len(real_networks)} simplified fallback networks")
        except Exception as fallback_error:
            print(f"Error creating fallback networks: {fallback_error}")

    # Build dataset from real networks
    print_section("BUILDING DATASET FROM REAL NETWORKS")
    real_network_count = 0

    for name, G in real_networks.items():
        try:
            print(f"\nProcessing {name}...")
            df = builder.build_dataset_from_network(G, PROBLEMS_PER_NETWORK)
            df['network_name'] = name
            df['network_type'] = 'real'
            all_data.append(df)
            real_network_count += 1
            print(f"  Generated {len(df)} samples from {name}")
        except Exception as e:
            print(f"  Warning: Error processing {name}: {e}")
            continue

    print(f"\nReal network processing complete: {real_network_count} networks")

    # Combine all data
    print_section("COMBINING AND SAVING DATASET")

    if not all_data:
        print("ERROR: No data was generated! Check errors above.")
        return

    complete_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset: {len(complete_df)} samples")

    # Save complete dataset
    train_file = f"{OUTPUT_DIR}/training_data.csv"
    complete_df.to_csv(train_file, index=False)
    print(f"\nDataset saved to: {train_file}")

    # Extract and save feature/label columns
    feature_cols = [col for col in complete_df.columns
                   if col not in ['path', 'source', 'target', 'network_id',
                                 'network_name', 'network_type', 'path_cost',
                                 'path_distance', 'path_hops', 'path_regenerators',
                                 'path_found', 'path_osnr', 'path_viable']]

    label_cols = ['path_cost', 'path_distance', 'path_hops',
                 'path_regenerators', 'path_osnr']

    # Save feature columns
    feature_file = f"{OUTPUT_DIR}/feature_columns.json"
    with open(feature_file, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Feature columns saved to: {feature_file}")

    # Save label columns
    label_file = f"{OUTPUT_DIR}/label_columns.json"
    with open(label_file, 'w') as f:
        json.dump(label_cols, f, indent=2)
    print(f"Label columns saved to: {label_file}")

    # Print comprehensive statistics
    print_section("DATASET STATISTICS")

    print(f"\nOverall Statistics:")
    print(f"  Total samples: {len(complete_df)}")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Total labels: {len(label_cols)}")

    if 'network_id' in complete_df.columns:
        print(f"  Unique networks: {complete_df['network_id'].nunique()}")

    if 'network_type' in complete_df.columns:
        print(f"\nSamples by network type:")
        print(complete_df['network_type'].value_counts())

    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")

    print(f"\nLabel columns ({len(label_cols)}):")
    for col in label_cols:
        print(f"  - {col}")

    # Check for missing values
    print(f"\nMissing Values Check:")
    missing = complete_df[feature_cols + label_cols].isnull().sum()
    if missing.sum() == 0:
        print("  No missing values detected!")
    else:
        print("  Columns with missing values:")
        for col in missing[missing > 0].index:
            print(f"    {col}: {missing[col]} ({missing[col]/len(complete_df)*100:.2f}%)")

    # Label distributions
    print(f"\nLabel Distributions:")
    print(complete_df[label_cols].describe())

    # Show sample data
    print_section("SAMPLE DATA (First 5 Rows)")
    print("\nFeatures:")
    print(complete_df[feature_cols].head())
    print("\nLabels:")
    print(complete_df[label_cols].head())

    # Feature correlations with target
    print_section("FEATURE ANALYSIS")
    print("\nTop 10 features correlated with path_cost:")
    correlations = complete_df[feature_cols].corrwith(complete_df['path_cost']).abs()
    top_features = correlations.sort_values(ascending=False).head(10)
    for feat, corr in top_features.items():
        print(f"  {feat}: {corr:.4f}")

    # Success rate
    if 'path_found' in complete_df.columns:
        success_rate = complete_df['path_found'].mean() * 100
        print(f"\nPath Finding Success Rate: {success_rate:.2f}%")

    print_section("DATASET GENERATION COMPLETE")
    print(f"\nOutput files:")
    print(f"  1. Training data: {train_file}")
    print(f"  2. Feature columns: {feature_file}")
    print(f"  3. Label columns: {label_file}")
    print(f"\nDataset is ready for ML model training!")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        generate_dataset()
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
