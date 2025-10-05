"""
Machine Learning-Based Routing

Implements Random Forest, XGBoost, and Neural Network approaches
for optical network route optimization.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import pickle
import time
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from routing.traditional_routing import RoutingResult, TraditionalRouter


@dataclass
class MLModelResults:
    """Container for ML model training results."""
    model_name: str
    train_mse: float
    test_mse: float
    train_r2: float
    test_r2: float
    train_mae: float
    test_mae: float
    training_time: float


class MLRouter:
    """Base class for ML-based routing."""

    def __init__(self):
        """Initialize ML router."""
        self.model = None
        self.feature_columns = None
        self.is_trained = False

    def extract_features_for_prediction(
        self,
        G: nx.Graph,
        source: int,
        target: int
    ) -> np.ndarray:
        """
        Extract features for a routing problem.

        Args:
            G: NetworkX graph
            source: Source node
            target: Target node

        Returns:
            Feature vector
        """
        features = {}

        # Network-level features
        features['n_nodes'] = G.number_of_nodes()
        features['n_edges'] = G.number_of_edges()
        features['avg_degree'] = np.mean([d for _, d in G.degree()])
        features['network_density'] = nx.density(G)

        # Node features
        features['source_degree'] = G.degree(source)
        features['target_degree'] = G.degree(target)

        # Geographic features
        pos = nx.get_node_attributes(G, 'pos')
        if source in pos and target in pos:
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            features['euclidean_distance'] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        else:
            features['euclidean_distance'] = 0

        # Shortest path length
        try:
            features['shortest_path_hops'] = nx.shortest_path_length(G, source, target)
        except:
            features['shortest_path_hops'] = -1

        # Simplified edge statistics (use neighborhood)
        neighbors_source = list(G.neighbors(source))
        neighbors_target = list(G.neighbors(target))

        if neighbors_source:
            source_edge_costs = [G[source][n]['total_cost'] for n in neighbors_source]
            features['avg_edge_cost'] = np.mean(source_edge_costs)
            features['max_edge_cost'] = np.max(source_edge_costs)
            features['min_edge_cost'] = np.min(source_edge_costs)
        else:
            features['avg_edge_cost'] = 0
            features['max_edge_cost'] = 0
            features['min_edge_cost'] = 0

        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G, k=min(20, G.number_of_nodes()))
            features['source_betweenness'] = betweenness.get(source, 0)
            features['target_betweenness'] = betweenness.get(target, 0)
        except:
            features['source_betweenness'] = 0
            features['target_betweenness'] = 0

        # Alternative paths estimate
        features['n_alternative_paths'] = len(neighbors_source) * len(neighbors_target)

        # Create feature vector in consistent order
        if self.feature_columns is None:
            self.feature_columns = sorted(features.keys())

        feature_vector = np.array([features.get(col, 0) for col in self.feature_columns])

        return feature_vector.reshape(1, -1)

    def route(
        self,
        G: nx.Graph,
        source: int,
        target: int
    ) -> RoutingResult:
        """
        Predict route cost and use ML-guided greedy path construction.

        Args:
            G: NetworkX graph
            source: Source node
            target: Target node

        Returns:
            RoutingResult object

        Raises:
            RuntimeError: If model is not trained
        """
        start_time = time.time()

        if not self.is_trained:
            raise RuntimeError(
                f"{self.model_name} model is not trained. "
                f"Call train() method or load a trained model first."
            )

        if self.model is None:
            raise RuntimeError(
                f"{self.model_name} model is None. "
                f"Model must be initialized before routing."
            )

        # Extract features
        features = self.extract_features_for_prediction(G, source, target)

        # Predict cost using ML model
        predicted_cost = float(self.model.predict(features)[0])

        # ML-guided greedy path construction
        path = self._construct_ml_path(G, source, target, predicted_cost)

        # Calculate actual path metrics
        if len(path) < 2:
            raise RuntimeError(
                f"ML routing failed to find path from {source} to {target}. "
                f"Network may be disconnected or model prediction is invalid."
            )

        actual_cost = sum(G[path[i]][path[i+1]]['total_cost'] for i in range(len(path)-1))
        total_distance = sum(G[path[i]][path[i+1]]['distance'] for i in range(len(path)-1))
        total_regenerators = sum(G[path[i]][path[i+1]]['n_regenerators'] for i in range(len(path)-1))

        computation_time = time.time() - start_time

        return RoutingResult(
            algorithm=f'{self.model_name} (ML)',
            path=path,
            cost=actual_cost,
            distance=total_distance,
            n_regenerators=total_regenerators,
            computation_time=computation_time,
            success=True,
            metrics={
                'predicted_cost': predicted_cost,
                'actual_cost': actual_cost,
                'prediction_error': abs(actual_cost - predicted_cost),
                'prediction_error_pct': abs(actual_cost - predicted_cost) / actual_cost * 100
            }
        )

    def _construct_ml_path(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        predicted_cost: float
    ) -> List[int]:
        """
        Construct path using ML-guided greedy selection.

        Uses the ML model to predict cost-to-target for each neighbor,
        then greedily selects the best next hop at each step.

        Args:
            G: Network graph
            source: Source node
            target: Target node
            predicted_cost: ML predicted total cost

        Returns:
            List of nodes forming the path
        """
        path = [source]
        current = source
        visited = {source}
        max_hops = G.number_of_nodes()  # Prevent infinite loops

        for _ in range(max_hops):
            if current == target:
                return path

            # Get unvisited neighbors
            neighbors = [n for n in G.neighbors(current) if n not in visited]

            if not neighbors:
                # Dead end - use shortest path from here
                try:
                    remaining_path = nx.shortest_path(G, current, target, weight='total_cost')
                    return path + remaining_path[1:]  # Skip current node
                except nx.NetworkXNoPath:
                    return []

            # Use ML model to predict cost from each neighbor to target
            neighbor_scores = []
            for neighbor in neighbors:
                # Extract features for neighbor -> target
                features = self.extract_features_for_prediction(G, neighbor, target)

                # Predict remaining cost
                predicted_remaining = float(self.model.predict(features)[0])

                # Add edge cost from current to neighbor
                edge_cost = G[current][neighbor]['total_cost']
                total_predicted_cost = edge_cost + predicted_remaining

                neighbor_scores.append((neighbor, total_predicted_cost))

            # Select neighbor with lowest predicted total cost
            best_neighbor = min(neighbor_scores, key=lambda x: x[1])[0]

            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor

        # If we've exhausted max hops, use shortest path
        if current != target:
            try:
                remaining_path = nx.shortest_path(G, current, target, weight='total_cost')
                return path + remaining_path[1:]
            except nx.NetworkXNoPath:
                return []

        return path


class RandomForestRouter(MLRouter):
    """Random Forest-based route cost prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 20,
        random_state: int = 42
    ):
        """
        Initialize Random Forest router.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.model_name = 'Random Forest'

    def train(self, X_train, y_train, X_test, y_test) -> MLModelResults:
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            MLModelResults with metrics
        """
        print(f"Training {self.model_name}...")
        start_time = time.time()

        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        self.is_trained = True

        print(f"  Training time: {training_time:.2f}s")
        print(f"  Test R²: {test_r2:.4f}, Test MAE: {test_mae:.2f}")

        return MLModelResults(
            model_name=self.model_name,
            train_mse=train_mse,
            test_mse=test_mse,
            train_r2=train_r2,
            test_r2=test_r2,
            train_mae=train_mae,
            test_mae=test_mae,
            training_time=training_time
        )

    # RandomForest uses inherited route() method from MLRouter

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained or self.feature_columns is None:
            return {}

        importances = self.model.feature_importances_
        return dict(zip(self.feature_columns, importances))


class XGBoostRouter(MLRouter):
    """XGBoost-based route optimization."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize XGBoost router.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random seed
        """
        super().__init__()
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            tree_method='hist'
        )
        self.model_name = 'XGBoost'

    def train(self, X_train, y_train, X_test, y_test) -> MLModelResults:
        """Train XGBoost model."""
        print(f"Training {self.model_name}...")
        start_time = time.time()

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        training_time = time.time() - start_time

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        self.is_trained = True

        print(f"  Training time: {training_time:.2f}s")
        print(f"  Test R²: {test_r2:.4f}, Test MAE: {test_mae:.2f}")

        return MLModelResults(
            model_name=self.model_name,
            train_mse=train_mse,
            test_mse=test_mse,
            train_r2=train_r2,
            test_r2=test_r2,
            train_mae=train_mae,
            test_mae=test_mae,
            training_time=training_time
        )

    # XGBoost uses inherited route() method from MLRouter
    # No need to override

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained or self.feature_columns is None:
            return {}

        importances = self.model.feature_importances_
        return dict(zip(self.feature_columns, importances))


class NeuralNetworkModel(nn.Module):
    """Neural Network for route cost prediction."""

    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32]):
        """
        Initialize neural network.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
        """
        super(NeuralNetworkModel, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class NeuralNetworkRouter(MLRouter):
    """Neural Network-based routing."""

    def __init__(
        self,
        hidden_sizes: List[int] = [64, 32],
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Initialize Neural Network router.

        Args:
            hidden_sizes: Hidden layer sizes
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
        """
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_name = 'Neural Network'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, X_train, y_train, X_test, y_test) -> MLModelResults:
        """Train Neural Network model."""
        print(f"Training {self.model_name} on {self.device}...")
        start_time = time.time()

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).reshape(-1, 1).to(self.device)

        # Initialize model
        input_size = X_train.shape[1]
        self.model = NeuralNetworkModel(input_size, self.hidden_sizes).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()

            # Mini-batch training
            indices = np.random.permutation(len(X_train_t))
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                batch_X = X_train_t[batch_idx]
                batch_y = y_train_t[batch_idx]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        training_time = time.time() - start_time

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            y_train_pred = self.model(X_train_t).cpu().numpy()
            y_test_pred = self.model(X_test_t).cpu().numpy()

        # Metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        self.is_trained = True

        print(f"  Training time: {training_time:.2f}s")
        print(f"  Test R²: {test_r2:.4f}, Test MAE: {test_mae:.2f}")

        return MLModelResults(
            model_name=self.model_name,
            train_mse=train_mse,
            test_mse=test_mse,
            train_r2=train_r2,
            test_r2=test_r2,
            train_mae=train_mae,
            test_mae=test_mae,
            training_time=training_time
        )

    def route(
        self,
        G: nx.Graph,
        source: int,
        target: int
    ) -> RoutingResult:
        """
        Route using Neural Network prediction.

        Args:
            G: NetworkX graph
            source: Source node
            target: Target node

        Returns:
            RoutingResult object

        Raises:
            RuntimeError: If model is not trained
        """
        start_time = time.time()

        if not self.is_trained:
            raise RuntimeError(
                f"{self.model_name} model is not trained. "
                f"Call train() method or load a trained model first."
            )

        if self.model is None:
            raise RuntimeError(
                f"{self.model_name} model is None. "
                f"Model must be initialized before routing."
            )

        # Extract features
        features = self.extract_features_for_prediction(G, source, target)

        # Predict using Neural Network
        self.model.eval()
        with torch.no_grad():
            features_t = torch.FloatTensor(features).to(self.device)
            predicted_cost = float(self.model(features_t).cpu().numpy()[0][0])

        # ML-guided path construction
        path = self._construct_ml_path(G, source, target, predicted_cost)

        if len(path) < 2:
            raise RuntimeError(
                f"Neural Network routing failed to find path from {source} to {target}. "
                f"Network may be disconnected or model prediction is invalid."
            )

        # Calculate actual metrics
        actual_cost = sum(G[path[i]][path[i+1]]['total_cost'] for i in range(len(path)-1))
        total_distance = sum(G[path[i]][path[i+1]]['distance'] for i in range(len(path)-1))
        total_regenerators = sum(G[path[i]][path[i+1]]['n_regenerators'] for i in range(len(path)-1))

        computation_time = time.time() - start_time

        return RoutingResult(
            algorithm=f'{self.model_name} (ML)',
            path=path,
            cost=actual_cost,
            distance=total_distance,
            n_regenerators=total_regenerators,
            computation_time=computation_time,
            success=True,
            metrics={
                'predicted_cost': predicted_cost,
                'actual_cost': actual_cost,
                'prediction_error': abs(actual_cost - predicted_cost),
                'prediction_error_pct': abs(actual_cost - predicted_cost) / actual_cost * 100
            }
        )


if __name__ == '__main__':
    print("ML routing algorithms ready!")
