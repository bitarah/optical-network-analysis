"""
Optical Signal Quality Calculations

Implements OSNR (Optical Signal-to-Noise Ratio) calculations,
signal degradation modeling, and regenerator placement logic.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict


class OpticalSignalAnalyzer:
    """Analyze and calculate optical signal quality metrics."""

    def __init__(self):
        """Initialize optical signal analyzer with standard parameters."""
        # Standard optical parameters
        self.fiber_attenuation_db_km = 0.25  # dB/km
        self.connector_loss_db = 0.5  # dB per connector
        self.splice_loss_db = 0.1  # dB per splice
        self.ase_noise_figure = 5.0  # dB (Amplified Spontaneous Emission)

        # Transmitter parameters
        self.tx_power_dbm = 0.0  # dBm (1 mW)

        # Receiver sensitivity
        self.rx_sensitivity_dbm = -28.0  # dBm
        self.osnr_threshold_db = 15.0  # dB (minimum required OSNR)

        # Regenerator parameters
        self.regenerator_spacing_km = 120  # km
        self.regenerator_osnr_improvement = 20.0  # dB

    def calculate_path_loss(
        self,
        distance_km: float,
        n_connectors: int = 2,
        n_splices: int = 0
    ) -> float:
        """
        Calculate total optical loss along a fiber path.

        Args:
            distance_km: Fiber length in kilometers
            n_connectors: Number of connectors
            n_splices: Number of fiber splices

        Returns:
            Total loss in dB
        """
        fiber_loss = distance_km * self.fiber_attenuation_db_km
        connector_loss = n_connectors * self.connector_loss_db
        splice_loss = n_splices * self.splice_loss_db

        total_loss = fiber_loss + connector_loss + splice_loss
        return round(total_loss, 3)

    def calculate_osnr(
        self,
        distance_km: float,
        n_amplifiers: int = 0
    ) -> float:
        """
        Calculate OSNR (Optical Signal-to-Noise Ratio) for a path.

        OSNR degrades with distance and number of optical amplifiers.

        Args:
            distance_km: Total path length
            n_amplifiers: Number of optical amplifiers

        Returns:
            OSNR in dB
        """
        # Base OSNR degradation with distance
        osnr = 30.0 - (distance_km / 100.0)  # Simple linear model

        # Each amplifier adds ASE noise
        if n_amplifiers > 0:
            ase_degradation = 10 * np.log10(n_amplifiers) + self.ase_noise_figure
            osnr -= ase_degradation

        return max(round(osnr, 2), 0.0)  # OSNR can't be negative

    def calculate_received_power(
        self,
        path_loss_db: float,
        tx_power_dbm: float = None
    ) -> float:
        """
        Calculate received optical power.

        Args:
            path_loss_db: Total path loss
            tx_power_dbm: Transmit power (if None, use default)

        Returns:
            Received power in dBm
        """
        if tx_power_dbm is None:
            tx_power_dbm = self.tx_power_dbm

        rx_power = tx_power_dbm - path_loss_db
        return round(rx_power, 2)

    def is_path_viable(
        self,
        distance_km: float,
        n_regenerators: int = 0
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if an optical path is viable based on signal quality.

        Args:
            distance_km: Total path length
            n_regenerators: Number of regenerators in path

        Returns:
            Tuple of (is_viable, metrics_dict)
        """
        # Calculate per-span metrics
        if n_regenerators == 0:
            span_length = distance_km
        else:
            span_length = distance_km / (n_regenerators + 1)

        # Calculate metrics
        path_loss = self.calculate_path_loss(span_length)
        rx_power = self.calculate_received_power(path_loss)
        osnr = self.calculate_osnr(span_length)

        # Check viability criteria
        power_ok = rx_power >= self.rx_sensitivity_dbm
        osnr_ok = osnr >= self.osnr_threshold_db

        is_viable = power_ok and osnr_ok

        metrics = {
            'distance_km': distance_km,
            'n_regenerators': n_regenerators,
            'span_length_km': round(span_length, 2),
            'path_loss_db': path_loss,
            'rx_power_dbm': rx_power,
            'osnr_db': osnr,
            'power_margin_db': round(rx_power - self.rx_sensitivity_dbm, 2),
            'osnr_margin_db': round(osnr - self.osnr_threshold_db, 2),
            'is_viable': is_viable
        }

        return is_viable, metrics

    def calculate_minimum_regenerators(self, distance_km: float) -> int:
        """
        Calculate minimum number of regenerators needed for a path.

        Args:
            distance_km: Total path length

        Returns:
            Minimum number of regenerators
        """
        if distance_km <= self.regenerator_spacing_km:
            return 0

        # Binary search for minimum regenerators
        min_regen = int(np.ceil(distance_km / self.regenerator_spacing_km)) - 1
        max_regen = int(np.ceil(distance_km / 50))  # Conservative upper bound

        for n_regen in range(min_regen, max_regen + 1):
            viable, _ = self.is_path_viable(distance_km, n_regen)
            if viable:
                return n_regen

        return max_regen  # Return conservative estimate

    def analyze_network_path(
        self,
        G: nx.Graph,
        path: List[int]
    ) -> Dict[str, any]:
        """
        Analyze signal quality for a complete network path.

        Args:
            G: NetworkX graph with optical parameters
            path: List of node IDs representing the route

        Returns:
            Dictionary with comprehensive path metrics
        """
        if len(path) < 2:
            return {'error': 'Path must have at least 2 nodes'}

        # Calculate cumulative metrics
        total_distance = 0.0
        total_cost = 0.0
        total_regenerators = 0
        min_osnr = float('inf')
        edge_details = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]

            if not G.has_edge(u, v):
                return {'error': f'No edge between {u} and {v}'}

            edge = G[u][v]
            total_distance += edge['distance']
            total_cost += edge['total_cost']
            total_regenerators += edge['n_regenerators']

            # Calculate OSNR for this segment
            osnr = self.calculate_osnr(edge['distance'], edge['n_regenerators'])
            min_osnr = min(min_osnr, osnr)

            edge_details.append({
                'from': u,
                'to': v,
                'distance': edge['distance'],
                'cost': edge['total_cost'],
                'osnr': osnr
            })

        # Overall path viability
        viable, metrics = self.is_path_viable(total_distance, total_regenerators)

        result = {
            'path': path,
            'n_hops': len(path) - 1,
            'total_distance_km': round(total_distance, 2),
            'total_cost': round(total_cost, 2),
            'total_regenerators': total_regenerators,
            'min_osnr_db': round(min_osnr, 2),
            'avg_osnr_db': round(np.mean([e['osnr'] for e in edge_details]), 2),
            'is_viable': viable,
            'power_margin_db': metrics['power_margin_db'],
            'osnr_margin_db': metrics['osnr_margin_db'],
            'edge_details': edge_details
        }

        return result

    def calculate_q_factor(self, osnr_db: float) -> float:
        """
        Calculate Q-factor from OSNR (quality metric).

        Q-factor relates OSNR to bit error rate (BER).

        Args:
            osnr_db: OSNR in dB

        Returns:
            Q-factor in dB
        """
        # Simplified relationship: Q(dB) ≈ OSNR(dB) - 3
        q_factor = osnr_db - 3.0
        return round(max(q_factor, 0.0), 2)

    def estimate_ber(self, osnr_db: float) -> float:
        """
        Estimate Bit Error Rate (BER) from OSNR.

        Args:
            osnr_db: OSNR in dB

        Returns:
            BER (probability)
        """
        q_factor_db = self.calculate_q_factor(osnr_db)
        q_factor_linear = 10 ** (q_factor_db / 20.0)

        # BER approximation using Q-factor
        # BER ≈ 0.5 * erfc(Q / sqrt(2))
        from scipy.special import erfc
        ber = 0.5 * erfc(q_factor_linear / np.sqrt(2))

        return max(ber, 1e-15)  # Avoid zero BER


if __name__ == '__main__':
    # Example usage
    analyzer = OpticalSignalAnalyzer()

    print("Optical Signal Quality Analyzer")
    print("=" * 50)

    # Test various distances
    distances = [50, 100, 200, 500]

    for dist in distances:
        min_regen = analyzer.calculate_minimum_regenerators(dist)
        viable, metrics = analyzer.is_path_viable(dist, min_regen)

        print(f"\nDistance: {dist} km")
        print(f"  Regenerators needed: {min_regen}")
        print(f"  OSNR: {metrics['osnr_db']} dB")
        print(f"  Viable: {viable}")

    print("\nSignal quality analyzer ready!")
