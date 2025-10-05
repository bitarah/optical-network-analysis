"""
Evaluation Module

Provides comprehensive evaluation framework for comparing routing algorithms.
"""

from .evaluator import (
    AlgorithmEvaluator,
    AlgorithmMetrics,
    NetworkMetrics,
    generate_test_problems
)

__all__ = [
    'AlgorithmEvaluator',
    'AlgorithmMetrics',
    'NetworkMetrics',
    'generate_test_problems'
]
