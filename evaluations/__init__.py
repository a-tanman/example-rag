"""
RAG Evaluation Framework

This module provides tools for evaluating the performance of RAG (Retrieval-Augmented Generation) systems.
It includes basic metrics, test datasets, and evaluation runners for assessing retrieval quality,
response accuracy, and system performance.
"""

__version__ = "1.0.0"
__author__ = "RAG Evaluation Framework"

from .basic_evaluator import BasicRAGEvaluator
from .metrics import EvaluationMetrics
from .test_datasets import TestDataset

__all__ = [
    "BasicRAGEvaluator",
    "EvaluationMetrics", 
    "TestDataset",
]
