"""Unified dataset merging, splitting, and validation.

This module provides tools for:
- Merging multiple canonical JSONL datasets with deduplication
- Splitting unified datasets into train/dev/test splits with stratification
- Validating dataset integrity and schema consistency
- Computing and exporting dataset statistics
"""

from .merger import MergeStats, merge_datasets
from .splitter import SplitStats, split_dataset
from .stats import DatasetStats, compute_stats, export_stats_json, print_stats
from .validator import ValidationReport, validate_dataset

__all__ = [
    "merge_datasets",
    "MergeStats",
    "split_dataset",
    "SplitStats",
    "validate_dataset",
    "ValidationReport",
    "compute_stats",
    "print_stats",
    "export_stats_json",
    "DatasetStats",
]
