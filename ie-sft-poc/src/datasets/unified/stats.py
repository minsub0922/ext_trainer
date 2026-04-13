"""Compute and export dataset statistics."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.io import read_jsonl, write_json
from src.common.logging_utils import get_logger
from src.common.schema import CanonicalIERecord

logger = get_logger(__name__)


@dataclass
class DatasetStats:
    """Comprehensive dataset statistics.

    Attributes:
        total: Total number of records
        by_lang: Record count by language code
        by_source: Record count by source
        by_task_type: Record count by task type
        by_split: Record count by split (train/dev/test)
        avg_text_length: Average text length in characters
        min_text_length: Minimum text length
        max_text_length: Maximum text length
        schema_coverage: Coverage statistics for schema elements
    """

    total: int = 0
    by_lang: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    by_task_type: dict[str, int] = field(default_factory=dict)
    by_split: dict[str, int] = field(default_factory=dict)
    avg_text_length: float = 0.0
    min_text_length: int = 0
    max_text_length: int = 0
    schema_coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total": self.total,
            "by_lang": self.by_lang,
            "by_source": self.by_source,
            "by_task_type": self.by_task_type,
            "by_split": self.by_split,
            "avg_text_length": round(self.avg_text_length, 2),
            "min_text_length": self.min_text_length,
            "max_text_length": self.max_text_length,
            "schema_coverage": self.schema_coverage,
        }

    def __str__(self) -> str:
        """Return human-readable summary."""
        lines = [
            "Dataset Statistics:",
            f"  Total records: {self.total}",
        ]

        if self.by_lang:
            lines.append("  By language:")
            for lang, count in sorted(self.by_lang.items()):
                pct = count / self.total * 100 if self.total > 0 else 0
                lines.append(f"    {lang}: {count} ({pct:.1f}%)")

        if self.by_source:
            lines.append("  By source:")
            for source, count in sorted(self.by_source.items()):
                pct = count / self.total * 100 if self.total > 0 else 0
                lines.append(f"    {source}: {count} ({pct:.1f}%)")

        if self.by_task_type:
            lines.append("  By task type:")
            for task_type, count in sorted(self.by_task_type.items()):
                pct = count / self.total * 100 if self.total > 0 else 0
                lines.append(f"    {task_type}: {count} ({pct:.1f}%)")

        if self.by_split:
            lines.append("  By split:")
            for split, count in sorted(self.by_split.items()):
                pct = count / self.total * 100 if self.total > 0 else 0
                lines.append(f"    {split}: {count} ({pct:.1f}%)")

        lines.append("  Text statistics:")
        lines.append(f"    Average length: {self.avg_text_length:.1f} chars")
        lines.append(f"    Min length: {self.min_text_length} chars")
        lines.append(f"    Max length: {self.max_text_length} chars")

        if self.schema_coverage:
            lines.append("  Schema coverage:")
            for task_type, coverage in sorted(self.schema_coverage.items()):
                if isinstance(coverage, dict):
                    lines.append(f"    {task_type}:")
                    for item, count in coverage.items():
                        lines.append(f"      {item}: {count}")
                else:
                    lines.append(f"    {task_type}: {coverage}")

        return "\n".join(lines)


def compute_stats(
    path_or_records: Path | str | list[dict[str, Any]] | list[CanonicalIERecord],
) -> DatasetStats:
    """Compute comprehensive dataset statistics.

    Args:
        path_or_records: Either a path to a JSONL file or a list of records

    Returns:
        DatasetStats with computed statistics
    """
    stats = DatasetStats()

    # Load records
    if isinstance(path_or_records, (str, Path)):
        path = Path(path_or_records)
        logger.info(f"Computing stats for {path}")
        records = []
        for record_data in read_jsonl(path):
            try:
                record = CanonicalIERecord.from_dict(record_data)
                records.append(record)
            except Exception as e:
                logger.warning(f"Skipping invalid record: {e}")
                continue
    else:
        records = path_or_records
        # Convert dicts to CanonicalIERecord if needed
        converted = []
        for item in records:
            if isinstance(item, dict):
                try:
                    converted.append(CanonicalIERecord.from_dict(item))
                except Exception as e:
                    logger.warning(f"Skipping invalid record: {e}")
            else:
                converted.append(item)
        records = converted

    if not records:
        logger.warning("No valid records found")
        return stats

    stats.total = len(records)

    # Compute statistics
    text_lengths = []
    task_type_examples = defaultdict(set)
    schema_items = {
        "kv": defaultdict(int),
        "entity": defaultdict(int),
        "relation": defaultdict(int),
    }

    for record in records:
        # Language stats
        lang = record.lang or "unknown"
        stats.by_lang[lang] = stats.by_lang.get(lang, 0) + 1

        # Source stats
        source = record.source or "unknown"
        stats.by_source[source] = stats.by_source.get(source, 0) + 1

        # Task type stats
        for task_type in record.task_types:
            stats.by_task_type[task_type] = stats.by_task_type.get(task_type, 0) + 1

        # Split stats
        split = record.meta.split or "unknown"
        stats.by_split[split] = stats.by_split.get(split, 0) + 1

        # Text length stats
        text_lengths.append(len(record.text))

        # Schema coverage
        for key in record.answer.kv:
            schema_items["kv"][key] += 1

        for entity in record.answer.entity:
            schema_items["entity"][entity.type] += 1

        for relation in record.answer.relation:
            schema_items["relation"][relation.relation] += 1

    # Compute text length stats
    if text_lengths:
        stats.avg_text_length = sum(text_lengths) / len(text_lengths)
        stats.min_text_length = min(text_lengths)
        stats.max_text_length = max(text_lengths)

    # Build schema coverage dict
    for task_type in ["kv", "entity", "relation"]:
        if schema_items[task_type]:
            stats.schema_coverage[task_type] = dict(schema_items[task_type])

    logger.info(f"Computed stats for {stats.total} records")

    return stats


def print_stats(stats: DatasetStats) -> None:
    """Print dataset statistics to stdout.

    Args:
        stats: DatasetStats object to print
    """
    print(stats)


def export_stats_json(
    stats: DatasetStats,
    output_path: Path | str,
) -> None:
    """Export dataset statistics to JSON file.

    Args:
        stats: DatasetStats object to export
        output_path: Path to write JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_json(stats.to_dict(), output_path)
    logger.info(f"Wrote statistics to {output_path}")
