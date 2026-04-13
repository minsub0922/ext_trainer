"""Validate canonical dataset files for schema and content integrity."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.io import read_jsonl
from src.common.logging_utils import get_logger
from src.common.schema import CanonicalIERecord
from src.common.constants import SUPPORTED_TASK_TYPES

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Report from dataset validation.

    Attributes:
        path: Path to the validated file
        total: Total records checked
        valid: Number of valid records
        invalid: Number of invalid records
        errors: List of error messages
        warnings: List of warning messages
        issues_by_record: Dictionary mapping record ID to issues
    """

    path: str = ""
    total: int = 0
    valid: int = 0
    invalid: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    issues_by_record: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.invalid == 0 and len(self.errors) == 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path": self.path,
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "errors": self.errors,
            "warnings": self.warnings,
            "issues_by_record": self.issues_by_record,
        }

    def __str__(self) -> str:
        """Return human-readable summary."""
        lines = [
            "Validation Report:",
            f"  File: {self.path}",
            f"  Total records: {self.total}",
            f"  Valid: {self.valid}",
            f"  Invalid: {self.invalid}",
        ]

        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10
                lines.append(f"    - {error}")
            if len(self.errors) > 10:
                lines.append(f"    ... and {len(self.errors) - 10} more")

        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                lines.append(f"    - {warning}")
            if len(self.warnings) > 10:
                lines.append(f"    ... and {len(self.warnings) - 10} more")

        status = "PASS" if self.is_valid else "FAIL"
        lines.insert(0, f"Validation Status: {status}")

        return "\n".join(lines)


def validate_dataset(
    path: Path | str,
    strict: bool = False,
) -> ValidationReport:
    """Validate a canonical dataset file.

    Checks:
    - Schema validity (JSON structure)
    - Required fields (id, text)
    - Task type validity
    - Consistency between task_types and answer content

    Args:
        path: Path to JSONL file to validate
        strict: If True, treat warnings as errors

    Returns:
        ValidationReport with detailed validation results

    Raises:
        FileNotFoundError: If file does not exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Validating dataset: {path}")

    report = ValidationReport(path=str(path))

    try:
        for record_data in read_jsonl(path):
            report.total += 1
            record_issues = []

            # Validate schema and required fields
            try:
                record = CanonicalIERecord.from_dict(record_data)
            except Exception as e:
                report.invalid += 1
                record_issues.append(f"Schema validation failed: {str(e)}")
                report.errors.append(
                    f"Record {report.total}: Schema validation failed: {str(e)}"
                )
                report.issues_by_record[str(report.total)] = record_issues
                continue

            # Check required fields
            if not record.id:
                record_issues.append("Missing or empty id")
                report.invalid += 1

            if not record.text or not record.text.strip():
                record_issues.append("Missing or empty text")
                report.invalid += 1

            # Validate task types
            for task_type in record.task_types:
                if task_type not in SUPPORTED_TASK_TYPES:
                    record_issues.append(
                        f"Invalid task type: {task_type} "
                        f"(must be one of {SUPPORTED_TASK_TYPES})"
                    )
                    report.invalid += 1

            # Check consistency between task_types and answer content
            active_types = record.get_active_task_types()
            if record.task_types and active_types != record.task_types:
                warning = (
                    f"Inconsistent task_types: declared={record.task_types}, "
                    f"actual={active_types}"
                )
                record_issues.append(warning)
                report.warnings.append(f"Record {record.id}: {warning}")

            # Check for empty answers
            if not active_types:
                warning = "Record has no extraction results (empty answer)"
                record_issues.append(warning)
                report.warnings.append(f"Record {record.id}: {warning}")

            # Check text length constraints
            text_length = len(record.text)
            if text_length < 1:
                record_issues.append("Text is too short")
                report.invalid += 1
            elif text_length > 10000:
                record_issues.append(
                    f"Text is very long ({text_length} chars), may cause issues"
                )
                report.warnings.append(f"Record {record.id}: Text too long ({text_length} chars)")

            # Check entity annotations for validity
            for entity in record.answer.entity:
                if not entity.text:
                    record_issues.append(f"Entity has empty text: {entity}")
                    report.invalid += 1
                if not entity.type:
                    record_issues.append(f"Entity missing type: {entity}")
                    report.invalid += 1

            # Check relation annotations
            for relation in record.answer.relation:
                if not all([relation.head, relation.head_type, relation.relation, relation.tail, relation.tail_type]):
                    record_issues.append(f"Relation missing required fields: {relation}")
                    report.invalid += 1

            # Update report
            if not record_issues:
                report.valid += 1
            else:
                report.issues_by_record[record.id] = record_issues
                if strict:
                    report.errors.extend(
                        [f"Record {record.id}: {issue}" for issue in record_issues]
                    )

    except Exception as e:
        report.errors.append(f"Unexpected error during validation: {str(e)}")
        logger.error(f"Validation error: {e}")
        raise

    logger.info(
        f"Validation complete: {report.valid} valid, "
        f"{report.invalid} invalid, {len(report.warnings)} warnings"
    )

    return report
