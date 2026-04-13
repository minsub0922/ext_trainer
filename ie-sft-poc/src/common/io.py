"""I/O utilities for reading and writing various file formats.

Provides convenient functions for working with JSON, JSONL, and YAML files
with pathlib support and robust encoding handling.
"""

import json
from pathlib import Path
from typing import Any, Generator

import yaml


def read_json(path: Path | str, encoding: str = "utf-8") -> Any:
    """
    Read a JSON file.

    Args:
        path: Path to the JSON file
        encoding: File encoding (default: utf-8)

    Returns:
        Parsed JSON content

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def write_json(
    data: Any,
    path: Path | str,
    encoding: str = "utf-8",
    indent: int | None = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Write data to a JSON file.

    Args:
        data: Data to serialize
        path: Path where JSON will be written
        encoding: File encoding (default: utf-8)
        indent: JSON indentation level (None for compact, default: 2)
        ensure_ascii: Whether to escape non-ASCII characters (default: False)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def read_jsonl(path: Path | str, encoding: str = "utf-8") -> list[dict]:
    """
    Read a JSONL file (one JSON object per line).

    Args:
        path: Path to the JSONL file
        encoding: File encoding (default: utf-8)

    Returns:
        List of parsed JSON objects

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If any line is not valid JSON
    """
    path = Path(path)
    records = []

    with open(path, "r", encoding=encoding) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Error on line {line_num}: {e.msg}", e.doc, e.pos
                ) from e

    return records


def read_jsonl_iter(path: Path | str, encoding: str = "utf-8") -> Generator[dict, None, None]:
    """
    Read a JSONL file lazily, yielding one object at a time.

    Useful for processing large files without loading everything into memory.

    Args:
        path: Path to the JSONL file
        encoding: File encoding (default: utf-8)

    Yields:
        Parsed JSON objects

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If any line is not valid JSON
    """
    path = Path(path)

    with open(path, "r", encoding=encoding) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Error on line {line_num}: {e.msg}", e.doc, e.pos
                ) from e


def write_jsonl(
    records: list[dict] | Generator,
    path: Path | str,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
) -> int:
    """
    Write records to a JSONL file (one JSON object per line).

    Args:
        records: List or generator of dictionaries to write
        path: Path where JSONL will be written
        encoding: File encoding (default: utf-8)
        ensure_ascii: Whether to escape non-ASCII characters (default: False)

    Returns:
        Number of records written
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(path, "w", encoding=encoding) as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")
            count += 1

    return count


def read_yaml(path: Path | str, encoding: str = "utf-8") -> Any:
    """
    Read a YAML file.

    Args:
        path: Path to the YAML file
        encoding: File encoding (default: utf-8)

    Returns:
        Parsed YAML content

    Raises:
        FileNotFoundError: If file does not exist
        yaml.YAMLError: If file is not valid YAML
    """
    path = Path(path)

    with open(path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def write_yaml(
    data: Any,
    path: Path | str,
    encoding: str = "utf-8",
    default_flow_style: bool = False,
    sort_keys: bool = False,
) -> None:
    """
    Write data to a YAML file.

    Args:
        data: Data to serialize
        path: Path where YAML will be written
        encoding: File encoding (default: utf-8)
        default_flow_style: Use flow style for collections (default: False, uses block style)
        sort_keys: Sort dictionary keys (default: False)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding=encoding) as f:
        yaml.dump(
            data,
            f,
            default_flow_style=default_flow_style,
            sort_keys=sort_keys,
            allow_unicode=True,
        )


def read_text(path: Path | str, encoding: str = "utf-8") -> str:
    """
    Read a text file.

    Args:
        path: Path to the text file
        encoding: File encoding (default: utf-8)

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file does not exist
    """
    path = Path(path)
    return path.read_text(encoding=encoding)


def write_text(
    content: str,
    path: Path | str,
    encoding: str = "utf-8",
) -> None:
    """
    Write text content to a file.

    Args:
        content: Text content to write
        path: Path where text will be written
        encoding: File encoding (default: utf-8)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


if __name__ == "__main__":
    # Example usage
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test JSON
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        json_path = tmpdir / "test.json"
        write_json(test_data, json_path)
        loaded_json = read_json(json_path)
        print(f"JSON test: {loaded_json == test_data}")

        # Test JSONL
        jsonl_records = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"},
        ]
        jsonl_path = tmpdir / "test.jsonl"
        count = write_jsonl(jsonl_records, jsonl_path)
        loaded_jsonl = read_jsonl(jsonl_path)
        print(f"JSONL test: {loaded_jsonl == jsonl_records}, wrote {count} records")

        # Test YAML
        yaml_data = {"title": "Test", "items": [1, 2, 3], "config": {"debug": True}}
        yaml_path = tmpdir / "test.yaml"
        write_yaml(yaml_data, yaml_path)
        loaded_yaml = read_yaml(yaml_path)
        print(f"YAML test: {loaded_yaml == yaml_data}")

        # Test text
        text_content = "This is a test file.\nWith multiple lines."
        text_path = tmpdir / "test.txt"
        write_text(text_content, text_path)
        loaded_text = read_text(text_path)
        print(f"Text test: {loaded_text == text_content}")
