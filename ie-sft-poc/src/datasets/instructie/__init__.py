"""InstructIE dataset module.

Provides tools for downloading, parsing, and converting the InstructIE dataset
from HuggingFace to the canonical IE record format.

The InstructIE dataset is a large-scale, multi-lingual Information Extraction dataset
covering entity and relation extraction tasks across 12 languages and multiple domains.

Usage:
    from src.datasets.instructie import InstructIEDownloader, parse_instructie_record
    from src.datasets.instructie.converter import convert_dataset
"""

from src.datasets.instructie.converter import convert_dataset, convert_file, convert_record
from src.datasets.instructie.downloader import InstructIEDownloader
from src.datasets.instructie.parser import parse_instructie_file, parse_instructie_record

__all__ = [
    "InstructIEDownloader",
    "parse_instructie_record",
    "parse_instructie_file",
    "convert_record",
    "convert_file",
    "convert_dataset",
]
