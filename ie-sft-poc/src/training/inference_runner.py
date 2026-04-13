"""Python wrapper for LLaMA-Factory inference and batch inference.

Provides tools for single inference and batch processing using
LLaMA-Factory CLI with optional LoRA adapter merging.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.common.io import read_jsonl, write_jsonl
from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def _check_llamafactory_installed() -> bool:
    """Check if llamafactory-cli is available in the current environment.

    Returns:
        True if llamafactory-cli is installed, False otherwise
    """
    try:
        result = subprocess.run(
            ["llamafactory-cli", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_inference(
    model_path: str | Path,
    input_text: str,
    template: str = "default",
    adapter_path: Optional[str | Path] = None,
    max_length: int = 2048,
) -> str:
    """Run single inference with optional LoRA adapter.

    Args:
        model_path: Path to model or HuggingFace model identifier
        input_text: Input prompt text
        template: Chat template name
        adapter_path: Optional path to LoRA adapter weights
        max_length: Maximum output length

    Returns:
        Generated text from the model

    Raises:
        ValueError: If llamafactory-cli is not installed
        subprocess.CalledProcessError: If inference fails
    """
    if not _check_llamafactory_installed():
        raise ValueError(
            "llamafactory-cli not found. "
            "Install with: pip install llamafactory"
        )

    logger.info(f"Running inference with model: {model_path}")
    if adapter_path:
        logger.info(f"  Using LoRA adapter: {adapter_path}")

    # Build inference command
    cmd = [
        "llamafactory-cli",
        "api",
        "--model_name_or_path", str(model_path),
        "--template", template,
        "--max_length", str(max_length),
    ]

    if adapter_path:
        cmd.extend(["--adapter_name_or_path", str(adapter_path)])

    logger.debug(f"Inference command: {' '.join(cmd)}")

    try:
        # Note: This is a simplified example. Actual implementation would use
        # the LLaMA-Factory Python API or a proper inference server setup
        logger.warning(
            "Note: This is a placeholder implementation. "
            "Use LLaMA-Factory API or CLI directly for actual inference."
        )
        return f"[Inference output for: {input_text[:50]}...]"

    except subprocess.CalledProcessError as e:
        logger.error(f"Inference failed: {e}")
        raise


def run_batch_inference(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    template: str = "default",
    adapter_path: Optional[str | Path] = None,
    max_length: int = 2048,
    batch_size: int = 32,
) -> int:
    """Run batch inference on JSONL dataset.

    Reads input JSONL file, runs inference on each example,
    and writes results to output JSONL.

    Args:
        model_path: Path to model or HuggingFace model identifier
        input_path: Path to input JSONL file with "text" or "input" field
        output_path: Path to output JSONL file
        template: Chat template name
        adapter_path: Optional path to LoRA adapter weights
        max_length: Maximum output length
        batch_size: Number of samples to process at once

    Returns:
        Number of samples processed

    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If llamafactory-cli is not installed
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not _check_llamafactory_installed():
        raise ValueError(
            "llamafactory-cli not found. "
            "Install with: pip install llamafactory"
        )

    logger.info(f"Loading input data from: {input_path}")
    try:
        input_data = read_jsonl(input_path)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        raise

    logger.info(f"Processing {len(input_data)} samples")
    logger.info(f"Using model: {model_path}")
    if adapter_path:
        logger.info(f"Using LoRA adapter: {adapter_path}")

    # Process in batches
    output_data = []
    for i, sample in enumerate(input_data):
        # Get input text
        text = sample.get("text") or sample.get("input")
        if not text:
            logger.warning(f"Sample {i} missing 'text' or 'input' field")
            continue

        logger.debug(f"Processing sample {i+1}/{len(input_data)}")

        # Run inference
        try:
            output = run_inference(
                model_path=model_path,
                input_text=text,
                template=template,
                adapter_path=adapter_path,
                max_length=max_length,
            )

            output_sample = {
                **sample,
                "output": output,
            }
            output_data.append(output_sample)

        except Exception as e:
            logger.error(f"Inference failed for sample {i}: {e}")
            continue

    # Write output
    logger.info(f"Writing {len(output_data)} results to: {output_path}")
    write_jsonl(output_data, output_path)

    logger.info(f"Batch inference completed: {len(output_data)} processed")
    return len(output_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLaMA-Factory inference"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single inference
    single_parser = subparsers.add_parser(
        "single",
        help="Run single inference",
    )
    single_parser.add_argument("model_path", help="Model path or identifier")
    single_parser.add_argument("input_text", help="Input prompt")
    single_parser.add_argument("--adapter-path", help="LoRA adapter path")
    single_parser.add_argument("--template", default="default")
    single_parser.add_argument("--max-length", type=int, default=2048)

    # Batch inference
    batch_parser = subparsers.add_parser(
        "batch",
        help="Run batch inference",
    )
    batch_parser.add_argument("model_path", help="Model path or identifier")
    batch_parser.add_argument("input_path", help="Input JSONL path")
    batch_parser.add_argument("output_path", help="Output JSONL path")
    batch_parser.add_argument("--adapter-path", help="LoRA adapter path")
    batch_parser.add_argument("--template", default="default")
    batch_parser.add_argument("--max-length", type=int, default=2048)
    batch_parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    try:
        if args.command == "single":
            output = run_inference(
                model_path=args.model_path,
                input_text=args.input_text,
                template=args.template,
                adapter_path=args.adapter_path,
                max_length=args.max_length,
            )
            print(output)
        elif args.command == "batch":
            count = run_batch_inference(
                model_path=args.model_path,
                input_path=args.input_path,
                output_path=args.output_path,
                template=args.template,
                adapter_path=args.adapter_path,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
            print(f"Processed {count} samples")
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
