#!/usr/bin/env python3
"""Check environment setup for IE SFT PoC.

Validates Python version, CUDA availability, and required package installation.
"""

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import platform


def check_python_version() -> bool:
    """Check Python version is 3.8+.

    Returns:
        True if Python version is acceptable, False otherwise
    """
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ERROR: Python 3.8+ required")
        return False

    print("  OK")
    return True


def check_cuda_availability() -> bool:
    """Check if CUDA is available.

    Returns:
        True if CUDA/GPU is detected, False otherwise
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print("CUDA/GPU: Available")
            # Extract GPU info from nvidia-smi output
            for line in result.stdout.split("\n"):
                if "NVIDIA-SMI" in line or "Driver Version" in line:
                    print(f"  {line.strip()}")
            print("  OK")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("CUDA/GPU: Not available (CPU mode)")
    print("  WARNING: Training on CPU will be very slow")
    return False


def check_package(package_name: str, import_name: str | None = None) -> bool:
    """Check if a Python package is installed.

    Args:
        package_name: Name as shown in pip (e.g., "transformers")
        import_name: Name for import if different (e.g., "transformers")

    Returns:
        True if package is installed, False otherwise
    """
    if import_name is None:
        import_name = package_name

    import_name = import_name.replace("-", "_")

    if find_spec(import_name) is not None:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{package_name}: {version}")
            print("  OK")
            return True
        except ImportError:
            pass

    print(f"{package_name}: Not installed")
    print(f"  ERROR: Install with: pip install {package_name}")
    return False


def check_llamafactory_cli() -> bool:
    """Check if llamafactory-cli command is available.

    Returns:
        True if llamafactory-cli is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["llamafactory-cli", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"llamafactory-cli: {version}")
            print("  OK")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("llamafactory-cli: Not available")
    print("  ERROR: Install with: pip install llamafactory")
    return False


def print_system_info() -> None:
    """Print system information."""
    print("\nSystem Information:")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python executable: {sys.executable}")
    print()


def main() -> int:
    """Run all environment checks.

    Returns:
        0 if all checks pass, 1 if any check fails
    """
    print("IE SFT PoC Environment Check")
    print("=" * 60)
    print()

    # System info
    print_system_info()

    # Checks
    all_ok = True

    print("Python:")
    print("-" * 60)
    if not check_python_version():
        all_ok = False
    print()

    print("CUDA/GPU:")
    print("-" * 60)
    check_cuda_availability()  # Warning but not critical
    print()

    print("Required Packages:")
    print("-" * 60)

    required_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("pydantic", "pydantic"),
        ("pyyaml", "yaml"),
    ]

    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_ok = False
        print()

    print("CLI Tools:")
    print("-" * 60)
    if not check_llamafactory_cli():
        all_ok = False
    print()

    # Summary
    print("=" * 60)
    if all_ok:
        print("All checks passed!")
        print("Your environment is ready for IE SFT training.")
        return 0
    else:
        print("Some checks failed!")
        print("Please install missing dependencies before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
