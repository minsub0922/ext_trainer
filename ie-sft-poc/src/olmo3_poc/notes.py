"""Status tracking, TODO items, and known differences for OLMo3 expansion.

This module provides structured documentation of the OLMo3 PoC status,
outstanding work items, and known differences from other supported models.
Use this as a reference for understanding what has been done and what
remains to be completed.
"""

from typing import Any

# Current implementation status of OLMo3 support
OLMO3_STATUS: dict[str, Any] = {
    "overall_status": "experimental",
    "last_updated": "2024-03-15",
    "version": "0.1.0",
    "completed": [
        "Adapter abstraction layer design",
        "QwenAdapter reference implementation",
        "OLMoAdapter skeleton with TODO items",
        "Conversion utilities placeholder",
        "Documentation of unknowns and gaps",
    ],
    "in_progress": [
        "Tokenizer compatibility testing",
        "Training configuration benchmarking",
        "Template format validation",
    ],
    "blocked_on": [
        "Official OLMo3 model release (needed for real testing)",
        "Clear documentation of OLMo3 template format",
        "Access to OLMo3 reference implementations",
    ],
    "summary": (
        "OLMo3 support is in early PoC stage. The adapter framework is in place, "
        "and the extension points are documented, but actual training/inference "
        "testing awaits the public OLMo3 release. Most TODO items are placeholders "
        "that will be filled in once real OLMo3 models are available."
    ),
}

# High-level tasks needed to complete OLMo3 support
OLMO3_TODO_LIST: list[str] = [
    # Phase 1: Setup and validation
    "Wait for public OLMo3 model release",
    "Download and test OLMo3 tokenizer",
    "Verify tokenizer special tokens (BOS, EOS, PAD, etc.)",
    "Confirm chat template format and markers",
    "Test model loading and generation",

    # Phase 2: Integration
    "Update OLMoAdapter.tokenizer_quirks() with actual values",
    "Update OLMoAdapter.default_template with correct name",
    "Update OLMoAdapter.validate_environment() to test real OLMo3",
    "Implement actual logic in convert_prompt_for_olmo()",
    "Test prompt conversion with real OLMo3 model",

    # Phase 3: Training setup
    "Run baseline training to establish baseline metrics",
    "Benchmark learning rate sensitivity",
    "Test different warmup strategies",
    "Evaluate gradient accumulation impact",
    "Test mixed precision (bf16) stability",
    "Determine optimal batch size",

    # Phase 4: Evaluation
    "Measure fine-tuned model performance on IE tasks",
    "Compare quality against Qwen3 baseline",
    "Evaluate inference speed and throughput",
    "Test memory usage and identify optimization opportunities",
    "Document any model family-specific quirks found",

    # Phase 5: Documentation
    "Update README with OLMo3 setup instructions",
    "Create troubleshooting guide for OLMo3 issues",
    "Document performance characteristics",
    "Add OLMo3 examples to example datasets",
    "Record recommended settings for different hardware",
]

# Known differences between Qwen3 and OLMo3
OLMO3_KNOWN_DIFFERENCES: dict[str, Any] = {
    "tokenizer": {
        "implementation": {
            "qwen3": "HuggingFace standard tokenizer",
            "olmo3": "Likely custom tokenizer requiring trust_remote_code=True",
        },
        "special_tokens": {
            "status": "NOT YET CONFIRMED",
            "notes": [
                "OLMo3 may use different BOS/EOS/PAD token IDs",
                "Padding side preference is unknown",
                "May have custom tokens not in standard vocab",
            ],
        },
        "vocab_size": {
            "qwen3": "152064 (approximately)",
            "olmo3": "[TODO: Confirm actual vocab size]",
        },
    },
    "template": {
        "format": {
            "qwen3": "Uses <|im_start|>/<|im_end|> markers",
            "olmo3": "[TODO: Determine actual format]",
        },
        "role_markers": {
            "qwen3": "<|im_start|>user/assistant",
            "olmo3": "[TODO: Confirm role markers]",
        },
        "system_prompt": {
            "qwen3": "Integrated into template via system role",
            "olmo3": "[TODO: Determine system prompt handling]",
        },
    },
    "training": {
        "learning_rate": {
            "qwen3": "5e-5 is a good starting point",
            "olmo3": "[TODO: Benchmark optimal LR]",
        },
        "warmup": {
            "qwen3": "0.1 warmup ratio works well",
            "olmo3": "[TODO: Test different warmup strategies]",
        },
        "context_length": {
            "qwen3": "2048 tokens typical",
            "olmo3": "[TODO: Confirm max context]",
        },
        "bf16_stability": {
            "qwen3": "Stable with standard settings",
            "olmo3": "[TODO: Test mixed precision training]",
        },
    },
    "inference": {
        "temperature": {
            "status": "Likely similar to other models",
            "note": "[TODO: Verify temperature response]",
        },
        "generation_quality": {
            "status": "UNKNOWN",
            "note": "Will be evaluated once model is available",
        },
        "speed": {
            "status": "UNKNOWN",
            "note": "Depends on model architecture details",
        },
    },
    "environment": {
        "dependencies": {
            "note": "OLMo3 may require additional packages",
            "examples": [
                "Custom tokenizer modules",
                "Specific versions of transformers",
                "Additional AI2 packages",
            ],
        },
        "compatibility": {
            "cuda": "Likely compatible with standard PyTorch",
            "dtype": "Should support bf16, test fp32 if needed",
            "note": "[TODO: Verify with real OLMo3]",
        },
    },
}


def print_olmo3_status() -> None:
    """Print a formatted summary of OLMo3 support status.

    This function provides a user-friendly overview of:
    - Current implementation status
    - Completed work
    - In-progress items
    - Blockers and dependencies
    - High-level TODO list
    - Known differences requiring attention

    Useful for understanding where OLMo3 support stands and what to expect.
    """
    print("\n" + "=" * 80)
    print("OLMo3 SUPPORT STATUS")
    print("=" * 80)

    print(f"\nStatus: {OLMO3_STATUS['overall_status'].upper()}")
    print(f"Last Updated: {OLMO3_STATUS['last_updated']}")
    print(f"Version: {OLMO3_STATUS['version']}")

    print("\nSummary:")
    print(f"  {OLMO3_STATUS['summary']}")

    print("\nCompleted Items:")
    for item in OLMO3_STATUS["completed"]:
        print(f"  ✓ {item}")

    print("\nIn Progress:")
    for item in OLMO3_STATUS["in_progress"]:
        print(f"  ⚙ {item}")

    print("\nBlockers:")
    for blocker in OLMO3_STATUS["blocked_on"]:
        print(f"  ⛔ {blocker}")

    print("\nKey TODO Items (Priority Order):")
    for i, item in enumerate(OLMO3_TODO_LIST[:5], 1):
        print(f"  {i}. {item}")
    remaining = len(OLMO3_TODO_LIST) - 5
    if remaining > 0:
        print(f"  ... and {remaining} more items")

    print("\nKnown Differences from Qwen3:")
    print("  Tokenizer:")
    print(f"    - Qwen3: {OLMO3_KNOWN_DIFFERENCES['tokenizer']['implementation']['qwen3']}")
    print(f"    - OLMo3: {OLMO3_KNOWN_DIFFERENCES['tokenizer']['implementation']['olmo3']}")
    print("  Template:")
    print(f"    - Qwen3: {OLMO3_KNOWN_DIFFERENCES['template']['format']['qwen3']}")
    print(f"    - OLMo3: {OLMO3_KNOWN_DIFFERENCES['template']['format']['olmo3']}")

    print("\n" + "=" * 80)
    print("For more details, see:")
    print("  - src/olmo3_poc/adapter.py (implementation details)")
    print("  - src/olmo3_poc/conversion.py (format conversion notes)")
    print("  - This file (notes.py)")
    print("=" * 80 + "\n")
