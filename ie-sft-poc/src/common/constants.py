"""Constants and configuration values for the IE SFT PoC.

Centralized storage of all project constants including supported task types,
languages, and other configuration values.
"""

# Task types
SUPPORTED_TASK_TYPES = ["kv", "entity", "relation"]

# Supported languages (ISO 639-1 codes)
SUPPORTED_LANGUAGES = ["ko", "en", "zh", "ja", "es", "fr", "de", "ru", "ar", "pt"]

# Data processing
DEFAULT_SEED = 42
DEFAULT_ENCODING = "utf-8"

# Training defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_EPOCHS = 3
DEFAULT_MAX_LENGTH = 512
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01

# Model defaults
DEFAULT_MODEL_NAME = "bert-base-multilingual-cased"
SUPPORTED_MODEL_TYPES = ["bert", "roberta", "electra", "xlm-roberta"]

# Evaluation metrics
EVAL_METRICS = ["precision", "recall", "f1", "accuracy"]

# File formats
SUPPORTED_DATA_FORMATS = ["jsonl", "json", "csv", "parquet"]
DEFAULT_DATA_FORMAT = "jsonl"

# Checkpoint configuration
CHECKPOINT_KEEP_BEST = 3
CHECKPOINT_SAVE_INTERVAL = 500  # steps

# Dataset splits
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_DEV_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1

# Entity types (commonly supported)
COMMON_ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "DATE",
    "TIME",
    "MONEY",
    "PERCENT",
    "FACILITY",
    "PRODUCT",
    "EVENT",
    "LANGUAGE",
    "LAW",
]

# Relation types (commonly supported)
COMMON_RELATION_TYPES = [
    "works_for",
    "located_in",
    "born_in",
    "date_of_birth",
    "headquarters",
    "country",
    "capital",
    "founded",
    "member_of",
    "owned_by",
]

# Validation constraints
MIN_TEXT_LENGTH = 1
MAX_TEXT_LENGTH = 10000
MIN_ENTITY_LENGTH = 1
MAX_ENTITY_LENGTH = 1000
MIN_RELATION_TYPES = 1
MAX_RELATION_TYPES = 100

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Cache
CACHE_VERSION = "1.0"
ENABLE_CACHE = True
CACHE_TIMEOUT = 3600 * 24  # 24 hours in seconds

if __name__ == "__main__":
    # Print all constants for documentation
    import inspect

    print("IE SFT PoC Constants")
    print("=" * 60)

    # Get all module-level constants (all caps variables)
    current_module = inspect.getmodule(inspect.currentframe())
    constants = {
        name: value
        for name, value in vars(current_module).items()
        if name.isupper() and not name.startswith("_")
    }

    for name, value in sorted(constants.items()):
        if isinstance(value, (list, dict)):
            print(f"{name}:")
            if isinstance(value, list):
                for item in value:
                    print(f"  - {item}")
            else:
                for key, val in value.items():
                    print(f"  {key}: {val}")
        else:
            print(f"{name}: {value}")
        print()
