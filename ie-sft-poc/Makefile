.PHONY: help install test test-cov lint env-check download-instructie preprocess unify split export-lf train-qwen3 train-qwen35 eval clean validate stats

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies from requirements.txt
	pip install -r requirements.txt

test:  ## Run all tests with pytest
	pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing --tb=short

lint:  ## Check Python syntax of key source files
	python -m py_compile src/common/schema.py
	python -m py_compile src/common/io.py
	python -m py_compile src/common/constants.py
	python -m py_compile src/datasets/instructie/parser.py
	python -m py_compile src/datasets/instructie/converter.py
	python -m py_compile src/datasets/unified/merger.py
	python -m py_compile src/datasets/unified/splitter.py
	@echo "All Python files have valid syntax"

env-check:  ## Check environment setup
	python scripts/utils/env_check.py

download-instructie:  ## Download InstructIE dataset from HuggingFace
	python scripts/download/download_instructie.py

preprocess:  ## Normalize InstructIE dataset to canonical format
	python scripts/preprocess/normalize_instructie.py

unify:  ## Merge all preprocessed datasets into unified format
	python scripts/preprocess/unify_ie_datasets.py --input data/interim/instructie/*.jsonl --output data/processed/unified.jsonl

split:  ## Split unified dataset into train/dev/test
	python scripts/export/export_train_dev_test.py --input data/processed/unified.jsonl

export-lf:  ## Export splits to LLaMA-Factory format
	python scripts/export/export_to_llamafactory.py --input data/processed/splits/

train-qwen3:  ## Train on Qwen3-0.6B using LLaMA-Factory
	bash scripts/train/run_sft_qwen3.sh

train-qwen35:  ## Train on Qwen3.5-0.8B using LLaMA-Factory
	bash scripts/train/run_sft_qwen35.sh

eval:  ## Run evaluation on trained model
	bash scripts/train/run_eval_qwen3.sh

clean:  ## Clean all generated files (data and outputs)
	rm -rf data/interim/* data/processed/* outputs/*

validate:  ## Validate canonical unified dataset
	python scripts/preprocess/validate_canonical_dataset.py --input data/processed/unified.jsonl

stats:  ## Print dataset statistics
	python scripts/utils/print_dataset_stats.py --input data/processed/unified.jsonl
