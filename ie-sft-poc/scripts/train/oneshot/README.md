# One-Shot Training Launchers

Zero-to-trained scripts: one command runs the full data pipeline (download → normalize → unify → split → export) followed by SFT. Each phase is idempotent — if its expected output already exists it's skipped, so reruns are cheap and resumable.

## Scripts

| Script | Model | Default config |
| --- | --- | --- |
| `run_oneshot_qwen3.sh` | Qwen3-0.6B | `configs/sft/qwen3_lora_sft.yaml` (LoRA) / `configs/sft/qwen3_full_sft_ds.yaml` (full) |
| `run_oneshot_qwen35.sh` | Qwen3.5 | `configs/sft/qwen3_5_lora_sft.yaml` (LoRA) / `configs/sft/qwen3_5_full_sft_ds.yaml` (full) |
| `run_oneshot_olmo3.sh` | OLMo3 PoC | `configs/sft/olmo3_poc_sft.yaml` |
| `_common.sh` | — | Shared helpers: `run_data_pipeline`, `run_train`, path constants. Sourced by the launchers. |

For the full 4-stage OLMo3-style recipe (mid-train → SFT → DPO → RLVR), use [`../olmo3_style/`](../olmo3_style/) instead.

## Usage

```bash
# Qwen3 LoRA (default)
bash scripts/train/oneshot/run_oneshot_qwen3.sh

# Qwen3 full SFT
MODE=full bash scripts/train/oneshot/run_oneshot_qwen3.sh

# Qwen3.5 LoRA
bash scripts/train/oneshot/run_oneshot_qwen35.sh

# Override config entirely
CONFIG=configs/sft/my_custom.yaml bash scripts/train/oneshot/run_oneshot_qwen3.sh
```

## What the data pipeline does

Phases run in order; each is skipped if its expected artifact exists:

1. **Download** → `data/raw/instructie/` (via `scripts/download/download_instructie.py`)
2. **Normalize** → `data/processed/canonical/instructie.jsonl`
3. **Unify** → `data/processed/unified.jsonl`
4. **Split** → `data/processed/splits/{train,dev,test}.jsonl`
5. **Export** → `data/processed/llamafactory/{train,dev,test}.jsonl` + `dataset_info.json` (registered as `ie_sft_unified`)

To force a phase to re-run, delete its output file(s) and rerun the script.

## Env vars

| Var | Purpose | Default |
| --- | --- | --- |
| `MODE` | `lora` or `full` — selects the default config. Ignored if `CONFIG` is set. | `lora` |
| `CONFIG` | Absolute/relative path to a LLaMA-Factory YAML. | per-script default |
| `CUDA_VISIBLE_DEVICES` | Standard PyTorch knob; respected by `llamafactory-cli`. | unset |

## Requirements

- `llamafactory-cli` on PATH (install LLaMA-Factory into the active venv).
- `HF_TOKEN` in `.env` if the raw InstructIE download needs auth.
- Disk space for raw + canonical + unified + splits + llamafactory copies (~GB-range for InstructIE).

## Troubleshooting

- **Pipeline repeats a phase you expected to skip** — check the artifact actually exists and is non-empty (`wc -l data/processed/llamafactory/train.jsonl`). The export step specifically re-runs if `train.jsonl` is zero bytes.
- **Training starts before you wanted** — run with only the data pipeline by commenting out `run_train` at the bottom of the launcher, or use the individual scripts in `scripts/download/`, `scripts/preprocess/`, `scripts/export/` directly.
- **`llamafactory-cli not found`** — `pip install -e LLaMA-Factory` into the same venv.

## See also

- Repo root [`README.md`](../../../README.md) — full pipeline documentation, troubleshooting, and config reference.
- [`../olmo3_style/README.md`](../olmo3_style/README.md) — 4-stage OLMo3-style recipe.
