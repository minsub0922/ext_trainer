"""GRPO-lite RLVR trainer for IE tasks.

This is a light, readable reference implementation — NOT a drop-in
replacement for trl/verl/OpenRLHF. It exists so the PoC can demonstrate
the OLMo3-style RLVR loop end-to-end with ~500 lines and no extra deps
beyond transformers + peft + torch.

Core loop per step:
    1. Sample a batch of prompts from prompts_file.
    2. Generate K completions per prompt with policy π_θ.
    3. Score each completion with ie_metrics (verifiable, rule-based).
    4. Compute group-relative advantages (GRPO): (r - mean_g) / (std_g + eps).
    5. Policy gradient update with PPO-style clip + KL to reference model.
    6. Periodically save and log reward stats.

For serious runs swap this for trl.GRPOTrainer or verl. The config
schema (configs/olmo3_style/*/stage4_rlvr.yaml) is intentionally a
superset of trl.GRPOConfig's common fields to make migration easy.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclass
class RLVRConfig:
    # model
    model_path: str
    ref_model_path: str | None = None
    tokenizer_template: str = "qwen"
    dtype: str = "bf16"
    attn_impl: str = "flash_attention_2"
    gradient_checkpointing: bool = True

    # data
    prompts_file: str = "data/processed/olmo3_style/rlvr_prompts.jsonl"
    output_dir: str = "outputs/olmo3_style/rlvr"

    # sampling
    num_generations_per_prompt: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.9
    top_p: float = 0.95
    prompt_batch_size: int = 4

    # optimization
    learning_rate: float = 1e-6
    kl_coef: float = 0.04
    clip_range: float = 0.2
    num_epochs: int = 1
    total_steps: int = 500

    # reward
    reward_mode: str = "auto"              # auto | kv | entity | relation
    reward_normalize: str = "groupwise"    # groupwise | running_mean | none
    parse_failure_penalty: float = -0.1

    # logging
    logging_steps: int = 5
    save_steps: int = 100
    report_to: str = "tensorboard"
    seed: int = 42

    # algorithm: keep for config compat; only grpo implemented here.
    algorithm: str = "grpo"

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RLVRConfig":
        import yaml  # local import to keep the module importable without yaml
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        known = {k: v for k, v in raw.items() if k in fields}
        extra = {k: v for k, v in raw.items() if k not in fields}
        return cls(**known, extra=extra)


def _detect_attn_implementation(requested: str = "flash_attention_2") -> str:
    """Return ``"flash_attention_2"`` only if transformers confirms it works.

    Uses ``transformers.utils.is_flash_attn_2_available()`` — the **same**
    check that ``from_pretrained`` runs internally — so we never pass a
    value that will blow up at model-init time.
    """
    if requested not in ("flash_attention_2", "fa2", "auto"):
        return requested
    try:
        from transformers.utils import is_flash_attn_2_available

        if is_flash_attn_2_available():
            logger.info("Flash Attention 2 available (transformers check passed)")
            return "flash_attention_2"
        else:
            logger.info("Flash Attention 2 NOT available; using sdpa")
    except Exception as exc:
        logger.info("Could not check flash-attn availability (%s); using sdpa", exc)
    return "sdpa"


def _load_prompts(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _groupwise_advantages(rewards: list[float]) -> list[float]:
    """r - mean(group) / (std(group) + 1e-6)."""
    if not rewards:
        return []
    mu = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    sigma = math.sqrt(var)
    denom = sigma + 1e-6
    return [(r - mu) / denom for r in rewards]


def _score_completion(completion: str, gold: dict, task_type: str,
                       parse_failure_penalty: float) -> float:
    """Verifiable reward — defer to ie_metrics."""
    from ..ie_metrics import evaluate
    m = evaluate([completion], [gold], task_types=[task_type])
    if task_type == "kv" and m.kv is not None:
        r = m.kv.f1
    elif task_type == "entity" and m.entity is not None:
        r = m.entity.f1
    elif task_type == "relation" and m.relation is not None:
        r = m.relation.f1
    else:
        r = max(
            m.kv.f1 if m.kv else 0.0,
            m.entity.f1 if m.entity else 0.0,
            m.relation.f1 if m.relation else 0.0,
        )
    if m.n_parse_failures > 0:
        r = r + parse_failure_penalty
    return float(r)


class RLVRTrainer:
    """Reference GRPO-lite trainer.

    Heavy imports (torch, transformers) are done lazily in `.train()` so
    the class is importable in environments without a GPU.
    """

    def __init__(self, cfg: RLVRConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._policy = None
        self._ref = None
        self._tok = None

    # ---- lifecycle --------------------------------------------------

    @staticmethod
    def _load_model_with_fallback(path, dtype, attn_impl):
        """Load model; if *attn_impl* causes an error, retry with sdpa."""
        from transformers import AutoModelForCausalLM

        try:
            return AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=dtype, attn_implementation=attn_impl,
                device_map="auto", trust_remote_code=True,
            )
        except (ImportError, ValueError) as exc:
            if attn_impl == "sdpa":
                raise
            logger.warning("attn_implementation=%s failed (%s); retrying with sdpa",
                           attn_impl, exc)
            return AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=dtype, attn_implementation="sdpa",
                device_map="auto", trust_remote_code=True,
            )

    def _load(self) -> None:
        import torch
        from transformers import AutoTokenizer

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[self.cfg.dtype]

        attn_impl = _detect_attn_implementation(self.cfg.attn_impl)

        logger.info("loading policy from %s", self.cfg.model_path)
        self._tok = AutoTokenizer.from_pretrained(self.cfg.model_path,
                                                  trust_remote_code=True)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token

        self._policy = self._load_model_with_fallback(
            self.cfg.model_path, dtype, attn_impl)
        if self.cfg.gradient_checkpointing:
            self._policy.gradient_checkpointing_enable()

        ref_path = self.cfg.ref_model_path or self.cfg.model_path
        logger.info("loading ref from %s", ref_path)
        self._ref = self._load_model_with_fallback(ref_path, dtype, attn_impl)
        self._ref.eval()
        for p in self._ref.parameters():
            p.requires_grad = False

    # ---- sampling ---------------------------------------------------

    def _sample_completions(self, prompts: list[str]) -> list[list[str]]:
        """For each prompt return K completions."""
        import torch
        K = self.cfg.num_generations_per_prompt
        outputs: list[list[str]] = []
        device = next(self._policy.parameters()).device
        for prompt in prompts:
            enc = self._tok(prompt, return_tensors="pt").to(device)
            batch = enc["input_ids"].expand(K, -1)
            attn = enc["attention_mask"].expand(K, -1)
            with torch.no_grad():
                gen = self._policy.generate(
                    input_ids=batch,
                    attention_mask=attn,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=True,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
                )
            gen_only = gen[:, batch.shape[1]:]
            decoded = self._tok.batch_decode(gen_only, skip_special_tokens=True)
            outputs.append(decoded)
        return outputs

    # ---- losses -----------------------------------------------------

    def _logprobs(self, model, input_ids, attention_mask):
        import torch
        import torch.nn.functional as F
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]
        tgt = input_ids[:, 1:]
        lp = F.log_softmax(logits, dim=-1)
        return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    def _policy_loss(self, logp_new, logp_old, advantages, logp_ref):
        import torch
        ratio = torch.exp(logp_new - logp_old)
        clipped = torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range)
        pg = -torch.minimum(ratio * advantages, clipped * advantages).mean()
        kl = (logp_new - logp_ref).mean()
        return pg + self.cfg.kl_coef * kl, pg.item(), kl.item()

    # ---- loop -------------------------------------------------------

    def _build_full_sequences(self, prompt_strs: list[str],
                               completions: list[str]):
        """Tokenize prompt+completion pairs, returning input_ids and the
        boundary index so we can mask the prompt portion of the loss."""
        import torch

        device = next(self._policy.parameters()).device
        all_ids: list[list[int]] = []
        prompt_lens: list[int] = []

        for prompt, comp in zip(prompt_strs, completions):
            p_ids = self._tok.encode(prompt, add_special_tokens=False)
            c_ids = self._tok.encode(comp, add_special_tokens=False)
            all_ids.append(p_ids + c_ids)
            prompt_lens.append(len(p_ids))

        max_len = max(len(ids) for ids in all_ids)
        pad_id = self._tok.pad_token_id or self._tok.eos_token_id
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]
        attn = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_ids]

        input_ids = torch.tensor(padded, device=device)
        attention_mask = torch.tensor(attn, device=device)
        return input_ids, attention_mask, prompt_lens

    def train(self) -> None:
        import torch
        from torch.optim import AdamW

        if self._policy is None:
            self._load()

        prompts = _load_prompts(self.cfg.prompts_file)
        if not prompts:
            raise RuntimeError(f"no prompts in {self.cfg.prompts_file}")

        opt = AdamW(self._policy.parameters(), lr=self.cfg.learning_rate)

        step = 0
        total_reward_sum = 0.0
        total_reward_count = 0

        for epoch in range(self.cfg.num_epochs):
            for start in range(0, len(prompts), self.cfg.prompt_batch_size):
                if step >= self.cfg.total_steps:
                    break
                batch = prompts[start:start + self.cfg.prompt_batch_size]
                prompt_strs = [b["prompt"] for b in batch]
                groups = self._sample_completions(prompt_strs)

                # ---- score completions and compute advantages ----
                flat_prompt_strs: list[str] = []
                flat_completions: list[str] = []
                flat_advantages: list[float] = []
                flat_rewards: list[float] = []

                for b, comps in zip(batch, groups):
                    task_type = b.get("task_type", "kv")
                    if self.cfg.reward_mode not in ("auto", task_type):
                        task_type = self.cfg.reward_mode
                    rewards = [
                        _score_completion(c, b["gold"], task_type,
                                          self.cfg.parse_failure_penalty)
                        for c in comps
                    ]
                    if self.cfg.reward_normalize == "groupwise":
                        advs = _groupwise_advantages(rewards)
                    else:
                        advs = rewards

                    for c, a, r in zip(comps, advs, rewards):
                        flat_prompt_strs.append(b["prompt"])
                        flat_completions.append(c)
                        flat_advantages.append(a)
                        flat_rewards.append(r)

                if not flat_completions:
                    step += 1
                    continue

                # ---- PPO-clip policy gradient update ----
                device = next(self._policy.parameters()).device
                adv_t = torch.tensor(flat_advantages, device=device,
                                     dtype=torch.float32)

                input_ids, attn_mask, prompt_lens = self._build_full_sequences(
                    flat_prompt_strs, flat_completions)

                # Compute old log-probs (detached) and ref log-probs
                with torch.no_grad():
                    old_lp = self._logprobs(self._policy, input_ids, attn_mask)
                    ref_lp = self._logprobs(self._ref, input_ids, attn_mask)

                # Mask: only compute loss on completion tokens, not prompt
                completion_mask = torch.zeros_like(old_lp)
                for i, plen in enumerate(prompt_lens):
                    # _logprobs shifts by 1, so completion starts at plen-1
                    seq_len = attn_mask[i].sum().item() - 1  # -1 for shift
                    start_idx = max(0, plen - 1)
                    completion_mask[i, start_idx:int(seq_len)] = 1.0

                # New forward pass with gradients
                new_lp = self._logprobs(self._policy, input_ids, attn_mask)

                # Per-token PPO-clip loss, masked to completion only
                ratio = torch.exp(new_lp - old_lp)
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_range,
                                      1 + self.cfg.clip_range)
                # Expand advantages to per-token
                adv_expanded = adv_t.unsqueeze(1).expand_as(ratio)

                pg_loss = -torch.minimum(
                    ratio * adv_expanded,
                    clipped * adv_expanded
                )
                kl = new_lp - ref_lp

                token_loss = pg_loss + self.cfg.kl_coef * kl
                # Apply completion mask
                masked_loss = (token_loss * completion_mask).sum() / \
                              completion_mask.sum().clamp(min=1.0)

                opt.zero_grad()
                masked_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._policy.parameters(), max_norm=1.0)
                opt.step()

                # ---- logging ----
                mean_r = sum(flat_rewards) / max(1, len(flat_rewards))
                total_reward_sum += sum(flat_rewards)
                total_reward_count += len(flat_rewards)

                if step % self.cfg.logging_steps == 0:
                    logger.info(
                        "step=%d  loss=%.4f  pg=%.4f  kl=%.4f  "
                        "mean_reward=%.4f  n=%d",
                        step, masked_loss.item(),
                        pg_loss[completion_mask.bool()].mean().item(),
                        kl[completion_mask.bool()].mean().item(),
                        mean_r, len(flat_rewards),
                    )

                if step > 0 and step % self.cfg.save_steps == 0:
                    save_dir = Path(self.cfg.output_dir) / f"step_{step}"
                    self._policy.save_pretrained(save_dir)
                    self._tok.save_pretrained(save_dir)
                    logger.info("checkpoint saved → %s", save_dir)

                step += 1

        final_dir = Path(self.cfg.output_dir) / "final"
        self._policy.save_pretrained(final_dir)
        self._tok.save_pretrained(final_dir)
        avg_reward = total_reward_sum / max(1, total_reward_count)
        logger.info("RLVR training done → %s (avg_reward=%.4f, steps=%d)",
                    final_dir, avg_reward, step)


def run_rlvr(config_path: str | Path) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = RLVRConfig.from_yaml(config_path)
    RLVRTrainer(cfg).train()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: python -m {__spec__.name if __spec__ else __name__} <config.yaml>")
        sys.exit(1)
    run_rlvr(sys.argv[1])
