"""OLMo3-style 4-stage pipeline utilities (mid-train -> SFT -> DPO -> RLVR).

Only the RLVR trainer and preference builder live here. Stages 1 and 2
are pure LLaMA-Factory runs driven by YAML in configs/olmo3_style/.
Stage 3 (DPO) is also LLaMA-Factory-native; this package provides the
preference-pair builder that feeds it.
"""

from .preference_builder import build_preference_pairs
from .rlvr_trainer import RLVRConfig, RLVRTrainer

__all__ = ["build_preference_pairs", "RLVRConfig", "RLVRTrainer"]
