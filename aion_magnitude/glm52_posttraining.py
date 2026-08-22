"""Architecture-specific PEFT targets for the GLM-5.2 0.8B frozen mapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .Inference_Opt_TFM import get_model_spec, resolve_model_path

try:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel
except ImportError:
    init_empty_weights = AutoConfig = AutoModel = None


DEFAULT_GLM52_MODEL = "GLM-5.2-0.8B-A0.8B"
GLM52_MODEL_TYPE = "glm_moe_dsa"

# Compatible MLA attention and DSA-indexer linears. PEFT converts GLM MoE MLP
# gate/up targets into packed expert parameters, which DoRA cannot wrap, while
# down_proj suffix targeting also aliases a packed expert parameter. Therefore
# all MLP and routed-expert tensors remain frozen for a common valid scope.
GLM52_LORA_TARGET_MODULES = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "wq_b",
    "wk",
    "weights_proj",
)

# MLA has no independent k_proj/v_proj. Scaling kv_b_proj is the closest
# architecture-correct analogue of IA3 key/value scaling; down_proj receives
# the feed-forward activation and is therefore the IA3 feed-forward target.
GLM52_IA3_TARGET_MODULES = ("kv_b_proj", "down_proj")
GLM52_IA3_FEEDFORWARD_MODULES = ("down_proj",)
GLM52_LORA_TARGET_SETTING = (
    "linear-suffixes:" + ",".join(GLM52_LORA_TARGET_MODULES)
)


def comma_join(names: tuple[str, ...]) -> str:
    return ",".join(names)


def inspect_glm52_architecture(model_path: str | Path = DEFAULT_GLM52_MODEL) -> dict[str, Any]:
    """Validate the local checkpoint and return a serializable adapter report."""
    if AutoConfig is None or AutoModel is None or init_empty_weights is None:
        raise ImportError("GLM architecture inspection requires transformers and accelerate.")
    resolved = resolve_model_path(model_path)
    config = AutoConfig.from_pretrained(
        resolved,
        local_files_only=Path(resolved).expanduser().exists(),
        trust_remote_code=True,
    )
    if config.model_type != GLM52_MODEL_TYPE:
        raise ValueError(
            f"Expected model_type={GLM52_MODEL_TYPE!r}, got {config.model_type!r}: {resolved}"
        )

    with init_empty_weights():
        model = AutoModel.from_config(config, trust_remote_code=True)
    linear_names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    linear_suffixes = sorted({name.rsplit(".", 1)[-1] for name in linear_names})
    missing_lora = sorted(set(GLM52_LORA_TARGET_MODULES) - set(linear_suffixes))
    missing_ia3 = sorted(set(GLM52_IA3_TARGET_MODULES) - set(linear_suffixes))
    if missing_lora or missing_ia3:
        raise RuntimeError(
            "GLM PEFT targets do not match the installed architecture: "
            f"missing_lora={missing_lora}, missing_ia3={missing_ia3}"
        )

    routed_expert_parameters = [
        name for name, parameter in model.named_parameters()
        if ".experts." in name and parameter.ndim == 3
    ]
    if not routed_expert_parameters:
        raise RuntimeError("Expected packed 3-D routed-expert parameters were not found.")

    spec = get_model_spec(model_path)
    return {
        "requested_model": str(model_path),
        "resolved_model_path": str(resolved),
        "model_type": config.model_type,
        "architecture_family": (
            spec.architecture_family if spec is not None else config.model_type
        ),
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "total_parameters_reported": int(getattr(config, "num_params", 0) or 0),
        "linear_module_count": len(linear_names),
        "linear_module_suffixes": linear_suffixes,
        "qlora_dora_target_modules": list(GLM52_LORA_TARGET_MODULES),
        "ia3_target_modules": list(GLM52_IA3_TARGET_MODULES),
        "ia3_feedforward_modules": list(GLM52_IA3_FEEDFORWARD_MODULES),
        "routed_expert_parameters": routed_expert_parameters,
        "routed_experts_trainable_by_peft": False,
        "checkpoint_role": "architecture_test_checkpoint",
        "capability_warning": (
            spec.intended_use_note
            if spec is not None
            else "Capabilities must be established empirically for this checkpoint."
        ),
    }


__all__ = [
    "DEFAULT_GLM52_MODEL",
    "GLM52_IA3_FEEDFORWARD_MODULES",
    "GLM52_IA3_TARGET_MODULES",
    "GLM52_LORA_TARGET_MODULES",
    "GLM52_LORA_TARGET_SETTING",
    "GLM52_MODEL_TYPE",
    "comma_join",
    "inspect_glm52_architecture",
]


