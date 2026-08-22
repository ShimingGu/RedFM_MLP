import importlib.util
from pathlib import Path
import unittest

import torch

from aion_magnitude.glm52_posttraining import (
    DEFAULT_GLM52_MODEL,
    GLM52_IA3_FEEDFORWARD_MODULES,
    GLM52_IA3_TARGET_MODULES,
    GLM52_LORA_TARGET_MODULES,
    GLM52_LORA_TARGET_SETTING,
    inspect_glm52_architecture,
)


ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL = Path("/arc/home/gsm/hf_models/GLM-5.2-0.8B-A0.8B")


def load_driver():
    path = ROOT / "notebooks/glm52_posttraining.py"
    spec = importlib.util.spec_from_file_location("glm52_posttraining_driver", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestGlm52Posttraining(unittest.TestCase):
    def test_targets_encode_validated_scope(self):
        self.assertEqual(DEFAULT_GLM52_MODEL, "GLM-5.2-0.8B-A0.8B")
        self.assertIn("kv_b_proj", GLM52_LORA_TARGET_MODULES)
        self.assertIn("weights_proj", GLM52_LORA_TARGET_MODULES)
        self.assertNotIn("gate_proj", GLM52_LORA_TARGET_MODULES)
        self.assertNotIn("up_proj", GLM52_LORA_TARGET_MODULES)
        self.assertNotIn("down_proj", GLM52_LORA_TARGET_MODULES)
        self.assertEqual(GLM52_IA3_TARGET_MODULES, ("kv_b_proj", "down_proj"))
        self.assertEqual(GLM52_IA3_FEEDFORWARD_MODULES, ("down_proj",))
        self.assertTrue(GLM52_LORA_TARGET_SETTING.startswith("linear-suffixes:"))

    def test_glm_cli_aliases(self):
        driver = load_driver()
        args = driver.build_parser().parse_args(
            [
                "--stage",
                "prepare",
                "--glm-model",
                str(LOCAL_MODEL),
                "--glm-max-length",
                "384",
                "--glm-embedding-batch-size",
                "3",
                "--force-recompute-glm",
            ]
        )
        self.assertEqual(args.qwen_model, str(LOCAL_MODEL))
        self.assertEqual(args.qwen_max_length, 384)
        self.assertEqual(args.qwen_embedding_batch_size, 3)
        self.assertTrue(args.force_recompute_qwen)

    def test_launcher_plots_every_method_against_glm_head_only(self):
        launcher = (ROOT / "scripts/glm52-posttraining.sh").read_text()
        self.assertIn(
            '--baseline-result-path "$MODE_OUTPUT/head_only/result.pt"',
            launcher,
        )
        self.assertIn('--baseline-label "GLM-5.2-head-only"', launcher)
        self.assertIn("plot_qwen_posttraining.py", launcher)
        for result_dir in ("qlora", "dora", "ia3", "embedding_adapter", "rlvr"):
            self.assertIn(f'result_dir="{result_dir}"', launcher)

    @unittest.skipUnless(LOCAL_MODEL.is_dir(), "local GLM checkpoint is unavailable")
    def test_local_architecture_and_peft_injection(self):
        from accelerate import init_empty_weights
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoConfig, AutoModel

        report = inspect_glm52_architecture(LOCAL_MODEL)
        self.assertEqual(report["model_type"], "glm_moe_dsa")
        self.assertEqual(report["hidden_size"], 2048)
        self.assertEqual(report["num_hidden_layers"], 6)
        self.assertEqual(report["linear_module_count"], 57)
        self.assertEqual(len(report["routed_expert_parameters"]), 8)
        self.assertFalse(report["routed_experts_trainable_by_peft"])

        config = AutoConfig.from_pretrained(LOCAL_MODEL, local_files_only=True)
        for use_dora in (False, True):
            with init_empty_weights():
                model = AutoModel.from_config(config)
                targets = [
                    name
                    for name, _module in model.named_modules()
                    if any(
                        name == suffix or name.endswith(f".{suffix}")
                        for suffix in GLM52_LORA_TARGET_MODULES
                    )
                ]
                adapted = get_peft_model(
                    model,
                    LoraConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,
                        r=8,
                        lora_alpha=16,
                        target_modules=targets,
                        use_dora=use_dora,
                    ),
                )
            trainable = [
                name
                for name, parameter in adapted.named_parameters()
                if parameter.requires_grad
            ]
            self.assertEqual(len(targets), 39)
            self.assertFalse(any(".experts." in name for name in trainable))
            if use_dora:
                self.assertEqual(
                    sum("lora_magnitude_vector" in name for name in trainable),
                    39,
                )


if __name__ == "__main__":
    unittest.main()


