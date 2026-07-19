import unittest
from types import SimpleNamespace

import torch

from aion_magnitude.Inference_Opt_TFM import (
    CatalogueSerializationConfig,
    InferenceOptimizedEmbeddingConfig,
    build_embedding_metadata,
    extract_text_embeddings,
    get_model_spec,
    pool_hidden_states,
    serialize_catalogue_batch,
    serialize_catalogue_row,
)


class TestInferenceOptimizedTransformer(unittest.TestCase):
    def test_registered_model_and_alias(self):
        canonical = get_model_spec("GLM-5.2-0.8B-A0.8B")
        alias = get_model_spec("glm_5_2_0_8b_a0_8b")
        self.assertEqual(canonical, alias)
        self.assertEqual(canonical.hub_id, "inference-optimization/GLM-5.2-0.8B-A0.8B")

    def test_serialization_is_ordered_and_handles_missing_values(self):
        config = CatalogueSerializationConfig(decimals=2)
        text = serialize_catalogue_row({"g_mag": 24.126, "flag": None, "size": float("nan")}, config=config)
        self.assertEqual(text, "catalogue observation; g_mag=24.13; flag=NA; size=NA")

    def test_batch_serialization_validates_width(self):
        self.assertEqual(len(serialize_catalogue_batch(torch.ones(2, 3), ["a", "b", "c"])), 2)
        with self.assertRaises(ValueError):
            serialize_catalogue_batch([[1, 2]], ["a"])

    def test_pooling(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])
        torch.testing.assert_close(pool_hidden_states(hidden, mask, "mean"), torch.tensor([[2.0, 3.0]]))
        torch.testing.assert_close(pool_hidden_states(hidden, mask, "last"), torch.tensor([[3.0, 4.0]]))
        self.assertEqual(pool_hidden_states(hidden, mask, "mean_last").shape, (1, 4))

    def test_extraction_uses_base_hidden_states_without_cache(self):
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.ones((len(texts), 3), dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 0]] * len(texts)),
                }

        class Model:
            config = SimpleNamespace(hidden_size=2)

            def __init__(self):
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                batch = kwargs["input_ids"].shape[0]
                return SimpleNamespace(last_hidden_state=torch.ones((batch, 3, 2)))

        model = Model()
        result = extract_text_embeddings(["one", "two"], Tokenizer(), model, "cpu", batch_size=2)
        self.assertEqual(result.shape, (2, 2))
        self.assertFalse(model.kwargs["use_cache"])
        self.assertTrue(model.kwargs["return_dict"])

    def test_metadata_states_checkpoint_limit(self):
        metadata = build_embedding_metadata(InferenceOptimizedEmbeddingConfig())
        self.assertEqual(metadata["model_role"], "architecture_test_checkpoint")
        self.assertIn("does not reproduce", metadata["capability_warning"])


if __name__ == "__main__":
    unittest.main()
