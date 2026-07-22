import unittest

import torch

from aion_magnitude.FM_Qwen import QwenEmbeddingConfig
from aion_magnitude.FM_Qwen3 import (
    QWEN_PHYSICAL_CONTEXT_MODES,
    Qwen3SerializationConfig,
    qwen3_embedding_metadata,
    serialize_qwen3_observation,
    serialize_tokenized_galaxy_image,
)


class TestQwen3ImageCompactification(unittest.TestCase):
    def test_qwen_embedding_defaults_to_last_pooling(self):
        self.assertEqual(QwenEmbeddingConfig().pooling, "last")

    def test_center_crop_is_spatial_center_not_flat_prefix(self):
        grid = torch.arange(24 * 24).reshape(24, 24)
        config = Qwen3SerializationConfig(
            image_input_mode="center_crop",
            image_crop_size=16,
            include_image_context=False,
        )
        text = serialize_tokenized_galaxy_image(grid, config=config)
        expected_first_row = ",".join(str(value) for value in grid[4, 4:20].tolist())
        expected_last_row = ",".join(str(value) for value in grid[19, 4:20].tolist())
        self.assertIn("source_grid_shape=24x24", text)
        self.assertIn("serialized_grid_shape=16x16", text)
        self.assertIn(f"ordered_token_rows=[{expected_first_row};", text)
        self.assertIn(f";{expected_last_row}]", text)
        self.assertNotIn("ordered_token_rows=[0,1,2", text)

    def test_full_grid_remains_available(self):
        grid = torch.arange(24 * 24).reshape(24, 24)
        config = Qwen3SerializationConfig(
            image_input_mode="full_grid",
            include_image_context=False,
        )
        text = serialize_tokenized_galaxy_image(grid, config=config)
        self.assertIn("serialized_grid_shape=24x24", text)
        self.assertIn("ordered_token_rows=[0,1,2", text)
        self.assertIn(",575]", text)

    def test_final_marker_follows_image(self):
        config = Qwen3SerializationConfig(
            image_input_mode="center_crop",
            image_crop_size=16,
            include_physical_context=False,
            include_image_context=False,
            final_marker="Combined galaxy representation:",
        )
        text = serialize_qwen3_observation(
            {"g_mag": 24.0}, torch.zeros((24, 24), dtype=torch.long), config=config
        )
        self.assertTrue(text.endswith("Combined galaxy representation:"))

    def test_disabling_physical_context_emits_only_raw_magnitude_columns(self):
        config = Qwen3SerializationConfig(
            include_physical_context=False,
            include_image_context=False,
        )
        text = serialize_qwen3_observation({"g_mag": 24.0}, config=config)
        self.assertIn("Magnitude columns: g_mag=24.00000", text)
        self.assertNotIn("instrument=", text)
        self.assertNotIn("passband=", text)
        self.assertNotIn("region=", text)
        self.assertNotIn("observed-frame spectral energy distribution", text)

    def test_global_context_explains_physics_without_band_details(self):
        text = serialize_qwen3_observation(
            {"g_mag": 24.0},
            config=Qwen3SerializationConfig(physical_context_mode="global"),
        )
        self.assertIn("observed-frame spectral energy distribution", text)
        self.assertIn("g AB magnitude=24.00000", text)
        self.assertNotIn("instrument=", text)
        self.assertNotIn("passband=", text)
        self.assertNotIn("region=", text)

    def test_compact_context_keeps_passband_but_omits_instrument_and_notes(self):
        text = serialize_qwen3_observation(
            {"u_star": 25.0},
            config=Qwen3SerializationConfig(physical_context_mode="compact"),
        )
        self.assertIn("u_star AB magnitude=25.00000", text)
        self.assertIn("passband=", text)
        self.assertIn("region=", text)
        self.assertNotIn("instrument=", text)
        self.assertNotIn("red leak", text)

    def test_invalid_physical_context_mode_is_rejected(self):
        self.assertEqual(QWEN_PHYSICAL_CONTEXT_MODES, ("none", "global", "compact", "full"))
        with self.assertRaisesRegex(ValueError, "physical_context_mode"):
            serialize_qwen3_observation(
                {"g": 24.0},
                config=Qwen3SerializationConfig(physical_context_mode="unknown"),
            )

    def test_metadata_disambiguates_crop_and_full_grid(self):
        embedding = QwenEmbeddingConfig(model_path="test-model", pooling="last")
        cropped = qwen3_embedding_metadata(
            embedding,
            Qwen3SerializationConfig(image_input_mode="center_crop", image_crop_size=16),
        )
        full = qwen3_embedding_metadata(
            embedding, Qwen3SerializationConfig(image_input_mode="full_grid")
        )
        self.assertEqual(cropped["qwen_serialized_image_grid_size"], 16)
        self.assertEqual(full["qwen_serialized_image_grid_size"], 24)
        self.assertNotEqual(cropped["qwen_image_input_mode"], full["qwen_image_input_mode"])
        self.assertEqual(full["qwen_physical_context_mode"], "full")

    def test_invalid_center_crop_is_rejected(self):
        grid = torch.zeros((24, 24), dtype=torch.long)
        with self.assertRaises(ValueError):
            serialize_tokenized_galaxy_image(
                grid,
                config=Qwen3SerializationConfig(
                    image_input_mode="center_crop", image_crop_size=15
                ),
            )


if __name__ == "__main__":
    unittest.main()
