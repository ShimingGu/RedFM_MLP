import unittest
from unittest.mock import MagicMock, patch

import torch

import aion_magnitude.FM_Qwen as qwen


class TestQwenDeviceAssignment(unittest.TestCase):
    def test_four_bit_loader_uses_requested_visible_device_index(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        model = MagicMock()
        model.parameters.return_value = []
        auto_tokenizer = MagicMock()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        auto_model = MagicMock()
        auto_model.from_pretrained.return_value = model
        quantization_config = MagicMock()

        with (
            patch.object(qwen, "require_transformers"),
            patch.object(qwen, "resolve_torch_device", return_value=torch.device("cuda:2")),
            patch.object(qwen, "AutoTokenizer", auto_tokenizer),
            patch.object(qwen, "AutoModel", auto_model),
            patch.object(qwen, "BitsAndBytesConfig", return_value=quantization_config),
        ):
            qwen.load_frozen_qwen("local-model", device=torch.device("cuda:2"), load_in_4bit=True)

        kwargs = auto_model.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["device_map"], {"": 2})
        self.assertIs(kwargs["quantization_config"], quantization_config)


if __name__ == "__main__":
    unittest.main()
